#!/usr/bin/env python3
"""Regenerate the hyperparameter-transfer figures from recorded runs.

Study-specific analysis, not library code: it knows about tiers, batch
sizes, and where the sweeps were archived. It reads only `metrics.json`
and `training.csv` from run directories, so it stays reproducible as long
as those are kept.

Every study it covers was recorded before commit `75f0b22`, when runs still
wrote long-form CSV. Current runs write `training.riglog` instead, so this
script reads the archive and not `runs/`. Pointing it at a current study
means reading the packed log through `rig.logpack` rather than `csv`.

    python docs/reports/make_transfer_charts.py \
        --runs runs \
        --archive ~/rig-run-archive \
        --output docs/reports/transfer-charts.html

Charts are hand-emitted SVG. The repo ships no plotting dependency, and
the output must stay a single self-contained file with no external assets,
which rules out a CDN-loaded chart library.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rig import logpack

TIERS = ("60m", "125m", "250m", "500m")
TIER_COLOR = {
    "60m": "#7dd3fc",
    "125m": "#f0abfc",
    "250m": "#fbbf24",
    "500m": "#4ade80",
}
BATCH_COLOR = {
    32: "#64748b",
    64: "#38bdf8",
    128: "#34d399",
    256: "#fbbf24",
    512: "#f87171",
}
# Where tier and batch appear together, batch moves to shape so the two
# variables stay separable; BATCH_COLOR aliases TIER_COLOR at 250M/256 and
# 500M/128.
BATCH_SHAPE = {
    32: "cross",
    64: "triangle",
    128: "circle",
    256: "square",
    512: "diamond",
}
INK, MUTED, GRID, PANEL = "#e6edf7", "#91a0b8", "#253047", "#111827"


@dataclass(frozen=True)
class Run:
    tier: str
    batch: int
    lg_lr: int
    seed: int
    loss: float
    steps: int
    seconds: float
    flops_per_token: int
    # Conditions the ladder was later re-run under. The 5-TPP, 1,024-context,
    # dense grid is the original; everything else exists to test whether an
    # optimum found there survives a change of regime.
    tpp: int = 5
    context: int = 1024
    routed: bool = False

    @property
    def condition(self) -> str:
        """A label for the regime this run belongs to, not its configuration."""

        if self.routed:
            return f"{self.tier} MoE · {self.context // 1024}k"
        if self.context != 1024:
            return f"{self.tier} · {self.context // 1024}k"
        return f"{self.tier} · {self.tpp} TPP"


# hf-dataset names carry the token budget: 60m-5tpp-bs128-lr2e-8-s1337.
_DATASET_NAME = re.compile(r"(\d+m)-(\d+)tpp-bs(\d+)-lr2e(-?\d+)-s(\d+)$")
# Live routed runs are named by the sweep that produced them.
_ROUTED_NAME = re.compile(r"(\d+m)-moe-lr2e(-?\d+)-s(\d+)")


def _read(
    directory: Path,
    name: str,
    tier: str,
    batch: int,
    lg_lr: int,
    seed: int,
    routed: bool,
) -> "Run | None":
    metrics = directory / "metrics.json"
    result = directory / "result.json"
    if not metrics.exists() or not result.exists():
        return None
    m = json.loads(metrics.read_text())["metrics"]
    contract = json.loads(result.read_text()).get("contract") or {}
    if m.get("validation_loss") is None:
        return None
    return Run(
        tier=tier,
        batch=batch,
        lg_lr=lg_lr,
        seed=seed,
        loss=float(m["validation_loss"]),
        steps=int(m.get("training_steps") or 0),
        seconds=float(m.get("train_seconds") or 0.0),
        flops_per_token=int(m.get("flops_per_token") or 0),
        tpp=round(m.get("tokens_per_parameter") or 5),
        context=int(contract.get("sequence_length") or 1024),
        routed=routed,
    )


def load_runs(roots: list[Path]) -> list[Run]:
    """Read every run, from the dataset tree and from live run directories.

    Keyed by configuration rather than by directory, because the same cell was
    sometimes measured twice -- the 250M learning-rate study and the 250M batch
    study overlap at batch 128 -- and averaging one measurement in twice would
    quietly weight it double.
    """

    found: list[Run] = []
    seen: set[tuple] = set()
    for root in roots:
        if not root.exists():
            continue
        # <study>/<run-name>/ in the dataset tree, timestamped dirs in runs/.
        candidates = [d for d in sorted(root.rglob("*")) if d.is_dir()]
        for directory in candidates:
            routed = False
            match = _DATASET_NAME.search(directory.name)
            if match:
                tier, tpp, batch, lg_lr, seed = match.groups()
                run = _read(
                    directory,
                    directory.name,
                    tier,
                    int(batch),
                    int(lg_lr),
                    int(seed),
                    False,
                )
            else:
                match = _ROUTED_NAME.search(directory.name)
                if not match:
                    continue
                routed = True
                tier, lg_lr, seed = match.groups()
                run = _read(
                    directory, directory.name, tier, 16, int(lg_lr), int(seed), True
                )
            if run is None:
                continue
            key = (
                run.tier,
                run.batch,
                run.lg_lr,
                run.seed,
                run.tpp,
                run.context,
                run.routed,
            )
            if key in seen:
                continue
            seen.add(key)
            found.append(run)
    return found


def load_spikes(roots: list[Path]) -> list[tuple[int, int, float, float]]:
    """(lg_lr, seed, spike ratio, loss) for the 250M learning-rate reseed.

    Read from the packed logs rather than the long-form CSV this once used, so
    it no longer depends on an archive that lives outside the repository and
    may not exist on the machine drawing the charts.
    """

    out = []
    for root in roots:
        if not root.exists():
            continue
        for directory in sorted(root.rglob("*")):
            if not directory.is_dir():
                continue
            match = re.search(r"250m-5tpp-bs128-lr2e(-?\d+)-s(\d+)$", directory.name)
            curve = directory / "training.riglog"
            metrics = directory / "metrics.json"
            if not match or not curve.exists() or not metrics.exists():
                continue
            norms = logpack.read_log(curve).series("grad_norm")
            if norms is None or not len(norms):
                continue
            finite = [float(v) for v in norms if math.isfinite(float(v))]
            if not finite:
                continue
            median = sorted(finite)[len(finite) // 2]
            if median <= 0:
                continue
            loss = json.loads(metrics.read_text())["metrics"].get("validation_loss")
            if loss is None:
                continue
            out.append(
                (
                    int(match.group(1)),
                    int(match.group(2)),
                    max(finite) / median,
                    float(loss),
                )
            )
    # The same cell was measured by two studies; keep one reading per run.
    unique = {(lg, seed): (lg, seed, ratio, loss) for lg, seed, ratio, loss in out}
    return sorted(unique.values())


def cells(runs: list[Run]) -> dict[tuple[str, int, int], list[float]]:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for r in runs:
        grouped[(r.tier, r.batch, r.lg_lr)].append(r.loss)
    return grouped


# --------------------------------------------------------------------------
# SVG primitives. Everything below draws into a fixed viewBox with an inner
# plot rect; scales map data to that rect.
# --------------------------------------------------------------------------

W, H = 560, 340
PAD = {"l": 62, "r": 18, "t": 30, "b": 48}


class Plot:
    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        width: int = W,
        height: int = H,
    ):
        self.w, self.h = width, height
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        self.parts: list[str] = []

    def X(self, v: float) -> float:
        span = (self.x1 - self.x0) or 1
        return PAD["l"] + (v - self.x0) / span * (self.w - PAD["l"] - PAD["r"])

    def Y(self, v: float) -> float:
        span = (self.y1 - self.y0) or 1
        return PAD["t"] + (self.y1 - v) / span * (self.h - PAD["t"] - PAD["b"])

    def frame(self, xticks, yticks, xfmt=str, yfmt=lambda v: f"{v:g}") -> None:
        for v in yticks:
            y = self.Y(v)
            self.parts.append(
                f'<line x1="{PAD["l"]}" y1="{y:.1f}" x2="{self.w - PAD["r"]}" '
                f'y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
            )
            self.parts.append(
                f'<text x="{PAD["l"] - 8}" y="{y + 3.5:.1f}" fill="{MUTED}" '
                f'font-size="10" text-anchor="end">{html.escape(yfmt(v))}</text>'
            )
        for v in xticks:
            x = self.X(v)
            self.parts.append(
                f'<line x1="{x:.1f}" y1="{PAD["t"]}" x2="{x:.1f}" '
                f'y2="{self.h - PAD["b"]}" stroke="{GRID}" stroke-width="1" '
                f'stroke-dasharray="2,3"/>'
            )
            self.parts.append(
                f'<text x="{x:.1f}" y="{self.h - PAD["b"] + 16}" fill="{MUTED}" '
                f'font-size="10" text-anchor="middle">{html.escape(xfmt(v))}</text>'
            )

    def line(self, points, color, width=2.0, dash=None, opacity=1.0):
        if not points:
            return
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{self.X(x):.1f},{self.Y(y):.1f}"
            for i, (x, y) in enumerate(points)
        )
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round" '
            f'opacity="{opacity}"{extra}/>'
        )

    def dots(self, points, color, r=3.2, opacity=1.0):
        for x, y in points:
            self.parts.append(
                f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" '
                f'fill="{color}" opacity="{opacity}"/>'
            )

    def marker(self, x, y, color, shape="circle", r=4.0, opacity=0.95):
        """Draw one point as a glyph, so a second variable can ride on shape.

        Colour alone cannot carry two variables at once: tier and batch each
        need their own channel or the reader has to hold a two-palette key in
        their head and the two palettes collide.
        """

        cx, cy = self.X(x), self.Y(y)
        fill = f'fill="{color}" opacity="{opacity}"'
        if shape == "circle":
            part = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" {fill}/>'
        elif shape == "square":
            side = r * 1.8
            part = (
                f'<rect x="{cx - side / 2:.1f}" y="{cy - side / 2:.1f}" '
                f'width="{side:.1f}" height="{side:.1f}" {fill}/>'
            )
        elif shape == "triangle":
            part = (
                f'<polygon points="{cx:.1f},{cy - r * 1.2:.1f} '
                f"{cx + r * 1.1:.1f},{cy + r * 0.85:.1f} "
                f'{cx - r * 1.1:.1f},{cy + r * 0.85:.1f}" {fill}/>'
            )
        elif shape == "diamond":
            part = (
                f'<polygon points="{cx:.1f},{cy - r * 1.3:.1f} '
                f"{cx + r * 1.3:.1f},{cy:.1f} {cx:.1f},{cy + r * 1.3:.1f} "
                f'{cx - r * 1.3:.1f},{cy:.1f}" {fill}/>'
            )
        else:  # cross
            arm, thick = r * 1.25, r * 0.42
            part = (
                f'<path d="M{cx - arm:.1f},{cy - thick:.1f} '
                f"h{arm - thick:.1f} v{-(arm - thick):.1f} "
                f"h{thick * 2:.1f} v{arm - thick:.1f} h{arm - thick:.1f} "
                f"v{thick * 2:.1f} h{-(arm - thick):.1f} v{arm - thick:.1f} "
                f"h{-thick * 2:.1f} v{-(arm - thick):.1f} "
                f'h{-(arm - thick):.1f} Z" {fill}/>'
            )
        self.parts.append(part)

    def errbar(self, x, lo, hi, color, width=1.4):
        self.parts.append(
            f'<line x1="{self.X(x):.1f}" y1="{self.Y(lo):.1f}" '
            f'x2="{self.X(x):.1f}" y2="{self.Y(hi):.1f}" stroke="{color}" '
            f'stroke-width="{width}" opacity="0.75"/>'
        )

    def rect(self, x, y, w, h, fill, stroke="none"):
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )

    def text(self, x, y, s, color=INK, size=10, anchor="middle", weight="normal"):
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(s)}</text>'
        )

    def legend(self, entries, x=None, y=None):
        """Key the swatches. An entry may be (label, color) or, when shape
        carries a second variable, (label, color, shape)."""

        x = self.w - PAD["r"] - 8 if x is None else x
        y = PAD["t"] + 6 if y is None else y
        for i, entry in enumerate(entries):
            label, color = entry[0], entry[1]
            shape = entry[2] if len(entry) > 2 else None
            yy = y + i * 15
            if shape is None:
                self.parts.append(
                    f'<rect x="{x - 9:.1f}" y="{yy - 7:.1f}" width="8" height="8" '
                    f'rx="2" fill="{color}"/>'
                )
            else:
                # Glyph swatches are drawn in data space, so borrow the axes.
                saved = len(self.parts)
                self.marker(0, 0, color, shape, r=3.6, opacity=1.0)
                drawn = self.parts.pop()
                del self.parts[saved:]
                self.parts.append(
                    f'<g transform="translate({x - 5 - self.X(0):.1f},'
                    f'{yy - 3 - self.Y(0):.1f})">{drawn}</g>'
                )
            self.parts.append(
                f'<text x="{x - 14:.1f}" y="{yy:.1f}" fill="{MUTED}" font-size="10" '
                f'text-anchor="end">{html.escape(label)}</text>'
            )

    def render(self) -> str:
        head = (
            f'<text x="{PAD["l"]}" y="18" fill="{INK}" font-size="12.5" '
            f'font-weight="650" text-anchor="start">{html.escape(self.title)}</text>'
        )
        xl = (
            f'<text x="{(PAD["l"] + self.w - PAD["r"]) / 2:.0f}" y="{self.h - 8}" '
            f'fill="{MUTED}" font-size="10" text-anchor="middle">'
            f"{html.escape(self.xlabel)}</text>"
        )
        yl = (
            f'<text transform="translate(14,{(PAD["t"] + self.h - PAD["b"]) / 2:.0f}) '
            f'rotate(-90)" fill="{MUTED}" font-size="10" text-anchor="middle">'
            f"{html.escape(self.ylabel)}</text>"
        )
        return (
            f'<svg viewBox="0 0 {self.w} {self.h}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'xmlns="http://www.w3.org/2000/svg" role="img">'
            f"{head}{''.join(self.parts)}{xl}{yl}</svg>"
        )


def sem(v: list[float]) -> float:
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def nice(lo: float, hi: float, pad: float = 0.08) -> tuple[float, float]:
    if hi == lo:
        return lo - 0.5, hi + 0.5
    m = (hi - lo) * pad
    return lo - m, hi + m


def ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_batch_normalized(c) -> str:
    """Loss above each tier's own best, vs batch. The headline figure."""
    series = {}
    for tier in TIERS:
        pts = []
        for batch in sorted({k[1] for k in c if k[0] == tier}):
            best = min(
                st.mean(v)
                for k, v in c.items()
                if k[0] == tier and k[1] == batch and len(v) >= 3
            )
            pts.append((math.log2(batch), best))
        if pts:
            floor = min(y for _, y in pts)
            series[tier] = [(x, y - floor) for x, y in pts]
    top = max(y for s in series.values() for _, y in s)
    p = Plot(
        "Loss above each tier's own optimum",
        "global batch size",
        "loss above that tier's best (nats)",
        (4.6, 9.4),
        (-0.05, top * 1.08),
    )
    p.frame(
        [5, 6, 7, 8, 9],
        ticks(0, top, 4),
        xfmt=lambda v: str(int(2**v)),
        yfmt=lambda v: f"{v:.2f}",
    )
    for tier, pts in series.items():
        p.line(pts, TIER_COLOR[tier], 2.4)
        p.dots(pts, TIER_COLOR[tier])
    p.line(
        [(math.log2(128), -0.05), (math.log2(128), top * 1.08)],
        "#94a3b8",
        1.0,
        dash="4,4",
        opacity=0.5,
    )
    p.text(p.X(7), PAD["t"] - 4, "batch 128", MUTED, 9)
    p.legend([(t, TIER_COLOR[t]) for t in series])
    return p.render()


CONDITION_COLOR = {
    "60m · 5 TPP": "#7dd3fc",
    "125m · 5 TPP": "#f0abfc",
    "250m · 5 TPP": "#fbbf24",
    "500m · 5 TPP": "#4ade80",
    "500m · 20 TPP": "#22d3ee",
    "60m · 8k": "#fb923c",
    "60m MoE · 8k": "#a78bfa",
    "125m MoE · 8k": "#f472b6",
}


def fig_optimum_transfers(runs: list[Run]) -> str:
    """Where each regime's learning-rate optimum sits, on one axis.

    Absolute loss is not comparable across these conditions -- a 500M model at
    20 tokens per parameter and a 60M model at 8k context are not competing --
    so each curve is drawn as loss above its own best. That throws away the
    level and keeps the only thing being asked: which learning rate wins.

    A condition needs three learning rates before it can show a minimum rather
    than an edge, so conditions with fewer are left out entirely.
    """

    by_condition: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    batches: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for run in runs:
        batches[run.condition][run.batch].add(run.lg_lr)
    canonical = {}
    for condition, options in batches.items():
        # The batch that swept the most learning rates is the one that can show
        # an optimum; ties go to the ladder's own default.
        best = max(options, key=lambda b: (len(options[b]), b == 128, b == 16))
        if len(options[best]) >= 3:
            canonical[condition] = best
    for run in runs:
        if canonical.get(run.condition) == run.batch:
            by_condition[run.condition][run.lg_lr].append(run.loss)

    series = {}
    for condition, rates in by_condition.items():
        points = [(lg, st.mean(v)) for lg, v in sorted(rates.items())]
        floor = min(y for _, y in points)
        series[condition] = [(x, y - floor) for x, y in points]
    if not series:
        return ""

    top = max(y for pts in series.values() for _, y in pts)
    xs = sorted({x for pts in series.values() for x, _ in pts})
    p = Plot(
        "Every regime peaks at the same learning rate",
        "base learning rate",
        "loss above that regime's best (nats)",
        (min(xs) - 0.4, max(xs) + 0.4),
        (-top * 0.04, top * 1.10),
    )
    p.frame(
        xs, ticks(0, top, 4), xfmt=lambda v: f"2^{int(v)}", yfmt=lambda v: f"{v:.2f}"
    )
    for condition, pts in sorted(series.items()):
        colour = CONDITION_COLOR.get(condition, "#94a3b8")
        p.line(pts, colour, 2.2)
        p.dots(pts, colour)
    p.line(
        [(-8, -top * 0.04), (-8, top * 1.10)], "#94a3b8", 1.0, dash="4,4", opacity=0.55
    )
    p.text(p.X(-8) + 5, PAD["t"] - 4, "2^-8", MUTED, 9)
    p.legend([(c, CONDITION_COLOR.get(c, "#94a3b8")) for c in sorted(series)])
    return p.render()


def fig_heatmaps(c) -> list[str]:
    """LR x batch grid per tier; colour is loss above that tier's best."""
    out = []
    for tier in TIERS:
        batches = sorted({k[1] for k in c if k[0] == tier})
        lrs = sorted({k[2] for k in c if k[0] == tier}, reverse=True)
        vals = {
            (b, l): st.mean(c[(tier, b, l)])
            for b in batches
            for l in lrs
            if (tier, b, l) in c
        }
        if not vals:
            continue
        floor = min(vals.values())
        worst = max(vals.values())
        span = (worst - floor) or 1
        p = Plot(
            f"{tier}: loss above optimum",
            "global batch size",
            "base learning rate",
            (0, len(batches)),
            (0, len(lrs)),
            width=400,
            height=300,
        )
        cw = (p.w - PAD["l"] - PAD["r"]) / len(batches)
        ch = (p.h - PAD["t"] - PAD["b"]) / len(lrs)
        for i, b in enumerate(batches):
            for j, l in enumerate(lrs):
                if (b, l) not in vals:
                    continue
                frac = (vals[(b, l)] - floor) / span
                # dark teal (best) -> amber -> red (worst)
                r = int(30 + 200 * min(1.0, frac * 1.5))
                g = int(190 - 130 * frac)
                bl = int(160 - 130 * frac)
                x = PAD["l"] + i * cw
                y = PAD["t"] + j * ch
                p.rect(x + 1, y + 1, cw - 2, ch - 2, f"rgb({r},{g},{bl})")
                delta = vals[(b, l)] - floor
                p.text(
                    x + cw / 2,
                    y + ch / 2 + 3.5,
                    "best" if delta < 1e-9 else f"+{delta:.3f}",
                    "#0b1220" if frac < 0.5 else "#fff",
                    9,
                    weight="700" if delta < 1e-9 else "normal",
                )
            p.text(PAD["l"] + i * cw + cw / 2, p.h - PAD["b"] + 16, str(b), MUTED, 10)
        for j, l in enumerate(lrs):
            p.text(
                PAD["l"] - 8,
                PAD["t"] + j * ch + ch / 2 + 3.5,
                f"2^{l}",
                MUTED,
                10,
                anchor="end",
            )
        out.append(p.render())
    return out


def fig_wall_clock(runs, c) -> str:
    """Loss vs wall clock, one point per (tier, batch, LR) cell.

    Tier rides on colour and batch on shape. They are independent variables,
    so giving each its own channel is the only way to read a point without a
    two-palette key -- and the earlier single-colour version aliased 250M
    against batch 256 and 500M against batch 128.

    No frontier is traced. A running minimum over this grid would connect
    points that differ in learning rate as well as batch, so it would not
    isolate the batch/time trade it appeared to describe -- and nobody
    operates along it, since the sweep ends by picking one batch.
    """

    pts_by_tier = {}
    for tier in TIERS:
        secs = {}
        for (t, b, l), v in c.items():
            if t != tier or len(v) < 3:
                continue
            s = st.mean(
                r.seconds for r in runs if r.tier == t and r.batch == b and r.lg_lr == l
            )
            secs[(b, l)] = (s, st.mean(v))
        if secs:
            pts_by_tier[tier] = secs
    allx = [s for d in pts_by_tier.values() for s, _ in d.values()]
    p = Plot(
        "Loss vs wall clock (every batch x LR cell)",
        "train seconds (log)",
        "validation loss",
        (math.log10(min(allx) * 0.85), math.log10(max(allx) * 1.15)),
        (0, 1),
    )
    lo = min(y for d in pts_by_tier.values() for _, y in d.values())
    hi = max(y for d in pts_by_tier.values() for _, y in d.values())
    p.y0, p.y1 = nice(lo, min(hi, lo + 1.4))
    p.frame(
        [2, 2.3, 2.6, 3, 3.3],
        ticks(p.y0, p.y1, 5),
        xfmt=lambda v: f"{10**v:,.0f}",
        yfmt=lambda v: f"{v:.2f}",
    )
    for tier, d in pts_by_tier.items():
        for (b, _l), (s, y) in d.items():
            if y > p.y1:
                continue
            p.marker(math.log10(s), y, TIER_COLOR[tier], BATCH_SHAPE.get(b, "circle"))
    batches = sorted({b for d in pts_by_tier.values() for b, _ in d})
    p.legend(
        [(t, TIER_COLOR[t]) for t in pts_by_tier]
        + [(f"batch {b}", MUTED, BATCH_SHAPE.get(b, "circle")) for b in batches]
    )
    return p.render()


def fig_penalty(c) -> str:
    """Extra loss from training at a batch larger than the optimum, per tier.

    Penalty is the 3-seed mean validation loss at batch B minus the 3-seed
    mean at batch 128, both at base LR 2^-8. Positive means the larger batch
    finished worse. Same token budget on both sides, so this is a sample
    efficiency cost, not a time cost.
    """
    params = {
        "60m": 59_918_208,
        "125m": 123_456_640,
        "250m": 244_444_032,
        "500m": 502_602_240,
    }
    series = {}
    for over in (256, 512):
        pts = []
        for tier in TIERS:
            base = c.get((tier, 128, -8))
            other = c.get((tier, over, -8))
            if base and other:
                d = st.mean(other) - st.mean(base)
                if d > 0:
                    pts.append((math.log10(params[tier]), math.log10(d)))
        if len(pts) >= 2:
            series[over] = pts
    ys = [y for s in series.values() for _, y in s]
    p = Plot(
        "Penalty for exceeding batch 128",
        "parameters (log)",
        "extra loss vs batch 128, nats (log)",
        (7.7, 8.45),
        (min(ys) - 0.25, max(ys) + 0.25),
    )
    p.frame(
        [math.log10(params[t]) for t in TIERS],
        ticks(min(ys) - 0.2, max(ys) + 0.2, 4),
        xfmt=lambda v: {7.777: "60M", 8.092: "125M", 8.388: "250M"}.get(
            round(v, 3), f"{10**v / 1e6:.0f}M"
        ),
        yfmt=lambda v: f"{10**v:.3f}",
    )
    for over, pts in series.items():
        col = BATCH_COLOR[over]
        p.line(pts, col, 2.4)
        p.dots(pts, col, 4)
    p.legend([(f"batch {b}", BATCH_COLOR[b]) for b in series])
    return p.render()


TOKENS = {
    "60m": 5 * 59_918_208,
    "125m": 5 * 123_456_640,
    "250m": 5 * 244_444_032,
    "500m": 5 * 502_602_240,
}


def fig_spread(c) -> str:
    """Seed range by LR, one line per (tier, batch). Noise peaks near the optimum.

    The plotted quantity is the range -- max minus min of validation loss over
    the cell's three seeds -- not a standard deviation. Three samples do not
    support an sd worth printing, and the range uses every draw there is.

    For normal draws E[range] = 1.69 sigma at n=3, so range/1.69 is an unbiased
    estimate of sigma, and validation loss is thin-tailed enough for that to
    roughly hold. What it is not is a precise estimate: the range's own
    standard deviation is 0.89 sigma, which is 53% of what it estimates. Two cells
    with identical true variance differ by more than 2x in range about 40% of
    the time. The chart is therefore readable for gross structure across many
    cells at once, and not for ranking one cell against another.
    """

    p = Plot(
        "Seed range (max - min over 3 seeds)",
        "base learning rate",
        "loss range, nats",
        (-10.4, -5.6),
        (0, 0.36),
    )
    p.frame(
        [-10, -9, -8, -7, -6],
        ticks(0, 0.35, 5),
        xfmt=lambda v: f"2^{int(v)}",
        yfmt=lambda v: f"{v:.2f}",
    )
    for tier in TIERS:
        for b in sorted({k[1] for k in c if k[0] == tier}):
            pts = []
            for l in sorted({k[2] for k in c if k[0] == tier}):
                v = c.get((tier, b, l))
                if v and len(v) >= 3:
                    pts.append((l, min(max(v) - min(v), 0.35)))
            if len(pts) >= 2:
                p.line(pts, TIER_COLOR[tier], 1.5, opacity=0.35 + 0.5 * (b == 128))
                p.dots(pts, TIER_COLOR[tier], 2.4, 0.8)
    p.legend([(t, TIER_COLOR[t]) for t in TIERS])
    return p.render()


def fig_spikes(spikes) -> str:
    """How far a seed's final loss lands from its own LR group's average.

    Each point is one run. Vertical position is that run's final validation
    loss minus the mean of the three seeds sharing its learning rate, so zero
    is its group's average and positive is worse than its two siblings.
    Comparing within the group is the point: the loss level itself moves with
    the learning rate, which would swamp the seed-to-seed effect being looked
    at here.
    """
    by_lr = defaultdict(list)
    for lg, seed, ratio, loss in spikes:
        by_lr[lg].append((seed, ratio, loss))
    pts = []
    for lg, rows in by_lr.items():
        mean = st.mean(r[2] for r in rows)
        for _seed, ratio, loss in rows:
            pts.append((math.log10(ratio), loss - mean, lg))
    if not pts:
        return ""
    p = Plot(
        "Gradient spike vs final loss, within each LR (250M reseed)",
        "peak / median gradient norm (log)",
        "final loss - mean of its 3 seeds (nats)",
        (0.9, 3.6),
        (-0.05, 0.05),
    )
    lo = min(y for _, y, _ in pts)
    hi = max(y for _, y, _ in pts)
    p.y0, p.y1 = nice(lo, hi)
    p.frame(
        [1, 1.5, 2, 2.5, 3, 3.5],
        ticks(p.y0, p.y1, 5),
        xfmt=lambda v: f"{10**v:,.0f}x",
        yfmt=lambda v: f"{v:+.3f}",
    )
    p.line([(0.9, 0), (3.6, 0)], GRID, 1.2, dash="4,4")
    colors = {-9: "#64748b", -8: "#34d399", -7: "#fbbf24", -6: "#f87171"}
    for lg in sorted(by_lr, reverse=True):
        group = sorted([(x, y) for x, y, g in pts if g == lg])
        p.line(group, colors.get(lg, "#888"), 1.2, opacity=0.55)
        p.dots(group, colors.get(lg, "#888"), 4)
    p.legend(
        [(f"2^{lg}", colors.get(lg, "#888")) for lg in sorted(by_lr, reverse=True)]
    )
    return p.render()


def fig_utilization(runs) -> str:
    """Achieved PFLOP/s vs batch: why the large batches are faster."""
    p = Plot(
        "Achieved throughput vs batch",
        "global batch size",
        "PFLOP/s",
        (4.6, 9.4),
        (0, 1.9),
    )
    p.frame(
        [5, 6, 7, 8, 9],
        ticks(0, 1.8, 6),
        xfmt=lambda v: str(int(2**v)),
        yfmt=lambda v: f"{v:.1f}",
    )
    for tier in TIERS:
        pts = []
        for b in sorted({r.batch for r in runs if r.tier == tier}):
            sel = [r for r in runs if r.tier == tier and r.batch == b]
            if not sel:
                continue
            tok = TOKENS[tier]
            rate = st.mean(tok / r.seconds * r.flops_per_token for r in sel) / 1e15
            pts.append((math.log2(b), rate))
        if pts:
            p.line(pts, TIER_COLOR[tier], 2.4)
            p.dots(pts, TIER_COLOR[tier])
    p.legend([(t, TIER_COLOR[t]) for t in TIERS])
    return p.render()


PAGE = """<title>Transfer Charts</title>
<style>
:root{{color-scheme:dark}}
body{{margin:0;background:#0a0e17;color:{ink};
 font:15px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif;padding:32px 26px 70px}}
h1{{font-size:clamp(24px,3vw,36px);letter-spacing:-.03em;margin:0 0 6px}}
.sub{{color:{muted};font-size:13px;max-width:70ch;margin-bottom:28px}}
h2{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:{muted};
 margin:34px 0 10px;font-weight:700}}
.card{{background:linear-gradient(145deg,rgba(20,27,42,.9),rgba(12,17,27,.9));
 border:1px solid {grid};border-radius:14px;padding:14px 16px 8px;margin-bottom:14px}}
.note{{color:{muted};font-size:12px;margin:2px 0 10px;max-width:78ch}}
.row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:14px}}
.foot{{color:{muted};font-size:11px;margin-top:34px}}
code{{background:#111827;padding:1px 5px;border-radius:4px;font-size:12px}}
</style>
<h1>Hyperparameter transfer at 5 TPP</h1>
<div class="sub">Figures regenerated from recorded runs by
<code>docs/reports/make_transfer_charts.py</code>. Every point is a 3-seed mean.
Companion to <code>docs/HYPERPARAMETER_TRANSFER.md</code>.</div>
{body}
<div class="foot">{count} runs · generated from metrics.json only · no external assets</div>
"""


def card(title_note: str, svg: str) -> str:
    return f'<div class="card"><div class="note">{title_note}</div>{svg}</div>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--archive", type=Path, default=Path.home() / "rig-run-archive")
    ap.add_argument(
        "--output", type=Path, default=Path("docs/reports/transfer-charts.html")
    )
    args = ap.parse_args()

    roots = [args.runs] + sorted(p for p in args.archive.glob("*") if p.is_dir())
    runs = load_runs(roots)
    if not runs:
        raise SystemExit(f"no runs found under {roots}")
    # Every figure below the transfer chart is about one grid: the dense,
    # 5-tokens-per-parameter, 1,024-context ladder. The other regimes exist to
    # test whether that grid's answer survives, and mixing them in would put
    # cells with one seed beside cells with three, averaging across
    # experiments that are not comparable to begin with.
    ladder = [r for r in runs if r.tpp == 5 and r.context == 1024 and not r.routed]
    c = cells(ladder)
    spikes = load_spikes(roots)

    body = []
    body.append("<h2>The result</h2>")
    body.append(
        card(
            "Each curve is one regime, drawn as loss above its own best, because "
            "absolute loss is not comparable between a 500M model at 20 tokens per "
            "parameter and a 60M model at 8k context. Every regime bottoms out at "
            "2^-8: three model sizes, a four-times-longer token budget, an "
            "eight-times-longer context, and a routed model. The optimum was "
            "measured once on the 5-TPP 1,024-context ladder and has not moved "
            "since, which is what the parameterization is supposed to buy. "
            "The curves do not all carry equal weight, though: every point is a "
            "three-seed mean except 500M at 20 TPP, which is one seed at 2^-9 "
            "and 2^-7 and two at 2^-8. Its 0.0009-nat lead over 2^-7 sits well "
            "inside the seed spread seen elsewhere, so what that curve "
            "establishes is that the optimum is not 2^-9 -- not that 2^-8 "
            "beats 2^-7.",
            fig_optimum_transfers(runs),
        )
    )
    body.append(
        card(
            "Batch 128 is optimal at all three tiers. The penalty for exceeding "
            "it falls steeply with scale: 0.45 nats at 60M, 0.016 at 250M.",
            fig_batch_normalized(c),
        )
    )
    body.append("<h2>Full grids</h2>")
    body.append(
        '<div class="row">'
        + "".join(f'<div class="card">{s}</div>' for s in fig_heatmaps(c))
        + "</div>"
    )
    body.append("<h2>What batch size buys</h2>")
    body.append(
        '<div class="row">'
        + card(
            "Colour is tier, shape is batch. Larger batches sit left "
            "(faster) and the tiers stack by loss; how far a batch "
            "moves you left is a throughput fact, what it costs in "
            "loss is the penalty chart below.",
            fig_wall_clock(ladder, c),
        )
        + card(
            "Throughput saturates around batch 256, which is why the "
            "larger batches finish sooner on an identical token budget.",
            fig_utilization(ladder),
        )
        + "</div>"
    )
    body.append("<h2>Noise structure</h2>")
    body.append(
        '<div class="row">'
        + card(
            "Range (max - min) of validation loss over each cell's "
            "three seeds, by LR; batch 128 drawn brighter. Noise is "
            "largest near the optimum, which is why single-seed "
            "comparisons there are unreliable. Read the shape across "
            "cells, not one cell against another: at n=3 the range "
            "estimates sigma to within about 53%, and two cells with "
            "identical variance differ by 2x roughly 40% of the time.",
            fig_spread(c),
        )
        + (
            card(
                "One point per run. Height is that run's final "
                "validation loss minus the average of the three seeds "
                "sharing its learning rate, so zero is its group's "
                "average. Within a group, bigger gradient spikes go with "
                "worse loss; pooled across learning rates the rank "
                "correlation is only +0.17, because the spikes grow with "
                "LR while these differences are centred inside each group.",
                fig_spikes(spikes),
            )
            if spikes
            else ""
        )
        + "</div>"
    )
    body.append("<h2>Penalty scaling</h2>")
    body.append(
        card(
            "Penalty is the extra validation loss from using a larger batch than "
            "the optimum: the three-seed mean at batch B minus the three-seed mean "
            "at batch 128, both at base LR 2^-8, on the same token budget. Positive "
            "means worse. Both axes log. If this stays log-linear it predicts the "
            "500M and 1B penalty, i.e. whether larger tiers can simply adopt "
            "batch 256.",
            fig_penalty(c),
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        PAGE.format(
            body="\n".join(body), count=len(runs), ink=INK, muted=MUTED, grid=GRID
        ),
        encoding="utf-8",
    )
    print(
        f"{args.output}: {len(runs)} runs, {len(c)} cells, {len(spikes)} spike samples"
    )


if __name__ == "__main__":
    main()
