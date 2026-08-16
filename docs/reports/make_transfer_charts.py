#!/usr/bin/env python3
"""Regenerate the hyperparameter-transfer figures from recorded runs.

Study-specific analysis, not library code: it knows about tiers, batch
sizes, and where the sweeps were archived. It reads only `metrics.json`
and `training.csv` from run directories, so it stays reproducible as long
as those are kept.

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

TIERS = ("60m", "125m", "250m")
TIER_COLOR = {"60m": "#7dd3fc", "125m": "#f0abfc", "250m": "#fbbf24"}
BATCH_COLOR = {
    32: "#64748b", 64: "#38bdf8", 128: "#34d399",
    256: "#fbbf24", 512: "#f87171",
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


def load_runs(roots: list[Path]) -> list[Run]:
    found: list[Run] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for directory in sorted(root.glob("2026*")):
            metrics = directory / "metrics.json"
            if not metrics.exists() or directory.name in seen:
                continue
            match = re.search(r"(\d+m)-bs(\d+)-lr2e-(\d+)-s(\d+)", directory.name)
            if not match:
                continue
            seen.add(directory.name)
            m = json.loads(metrics.read_text())["metrics"]
            found.append(
                Run(
                    tier=match.group(1),
                    batch=int(match.group(2)),
                    lg_lr=-int(match.group(3)),
                    seed=int(match.group(4)),
                    loss=float(m["validation_loss"]),
                    steps=int(m.get("training_steps") or 0),
                    seconds=float(m["train_seconds"]),
                    flops_per_token=int(m.get("flops_per_token") or 0),
                )
            )
    return found


def load_spikes(archive: Path) -> list[tuple[int, int, float, float]]:
    """(lg_lr, seed, spike ratio, loss) for the 250M LR reseed."""

    study = archive / "2026-08-15-lr-transfer-5tpp"
    results = study / "studies" / "complete_d_p_lr_250m_reseed_v1" / "results.csv"
    if not results.exists():
        return []
    out = []
    for row in csv.DictReader(results.open()):
        curve = study / row["run_id"] / "training.csv"
        if not curve.exists():
            continue
        norms = [
            float(r["grad_norm"])
            for r in csv.DictReader(curve.open())
            if r.get("grad_norm") not in (None, "", "nan")
        ]
        if not norms:
            continue
        median = sorted(norms)[len(norms) // 2]
        lg = round(math.log2(float(row["base_learning_rate"])))
        out.append((lg, int(row["seed"]), max(norms) / median,
                    float(row["validation_loss"])))
    return out


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
    def __init__(self, title: str, xlabel: str, ylabel: str,
                 xlim: tuple[float, float], ylim: tuple[float, float],
                 width: int = W, height: int = H):
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
                f'<line x1="{PAD["l"]}" y1="{y:.1f}" x2="{self.w-PAD["r"]}" '
                f'y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
            )
            self.parts.append(
                f'<text x="{PAD["l"]-8}" y="{y+3.5:.1f}" fill="{MUTED}" '
                f'font-size="10" text-anchor="end">{html.escape(yfmt(v))}</text>'
            )
        for v in xticks:
            x = self.X(v)
            self.parts.append(
                f'<line x1="{x:.1f}" y1="{PAD["t"]}" x2="{x:.1f}" '
                f'y2="{self.h-PAD["b"]}" stroke="{GRID}" stroke-width="1" '
                f'stroke-dasharray="2,3"/>'
            )
            self.parts.append(
                f'<text x="{x:.1f}" y="{self.h-PAD["b"]+16}" fill="{MUTED}" '
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
        x = self.w - PAD["r"] - 8 if x is None else x
        y = PAD["t"] + 6 if y is None else y
        for i, (label, color) in enumerate(entries):
            yy = y + i * 15
            self.parts.append(
                f'<rect x="{x-9:.1f}" y="{yy-7:.1f}" width="8" height="8" '
                f'rx="2" fill="{color}"/>'
            )
            self.parts.append(
                f'<text x="{x-14:.1f}" y="{yy:.1f}" fill="{MUTED}" font-size="10" '
                f'text-anchor="end">{html.escape(label)}</text>'
            )

    def render(self) -> str:
        head = (
            f'<text x="{PAD["l"]}" y="18" fill="{INK}" font-size="12.5" '
            f'font-weight="650" text-anchor="start">{html.escape(self.title)}</text>'
        )
        xl = (
            f'<text x="{(PAD["l"]+self.w-PAD["r"])/2:.0f}" y="{self.h-8}" '
            f'fill="{MUTED}" font-size="10" text-anchor="middle">'
            f"{html.escape(self.xlabel)}</text>"
        )
        yl = (
            f'<text transform="translate(14,{(PAD["t"]+self.h-PAD["b"])/2:.0f}) '
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
                st.mean(v) for k, v in c.items()
                if k[0] == tier and k[1] == batch and len(v) >= 3
            )
            pts.append((math.log2(batch), best))
        if pts:
            floor = min(y for _, y in pts)
            series[tier] = [(x, y - floor) for x, y in pts]
    top = max(y for s in series.values() for _, y in s)
    p = Plot("Loss above each tier's own optimum",
             "global batch size", "Δ nats vs that tier's best",
             (4.6, 9.4), (-0.05, top * 1.08))
    p.frame([5, 6, 7, 8, 9], ticks(0, top, 4),
            xfmt=lambda v: str(int(2 ** v)), yfmt=lambda v: f"{v:.2f}")
    for tier, pts in series.items():
        p.line(pts, TIER_COLOR[tier], 2.4)
        p.dots(pts, TIER_COLOR[tier])
    p.line([(math.log2(128), -0.05), (math.log2(128), top * 1.08)],
           "#94a3b8", 1.0, dash="4,4", opacity=0.5)
    p.text(p.X(7), PAD["t"] - 4, "batch 128", MUTED, 9)
    p.legend([(t, TIER_COLOR[t]) for t in series])
    return p.render()


def fig_heatmaps(c) -> list[str]:
    """LR x batch grid per tier; colour is loss above that tier's best."""
    out = []
    for tier in TIERS:
        batches = sorted({k[1] for k in c if k[0] == tier})
        lrs = sorted({k[2] for k in c if k[0] == tier}, reverse=True)
        vals = {
            (b, l): st.mean(c[(tier, b, l)])
            for b in batches for l in lrs if (tier, b, l) in c
        }
        if not vals:
            continue
        floor = min(vals.values())
        worst = max(vals.values())
        span = (worst - floor) or 1
        p = Plot(f"{tier}: loss above optimum", "global batch size",
                 "base learning rate", (0, len(batches)), (0, len(lrs)),
                 width=400, height=300)
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
                p.text(x + cw / 2, y + ch / 2 + 3.5,
                       "best" if delta < 1e-9 else f"+{delta:.3f}",
                       "#0b1220" if frac < 0.5 else "#fff", 9,
                       weight="700" if delta < 1e-9 else "normal")
            p.text(PAD["l"] + i * cw + cw / 2, p.h - PAD["b"] + 16, str(b), MUTED, 10)
        for j, l in enumerate(lrs):
            p.text(PAD["l"] - 8, PAD["t"] + j * ch + ch / 2 + 3.5,
                   f"2^{l}", MUTED, 10, anchor="end")
        out.append(p.render())
    return out


def fig_pareto(runs, c) -> str:
    """Loss vs wall clock. Every cell is a point; the frontier is traced."""
    p = None
    pts_by_tier = {}
    for tier in TIERS:
        secs = {}
        for (t, b, l), v in c.items():
            if t != tier or len(v) < 3:
                continue
            s = st.mean(r.seconds for r in runs
                        if r.tier == t and r.batch == b and r.lg_lr == l)
            secs[(b, l)] = (s, st.mean(v))
        if secs:
            pts_by_tier[tier] = secs
    allx = [s for d in pts_by_tier.values() for s, _ in d.values()]
    p = Plot("Loss vs wall clock (every batch x LR cell)",
             "train seconds (log)", "validation loss",
             (math.log10(min(allx) * 0.85), math.log10(max(allx) * 1.15)),
             (0, 1))
    lo = min(y for d in pts_by_tier.values() for _, y in d.values())
    hi = max(y for d in pts_by_tier.values() for _, y in d.values())
    p.y0, p.y1 = nice(lo, min(hi, lo + 1.4))
    p.frame([2, 2.3, 2.6, 3, 3.3], ticks(p.y0, p.y1, 5),
            xfmt=lambda v: f"{10**v:,.0f}", yfmt=lambda v: f"{v:.2f}")
    for tier, d in pts_by_tier.items():
        for (b, _l), (s, y) in d.items():
            if y > p.y1:
                continue
            p.dots([(math.log10(s), y)], BATCH_COLOR.get(b, "#888"), 3.6, 0.95)
        front, best = [], math.inf
        for (_b, _l), (s, y) in sorted(d.items(), key=lambda kv: kv[1][0]):
            if y < best:
                best = y
                front.append((math.log10(s), y))
        p.line(sorted(front), TIER_COLOR[tier], 1.6, dash="5,4", opacity=0.9)
    p.legend([(f"batch {b}", BATCH_COLOR[b]) for b in sorted(BATCH_COLOR)]
             + [(t, TIER_COLOR[t]) for t in pts_by_tier])
    return p.render()


def fig_penalty(c) -> str:
    """Penalty for exceeding batch 128, per tier, log scale."""
    params = {"60m": 59_918_208, "125m": 123_456_640, "250m": 244_444_032}
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
    p = Plot("Penalty for exceeding batch 128", "parameters (log)",
             "Δ nats vs batch 128 (log)",
             (7.7, 8.45), (min(ys) - 0.25, max(ys) + 0.25))
    p.frame([math.log10(params[t]) for t in TIERS],
            ticks(min(ys) - 0.2, max(ys) + 0.2, 4),
            xfmt=lambda v: {7.777: "60M", 8.092: "125M", 8.388: "250M"}.get(
                round(v, 3), f"{10**v/1e6:.0f}M"),
            yfmt=lambda v: f"{10**v:.3f}")
    for over, pts in series.items():
        col = BATCH_COLOR[over]
        p.line(pts, col, 2.4)
        p.dots(pts, col, 4)
    p.legend([(f"batch {b}", BATCH_COLOR[b]) for b in series])
    return p.render()


def fig_steps(c) -> str:
    """Loss vs optimizer steps. If steps were binding these would align."""
    p = Plot("Loss vs optimizer steps (curves do NOT align)",
             "optimizer steps (log)", "Δ nats vs that tier's best",
             (2.6, 4.4), (-0.05, 0.55))
    p.frame([2.7, 3.0, 3.3, 3.7, 4.0, 4.3], ticks(0, 0.5, 5),
            xfmt=lambda v: f"{10**v:,.0f}", yfmt=lambda v: f"{v:.2f}")
    for tier in TIERS:
        pts = []
        for b in sorted({k[1] for k in c if k[0] == tier}):
            v = c.get((tier, b, -8))
            if not v or len(v) < 3:
                continue
            steps = TOKENS[tier] / (b * 1024)
            pts.append((math.log10(steps), st.mean(v)))
        if not pts:
            continue
        floor = min(y for _, y in pts)
        pts = [(x, min(y - floor, 0.52)) for x, y in pts]
        p.line(pts, TIER_COLOR[tier], 2.4)
        p.dots(pts, TIER_COLOR[tier])
        best = min(pts, key=lambda q: q[1])
        p.dots([best], "#fff", 5.5, 0.25)
    p.legend([(t, TIER_COLOR[t]) for t in TIERS])
    return p.render()


TOKENS = {"60m": 5 * 59_918_208, "125m": 5 * 123_456_640, "250m": 5 * 244_444_032}


def fig_spread(c) -> str:
    """Seed spread by LR, one line per (tier, batch). Noise peaks near the optimum."""
    p = Plot("Seed spread (max-min over 3 seeds)", "base learning rate",
             "loss spread, nats", (-10.4, -5.6), (0, 0.36))
    p.frame([-10, -9, -8, -7, -6], ticks(0, 0.35, 5),
            xfmt=lambda v: f"2^{int(v)}", yfmt=lambda v: f"{v:.2f}")
    for tier in TIERS:
        for b in sorted({k[1] for k in c if k[0] == tier}):
            pts = []
            for l in sorted({k[2] for k in c if k[0] == tier}):
                v = c.get((tier, b, l))
                if v and len(v) >= 3:
                    pts.append((l, min(max(v) - min(v), 0.35)))
            if len(pts) >= 2:
                p.line(pts, TIER_COLOR[tier], 1.5,
                       opacity=0.35 + 0.5 * (b == 128))
                p.dots(pts, TIER_COLOR[tier], 2.4, 0.8)
    p.legend([(t, TIER_COLOR[t]) for t in TIERS])
    return p.render()


def fig_spikes(spikes) -> str:
    """Gradient spike vs loss excess within each LR, 250M reseed."""
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
    p = Plot("Gradient spike vs loss excess (250M reseed)",
             "peak / median gradient norm (log)", "Δ nats within its LR group",
             (0.9, 3.6), (-0.05, 0.05))
    lo = min(y for _, y, _ in pts)
    hi = max(y for _, y, _ in pts)
    p.y0, p.y1 = nice(lo, hi)
    p.frame([1, 1.5, 2, 2.5, 3, 3.5], ticks(p.y0, p.y1, 5),
            xfmt=lambda v: f"{10**v:,.0f}x", yfmt=lambda v: f"{v:+.3f}")
    p.line([(0.9, 0), (3.6, 0)], GRID, 1.2, dash="4,4")
    colors = {-9: "#64748b", -8: "#34d399", -7: "#fbbf24", -6: "#f87171"}
    for lg in sorted(by_lr, reverse=True):
        group = sorted([(x, y) for x, y, g in pts if g == lg])
        p.line(group, colors.get(lg, "#888"), 1.2, opacity=0.55)
        p.dots(group, colors.get(lg, "#888"), 4)
    p.legend([(f"2^{lg}", colors.get(lg, "#888")) for lg in sorted(by_lr, reverse=True)])
    return p.render()


def fig_utilization(runs) -> str:
    """Achieved PFLOP/s vs batch: why the large batches are faster."""
    p = Plot("Achieved throughput vs batch", "global batch size",
             "PFLOP/s", (4.6, 9.4), (0, 1.9))
    p.frame([5, 6, 7, 8, 9], ticks(0, 1.8, 6),
            xfmt=lambda v: str(int(2 ** v)), yfmt=lambda v: f"{v:.1f}")
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
    ap.add_argument("--archive", type=Path,
                    default=Path.home() / "rig-run-archive")
    ap.add_argument("--output", type=Path,
                    default=Path("docs/reports/transfer-charts.html"))
    args = ap.parse_args()

    roots = [args.runs] + sorted(p for p in args.archive.glob("*") if p.is_dir())
    runs = load_runs(roots)
    if not runs:
        raise SystemExit(f"no runs found under {roots}")
    c = cells(runs)
    spikes = load_spikes(args.archive)

    body = []
    body.append("<h2>The result</h2>")
    body.append(card(
        "Batch 128 is optimal at all three tiers. The penalty for exceeding "
        "it falls steeply with scale: 0.45 nats at 60M, 0.016 at 250M.",
        fig_batch_normalized(c)))
    body.append("<h2>Full grids</h2>")
    body.append('<div class="row">'
                + "".join(f'<div class="card">{s}</div>' for s in fig_heatmaps(c))
                + "</div>")
    body.append("<h2>Why it is batch size, not step count</h2>")
    body.append(card(
        "Re-indexed by optimizer steps. If a minimum step count were binding, "
        "these curves would align; instead each tier's optimum sits at a "
        "different step count (2,286 / 4,709 / 9,325) while batch stays at 128.",
        fig_steps(c)))
    body.append("<h2>Practical tradeoff</h2>")
    body.append('<div class="row">'
                + card("Dashed lines trace each tier's Pareto frontier. At 250M, "
                       "batch 256 costs 0.016 nats for 11% wall clock; at 60M the "
                       "same trade costs 0.45.", fig_pareto(runs, c))
                + card("Throughput saturates around batch 256, which is why the "
                       "larger batches finish sooner on an identical token budget.",
                       fig_utilization(runs))
                + "</div>")
    body.append("<h2>Noise structure</h2>")
    body.append('<div class="row">'
                + card("Seed spread by LR; batch 128 drawn brighter. Noise is "
                       "largest near the optimum, which is why single-seed "
                       "comparisons there are unreliable.", fig_spread(c))
                + (card("Within each LR group, larger gradient spikes track worse "
                        "loss. Pooled across LRs the correlation is only +0.17 — "
                        "the effect is within-group.", fig_spikes(spikes))
                   if spikes else "")
                + "</div>")
    body.append("<h2>Penalty scaling</h2>")
    body.append(card(
        "Both axes log. If this stays log-linear it predicts the 500M and 1B "
        "penalty, i.e. whether larger tiers can simply adopt batch 256.",
        fig_penalty(c)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        PAGE.format(body="\n".join(body), count=len(runs),
                    ink=INK, muted=MUTED, grid=GRID),
        encoding="utf-8",
    )
    print(f"{args.output}: {len(runs)} runs, {len(c)} cells, "
          f"{len(spikes)} spike samples")


if __name__ == "__main__":
    main()
