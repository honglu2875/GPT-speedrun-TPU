"""Terminal rendering shared by every recipe.

A recipe's progress output is not part of what it measures, so it does not
belong in the entry program. Stable run-card groups, rendering, and colour
policy live here. Each recipe still supplies its architecture rows explicitly,
so a fork can describe new science without teaching a shared formatter how to
inspect arbitrary recipe configs.

Writes to stderr throughout, because stdout carries the machine-readable
result line that the harness parses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence
import os
import sys

import jax


if TYPE_CHECKING:
    from rig.evaluation import EvaluationReport


ConsoleRow = tuple[str, object]
ConsoleRows = tuple[ConsoleRow, ...]


def standard_identity_rows(
    *,
    config_filename: str,
    config_profile: str,
    config_sha256: str,
    devices: Sequence[jax.Device],
    process_count: int,
    process_index: int,
) -> ConsoleRows:
    """Return experiment config, devices, JAX processes, and mesh rows."""

    return (
        (
            "experiment config",
            f"{config_filename} · {config_profile} · sha256:{config_sha256[:12]}",
        ),
        ("devices", f"{len(devices)} × {device_label(devices)}"),
        ("JAX processes", f"{process_count} (this rank {process_index})"),
        ("mesh", f"data={len(devices)} (replicated model)"),
    )


def standard_data_rows(
    *,
    source: str,
    train_tokens: int,
    validation_tokens: int,
    downstream_domains: int,
    downstream_tokens: int,
) -> ConsoleRows:
    """Return dataset, train/validation token, and downstream rows."""

    downstream = (
        f"{downstream_domains} domains / {downstream_tokens:,} scored"
        if downstream_domains
        else "not requested"
    )
    return (
        ("dataset", source),
        ("train / val tokens", f"{train_tokens:,} / {validation_tokens:,}"),
        ("downstream", downstream),
    )


def standard_training_rows(
    *,
    parameterization: str,
    width_multiplier: float,
    depth_multiplier: float,
    data_multiplier: float,
    batch_size: int,
    seq_len: int,
    sampling: str,
    usable_tokens_per_epoch: int | None,
    dtype_name: str,
) -> ConsoleRows:
    """Return parameterization, global batch, sampling, and compute rows."""

    if sampling == "shuffled_epochs":
        if usable_tokens_per_epoch is None:
            raise ValueError("shuffled-epoch display needs usable tokens per epoch")
        sampling_detail = (
            f"shuffled epochs · {usable_tokens_per_epoch:,} unique targets/epoch"
        )
    elif sampling == "random_windows":
        sampling_detail = "random windows with replacement"
    else:
        raise ValueError(f"unsupported standard sampling display: {sampling!r}")
    return (
        (
            "parameterization",
            f"{parameterization} · mN={width_multiplier:.4g} · "
            f"mL={depth_multiplier:.4g} · mD={data_multiplier:.4g}",
        ),
        ("global batch", f"{batch_size} × {seq_len} tokens"),
        ("train sampling", sampling_detail),
        ("compute", dtype_name),
    )


def standard_kernel_rows(
    *,
    attention_backend: str,
    attention_rows: Sequence[ConsoleRow],
    loss_backend: str,
    semantic_vocab_size: int,
    vocab_tile_size: int,
) -> ConsoleRows:
    """Return attention, attention-detail, and output-loss rows."""

    if loss_backend == "tiled":
        loss = f"tiled CE (semantic {semantic_vocab_size:,}, tile {vocab_tile_size:,})"
    elif loss_backend == "dense":
        loss = f"dense CE ({semantic_vocab_size:,} classes)"
    else:
        raise ValueError(f"unsupported standard loss display: {loss_backend!r}")
    return (
        ("attention", attention_backend),
        *attention_rows,
        ("output loss", loss),
    )


def standard_schedule_rows(
    *,
    diagnostics_every: int,
    final_step: int,
    schedule_steps: int,
    early_stopped: bool,
    tokens_processed: int,
    total_flops: int,
    flop_breakdown: Iterable[tuple[str, str]],
    capture_window: tuple[int, int] | None,
    xprof_destination: object | None,
) -> ConsoleRows:
    """Return diagnostics, duration, tokens, FLOPs, breakdown, and XProf rows."""

    diagnostics = (
        f"step 1 / every {diagnostics_every} / final"
        if diagnostics_every
        else "disabled"
    )
    duration = (
        f"{final_step:,} of {schedule_steps:,} scheduled steps (early stop)"
        if early_stopped
        else f"{schedule_steps:,} steps"
    )
    breakdown = (
        " · ".join(f"{label} {share}" for label, share in flop_breakdown) or "none"
    )
    if capture_window is None:
        xprof = "disabled"
    else:
        if xprof_destination is None:
            raise ValueError("an XProf capture window needs a destination")
        xprof = f"steps {capture_window[0]}..{capture_window[1]} → {xprof_destination}"
    return (
        ("diagnostics", diagnostics),
        ("duration", duration),
        ("train tokens", format_count(tokens_processed)),
        ("traced FLOPs", format_count(total_flops)),
        ("FLOP breakdown", breakdown),
        ("XProf", xprof),
    )


class Console:
    """Tiny ANSI renderer; avoids adding a UI dependency to the reference."""

    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[38;5;81m",
        "blue": "\033[38;5;75m",
        "green": "\033[38;5;114m",
        "yellow": "\033[38;5;221m",
        "magenta": "\033[38;5;176m",
        "red": "\033[38;5;203m",
        "white": "\033[38;5;255m",
    }

    def __init__(self, mode: str, *, active: bool = True) -> None:
        auto = sys.stderr.isatty() and "NO_COLOR" not in os.environ
        self.enabled = mode == "always" or (mode == "auto" and auto)
        self.active = active

    def paint(self, text: object, *styles: str) -> str:
        raw = str(text)
        if not self.enabled or not styles:
            return raw
        prefix = "".join(self.COLORS[s] for s in styles)
        return f"{prefix}{raw}{self.COLORS['reset']}"

    def banner(self) -> None:
        if not self.active:
            return
        mark = self.paint("◆", "magenta", "bold")
        title = self.paint(" GPT TPU RIG ", "white", "bold")
        print(
            f"\n  {mark}{title}{self.paint('reference / jax', 'cyan')}\n",
            file=sys.stderr,
        )

    def table(self, title: str, rows: Sequence[tuple[str, object]]) -> None:
        if not self.active:
            return
        # Keep configuration cards readable in ordinary terminals. Provenance
        # remains complete in result.json/checkpoints; the live card is a
        # compact summary and must never grow to a digest- or JSON-sized width.
        width = min(
            78,
            max(52, *(max(20, len(str(k))) + len(str(v)) + 7 for k, v in rows)),
        )
        inner = width - 2
        heading = f" {title} "
        top_fill = max(0, inner - len(heading) - 1)
        print(
            "  "
            + self.paint("╭─", "blue")
            + self.paint(heading, "white", "bold")
            + self.paint("─" * top_fill + "╮", "blue"),
            file=sys.stderr,
        )
        for key, value in rows:
            raw_key = str(key)
            if len(raw_key) > 20:
                raw_key = raw_key[:19] + "…"
            key_text = f"{raw_key:<20}"
            value_text = str(value)
            value_limit = max(8, inner - 23)
            if len(value_text) > value_limit:
                value_text = value_text[: value_limit - 1] + "…"
            padding = max(1, inner - 2 - len(key_text) - len(value_text))
            print(
                "  "
                + self.paint("│", "blue")
                + " "
                + self.paint(key_text, "dim")
                + " " * padding
                + self.paint(value_text, "cyan", "bold")
                + " "
                + self.paint("│", "blue"),
                file=sys.stderr,
            )
        print(
            "  " + self.paint("╰" + "─" * inner + "╯", "blue"),
            file=sys.stderr,
        )

    def phase(self, label: str, detail: str = "") -> None:
        if not self.active:
            return
        suffix = f" {self.paint(detail, 'dim')}" if detail else ""
        print(
            f"\n  {self.paint('●', 'magenta')} {self.paint(label, 'white', 'bold')}{suffix}",
            file=sys.stderr,
        )

    def step(
        self,
        step: int,
        total: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_per_second: float,
    ) -> None:
        if not self.active:
            return
        fraction = step / total
        slots = 18
        filled = min(slots, int(round(fraction * slots)))
        bar = self.paint("━" * filled, "green") + self.paint(
            "─" * (slots - filled), "dim"
        )
        print(
            f"  {self.paint(f'{step:>4}/{total:<4}', 'white', 'bold')} "
            f"{bar}  loss {self.paint(f'{loss:.4f}', 'yellow', 'bold')}  "
            f"lr {lr:.2e}  |g| {grad_norm:.3f}  "
            f"{format_rate(tokens_per_second)} tok/s",
            file=sys.stderr,
        )

    def warn(self, message: str) -> None:
        """Surface something that did not fail but must not pass unnoticed."""

        if not self.active:
            return
        print(
            f"  {self.paint('!', 'yellow', 'bold')} {self.paint(message, 'yellow')}",
            file=sys.stderr,
        )

    def success(
        self,
        validation_loss: float,
        train_seconds: float,
        validation_seconds: float,
    ) -> None:
        if not self.active:
            return
        print(
            f"\n  {self.paint('✓', 'green', 'bold')} "
            f"synchronized training "
            f"{self.paint(f'{train_seconds:.3f}s', 'white', 'bold')} "
            f"{self.paint('(compilation excluded)', 'dim')}\n"
            f"    validation loss {self.paint(f'{validation_loss:.4f}', 'green', 'bold')} "
            f"in {self.paint(f'{validation_seconds:.3f}s', 'white', 'bold')}\n",
            file=sys.stderr,
        )

    def validation_probe(
        self, step: int, loss: float, batches: int, elapsed: float
    ) -> None:
        if not self.active:
            return
        print(
            f"  {self.paint('◇', 'cyan')} validation @ {step:,}  "
            f"loss {self.paint(f'{loss:.4f}', 'yellow', 'bold')}  "
            f"{batches} batches in {elapsed:.3f}s",
            file=sys.stderr,
        )

    def downstream(
        self, domain: str, loss: float, perplexity: float, tokens: int, elapsed: float
    ) -> None:
        if not self.active:
            return
        print(
            f"  {self.paint('◇', 'cyan')} {domain:<14} "
            f"loss {self.paint(f'{loss:.4f}', 'yellow', 'bold')}  "
            f"ppl {perplexity:.2f}  {tokens:,} tokens in {elapsed:.3f}s",
            file=sys.stderr,
        )

    def evaluations(self, report: EvaluationReport) -> None:
        """Render downstream domains and their fixed unweighted macro."""

        for entry in report.downstream:
            result = entry.result
            self.downstream(
                entry.name,
                result.loss,
                result.perplexity,
                result.scored_tokens,
                result.seconds,
            )
        macro = report.macro
        if macro is not None:
            self.downstream(
                f"{report.downstream_name} macro",
                macro.loss,
                macro.perplexity,
                macro.scored_tokens,
                macro.seconds,
            )


def format_count(value: float) -> str:
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.2f}{suffix}"
    return f"{value:.0f}"


def format_rate(value: float) -> str:
    return format_count(value)


def device_label(devices: Sequence[jax.Device]) -> str:
    kinds = sorted({str(device.device_kind) for device in devices})
    return ", ".join(kinds)
