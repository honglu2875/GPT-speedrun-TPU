"""Terminal rendering shared by every recipe.

A recipe's progress output is not part of what it measures, so it does not
belong in the entry program. What each recipe still owns is *what* to say --
the run-configuration table names its own config fields -- while the rendering,
the colour policy, and the shapes of the standard lines live here.

Writes to stderr throughout, because stdout carries the machine-readable
result line that the harness parses.
"""

from __future__ import annotations

from typing import Sequence
import os
import sys

import jax


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
        bar = self.paint("━" * filled, "green") + self.paint("─" * (slots - filled), "dim")
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
