"""Qualification, deterministic rankings, and dependency-free rendering."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from ..cohort import CohortError, validate_cohort


def rank_records(
    records: Iterable[Mapping[str, Any]],
    *,
    cohort_id: str | None = None,
    profile: str | None = None,
    best_per_recipe: bool = True,
) -> list[dict[str, Any]]:
    """Rank full, qualifying records from exactly one explicit cohort."""

    candidates: list[dict[str, Any]] = []
    observed_cohorts: set[str] = set()
    for source in records:
        record = dict(source)
        if record.get("status") != "ok" or record.get("run_kind") != "full":
            continue
        if profile is not None and record.get("profile") != profile:
            continue
        cohort = record.get("cohort")
        if not isinstance(cohort, Mapping):
            continue
        try:
            checked_cohort = validate_cohort(cohort)
        except CohortError:
            continue
        current_cohort = checked_cohort["cohort_id"]
        if record.get("cohort_id") != current_cohort:
            continue
        observed_cohorts.add(current_cohort)
        if cohort_id is not None and current_cohort != cohort_id:
            continue
        train_seconds = record.get("metrics", {}).get("train_seconds")
        if not bool(record.get("qualified")):
            continue
        if not _finite_nonnegative(train_seconds):
            continue
        candidates.append(record)

    if cohort_id is None and len(observed_cohorts) > 1:
        raise ValueError(
            "records span multiple cohorts; select one cohort_id before ranking"
        )

    def key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        seconds = float(record["metrics"]["train_seconds"])
        return (seconds, record["run_id"])

    candidates.sort(key=key)
    if best_per_recipe:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for record in candidates:
            recipe = str(record.get("recipe", ""))
            if recipe in seen:
                continue
            seen.add(recipe)
            unique.append(record)
        candidates = unique
    for rank, record in enumerate(candidates, 1):
        record["rank"] = rank
    return candidates


def render_leaderboard(
    records: Sequence[Mapping[str, Any]],
    *,
    cohort_id: str | None = None,
    color: bool = False,
) -> str:
    """Return a compact terminal table; callers decide whether ANSI is suitable."""

    headers = ["#", "recipe", "time", "tokens", "val loss", "run"]
    rows: list[list[str]] = []
    for position, record in enumerate(records, 1):
        metrics = record.get("metrics", {})
        rows.append(
            [
                str(record.get("rank", position)),
                str(record.get("recipe", "?")),
                _seconds(metrics.get("train_seconds")),
                _integer(metrics.get("tokens_processed")),
                _loss(metrics.get("validation_loss")),
                str(record.get("run_id", "?"))[:12],
            ]
        )
    widths = [
        max([len(headers[i]), *(len(row[i]) for row in rows)])
        for i in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(
            value.rjust(widths[i]) if i in (0, 2, 3, 4) else value.ljust(widths[i])
            for i, value in enumerate(row)
        )

    cohort_label = cohort_id[:12] if cohort_id else "unselected"
    title = f"Cohort {cohort_label} leaderboard"
    if color:
        title = f"\x1b[1;36m{title}\x1b[0m"
    divider = "  ".join("─" * width for width in widths)
    output = [title, format_row(headers), divider]
    output.extend(format_row(row) for row in rows)
    if not rows:
        output.append("No qualifying runs.")
    return "\n".join(output)


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _seconds(value: Any) -> str:
    return f"{float(value):.3f}s" if _finite_nonnegative(value) else "—"


def _integer(value: Any) -> str:
    return f"{value:,}" if _positive_int(value) else "—"


def _loss(value: Any) -> str:
    return f"{float(value):.4f}" if _finite_nonnegative(value) else "—"
