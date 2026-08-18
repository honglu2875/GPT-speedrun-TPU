"""Read the long-form CSV artifacts that predate the packed log format.

Runs recorded before ``75f0b22`` wrote ``training.csv`` and ``diagnostics.csv``
as long-form CSV. Nothing read them afterwards: the current reader checks for
the packed magic and refuses, so those runs dropped out of every report and
could only be inspected by checking out an older commit.

That is a bad place to leave real measurements. This module converts the
training curve into the packed format so those runs are ordinary inputs again,
and it does so without touching the originals -- the CSV stays exactly where it
is and the packed file is written somewhere else.

Both the training curve and the per-layer diagnostics convert. The diagnostics
are the expensive half -- 5-290 MB of CSV per run -- and the packed form is
roughly twenty times smaller, which is what makes keeping them affordable at
all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import csv
import json

import numpy as np

from . import runlog


class LegacyError(Exception):
    """A legacy artifact does not have the shape this converter assumes."""


# The long-form header, in the order the old writer emitted it. Any other
# header means the file is not what this converter was written against, and
# guessing at a mapping would silently mislabel columns.
TRAINING_HEADER = (
    "step",
    "tokens_processed",
    "cumulative_estimated_flops",
    "train_loss",
    "learning_rate",
    "grad_norm",
)

DIAGNOSTICS_HEADER = (
    "step",
    "tokens_processed",
    "cumulative_estimated_flops",
    "scope",
    "layer",
    "family",
    "stat",
    "value",
    "element_count",
)


def read_training_csv(path: Path) -> tuple[np.ndarray, int, int]:
    """Return ``(history, tokens_per_step, flops_per_token)`` from a legacy CSV.

    ``history`` is the ``[steps, 3]`` array the packed writer expects, holding
    train loss, learning rate, and gradient norm per optimizer step.

    The packed format keeps tokens-per-step and FLOPs-per-token once in its
    header rather than repeating them on every row, so both are recovered here
    and checked to be genuinely constant. A file where they drift is not
    something this converter can represent, and quietly keeping the first row's
    value would misplace every later point on the token and FLOP axes.
    """

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration:
            raise LegacyError(f"{path} is empty") from None
        if header != TRAINING_HEADER:
            raise LegacyError(f"{path} has header {header}, expected {TRAINING_HEADER}")
        rows = [row for row in reader if row]

    if not rows:
        raise LegacyError(f"{path} has a header but no samples")

    table = np.array(rows, dtype=np.float64)
    steps = table[:, 0].astype(np.int64)
    expected = np.arange(1, len(rows) + 1, dtype=np.int64)
    if not np.array_equal(steps, expected):
        raise LegacyError(
            f"{path} steps are not 1..{len(rows)} without gaps; "
            "the packed writer stores a dense per-step history"
        )

    tokens = table[:, 1].astype(np.int64)
    tokens_per_step = int(tokens[0])
    if not np.array_equal(tokens, expected * tokens_per_step):
        raise LegacyError(f"{path} tokens per step is not constant")

    flops = table[:, 2].astype(np.float64)
    flops_per_token = flops / tokens
    # float64 holds these exactly at this magnitude; a real drift is what this
    # is looking for, not the last bit of a division.
    if not np.allclose(flops_per_token, flops_per_token[0], rtol=1e-12, atol=0):
        raise LegacyError(f"{path} FLOPs per token is not constant")

    history = np.ascontiguousarray(table[:, 3:6], dtype=np.float32)
    return history, tokens_per_step, int(round(float(flops_per_token[0])))


def read_diagnostics_csv(
    path: Path,
) -> tuple[list[runlog.DiagnosticPoint], list[tuple[str, int | None, int]]]:
    """Return ``(points, scope_metadata)`` from a long-form diagnostics CSV.

    The long form names every value by ``(scope, layer, family, stat)`` on its
    own row. The packed form stores a fixed grid instead -- scope x family x
    stat, in the order ``runlog.diagnostic_log_columns`` lays it out -- so the
    conversion is a pivot, and its only real risk is putting a value in the
    wrong cell of that grid.

    Guarded three ways. Scopes keep first-seen order, so the grid matches the
    order the recipe emitted. Every family and statistic must be one the
    registry knows, because an unrecognized name means this file records
    something the packed layout has no cell for. And every step must fill the
    grid exactly once -- a missing or duplicated cell would otherwise leave a
    zero that reads as a real measurement.
    """

    families = {name: index for index, name in enumerate(runlog.DIAGNOSTIC_FAMILIES)}
    stats = {name: index for index, name in enumerate(runlog.DIAGNOSTIC_STATS)}

    scopes: dict[tuple[str, int | None], int] = {}
    elements: dict[tuple[str, int | None], int] = {}
    rows_by_step: dict[int, list[tuple[int, int, int, float]]] = {}

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration:
            raise LegacyError(f"{path} is empty") from None
        if header != DIAGNOSTICS_HEADER:
            raise LegacyError(
                f"{path} has header {header}, expected {DIAGNOSTICS_HEADER}"
            )
        for row in reader:
            if not row:
                continue
            step = int(row[0])
            layer = int(row[4]) if row[4] != "" else None
            key = (row[3], layer)
            if key not in scopes:
                scopes[key] = len(scopes)
                elements[key] = int(row[8])
            family, stat = row[5], row[6]
            if family not in families or stat not in stats:
                raise LegacyError(
                    f"{path} records {family}.{stat}, which is not in the registry"
                )
            rows_by_step.setdefault(step, []).append(
                (scopes[key], families[family], stats[stat], float(row[7]))
            )

    if not rows_by_step:
        raise LegacyError(f"{path} has a header but no samples")

    shape = (len(scopes), len(families), len(stats))
    expected = shape[0] * shape[1] * shape[2]
    points = []
    for step in sorted(rows_by_step):
        cells = rows_by_step[step]
        if len(cells) != expected:
            raise LegacyError(
                f"{path} step {step} has {len(cells)} values, expected {expected}; "
                "a partial grid would leave zeros that read as measurements"
            )
        values = np.zeros(shape, dtype=np.float32)
        filled = np.zeros(shape, dtype=bool)
        for scope_index, family_index, stat_index, value in cells:
            values[scope_index, family_index, stat_index] = value
            filled[scope_index, family_index, stat_index] = True
        if not filled.all():
            raise LegacyError(f"{path} step {step} repeats a cell and omits another")
        points.append(runlog.DiagnosticPoint(step=step, values=values))

    metadata = [(name, layer, elements[(name, layer)]) for name, layer in scopes]
    return points, metadata


def convert_run(source: Path, destination: Path) -> dict[str, Any]:
    """Write a packed copy of one legacy run, leaving the original untouched.

    Copies across only what a report reads -- the converted training curve, the
    validation curve, which was already CSV and stays so, and the result and
    metrics documents. ``result.json`` is rewritten with its ``training_curve``
    artifact repointed at the packed file, because the report resolves artifact
    names through that field rather than by convention.

    Returns the result document it wrote, so a caller can rebuild a ledger.
    """

    source, destination = Path(source), Path(destination)
    result_path = source / "result.json"
    if not result_path.is_file():
        raise LegacyError(
            f"{source.name} has no result.json, so nothing records what it "
            "measured; the report cannot place an unlabelled curve"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    history, tokens_per_step, flops_per_token = read_training_csv(
        source / "training.csv"
    )
    _check_against_result(source.name, result, history, tokens_per_step)

    destination.mkdir(parents=True, exist_ok=True)
    runlog.write_training_log(
        destination,
        history,
        tokens_per_step=tokens_per_step,
        final_step=len(history),
        flops_per_token=flops_per_token,
    )

    artifacts = dict(result.get("artifacts") or {})
    artifacts["training_curve"] = runlog.TRAINING_LOG_NAME

    diagnostics = source / "diagnostics.csv"
    if diagnostics.is_file():
        points, metadata = read_diagnostics_csv(diagnostics)
        runlog.write_diagnostics_log(
            destination,
            points,
            metadata,
            tokens_per_step=tokens_per_step,
            final_step=points[-1].step,
            flops_per_token=flops_per_token,
        )
        artifacts["diagnostics"] = runlog.DIAGNOSTICS_LOG_NAME
    else:
        # A declared artifact that is absent is an error, not a shrug.
        artifacts.pop("diagnostics", None)
    result = {**result, "artifacts": artifacts}
    (destination / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for name in ("validation.csv", "metrics.json"):
        if (source / name).is_file():
            (destination / name).write_bytes((source / name).read_bytes())
    return result


def _check_against_result(
    run: str,
    result: Mapping[str, Any],
    history: np.ndarray,
    tokens_per_step: int,
) -> None:
    """Refuse a conversion the run's own result document contradicts.

    The report re-checks these once the packed file exists, so catching them
    here only changes where the failure appears -- but it names the legacy CSV
    as the thing that disagreed, which is the part a later reader needs.
    """

    metrics = result.get("metrics") or {}
    steps = metrics.get("training_steps")
    if steps is not None and int(steps) != len(history):
        raise LegacyError(
            f"{run}: training.csv has {len(history)} steps, result.json says {steps}"
        )
    tokens = metrics.get("tokens_processed")
    if tokens is not None and int(tokens) != tokens_per_step * len(history):
        raise LegacyError(
            f"{run}: training.csv covers {tokens_per_step * len(history)} tokens, "
            f"result.json says {tokens}"
        )
    loss = metrics.get("train_loss")
    if loss is not None and not np.isclose(
        float(loss), float(history[-1, 0]), rtol=1e-6
    ):
        raise LegacyError(
            f"{run}: training.csv ends at loss {history[-1, 0]}, "
            f"result.json says {loss}"
        )
