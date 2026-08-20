"""Build the paired seed-variance browser from full-resolution rig logs.

The archived studies remain the source of truth.  This module aligns every run
at the exact logged optimizer step, computes the sample standard deviation
across seeds, and only then downsamples the derived curve for the browser.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rig import logpack
from rig.report import _lttb


DEFAULT_MAX_POINTS = 1_440
DEFAULT_PLANNED_SEEDS = 64
TRAINING_LOG = "training.riglog"
DIAGNOSTICS_LOG = "diagnostics.riglog"


class VarianceReportError(ValueError):
    """Raised when runs cannot be compared at identical training steps."""


@dataclass(frozen=True)
class StudySpec:
    """One seed cohort displayed in the paired report."""

    identifier: str
    label: str
    path: Path
    planned_seeds: int = DEFAULT_PLANNED_SEEDS


@dataclass(frozen=True)
class ReportSummary:
    """Small build result for CLI output and tests."""

    output: Path
    studies: tuple[tuple[str, int], ...]
    metrics: int
    chart_points: int


def _column_key(source: str, column: logpack.Column) -> str:
    return ":".join(
        str(value)
        for value in (
            source,
            column.metric_id,
            column.scope_id,
            column.layer,
            column.index,
        )
    )


def _scope_label(column: logpack.Column) -> str:
    scope = column.scope
    name = scope.name if scope is not None else f"scope {column.scope_id}"
    if name == "block" and column.layer >= 0:
        return f"block {column.layer}"
    if name == "expert" and column.layer >= 0 and column.index >= 0:
        return f"block {column.layer} / expert {column.index}"
    if column.layer >= 0:
        name = f"{name} {column.layer}"
    if column.index >= 0:
        name = f"{name} / index {column.index}"
    return name.replace("_", " ")


def _metric_metadata(source: str, column: logpack.Column) -> dict[str, Any]:
    metric = column.metric
    metric_name = (
        metric.name if metric is not None else f"metric {column.metric_id}"
    )
    readable_name = metric_name.replace("_", " ").replace(".", " / ")
    scope = _scope_label(column)
    return {
        "id": _column_key(source, column),
        "source": source,
        "metric": metric_name,
        "scope": scope,
        "label": f"{readable_name} — {scope}",
        "raw": column.describe(),
    }


def _sample_standard_deviation(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite-only sample SD and contributor count at every point."""

    finite = np.isfinite(values)
    counts = finite.sum(axis=0, dtype=np.int32)
    totals = np.where(finite, values, 0.0).sum(axis=0, dtype=np.float64)
    means = np.divide(
        totals,
        counts,
        out=np.zeros_like(totals, dtype=np.float64),
        where=counts > 0,
    )
    squared = np.where(finite, (values - means) ** 2, 0.0).sum(
        axis=0, dtype=np.float64
    )
    variances = np.divide(
        squared,
        counts - 1,
        out=np.full_like(squared, np.nan, dtype=np.float64),
        where=counts >= 2,
    )
    return np.sqrt(variances), counts


def _validate_log(
    candidate: logpack.Log,
    reference: logpack.Log,
    *,
    path: Path,
    reference_path: Path,
) -> None:
    if candidate.columns != reference.columns:
        raise VarianceReportError(
            f"{path} has a different column layout from {reference_path}"
        )
    if not np.array_equal(candidate.steps, reference.steps):
        raise VarianceReportError(
            f"{path} does not log the same optimizer steps as {reference_path}"
        )
    if candidate.tokens_per_step != reference.tokens_per_step:
        raise VarianceReportError(
            f"{path} has different token accounting from {reference_path}"
        )
    if not math.isclose(
        candidate.flops_per_token,
        reference.flops_per_token,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise VarianceReportError(
            f"{path} has different FLOP accounting from {reference_path}"
        )


def _log_variances(
    run_directories: Sequence[Path],
    filename: str,
    source: str,
    max_points: int,
) -> tuple[dict[str, dict[str, Any]], int, float]:
    paths = [directory / filename for directory in run_directories]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise VarianceReportError(
            f"{len(missing)} run(s) are missing {filename}; first: {missing[0]}"
        )

    reference_path = paths[0]
    reference = logpack.read_log(reference_path)
    matrices = [reference.values]
    for path in paths[1:]:
        candidate = logpack.read_log(path)
        _validate_log(
            candidate,
            reference,
            path=path,
            reference_path=reference_path,
        )
        matrices.append(candidate.values)

    values = np.stack(matrices, axis=0)
    deviations, counts = _sample_standard_deviation(values)
    flops = reference.axis("cumulative_flops")
    series: dict[str, dict[str, Any]] = {}
    retained = 0
    for position, column in enumerate(reference.columns):
        points = [
            [int(step), float(flop), float(deviation), int(count)]
            for step, flop, deviation, count in zip(
                reference.steps,
                flops,
                deviations[:, position],
                counts[:, position],
                strict=True,
            )
            if math.isfinite(float(deviation))
        ]
        if len(points) > max_points:
            points = _lttb(points, max_points, x_index=1)
        # FLOPs are exact step * header constant.  Keep the constant once per
        # study instead of repeating a 17-digit coordinate in every metric.
        compact_points = [[point[0], point[2], point[3]] for point in points]
        metadata = _metric_metadata(source, column)
        series[metadata["id"]] = {
            "metadata": metadata,
            "points": compact_points,
        }
        retained += len(compact_points)
    flops_per_step = reference.tokens_per_step * reference.flops_per_token
    return series, retained, flops_per_step


def _hardware_label(result: dict[str, Any]) -> str:
    system = result.get("system", {})
    kinds = system.get("device_kinds", [])
    kind = kinds[0] if kinds else system.get("platform", "unknown hardware")
    processes = int(system.get("process_count", 0))
    devices = int(system.get("device_count", 0))
    process_word = "process" if processes == 1 else "processes"
    return f"{kind} · {processes} {process_word} · {devices} chips"


def _load_study(
    spec: StudySpec, max_points: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], int]:
    run_directories = sorted(
        path.parent for path in spec.path.glob(f"*/{TRAINING_LOG}")
    )
    if len(run_directories) < 2:
        raise VarianceReportError(
            f"{spec.path} needs at least two runs, found {len(run_directories)}"
        )

    seeds: list[int] = []
    validation_losses: list[float] = []
    hardware: set[str] = set()
    for directory in run_directories:
        result_path = directory / "result.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VarianceReportError(f"cannot read {result_path}: {error}") from error
        if result.get("status") != "ok":
            raise VarianceReportError(
                f"{result_path} has status {result.get('status')!r}, not 'ok'"
            )
        seeds.append(int(result["seed"]))
        validation_losses.append(float(result["metrics"]["validation_loss"]))
        hardware.add(_hardware_label(result))

    if len(set(seeds)) != len(seeds):
        raise VarianceReportError(f"{spec.path} contains duplicate seeds")
    if len(hardware) != 1:
        raise VarianceReportError(
            f"{spec.path} mixes hardware/topologies: {sorted(hardware)}"
        )

    training, training_points, training_flops_per_step = _log_variances(
        run_directories, TRAINING_LOG, "Training", max_points
    )
    diagnostics, diagnostic_points, diagnostic_flops_per_step = _log_variances(
        run_directories, DIAGNOSTICS_LOG, "Diagnostics", max_points
    )
    if not math.isclose(
        training_flops_per_step,
        diagnostic_flops_per_step,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise VarianceReportError(
            f"{spec.path} training and diagnostics use different FLOP accounting"
        )
    validation = np.asarray(validation_losses, dtype=np.float64)
    study = {
        "id": spec.identifier,
        "label": spec.label,
        "completed": len(seeds),
        "planned": spec.planned_seeds,
        "progress": len(seeds) / spec.planned_seeds,
        "seeds": sorted(seeds),
        "hardware": next(iter(hardware)),
        "flopsPerStep": training_flops_per_step,
        "finalValidationMean": float(np.mean(validation)),
        "finalValidationSd": float(np.std(validation, ddof=1)),
    }
    return study, {**training, **diagnostics}, training_points + diagnostic_points


def _merge_metrics(
    study_series: Sequence[dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    keys = set().union(*(series.keys() for series in study_series))

    def sort_key(key: str) -> tuple[int, int, int, int, int]:
        source, metric_id, scope_id, layer, index = key.split(":")
        return (
            0 if source == "Training" else 1,
            int(metric_id),
            int(scope_id),
            int(layer),
            int(index),
        )

    metrics = []
    for key in sorted(keys, key=sort_key):
        available = [series[key] for series in study_series if key in series]
        metadata = available[0]["metadata"]
        if any(item["metadata"] != metadata for item in available[1:]):
            raise VarianceReportError(f"metric metadata disagrees for {key}")
        metrics.append(
            {
                **metadata,
                "series": [
                    series.get(key, {}).get("points", []) for series in study_series
                ],
            }
        )
    return metrics


def build_seed_variance_report(
    specs: Sequence[StudySpec], output: Path, *, max_points: int = DEFAULT_MAX_POINTS
) -> ReportSummary:
    """Build a deterministic, self-contained paired variance report."""

    if len(specs) != 2:
        raise VarianceReportError("the paired report requires exactly two studies")
    if max_points < 3:
        raise VarianceReportError("max_points must be at least 3")
    if any(spec.planned_seeds <= 0 for spec in specs):
        raise VarianceReportError("planned_seeds must be positive")
    if len({spec.identifier for spec in specs}) != len(specs):
        raise VarianceReportError("study identifiers must be unique")

    studies = []
    all_series = []
    chart_points = 0
    for spec in specs:
        study, series, retained = _load_study(spec, max_points)
        studies.append(study)
        all_series.append(series)
        chart_points += retained

    metrics = _merge_metrics(all_series)
    default_metric = next(
        (
            metric["id"]
            for metric in metrics
            if metric["source"] == "Training"
            and metric["metric"] == "train_loss"
            and metric["scope"] == "overall"
        ),
        metrics[0]["id"],
    )
    payload = {
        "meta": {
            "title": "MoE seed variance across training",
            "maxPoints": max_points,
            "defaultMetric": default_metric,
            "variance": "sample standard deviation (n - 1)",
            "xAxis": "cumulative training FLOPs",
        },
        "studies": studies,
        "metrics": metrics,
    }
    encoded = base64.b64encode(
        gzip.compress(
            json.dumps(
                payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8"),
            compresslevel=9,
            mtime=0,
        )
    ).decode("ascii")
    template_path = Path(__file__).with_name("template.html")
    template = template_path.read_text(encoding="utf-8")
    marker = "__PAYLOAD_BASE64__"
    if template.count(marker) != 1:
        raise VarianceReportError(f"{template_path} must contain one {marker}")
    rendered = template.replace(marker, encoded)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return ReportSummary(
        output=output,
        studies=tuple(
            (study["label"], int(study["completed"])) for study in studies
        ),
        metrics=len(metrics),
        chart_points=chart_points,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the paired 60M/125M seed-variance HTML."
    )
    parser.add_argument(
        "--study-60m",
        type=Path,
        default=Path("hf-dataset/seed-variance-60M"),
    )
    parser.add_argument(
        "--study-125m",
        type=Path,
        default=Path("hf-dataset/seed-variance-125M"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("hf-dataset/seed-variance.html")
    )
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    parser.add_argument(
        "--planned-seeds", type=int, default=DEFAULT_PLANNED_SEEDS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    specs = (
        StudySpec("60m", "60M MoE", args.study_60m, args.planned_seeds),
        StudySpec("125m", "125M MoE", args.study_125m, args.planned_seeds),
    )
    summary = build_seed_variance_report(
        specs, args.output, max_points=args.max_points
    )
    cohorts = ", ".join(f"{label}: {count}" for label, count in summary.studies)
    size_mb = summary.output.stat().st_size / (1024 * 1024)
    print(
        f"wrote {summary.output} ({size_mb:.2f} MB): {cohorts}; "
        f"{summary.metrics} metrics, {summary.chart_points:,} retained points"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
