"""Build a self-contained, dependency-free HTML report from benchmark run logs.

The report reader is intentionally conservative.  A successful result event and a
sound training CSV are required, ledger hashes are checked when available, and
malformed runs are listed rather than silently mixed into plots.  This is not a
replacement for ``speedrun verify``; it is a fast, read-only integrity gate for
visualization.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_CSV_BYTES = 128 * 1024 * 1024
_MAX_CHART_POINTS = 1_400
_OFFICIAL_OPEN_TOKENS = 624_984_064
_LAYER_KEY = re.compile(r"(?:^|/)(?:blocks?|layers?|h)(?:/|_)(\d+)(?:/|$)")
_COLORS = (
    "#7dd3fc",
    "#f9a8d4",
    "#86efac",
    "#fde047",
    "#c4b5fd",
    "#fb923c",
    "#67e8f9",
    "#fca5a5",
    "#a3e635",
    "#d8b4fe",
    "#fcd34d",
    "#5eead4",
)
_FRESH10_ORDER = (
    "science",
    "medicine",
    "software",
    "history",
    "fiction",
    "government",
    "legal",
    "economics",
    "climate",
    "education",
)
_TRAIN_REQUIRED = ("step", "tokens_processed", "train_loss")
_VALIDATION_REQUIRED = (
    "step",
    "tokens_processed",
    "kind",
    "domain",
    "validation_loss",
)
_PRIMARY_TRAIN_METRICS = ("train_loss", "learning_rate")
_LAYER_METRICS = (
    "l2_norm",
    "l1_norm",
    "mean",
    "std",
    "third_moment",
    "fourth_moment",
    "min",
    "max",
)


class ReportError(ValueError):
    """A report input or destination is invalid."""


@dataclass(frozen=True)
class ReportSummary:
    """Result of one report build."""

    output_path: Path
    included: tuple[str, ...]
    skipped: Mapping[str, str]
    notices: tuple[str, ...] = ()


@dataclass
class _Run:
    run_id: str
    result: dict[str, Any]
    record: dict[str, Any] | None
    training: list[dict[str, float]]
    validation: list[dict[str, Any]]
    train_metrics: tuple[str, ...]
    flop_source: str
    layer_stats: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)


def build_report(
    runs_dir: Path,
    output_path: Path,
    *,
    max_chart_points: int = _MAX_CHART_POINTS,
) -> ReportSummary:
    """Scan ``runs_dir`` and write one portable report HTML file.

    ``max_chart_points`` bounds the embedded data and canvas work per series.  The
    first and last points are always retained.
    """

    runs_dir = Path(runs_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if isinstance(max_chart_points, bool) or max_chart_points < 32:
        raise ReportError("max_chart_points must be an integer of at least 32")
    if not runs_dir.is_dir():
        raise ReportError(f"runs directory does not exist: {runs_dir}")
    if output_path.exists() and (output_path.is_dir() or output_path.is_symlink()):
        raise ReportError("report output must be a regular file, not a directory or symlink")

    records, duplicate_record_ids, ledger_notices = _read_ledger(
        runs_dir / "records.jsonl"
    )
    included: list[_Run] = []
    skipped: dict[str, str] = {}
    for candidate in sorted(runs_dir.iterdir()):
        if not candidate.is_dir() or candidate.is_symlink() or candidate.name.startswith("."):
            continue
        if not (candidate / "result.json").exists():
            # Failed/partial harness directories are expected and are not plot-able.
            continue
        if candidate.name in duplicate_record_ids:
            skipped[candidate.name] = (
                "records.jsonl contains duplicate entries for this run ID; "
                "the ledger identity is ambiguous"
            )
            continue
        try:
            run = _read_run(candidate, records.get(candidate.name), max_chart_points)
        except (OSError, ReportError, csv.Error, json.JSONDecodeError) as exc:
            skipped[candidate.name] = str(exc)
            continue
        included.append(run)

    notices = list(ledger_notices)
    notices.extend(notice for run in included for notice in run.notices)
    payload = _report_payload(included, skipped, notices)
    html = _render_html(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return ReportSummary(
        output_path=output_path,
        included=tuple(run.run_id for run in included),
        skipped=skipped,
        notices=tuple(notices),
    )


def _read_ledger(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], set[str], list[str]]:
    if not path.exists():
        return {}, set(), ["records.jsonl is absent; runs are marked as unledgered."]
    if not path.is_file() or path.is_symlink():
        return (
            {},
            set(),
            ["records.jsonl is not a regular file; ledger checks were skipped."],
        )
    records: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    notices: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                notices.append(f"Ignored malformed records.jsonl line {number}.")
                continue
            if not isinstance(value, dict) or not isinstance(value.get("run_id"), str):
                notices.append(f"Ignored non-record records.jsonl line {number}.")
                continue
            run_id = value["run_id"]
            if run_id in records or run_id in duplicate_ids:
                records.pop(run_id, None)
                if run_id not in duplicate_ids:
                    notices.append(
                        f"Duplicate records.jsonl run ID {run_id!r}; that run was excluded."
                    )
                duplicate_ids.add(run_id)
                continue
            records[run_id] = value
    return records, duplicate_ids, notices


def _read_run(path: Path, record: dict[str, Any] | None, limit: int) -> _Run:
    run_id = path.name
    result_path = _regular_file(path, "result.json")
    if result_path.stat().st_size > _MAX_JSON_BYTES:
        raise ReportError("result.json is implausibly large")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ReportError("result.json is not an object")
    if result.get("schema_version") != 1 or result.get("status") != "ok":
        raise ReportError("result.json is not a successful schema-v1 result")
    _check_identity(result, record, run_id)

    metrics = _object(result.get("metrics"), "result metrics")
    final_tokens = _positive_int(metrics.get("tokens_processed"), "tokens_processed")
    _positive_finite(metrics.get("train_seconds"), "train_seconds")
    final_train_loss = _finite(metrics.get("train_loss"), "train_loss")
    _finite(metrics.get("validation_loss"), "validation_loss")

    training_path = _artifact_path(path, result, "training_curve", "training.csv", required=True)
    _check_record_artifact(training_path, path, record, "training_curve")
    training, metric_names = _read_training(training_path)
    if int(training[-1]["tokens_processed"]) != final_tokens:
        raise ReportError("training.csv final token count disagrees with result.json")
    if not _close(training[-1]["train_loss"], final_train_loss):
        raise ReportError("training.csv final loss disagrees with result.json")

    flop_source = _attach_flops(training, metrics)
    validation_path = _artifact_path(
        path, result, "validation_curve", "validation.csv", required=False
    )
    if validation_path is not None:
        _check_record_artifact(validation_path, path, record, "validation_curve")
        validation = _read_validation(validation_path, training)
    else:
        validation = []
    _validate_and_fill_evaluations(validation, result, training[-1])

    # Downsample independently per scalar later; keep this shared row table bounded
    # using evenly spaced indices only after validation points have been derived.
    # Individual metric LTTB downsampling occurs in _training_series.
    run = _Run(
        run_id=run_id,
        result=result,
        record=record,
        training=training,
        validation=validation,
        train_metrics=metric_names,
        flop_source=flop_source,
    )
    if record is None:
        run.notices.append(f"{run_id}: not present in records.jsonl (shown as unledgered).")
    elif "qualified" in record and not isinstance(record["qualified"], bool):
        run.notices.append(f"{run_id}: invalid ledger qualified flag was ignored.")
    _read_checkpoint_layers(path, run)
    # Store the limit on the transient object without widening the public payload.
    run.result["_report_point_limit"] = limit
    return run


def _check_identity(
    result: Mapping[str, Any], record: Mapping[str, Any] | None, run_id: str
) -> None:
    for name in ("track", "profile"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ReportError(f"result {name} is missing")
    if isinstance(result.get("seed"), bool) or not isinstance(result.get("seed"), int):
        raise ReportError("result seed is invalid")
    if record is None:
        return
    if record.get("run_id") != run_id or record.get("status") != "ok":
        raise ReportError("ledger identity/status does not match the run directory")
    for name in ("track", "profile", "seed"):
        if record.get(name) != result.get(name):
            raise ReportError(f"ledger {name} disagrees with result.json")
    result_metrics = _object(result.get("metrics"), "result metrics")
    record_metrics = _object(record.get("metrics"), "ledger metrics")
    for name in ("tokens_processed", "train_seconds", "validation_loss"):
        left = result_metrics.get(name)
        right = record_metrics.get(name)
        if name == "tokens_processed":
            if left != right:
                raise ReportError(f"ledger {name} disagrees with result.json")
        elif not _close(_finite(left, name), _finite(right, name)):
            raise ReportError(f"ledger {name} disagrees with result.json")


def _artifact_path(
    run_dir: Path,
    result: Mapping[str, Any],
    artifact_name: str,
    fallback: str,
    *,
    required: bool,
) -> Path | None:
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ReportError("result artifacts is not an object")
    declared = artifacts.get(artifact_name)
    name = declared if declared is not None else fallback
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ReportError(f"artifact {artifact_name} has an unsafe path")
    unresolved = run_dir / name
    try:
        resolved = unresolved.resolve(strict=False)
        resolved.relative_to(run_dir.resolve())
    except (OSError, ValueError) as exc:
        raise ReportError(f"artifact {artifact_name} escapes its run directory") from exc
    if not resolved.exists():
        if declared is not None or required:
            raise ReportError(f"artifact {artifact_name} is missing")
        return None
    if not resolved.is_file() or unresolved.is_symlink():
        raise ReportError(f"artifact {artifact_name} is not a regular file")
    if resolved.stat().st_size > _MAX_CSV_BYTES:
        raise ReportError(f"artifact {artifact_name} is implausibly large")
    return resolved


def _regular_file(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    if not path.is_file() or path.is_symlink():
        raise ReportError(f"{name} is not a regular file")
    return path


def _check_record_artifact(
    path: Path, run_dir: Path, record: Mapping[str, Any] | None, name: str
) -> None:
    if record is None:
        return
    artifacts = _object(record.get("artifacts", {}), "ledger artifacts")
    value = artifacts.get(name)
    if not isinstance(value, dict):
        raise ReportError(f"ledger is missing the {name} artifact")
    try:
        relative = path.relative_to(run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ReportError(f"{name} is outside its run directory") from exc
    if value.get("path") != relative:
        raise ReportError(f"ledger path for {name} disagrees with result.json")
    if value.get("bytes") != path.stat().st_size:
        raise ReportError(f"ledger byte count for {name} no longer matches")
    expected = value.get("sha256")
    if not isinstance(expected, str) or _sha256(path) != expected:
        raise ReportError(f"ledger SHA-256 for {name} no longer matches")


def _read_training(path: Path) -> tuple[list[dict[str, float]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ReportError("training.csv has a missing or duplicate header")
        missing = [name for name in _TRAIN_REQUIRED if name not in fields]
        if missing:
            raise ReportError("training.csv is missing: " + ", ".join(missing))
        raw_rows = list(reader)
    if not raw_rows:
        raise ReportError("training.csv has no data rows")

    primary = set(_TRAIN_REQUIRED)
    parsed: list[dict[str, float]] = []
    numeric_extra = {name for name in fields if name not in primary}
    strict_extra = {
        name
        for name in ("learning_rate", "grad_norm", "cumulative_estimated_flops")
        if name in numeric_extra
    }
    for number, raw in enumerate(raw_rows, 2):
        row: dict[str, float] = {}
        row["step"] = float(_positive_int_text(raw.get("step"), f"training.csv:{number} step"))
        row["tokens_processed"] = float(
            _positive_int_text(raw.get("tokens_processed"), f"training.csv:{number} tokens")
        )
        row["train_loss"] = _finite_text(
            raw.get("train_loss"), f"training.csv:{number} train_loss"
        )
        for name in tuple(numeric_extra):
            value = raw.get(name)
            if value is None or not value.strip():
                if name in strict_extra:
                    raise ReportError(f"training.csv:{number} {name} is missing")
                numeric_extra.discard(name)
                continue
            try:
                row[name] = _finite_text(value, f"training.csv:{number} {name}")
            except ReportError:
                if name in strict_extra:
                    raise
                # A textual or sparse diagnostic column is harmless, but is not a
                # scalar series.  Primary metric corruption remains fatal above.
                numeric_extra.discard(name)
        parsed.append(row)

    previous_tokens = 0
    for expected_step, row in enumerate(parsed, 1):
        step, tokens = int(row["step"]), int(row["tokens_processed"])
        if step != expected_step:
            raise ReportError(
                "training.csv must contain exactly one row for every optimizer step "
                f"starting at 1 (expected {expected_step}, got {step})"
            )
        if tokens <= previous_tokens:
            raise ReportError("training.csv token counts must increase strictly")
        previous_tokens = tokens
    # Remove values parsed before a later row proved a column non-scalar.
    for row in parsed:
        for name in tuple(row):
            if name not in primary and name not in numeric_extra:
                del row[name]
    metric_names = tuple(
        name
        for name in fields
        if name in numeric_extra and name not in {"cumulative_estimated_flops"}
    )
    return parsed, ("train_loss", *metric_names)


def _attach_flops(rows: list[dict[str, float]], metrics: Mapping[str, Any]) -> str:
    explicit = all("cumulative_estimated_flops" in row for row in rows)
    if explicit:
        prior = 0.0
        for row in rows:
            value = row["cumulative_estimated_flops"]
            if value <= prior:
                raise ReportError(
                    "cumulative_estimated_flops must be positive and increase strictly"
                )
            row["estimated_flops"] = value
            prior = value
        declared_total = metrics.get("estimated_total_flops")
        if declared_total is not None:
            expected = _positive_finite(declared_total, "estimated_total_flops")
            if not _close(prior, expected):
                raise ReportError(
                    "training.csv final cumulative_estimated_flops disagrees with "
                    "result metrics.estimated_total_flops"
                )
        return "training.csv cumulative_estimated_flops"

    ratio_value = metrics.get("flops_per_token")
    if ratio_value is None:
        total = metrics.get("estimated_total_flops")
        tokens = metrics.get("tokens_processed")
        if total is not None and tokens:
            ratio_value = _finite(total, "estimated_total_flops") / _positive_int(
                tokens, "tokens_processed"
            )
    if ratio_value is None:
        raise ReportError(
            "no cumulative_estimated_flops or result metrics.flops_per_token for equi-FLOP plots"
        )
    ratio = _finite(ratio_value, "flops_per_token")
    if ratio <= 0:
        raise ReportError("flops_per_token must be greater than zero")
    for row in rows:
        row["estimated_flops"] = ratio * row["tokens_processed"]
    return "derived: result metrics.flops_per_token × tokens_processed"


def _read_validation(path: Path, training: Sequence[Mapping[str, float]]) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ReportError("validation.csv has a missing or duplicate header")
        missing = [name for name in _VALIDATION_REQUIRED if name not in fields]
        if missing:
            raise ReportError("validation.csv is missing: " + ", ".join(missing))
        raw_rows = list(reader)
    rows: list[dict[str, Any]] = []
    prior_step = 0
    final_step = int(training[-1]["step"])
    training_tokens = {int(row["step"]): int(row["tokens_processed"]) for row in training}
    for number, raw in enumerate(raw_rows, 2):
        step = _positive_int_text(raw.get("step"), f"validation.csv:{number} step")
        tokens = _positive_int_text(
            raw.get("tokens_processed"), f"validation.csv:{number} tokens"
        )
        if step < prior_step or step > final_step:
            raise ReportError("validation.csv steps must be ordered and within training")
        if training_tokens.get(step) != tokens:
            raise ReportError(
                f"validation.csv:{number} token count disagrees with training.csv step {step}"
            )
        kind, domain = raw.get("kind", "").strip(), raw.get("domain", "").strip()
        if not kind or not domain:
            raise ReportError(f"validation.csv:{number} kind/domain is empty")
        row = {
            "step": step,
            "tokens_processed": tokens,
            "kind": kind,
            "domain": domain,
            "validation_loss": _finite_text(
                raw.get("validation_loss"), f"validation.csv:{number} validation_loss"
            ),
            "estimated_flops": _flops_at_step(training, step),
        }
        rows.append(row)
        prior_step = step
    return rows


def _flops_at_step(training: Sequence[Mapping[str, float]], step: int) -> float:
    if step <= training[0]["step"]:
        scale = step / training[0]["step"]
        return float(training[0]["estimated_flops"] * scale)
    for left, right in zip(training, training[1:]):
        if step == int(left["step"]):
            return float(left["estimated_flops"])
        if left["step"] < step <= right["step"]:
            fraction = (step - left["step"]) / (right["step"] - left["step"])
            return float(
                left["estimated_flops"]
                + fraction * (right["estimated_flops"] - left["estimated_flops"])
            )
    return float(training[-1]["estimated_flops"])


def _validate_and_fill_evaluations(
    rows: list[dict[str, Any]],
    result: Mapping[str, Any],
    final_training: Mapping[str, float],
) -> None:
    metrics = _object(result.get("metrics"), "result metrics")
    validation_loss = _finite(metrics.get("validation_loss"), "validation_loss")
    final_step = int(final_training["step"])
    final_tokens = int(final_training["tokens_processed"])
    final_flops = float(final_training["estimated_flops"])
    fineweb_final = [
        row
        for row in rows
        if row["kind"] == "fineweb" or (
            row["domain"] == "fineweb" and str(row["kind"]).endswith("final")
        )
    ]
    if fineweb_final and not _close(fineweb_final[-1]["validation_loss"], validation_loss):
        raise ReportError("validation.csv final FineWeb loss disagrees with result.json")
    if not fineweb_final:
        rows.append(
            {
                "step": final_step,
                "tokens_processed": final_tokens,
                "kind": "fineweb",
                "domain": "fineweb",
                "validation_loss": validation_loss,
                "estimated_flops": final_flops,
                "synthetic": True,
            }
        )

    evaluations = result.get("evaluations")
    if evaluations is None:
        return
    evaluations = _object(evaluations, "evaluations")
    evaluated_fineweb = evaluations.get("fineweb")
    if evaluated_fineweb is not None:
        evaluated_loss = _finite(
            _object(evaluated_fineweb, "evaluations.fineweb").get("loss"),
            "evaluations.fineweb.loss",
        )
        if not _close(evaluated_loss, validation_loss):
            raise ReportError("evaluations.fineweb loss disagrees with result metrics")
    fresh = evaluations.get("fresh10")
    if fresh is None:
        return
    fresh = _object(fresh, "evaluations.fresh10")
    macro = _finite(fresh.get("macro_loss"), "Fresh10 macro_loss")
    macro_rows = [row for row in rows if row["kind"] == "downstream_macro"]
    if macro_rows and not _close(macro_rows[-1]["validation_loss"], macro):
        raise ReportError("validation.csv Fresh10 macro loss disagrees with result.json")
    if not macro_rows:
        rows.append(
            {
                "step": final_step,
                "tokens_processed": final_tokens,
                "kind": "downstream_macro",
                "domain": "fresh10_macro",
                "validation_loss": macro,
                "estimated_flops": final_flops,
                "synthetic": True,
            }
        )
    domains = _object(fresh.get("domains"), "evaluations.fresh10.domains")
    if len(domains) != 10:
        raise ReportError("evaluations.fresh10 must contain exactly ten domains")
    domain_losses = [
        _finite(_object(value, f"Fresh10 {domain}").get("loss"), f"Fresh10 {domain}")
        for domain, value in domains.items()
    ]
    if not _close(macro, math.fsum(domain_losses) / len(domain_losses)):
        raise ReportError("Fresh10 macro loss disagrees with its ten domain losses")
    for domain, value in domains.items():
        expected = _finite(_object(value, f"Fresh10 {domain}").get("loss"), f"Fresh10 {domain}")
        matches = [
            row for row in rows if row["kind"] == "downstream" and row["domain"] == domain
        ]
        if matches and not _close(matches[-1]["validation_loss"], expected):
            raise ReportError(f"validation.csv Fresh10 {domain} loss disagrees with result.json")
        if not matches:
            rows.append(
                {
                    "step": final_step,
                    "tokens_processed": final_tokens,
                    "kind": "downstream",
                    "domain": domain,
                    "validation_loss": expected,
                    "estimated_flops": final_flops,
                    "synthetic": True,
                }
            )


def _read_checkpoint_layers(run_dir: Path, run: _Run) -> None:
    checkpoint_name = run.result.get("checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        run.notices.append(f"{run.run_id}: result has no usable checkpoint path.")
        return
    try:
        checkpoint = _contained_optional(run_dir, checkpoint_name)
    except ReportError as exc:
        run.notices.append(f"{run.run_id}: checkpoint ignored ({exc}).")
        return
    if checkpoint is None:
        retained = (
            run.record.get("checkpoint", {}).get("retained")
            if isinstance(run.record, dict)
            else None
        )
        policy = "retention policy removed it" if retained is False else "file is absent"
        run.notices.append(f"{run.run_id}: per-layer plots unavailable ({policy}).")
        return

    if run.record is not None:
        ledger = run.record.get("checkpoint")
        if not isinstance(ledger, dict):
            run.notices.append(f"{run.run_id}: checkpoint lacks a ledger record; layer scan skipped.")
            return
        if ledger.get("bytes") != checkpoint.stat().st_size:
            run.notices.append(f"{run.run_id}: checkpoint size differs from ledger; layer scan skipped.")
            return
        expected = ledger.get("sha256")
        if not isinstance(expected, str) or _sha256(checkpoint) != expected:
            run.notices.append(f"{run.run_id}: checkpoint SHA-256 differs; layer scan skipped.")
            return
    try:
        run.layer_stats = _checkpoint_layer_stats(checkpoint)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        run.notices.append(f"{run.run_id}: checkpoint layer scan failed ({exc}).")
        return
    if not run.layer_stats:
        run.notices.append(f"{run.run_id}: checkpoint contains no recognized logical layer arrays.")


def _contained_optional(run_dir: Path, relative: str) -> Path | None:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ReportError("absolute checkpoint path")
    unresolved = run_dir / candidate
    resolved = unresolved.resolve(strict=False)
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ReportError("checkpoint path escapes run directory") from exc
    if not resolved.exists():
        return None
    if not resolved.is_file() or unresolved.is_symlink():
        raise ReportError("checkpoint is not a regular file")
    return resolved


def _checkpoint_layer_stats(path: Path) -> dict[str, list[dict[str, float]]]:
    # NumPy is deliberately imported lazily: `speedrun report` must not import JAX
    # or initialize a TPU, and ordinary CLI commands should remain instant.
    import numpy as np

    # Streaming raw sums avoid materializing multiple full-size power arrays.
    # With current reference checkpoints this keeps peak host memory near one
    # float64 copy of the largest tensor, rather than loading the whole NPZ.
    accumulators: dict[tuple[str, int], list[float]] = {}
    chunk_elements = 1_000_000
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            identity = _layer_identity(key)
            if identity is None:
                continue
            array = archive[key]
            numeric = np.issubdtype(array.dtype, np.number) or array.dtype.name == "bfloat16"
            if not numeric or array.size == 0:
                continue
            # count, sum(|x|), sum(x^2), sum(x), sum(x^3), sum(x^4), min, max
            acc = accumulators.setdefault(
                identity,
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.inf, -math.inf],
            )
            flattened = array.reshape(-1)
            for start in range(0, flattened.size, chunk_elements):
                values = np.asarray(
                    flattened[start : start + chunk_elements], dtype=np.float64
                )
                if not bool(np.all(np.isfinite(values))):
                    raise ValueError(f"non-finite layer array {key!r}")
                acc[0] += values.size
                acc[1] += float(np.sum(np.abs(values), dtype=np.float64))
                squared = values * values
                acc[2] += float(np.sum(squared, dtype=np.float64))
                acc[3] += float(np.sum(values, dtype=np.float64))
                acc[4] += float(np.sum(squared * values, dtype=np.float64))
                acc[5] += float(np.sum(squared * squared, dtype=np.float64))
                acc[6] = min(acc[6], float(np.min(values)))
                acc[7] = max(acc[7], float(np.max(values)))

    result: dict[str, list[dict[str, float]]] = {}
    for (family, layer), acc in sorted(accumulators.items()):
        count, l1, sum2, total, sum3, sum4, minimum, maximum = acc
        mean = total / count
        raw2, raw3, raw4 = sum2 / count, sum3 / count, sum4 / count
        variance = max(0.0, raw2 - mean * mean)
        third = raw3 - 3.0 * mean * raw2 + 2.0 * mean**3
        fourth = raw4 - 4.0 * mean * raw3 + 6.0 * mean * mean * raw2 - 3.0 * mean**4
        result.setdefault(family, []).append(
            {
                "layer": float(layer),
                "l1_norm": l1,
                "l2_norm": math.sqrt(max(0.0, sum2)),
                "mean": mean,
                "std": math.sqrt(variance),
                "third_moment": third,
                "fourth_moment": max(0.0, fourth),
                "min": minimum,
                "max": maximum,
            }
        )
    return result


def _layer_identity(key: str) -> tuple[str, int] | None:
    normalized = key.lower().replace(".", "/")
    first = normalized.split("/", 1)[0]
    family = {
        "params": "param",
        "parameters": "param",
        "param": "param",
        "grads": "grad",
        "gradients": "grad",
        "grad": "grad",
        "updates": "update",
        "update": "update",
    }.get(first)
    match = _LAYER_KEY.search(normalized)
    return (family, int(match.group(1))) if family is not None and match else None


def _report_payload(
    runs: Sequence[_Run], skipped: Mapping[str, str], notices: Sequence[str]
) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        result = run.result
        metrics = _object(result["metrics"], "result metrics")
        evaluation = result.get("evaluations", {})
        fresh = evaluation.get("fresh10") if isinstance(evaluation, dict) else None
        fresh_loss = fresh.get("macro_loss") if isinstance(fresh, dict) else None
        submission = (
            run.record.get("submission")
            if isinstance(run.record, dict) and isinstance(run.record.get("submission"), str)
            else _submission_from_run_id(run.run_id)
        )
        selected, classification = _default_run_selection(
            str(result["track"]), str(result["profile"]), int(metrics["tokens_processed"])
        )
        qualified = run.record.get("qualified") if run.record else None
        if not isinstance(qualified, bool):
            qualified = None
        run_rows.append(
            {
                "id": run.run_id,
                "submission": submission,
                "label": f"{submission} · {result['profile']} · {run.run_id[:15]}",
                "track": result["track"],
                "profile": result["profile"],
                "classification": classification,
                "selected": selected,
                "seed": result["seed"],
                "color": _COLORS[index % len(_COLORS)],
                "ledger": run.record is not None,
                "flopSource": run.flop_source,
                "finalStep": int(run.training[-1]["step"]),
                "tokens": int(metrics["tokens_processed"]),
                "trainSeconds": metrics.get("train_seconds"),
                "trainLoss": metrics.get("train_loss"),
                "validationLoss": metrics.get("validation_loss"),
                "fresh10Loss": fresh_loss,
                "qualified": qualified,
                "hasLayerStats": bool(run.layer_stats),
            }
        )

    time_charts: list[dict[str, Any]] = []
    time_charts.append(_training_chart(runs, "train_loss", "Training loss", "loss"))
    time_charts.append(_validation_chart(runs, "fineweb", "FineWeb validation loss"))
    time_charts.append(_validation_chart(runs, "fresh10_macro", "Fresh10 macro loss"))
    domains = list(_FRESH10_ORDER)
    discovered = sorted(
        {
            str(row["domain"])
            for run in runs
            for row in run.validation
            if row["kind"] == "downstream"
        }
        - set(domains)
    )
    domains.extend(discovered)
    for domain in domains:
        chart = _validation_chart(runs, domain, f"Fresh10 · {_humanize(domain)}")
        if chart["series"]:
            time_charts.append(chart)
    time_charts.append(_training_chart(runs, "learning_rate", "Learning rate", "rate"))

    reserved = {
        "step",
        "tokens_processed",
        "estimated_flops",
        "cumulative_estimated_flops",
        "train_loss",
        "learning_rate",
    }
    extras: list[str] = []
    for run in runs:
        for metric in run.train_metrics:
            if metric not in reserved and metric not in extras and _diagnostic_metric(metric):
                extras.append(metric)
    for metric in sorted(extras, key=_metric_sort_key):
        time_charts.append(
            _training_chart(runs, metric, _humanize(metric), _metric_unit(metric))
        )
    time_charts = [chart for chart in time_charts if chart["series"]]

    recorded_overall = {
        identity
        for run in runs
        for metric in run.train_metrics
        if (identity := _overall_metric_identity(metric)) is not None
    }
    requested_overall = [
        (family, statistic)
        for family in ("grad", "update", "param")
        for statistic in (
            "l1_norm",
            "l2_norm",
            "mean",
            "std",
            "third_moment",
            "fourth_moment",
        )
    ]
    missing_overall = [
        _overall_metric_label(identity)
        for identity in requested_overall
        if identity not in recorded_overall
    ]
    recorded_labels = [
        _overall_metric_label(identity)
        for identity in requested_overall
        if identity in recorded_overall
    ]
    notices = list(notices) + [
        "Overall training diagnostic coverage: recorded "
        + (", ".join(recorded_labels) if recorded_labels else "none")
        + "; not recorded: "
        + (", ".join(missing_overall) if missing_overall else "none")
        + ". Future numeric training.csv columns with grad/update/param metric names "
        "are discovered automatically."
    ]

    layer_charts: list[dict[str, Any]] = []
    for family in ("param", "grad", "update"):
        for metric in _LAYER_METRICS:
            series = []
            for run in runs:
                points = [
                    [int(row["layer"]), row[metric]]
                    for row in run.layer_stats.get(family, [])
                    if metric in row
                ]
                if points:
                    series.append({"run": run.run_id, "points": points})
            if series:
                layer_charts.append(
                    {
                        "key": f"layer_{family}_{metric}",
                        "title": f"{family.title()} · {_humanize(metric)} by layer",
                        "yLabel": _metric_unit(metric),
                        "series": series,
                    }
                )

    missing_final_families = []
    for family in ("grad", "update"):
        if not any(family in run.layer_stats for run in runs):
            missing_final_families.append(family)
    if missing_final_families:
        notices = list(notices) + [
            "Final-checkpoint "
            + "/".join(missing_final_families)
            + " layer metrics are unavailable; they will appear when checkpoints record those arrays."
        ]

    return {
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "included": len(runs),
            "skipped": len(skipped),
            "defaultXAxis": "flops",
            "flopsLabel": "Estimated cumulative FLOPs",
        },
        "runs": run_rows,
        "timeCharts": time_charts,
        "layerCharts": layer_charts,
        "notices": list(dict.fromkeys(notices)),
        "skipped": [{"run": key, "reason": value} for key, value in skipped.items()],
    }


def _training_chart(
    runs: Sequence[_Run], metric: str, title: str, y_label: str
) -> dict[str, Any]:
    series: list[dict[str, Any]] = []
    for run in runs:
        points = [
            [row["step"], row["estimated_flops"], row[metric]]
            for row in run.training
            if metric in row
        ]
        if points:
            limit = int(run.result.get("_report_point_limit", _MAX_CHART_POINTS))
            # The report opens in equi-FLOP mode, so point selection must preserve
            # shape in that coordinate rather than silently optimizing for steps.
            points = _lttb(points, limit, x_index=1)
            series.append({"run": run.run_id, "points": points})
    return {
        "key": metric,
        "title": title,
        "yLabel": y_label,
        "series": series,
    }


def _validation_chart(
    runs: Sequence[_Run], identity: str, title: str
) -> dict[str, Any]:
    series = []
    for run in runs:
        if identity == "fineweb":
            selected = [
                row
                for row in run.validation
                if row["domain"] == "fineweb"
                and row["kind"] in {"fineweb_probe", "fineweb", "fineweb_final"}
            ]
        elif identity == "fresh10_macro":
            selected = [row for row in run.validation if row["kind"] == "downstream_macro"]
        else:
            selected = [
                row
                for row in run.validation
                if row["kind"] == "downstream" and row["domain"] == identity
            ]
        if selected:
            series.append(
                {
                    "run": run.run_id,
                    "points": [
                        [row["step"], row["estimated_flops"], row["validation_loss"]]
                        for row in selected
                    ],
                }
            )
    return {
        "key": f"validation_{identity}",
        "title": title,
        "yLabel": "loss",
        "series": series,
    }


def _lttb(
    points: Sequence[Sequence[float]], threshold: int, *, x_index: int = 1
) -> list[list[float]]:
    """Largest-Triangle-Three-Buckets downsampling, preserving endpoints.

    Time-series rows are ``[step, cumulative FLOPs, value]`` and therefore use
    the report's default FLOP coordinate unless a caller explicitly chooses a
    different x column.
    """

    if threshold >= len(points) or threshold == 0:
        return [list(point) for point in points]
    sampled: list[list[float]] = [list(points[0])]
    every = (len(points) - 2) / (threshold - 2)
    anchor = 0
    for bucket in range(threshold - 2):
        average_start = int(math.floor((bucket + 1) * every)) + 1
        average_end = min(int(math.floor((bucket + 2) * every)) + 1, len(points))
        average_range = points[average_start:average_end]
        if average_range:
            average_x = (
                math.fsum(point[x_index] for point in average_range) / len(average_range)
            )
            average_y = math.fsum(point[2] for point in average_range) / len(average_range)
        else:
            average_x, average_y = points[-1][x_index], points[-1][2]
        range_start = int(math.floor(bucket * every)) + 1
        range_end = min(int(math.floor((bucket + 1) * every)) + 1, len(points) - 1)
        point_a = points[anchor]
        max_area = -1.0
        selected = range_start
        for index in range(range_start, max(range_start + 1, range_end)):
            point = points[index]
            area = abs(
                (point_a[x_index] - average_x) * (point[2] - point_a[2])
                - (point_a[x_index] - point[x_index]) * (average_y - point_a[2])
            )
            if area > max_area:
                max_area = area
                selected = index
        sampled.append(list(points[selected]))
        anchor = selected
    sampled.append(list(points[-1]))
    return sampled


def _diagnostic_metric(name: str) -> bool:
    lowered = name.lower()
    if lowered in {"grad_norm", "update_norm", "parameter_norm", "param_norm"}:
        return True
    family = any(token in lowered for token in ("grad", "update", "param"))
    statistic = any(
        token in lowered
        for token in (
            "norm",
            "mean",
            "std",
            "moment",
            "min",
            "max",
            "percentile",
            "hist",
            "skew",
            "kurt",
        )
    )
    return family and statistic


def _overall_metric_identity(name: str) -> tuple[str, str] | None:
    """Map a numeric diagnostic column name to the requested coverage grid."""

    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    tokens = set(normalized.split("_"))
    if tokens & {"grad", "grads", "gradient", "gradients"}:
        family = "grad"
    elif tokens & {"update", "updates"}:
        family = "update"
    elif tokens & {"param", "params", "parameter", "parameters"}:
        family = "param"
    else:
        return None

    compact = normalized.replace("_", "")
    if "l1norm" in compact or "norml1" in compact:
        statistic = "l1_norm"
    elif "l2norm" in compact or "norml2" in compact:
        statistic = "l2_norm"
    elif "thirdmoment" in compact or "moment3" in compact or "m3" in tokens:
        statistic = "third_moment"
    elif "fourthmoment" in compact or "moment4" in compact or "m4" in tokens:
        statistic = "fourth_moment"
    elif tokens & {"std", "stdev", "stddev"}:
        statistic = "std"
    elif "mean" in tokens:
        statistic = "mean"
    elif "norm" in tokens or normalized.endswith("norm"):
        # The existing reference `grad_norm` is the global L2 norm.
        statistic = "l2_norm"
    else:
        return None
    return family, statistic


def _overall_metric_label(identity: tuple[str, str]) -> str:
    family, statistic = identity
    family_label = {"grad": "gradient", "update": "update", "param": "parameter"}[
        family
    ]
    statistic_label = {
        "l1_norm": "L1 norm",
        "l2_norm": "L2 norm",
        "mean": "mean",
        "std": "std",
        "third_moment": "third moment",
        "fourth_moment": "fourth moment",
    }[statistic]
    return f"{family_label} {statistic_label}"


def _default_run_selection(track: str, profile: str, tokens: int) -> tuple[bool, str]:
    """Return the initial visibility and honest display classification."""

    if profile == "official" and track == "sample_efficiency":
        return True, "official"
    if profile == "official" and track == "open":
        if tokens == _OFFICIAL_OPEN_TOKENS:
            return True, "official"
        return False, "partial"
    if profile in {"smoke", "dev"}:
        return False, "diagnostic"
    return False, "partial"


def _metric_sort_key(name: str) -> tuple[int, str]:
    lowered = name.lower()
    family = 0 if "grad" in lowered else 1 if "update" in lowered else 2
    return family, lowered


def _metric_unit(name: str) -> str:
    lowered = name.lower()
    if "loss" in lowered:
        return "loss"
    if "learning_rate" in lowered or lowered == "lr":
        return "rate"
    if "norm" in lowered:
        return "norm"
    return "value"


def _humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def _submission_from_run_id(run_id: str) -> str:
    # Run IDs are timestamp-submission-random.  This is display-only; the ledger is
    # authoritative whenever it exists.
    match = re.match(r"^\d{8}T\d{6}\.\d+Z-(.+)-[0-9a-f]{8}$", run_id)
    return match.group(1) if match else run_id


def _render_html(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    # Data lives in a script text node; escape HTML-significant characters so even
    # a deliberately strange submission name cannot terminate it.
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return _HTML.replace("__REPORT_DATA__", encoded)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportError(f"{name} is not an object")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReportError(f"{name} is not a positive integer")
    return value


def _positive_int_text(value: str | None, name: str) -> int:
    if value is None or not re.fullmatch(r"[1-9][0-9]*", value.strip()):
        raise ReportError(f"{name} is not a positive integer")
    return int(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReportError(f"{name} is not finite")
    return number


def _positive_finite(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ReportError(f"{name} must be greater than zero")
    return number


def _finite_text(value: str | None, name: str) -> float:
    if value is None:
        raise ReportError(f"{name} is missing")
    try:
        number = float(value)
    except ValueError as exc:
        raise ReportError(f"{name} is not numeric") from exc
    if not math.isfinite(number):
        raise ReportError(f"{name} is not finite")
    return number


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>GPT TPU Speedrun Report</title>
<style>
:root{--bg:#080b12;--panel:#101521;--panel2:#151b29;--line:#273247;--text:#edf4ff;--muted:#91a0b8;--accent:#7dd3fc;--good:#86efac;--warn:#fde047;--bad:#fca5a5;--radius:14px;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--text);background:var(--bg);font-synthesis:none}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 75% -10%,#17233a 0,transparent 35rem),var(--bg);min-height:100vh}button,input{font:inherit}.shell{display:grid;grid-template-columns:280px minmax(0,1fr);min-height:100vh}.sidebar{position:sticky;top:0;height:100vh;overflow:auto;border-right:1px solid var(--line);background:rgba(10,14,23,.94);backdrop-filter:blur(16px);padding:20px}.brand{font-weight:800;letter-spacing:.02em;font-size:17px}.brand b{color:var(--accent)}.subtle{color:var(--muted);font-size:12px;line-height:1.5}.side-actions{display:flex;gap:7px;margin:18px 0 10px}.ghost{border:1px solid var(--line);color:var(--muted);background:var(--panel);border-radius:8px;padding:6px 9px;cursor:pointer}.ghost:hover{color:var(--text);border-color:#41516d}.search{width:100%;border:1px solid var(--line);background:#090d16;color:var(--text);padding:9px 10px;border-radius:9px;outline:none}.search:focus{border-color:var(--accent)}.run-list{display:grid;gap:7px;margin-top:12px}.run-toggle{display:grid;grid-template-columns:auto 10px 1fr;gap:9px;align-items:start;padding:9px;border:1px solid transparent;border-radius:10px;cursor:pointer}.run-toggle:hover{background:var(--panel);border-color:var(--line)}.run-toggle input{margin-top:3px;accent-color:var(--accent)}.dot{width:9px;height:9px;border-radius:50%;margin-top:4px;box-shadow:0 0 14px currentColor}.run-name{font-size:12px;font-weight:650;overflow-wrap:anywhere}.run-meta{display:block;color:var(--muted);font-size:10px;margin-top:3px}.main{min-width:0;padding:26px 28px 60px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:20px}.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:750}.top h1{font-size:clamp(25px,3vw,42px);letter-spacing:-.04em;margin:5px 0 7px}.stats{display:flex;gap:8px;flex-wrap:wrap}.pill{border:1px solid var(--line);background:rgba(16,21,33,.8);border-radius:999px;padding:7px 10px;color:var(--muted);font-size:12px}.pill strong{color:var(--text)}.axis-control{flex:none;border:1px solid var(--line);border-radius:11px;padding:4px;background:#090d16;display:flex}.axis-control label{cursor:pointer}.axis-control input{position:absolute;opacity:0;pointer-events:none}.axis-control span{display:block;padding:8px 11px;border-radius:7px;color:var(--muted);font-size:12px;font-weight:700}.axis-control input:checked+span{background:var(--panel2);color:var(--text);box-shadow:0 1px 5px #0008}.axis-hint{text-align:right;color:var(--muted);font-size:10px;margin-top:6px}.notice-wrap{display:grid;gap:7px;margin:16px 0}.notice{padding:10px 12px;border:1px solid #3d3940;background:#18161c;color:#c6b9c6;border-radius:10px;font-size:12px}.section-title{margin:30px 0 12px;font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,410px),1fr));gap:12px}.chart{background:linear-gradient(145deg,rgba(20,27,42,.92),rgba(12,17,27,.92));border:1px solid var(--line);border-radius:var(--radius);padding:15px;min-width:0}.chart-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:7px}.chart h2{font-size:14px;margin:0;letter-spacing:-.01em}.chart-unit{color:var(--muted);font-size:10px}.canvas-wrap{height:270px;position:relative}.chart canvas{display:block;width:100%;height:100%;touch-action:none}.tooltip{position:fixed;z-index:20;pointer-events:none;background:#070a11ec;border:1px solid #36435b;border-radius:8px;padding:7px 9px;box-shadow:0 12px 30px #000a;font-size:11px;line-height:1.45;display:none;max-width:270px}.empty{border:1px dashed var(--line);border-radius:var(--radius);padding:30px;color:var(--muted);text-align:center}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:var(--radius);background:var(--panel)}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}th:first-child,td:first-child{text-align:left}tbody tr:last-child td{border-bottom:0}.status{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:7px}.footer{color:var(--muted);font-size:11px;margin-top:26px}.mobile-runs{display:none}.skip details{color:var(--muted);font-size:12px}.skip summary{cursor:pointer}.skip code{color:var(--bad);white-space:normal;overflow-wrap:anywhere}@media(max-width:850px){.shell{grid-template-columns:1fr}.sidebar{display:none;position:fixed;z-index:30;width:min(88vw,320px);box-shadow:20px 0 70px #000;height:100vh}.sidebar.open{display:block}.main{padding:20px 14px 50px}.mobile-runs{display:inline-flex}.top{align-items:stretch;flex-direction:column}.axis-control{align-self:flex-start}.axis-hint{text-align:left}.canvas-wrap{height:240px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style>
</head>
<body>
<div class="shell">
<aside class="sidebar" id="sidebar">
  <div class="brand"><b>◆</b> GPT TPU SPEEDRUN</div>
  <p class="subtle">Toggle any number of complete, integrity-checked runs. Colors stay consistent across every plot.</p>
  <input class="search" id="run-search" type="search" placeholder="Filter runs…" aria-label="Filter runs">
  <div class="side-actions"><button class="ghost" id="all-runs">All</button><button class="ghost" id="no-runs">None</button><button class="ghost mobile-runs" id="close-runs">Close</button></div>
  <div class="run-list" id="run-list"></div>
</aside>
<main class="main">
  <div class="top">
    <div><div class="eyebrow">Static performance dossier</div><h1>Training, at a glance.</h1><div class="stats" id="stats"></div></div>
    <div><button class="ghost mobile-runs" id="open-runs">Choose runs</button><div class="axis-control" role="radiogroup" aria-label="Time-series x-axis"><label><input type="radio" name="axis" value="flops" checked><span>equi-FLOP</span></label><label><input type="radio" name="axis" value="step"><span>equi-step</span></label></div><div class="axis-hint" id="axis-hint">Estimated cumulative FLOPs</div></div>
  </div>
  <div class="notice-wrap" id="notices"></div>
  <div class="section-title">Run summary</div><div id="summary"></div>
  <div class="section-title">Training timeline</div><div class="charts" id="time-charts"></div>
  <div class="section-title">Final checkpoint · x = logical layer</div><div class="charts" id="layer-charts"></div>
  <div class="skip" id="skipped"></div>
  <div class="footer" id="footer"></div>
</main>
</div>
<div class="tooltip" id="tooltip"></div>
<script type="application/json" id="report-data">__REPORT_DATA__</script>
<script>
(()=>{'use strict';
const D=JSON.parse(document.getElementById('report-data').textContent), runMap=new Map(D.runs.map(r=>[r.id,r]));
const visible=new Set(D.runs.filter(r=>r.selected).map(r=>r.id)); let axis='flops', charts=[], frame=0;
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(n,d=3)=>{if(n==null||!Number.isFinite(+n))return '—';n=+n;const a=Math.abs(n);if(a>=1e18)return (n/1e18).toFixed(2)+' EF';if(a>=1e15)return (n/1e15).toFixed(2)+' PF';if(a>=1e12)return (n/1e12).toFixed(2)+' T';if(a>=1e9)return (n/1e9).toFixed(2)+' B';if(a>=1e6)return (n/1e6).toFixed(2)+' M';if(a>=1e3)return (n/1e3).toFixed(2)+' k';if(a>0&&a<.001)return n.toExponential(2);const f=n.toFixed(d);return d?f.replace(/(\.\d*?[1-9])0+$|\.0+$/,'$1'):f};
function init(){
 $('stats').innerHTML=`<span class="pill"><strong>${D.meta.included}</strong> plotted</span><span class="pill"><strong>${D.meta.skipped}</strong> skipped</span><span class="pill">default <strong>equi-FLOP</strong></span>`;
 $('notices').innerHTML=D.notices.map(n=>`<div class="notice">${esc(n)}</div>`).join('');
 buildRuns(); buildSummary(); buildCharts(); buildSkipped();
 $('footer').textContent=`Generated ${new Date(D.meta.generatedAt).toLocaleString()} · portable HTML · no network or external JavaScript`;
 document.querySelectorAll('input[name=axis]').forEach(r=>r.addEventListener('change',()=>{axis=r.value;$('axis-hint').textContent=axis==='flops'?'Estimated cumulative FLOPs':'Optimizer step';schedule()}));
 $('all-runs').onclick=()=>{D.runs.forEach(r=>visible.add(r.id));syncChecks();schedule()}; $('no-runs').onclick=()=>{visible.clear();syncChecks();schedule()};
 $('run-search').oninput=e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('.run-toggle').forEach(x=>x.hidden=!x.dataset.search.includes(q))};
 $('open-runs').onclick=()=>{$('sidebar').classList.add('open')}; $('close-runs').onclick=()=>{$('sidebar').classList.remove('open')};
 new ResizeObserver(schedule).observe(document.querySelector('.main')); schedule();
}
function buildRuns(){$('run-list').innerHTML=D.runs.map(r=>`<label class="run-toggle" data-search="${esc((r.label+' '+r.id+' '+r.classification).toLowerCase())}"><input type="checkbox" data-run="${esc(r.id)}" ${r.selected?'checked':''}><span class="dot" style="color:${r.color};background:${r.color}"></span><span class="run-name">${esc(r.label)}<small class="run-meta">${esc(r.classification)} · step ${fmt(r.finalStep,0)} · val ${fmt(r.validationLoss,4)}${r.ledger?' · ledger ✓':' · unledgered'}</small></span></label>`).join('')||'<p class="subtle">No plot-able runs found.</p>';document.querySelectorAll('[data-run]').forEach(x=>x.onchange=()=>{x.checked?visible.add(x.dataset.run):visible.delete(x.dataset.run);schedule()})}
function syncChecks(){document.querySelectorAll('[data-run]').forEach(x=>x.checked=visible.has(x.dataset.run))}
function buildSummary(){if(!D.runs.length){$('summary').innerHTML='<div class="empty">No completed runs passed the report integrity checks.</div>';return}$('summary').innerHTML=`<div class="table-wrap"><table><thead><tr><th>Run</th><th>Class</th><th>Steps</th><th>Tokens</th><th>Train s</th><th>Train loss</th><th>Val loss</th><th>Fresh10</th><th>FLOP x</th><th>Checkpoint layers</th></tr></thead><tbody>${D.runs.map(r=>`<tr><td><span class="status" style="background:${r.color}"></span>${esc(r.label)}</td><td>${esc(r.classification)}</td><td>${fmt(r.finalStep,0)}</td><td>${fmt(r.tokens,0)}</td><td>${fmt(r.trainSeconds,2)}</td><td>${fmt(r.trainLoss,4)}</td><td>${fmt(r.validationLoss,4)}</td><td>${fmt(r.fresh10Loss,4)}</td><td title="${esc(r.flopSource)}">${r.flopSource.startsWith('derived:')?'derived':'logged'}</td><td>${r.hasLayerStats?'yes':'—'}</td></tr>`).join('')}</tbody></table></div>`}
function buildCharts(){charts=[];makeGroup($('time-charts'),D.timeCharts,'time');makeGroup($('layer-charts'),D.layerCharts,'layer');if(!D.layerCharts.length)$('layer-charts').innerHTML='<div class="empty">No retained checkpoint contains recognized per-layer arrays. Qualifying retention may intentionally remove checkpoints.</div>'}
function makeGroup(root,data,type){root.innerHTML=data.map((c,i)=>`<article class="chart"><div class="chart-head"><h2>${esc(c.title)}</h2><span class="chart-unit">${esc(c.yLabel)}</span></div><div class="canvas-wrap"><canvas aria-label="${esc(c.title)}"></canvas></div></article>`).join('');root.querySelectorAll('canvas').forEach((canvas,i)=>{const item={canvas,data:data[i],type};canvas.onpointermove=e=>tip(e,item);canvas.onpointerleave=hideTip;charts.push(item)})}
function buildSkipped(){if(!D.skipped.length)return;$('skipped').innerHTML=`<div class="section-title">Excluded by integrity scan</div>${D.skipped.map(x=>`<details><summary>${esc(x.run)}</summary><code>${esc(x.reason)}</code></details>`).join('')}`}
function schedule(){cancelAnimationFrame(frame);frame=requestAnimationFrame(()=>charts.forEach(draw))}
function bounds(item){let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity,count=0;for(const s of item.data.series){if(!visible.has(s.run))continue;for(const p of s.points){const x=item.type==='layer'?p[0]:p[axis==='flops'?1:0],y=item.type==='layer'?p[1]:p[2];if(Number.isFinite(x)&&Number.isFinite(y)){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);count++}}}if(!count)return null;if(x0===x1){x0-=.5;x1+=.5}if(y0===y1){const q=Math.abs(y0)*.05||.5;y0-=q;y1+=q}else{const q=(y1-y0)*.08;y0-=q;y1+=q}return{x0,x1,y0,y1}}
function draw(item){const c=item.canvas,rect=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,rect.width),h=Math.max(1,rect.height);if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr)}const g=c.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);const m={l:54,r:14,t:12,b:31},b=bounds(item);if(!b){g.fillStyle='#91a0b8';g.font='12px system-ui';g.textAlign='center';g.fillText('No selected run has this metric',w/2,h/2);return}const X=x=>m.l+(x-b.x0)/(b.x1-b.x0)*(w-m.l-m.r),Y=y=>m.t+(b.y1-y)/(b.y1-b.y0)*(h-m.t-m.b);g.strokeStyle='#253047';g.lineWidth=1;g.fillStyle='#7e8da6';g.font='10px ui-monospace,monospace';for(let i=0;i<=4;i++){const y=m.t+(h-m.t-m.b)*i/4,v=b.y1-(b.y1-b.y0)*i/4;g.beginPath();g.moveTo(m.l,y);g.lineTo(w-m.r,y);g.stroke();g.textAlign='right';g.fillText(fmt(v,3),m.l-7,y+3)}for(let i=0;i<=4;i++){const x=m.l+(w-m.l-m.r)*i/4,v=b.x0+(b.x1-b.x0)*i/4;g.textAlign=i===0?'left':i===4?'right':'center';g.fillText(fmt(v,item.type==='layer'?0:2),x,h-9)}for(const s of item.data.series){if(!visible.has(s.run))continue;const r=runMap.get(s.run);g.strokeStyle=r.color;g.fillStyle=r.color;g.lineWidth=1.7;g.globalAlpha=.9;g.beginPath();let n=0;for(const p of s.points){const x=item.type==='layer'?p[0]:p[axis==='flops'?1:0],y=item.type==='layer'?p[1]:p[2];if(!Number.isFinite(x)||!Number.isFinite(y))continue;n?g.lineTo(X(x),Y(y)):g.moveTo(X(x),Y(y));n++}g.stroke();if(n<=20)for(const p of s.points){const x=item.type==='layer'?p[0]:p[axis==='flops'?1:0],y=item.type==='layer'?p[1]:p[2];g.beginPath();g.arc(X(x),Y(y),2.3,0,Math.PI*2);g.fill()}}g.globalAlpha=1;item._plot={b,m,w,h,X,Y}}
function tip(e,item){if(!item._plot)return;const rect=item.canvas.getBoundingClientRect(),px=e.clientX-rect.left;let best=null;for(const s of item.data.series){if(!visible.has(s.run))continue;for(const p of s.points){const x=item.type==='layer'?p[0]:p[axis==='flops'?1:0],y=item.type==='layer'?p[1]:p[2],dist=Math.abs(item._plot.X(x)-px);if(!best||dist<best.dist)best={dist,p,x,y,run:runMap.get(s.run)}}}if(!best||best.dist>30){hideTip();return}const xname=item.type==='layer'?'layer':axis==='flops'?'estimated FLOPs':'step',t=$('tooltip');t.innerHTML=`<b style="color:${best.run.color}">${esc(best.run.label)}</b><br>${xname}: ${fmt(best.x,item.type==='layer'||axis==='step'?0:2)}<br>${esc(item.data.title)}: ${fmt(best.y,5)}`;t.style.display='block';t.style.left=Math.min(innerWidth-285,e.clientX+13)+'px';t.style.top=Math.min(innerHeight-90,e.clientY+13)+'px'}function hideTip(){$('tooltip').style.display='none'}
init();
})();
</script>
</body>
</html>
'''
