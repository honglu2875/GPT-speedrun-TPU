"""Validation of result events, contracts, paths, and evaluator output."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .errors import ResultValidationError
from .models import Evaluator, ReferenceContract, ValidationResult


RESULT_PREFIX = "SPEEDRUN_RESULT="
SCHEMA_VERSION = 1
MAX_RESULT_BYTES = 1_000_000
_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def parse_result_line(stdout: str) -> dict[str, Any]:
    """Parse the final non-empty stdout line as a v1 result event."""

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines or not lines[-1].startswith(RESULT_PREFIX):
        raise ResultValidationError(
            f"final non-empty stdout line must begin with {RESULT_PREFIX!r}"
        )
    encoded = lines[-1][len(RESULT_PREFIX) :]
    if not encoded:
        raise ResultValidationError("result event contains no JSON payload")
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ResultValidationError("result event is larger than 1 MB")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ResultValidationError(f"result event is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultValidationError("result payload must be a JSON object")
    return payload


def _finite_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultValidationError(f"{name} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ResultValidationError(f"{name} must be finite and >= {minimum:g}") from exc
    if not math.isfinite(number) or number < minimum:
        raise ResultValidationError(f"{name} must be finite and >= {minimum:g}")
    return number


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ResultValidationError(f"{name} must be a positive integer")
    return value


def _plain_object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ResultValidationError(f"{name} must be a JSON object")
    return value


def reference_contract_dict(
    contract: ReferenceContract | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if contract is None:
        return None
    value = contract.as_dict() if isinstance(contract, ReferenceContract) else dict(contract)
    required = ("model_id", "dataset_id", "tokenizer_id", "sequence_length")
    missing = [key for key in required if key not in value]
    if missing:
        raise ResultValidationError(
            "reference contract is missing: " + ", ".join(sorted(missing))
        )
    for key in required[:3]:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ResultValidationError(f"reference contract {key} must be non-empty")
    _positive_integer(value["sequence_length"], "reference contract sequence_length")
    _ensure_json(value, "reference contract")
    return value


def contained_file(run_dir: Path, relative: Any) -> Path:
    """Resolve a result artifact without permitting absolute paths or escapes."""

    if not isinstance(relative, str) or not relative.strip():
        raise ResultValidationError("checkpoint must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ResultValidationError("checkpoint path must be relative to the run directory")
    root = run_dir.resolve()
    unresolved = root / candidate
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ResultValidationError("checkpoint path escapes the run directory") from exc
    if not resolved.is_file():
        raise ResultValidationError("checkpoint is not a regular file")
    if unresolved.is_symlink():
        raise ResultValidationError("checkpoint may not be a symbolic link")
    # A symlink in a parent is caught by the resolved containment check. Reject a
    # changed target between resolution and hashing as well as practical stdlib can.
    try:
        if not os.path.samefile(resolved, root / candidate):
            raise ResultValidationError("checkpoint changed while being validated")
    except OSError as exc:
        raise ResultValidationError("checkpoint could not be inspected") from exc
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result(
    payload: Mapping[str, Any],
    *,
    run_dir: Path,
    track: str,
    reference_contract: ReferenceContract | Mapping[str, Any] | None = None,
    expected_validation_tokens: int | None = None,
    evaluator: Evaluator | None = None,
) -> ValidationResult:
    """Validate a trainer result and, optionally, independently evaluate it."""

    _ensure_json(payload, "result payload")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ResultValidationError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("status") != "ok":
        raise ResultValidationError("result status must be 'ok'")
    metrics = _plain_object(payload.get("metrics"), "metrics")
    declared_time = _finite_number(
        metrics.get("train_seconds"), "metrics.train_seconds", minimum=0.0
    )
    if declared_time <= 0:
        raise ResultValidationError("metrics.train_seconds must be greater than zero")
    tokens = _positive_integer(metrics.get("tokens_processed"), "metrics.tokens_processed")
    loss = _finite_number(metrics.get("validation_loss"), "metrics.validation_loss")
    if expected_validation_tokens is not None:
        validation_tokens = _positive_integer(
            metrics.get("validation_tokens"), "metrics.validation_tokens"
        )
        if validation_tokens != expected_validation_tokens:
            raise ResultValidationError(
                "metrics.validation_tokens must match the fixed validation prefix: "
                f"expected {expected_validation_tokens:,}, got {validation_tokens:,}"
            )
    checkpoint = contained_file(run_dir, payload.get("checkpoint"))
    artifact_paths: dict[str, Path] = {}
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ResultValidationError("artifacts must be a JSON object")
    for name, relative in artifacts.items():
        if not isinstance(name, str) or not _ARTIFACT_NAME.fullmatch(name):
            raise ResultValidationError(f"invalid artifact name: {name!r}")
        artifact_paths[name] = contained_file(run_dir, relative)

    expected = reference_contract_dict(reference_contract)
    if track == "sample_efficiency":
        if expected is None:
            raise ResultValidationError(
                "sample_efficiency requires a configured reference contract"
            )
        submitted = _plain_object(payload.get("contract"), "contract")
        for key, expected_value in expected.items():
            if submitted.get(key) != expected_value:
                raise ResultValidationError(
                    f"contract mismatch for {key}: expected {expected_value!r}, "
                    f"got {submitted.get(key)!r}"
                )
    elif track != "open":
        raise ResultValidationError(f"unknown track: {track!r}")

    evaluator_metrics: Mapping[str, Any] = {}
    if evaluator is not None:
        try:
            evaluated = evaluator(checkpoint, payload)
        except ResultValidationError:
            raise
        except Exception as exc:
            raise ResultValidationError(f"independent evaluator failed: {exc}") from exc
        if evaluated is not None:
            evaluator_metrics = _plain_object(evaluated, "evaluator result")
            _ensure_json(evaluator_metrics, "evaluator result")
            if "validation_loss" in evaluator_metrics:
                # When a harness-owned evaluator is available, its loss is the
                # canonical qualification value. The raw declared value remains
                # preserved in result.json and inside declared_metrics.
                loss = _finite_number(
                    evaluator_metrics["validation_loss"],
                    "evaluator result validation_loss",
                )

    checkpoint_size = checkpoint.stat().st_size
    return ValidationResult(
        payload=dict(payload),
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        checkpoint_bytes=checkpoint_size,
        declared_train_seconds=declared_time,
        tokens_processed=tokens,
        validation_loss=loss,
        declared_metrics=dict(metrics),
        evaluator_metrics=dict(evaluator_metrics),
        artifacts=artifact_paths,
    )


def _ensure_json(value: Any, name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(f"{name} must contain only finite JSON values") from exc


def verify_run(
    run_dir: Path,
    *,
    track: str = "open",
    reference_contract: ReferenceContract | Mapping[str, Any] | None = None,
    expected_validation_tokens: int | None = None,
    evaluator: Evaluator | None = None,
) -> ValidationResult:
    """Re-validate an existing run from its captured stdout log."""

    stdout_path = run_dir / "stdout.log"
    if not stdout_path.is_file():
        raise ResultValidationError(f"missing captured stdout log: {stdout_path}")
    payload = parse_result_line(stdout_path.read_text(encoding="utf-8", errors="replace"))
    return validate_result(
        payload,
        run_dir=run_dir,
        track=track,
        reference_contract=reference_contract,
        expected_validation_tokens=expected_validation_tokens,
        evaluator=evaluator,
    )
