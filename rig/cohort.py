"""Comparable-run cohort identities.

A cohort names the conditions that make two completed runs comparable while
leaving recipe, architecture, optimizer, batch size, learning rate, and seed as
experimental dimensions.  The identity is explicit in each new record, so a
leaderboard never guesses from a subset of historical fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .plan import RecipePlan


COHORT_SCHEMA_VERSION = 1
HORIZON_ROUNDING = "nearest_complete_global_step"


class CohortError(ValueError):
    """The inputs do not define a valid comparable-run cohort."""


def build_cohort(
    *,
    plan: RecipePlan,
    dataset_id: str,
    tokenizer_id: str,
    dataset_provenance: Mapping[str, Any],
    accelerator: str,
    tpu_vm_count: int,
    chips_per_host: int,
    target_loss: float,
) -> dict[str, Any] | None:
    """Return a canonical cohort object, or ``None`` for non-rankable runs."""

    if plan.run_kind != "full":
        return None
    payload = plan.payload
    declared_parameters = _positive_integer(
        payload.get("declared_parameters"), "declared_parameters"
    )
    target_tpp = _positive_number(
        payload.get("target_tokens_per_parameter"), "target_tokens_per_parameter"
    )
    dataset = _mapping(dataset_provenance.get("dataset"), "dataset provenance")
    manifest = _mapping(dataset.get("manifest"), "dataset manifest provenance")
    canonical_manifest = _digest(
        manifest.get("canonical_sha256"), "dataset manifest canonical_sha256"
    )
    train_files = _string_sequence(dataset.get("train_files"), "dataset train_files")
    validation_files = _string_sequence(
        dataset.get("validation_files"), "dataset validation_files"
    )
    validation_prefix = _positive_integer(
        dataset.get("validation_prefix_tokens"), "validation_prefix_tokens"
    )
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise CohortError("dataset_id must be a non-empty string")
    if not isinstance(tokenizer_id, str) or not tokenizer_id.strip():
        raise CohortError("tokenizer_id must be a non-empty string")
    if not isinstance(accelerator, str) or not accelerator.strip():
        raise CohortError("accelerator must be a non-empty string")
    _positive_integer(tpu_vm_count, "tpu_vm_count")
    _positive_integer(chips_per_host, "chips_per_host")
    target_loss_value = _nonnegative_number(target_loss, "target_loss")

    body: dict[str, Any] = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "profile": str(payload["profile"]),
        "tier": str(payload["tier"]),
        "declared_parameters": declared_parameters,
        "horizon": {
            "target_tokens_per_parameter": _canonical_number(target_tpp),
            "rounding": HORIZON_ROUNDING,
        },
        "data": {
            "dataset_id": dataset_id.strip(),
            "tokenizer_id": tokenizer_id.strip(),
            "manifest_canonical_sha256": canonical_manifest,
            "train_files": train_files,
            "train_files_sha256": _sequence_sha256(train_files),
            "validation_files": validation_files,
            "validation_prefix_tokens": validation_prefix,
        },
        "hardware": {
            "accelerator": accelerator.strip(),
            "tpu_vm_count": tpu_vm_count,
            "chips_per_host": chips_per_host,
            "total_chips": tpu_vm_count * chips_per_host,
        },
        "qualification": {"target_loss": _canonical_number(target_loss_value)},
    }
    fresh10 = dataset_provenance.get("fresh10")
    if fresh10 is not None:
        fresh = _mapping(fresh10, "Fresh10 provenance")
        fresh_manifest = _mapping(fresh.get("manifest"), "Fresh10 manifest provenance")
        body["evaluation"] = {
            "name": _string(fresh.get("name"), "Fresh10 name"),
            "manifest_canonical_sha256": _digest(
                fresh_manifest.get("canonical_sha256"),
                "Fresh10 manifest canonical_sha256",
            ),
            "scored_tokens": _positive_integer(
                fresh.get("scored_tokens"), "Fresh10 scored_tokens"
            ),
        }
    cohort_id = _canonical_sha256(body)
    return {**body, "cohort_id": cohort_id}


def validate_cohort(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a stored cohort's schema and recompute its content address."""

    if not isinstance(value, Mapping):
        raise CohortError("cohort must be a JSON object")
    copied = _json_copy(value)
    cohort_id = _digest(copied.pop("cohort_id", None), "cohort_id")
    _exact_keys(
        copied,
        "cohort",
        {
            "schema_version",
            "profile",
            "tier",
            "declared_parameters",
            "horizon",
            "data",
            "hardware",
            "qualification",
        },
        optional={"evaluation"},
    )
    if type(copied["schema_version"]) is not int or (
        copied["schema_version"] != COHORT_SCHEMA_VERSION
    ):
        raise CohortError(f"cohort schema_version must be {COHORT_SCHEMA_VERSION}")
    _string(copied["profile"], "cohort profile")
    _string(copied["tier"], "cohort tier")
    _positive_integer(copied["declared_parameters"], "cohort declared_parameters")

    horizon = _mapping(copied["horizon"], "cohort horizon")
    _exact_keys(
        horizon,
        "cohort horizon",
        {"target_tokens_per_parameter", "rounding"},
    )
    _canonical_numeric_string(
        horizon["target_tokens_per_parameter"],
        "cohort target_tokens_per_parameter",
        positive=True,
    )
    if horizon["rounding"] != HORIZON_ROUNDING:
        raise CohortError(f"cohort rounding must be {HORIZON_ROUNDING!r}")

    data = _mapping(copied["data"], "cohort data")
    _exact_keys(
        data,
        "cohort data",
        {
            "dataset_id",
            "tokenizer_id",
            "manifest_canonical_sha256",
            "train_files",
            "train_files_sha256",
            "validation_files",
            "validation_prefix_tokens",
        },
    )
    _string(data["dataset_id"], "cohort dataset_id")
    _string(data["tokenizer_id"], "cohort tokenizer_id")
    _digest(data["manifest_canonical_sha256"], "cohort manifest_canonical_sha256")
    train_files = _string_sequence(data["train_files"], "cohort train_files")
    train_files_sha256 = _digest(
        data["train_files_sha256"], "cohort train_files_sha256"
    )
    if train_files_sha256 != _sequence_sha256(train_files):
        raise CohortError("cohort train_files_sha256 does not match train_files")
    _string_sequence(data["validation_files"], "cohort validation_files")
    _positive_integer(
        data["validation_prefix_tokens"], "cohort validation_prefix_tokens"
    )

    hardware = _mapping(copied["hardware"], "cohort hardware")
    _exact_keys(
        hardware,
        "cohort hardware",
        {"accelerator", "tpu_vm_count", "chips_per_host", "total_chips"},
    )
    _string(hardware["accelerator"], "cohort accelerator")
    hosts = _positive_integer(hardware["tpu_vm_count"], "cohort tpu_vm_count")
    chips = _positive_integer(hardware["chips_per_host"], "cohort chips_per_host")
    total = _positive_integer(hardware["total_chips"], "cohort total_chips")
    if total != hosts * chips:
        raise CohortError("cohort total_chips must equal tpu_vm_count * chips_per_host")

    qualification = _mapping(copied["qualification"], "cohort qualification")
    _exact_keys(qualification, "cohort qualification", {"target_loss"})
    _canonical_numeric_string(
        qualification["target_loss"], "cohort target_loss", positive=False
    )

    if "evaluation" in copied:
        evaluation = _mapping(copied["evaluation"], "cohort evaluation")
        _exact_keys(
            evaluation,
            "cohort evaluation",
            {"name", "manifest_canonical_sha256", "scored_tokens"},
        )
        _string(evaluation["name"], "cohort evaluation name")
        _digest(
            evaluation["manifest_canonical_sha256"],
            "cohort evaluation manifest_canonical_sha256",
        )
        _positive_integer(
            evaluation["scored_tokens"], "cohort evaluation scored_tokens"
        )

    computed = _canonical_sha256(copied)
    if cohort_id != computed:
        raise CohortError(
            f"cohort_id does not match cohort content: expected {computed}, got {cohort_id}"
        )
    return {**copied, "cohort_id": cohort_id}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sequence_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        copied = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CohortError(f"cohort must contain finite JSON values: {exc}") from exc
    if not isinstance(copied, dict):
        raise CohortError("cohort must be a JSON object")
    return copied


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CohortError(f"{name} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    name: str,
    required: set[str],
    *,
    optional: set[str] = frozenset(),
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise CohortError(f"{name} is missing: {', '.join(missing)}")
    if unknown:
        raise CohortError(f"{name} has unknown field(s): {', '.join(unknown)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CohortError(f"{name} must be a non-empty trimmed string")
    return value


def _string_sequence(value: Any, name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise CohortError(f"{name} must be a non-empty sequence")
    result = [_string(item, name) for item in value]
    if len(result) != len(set(result)):
        raise CohortError(f"{name} must not contain duplicates")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CohortError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    result = _nonnegative_number(value, name)
    if result <= 0.0:
        raise CohortError(f"{name} must be positive")
    return result


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CohortError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CohortError(f"{name} must be finite and non-negative")
    return result


def _canonical_number(value: float) -> str:
    return format(value, ".15g")


def _canonical_numeric_string(value: Any, name: str, *, positive: bool) -> str:
    if not isinstance(value, str):
        raise CohortError(f"{name} must be a canonical numeric string")
    try:
        number = float(value)
    except ValueError as exc:
        raise CohortError(f"{name} must be a canonical numeric string") from exc
    if not math.isfinite(number) or number < 0.0 or (positive and number <= 0.0):
        requirement = "positive" if positive else "non-negative"
        raise CohortError(f"{name} must be finite and {requirement}")
    if value != _canonical_number(number):
        raise CohortError(f"{name} is not canonically encoded")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CohortError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CohortError(f"{name} must be a lowercase SHA-256 digest") from exc
    if value != value.lower():
        raise CohortError(f"{name} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "COHORT_SCHEMA_VERSION",
    "CohortError",
    "HORIZON_ROUNDING",
    "build_cohort",
    "validate_cohort",
]
