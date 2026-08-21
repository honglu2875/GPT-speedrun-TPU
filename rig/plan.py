"""Resolve and validate a recipe's data-independent execution plan.

Recipes own their scientific configuration.  The rig asks an entry point to
resolve that configuration before it starts a run, then treats the returned
JSON as the token-budget contract for validation and record keeping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence


PLAN_SCHEMA_VERSION = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_KINDS = {"smoke", "full", "diagnostic"}


class PlanError(ValueError):
    """A recipe could not produce a valid deterministic plan."""


@dataclass(frozen=True)
class RecipePlan:
    """Validated plan plus its canonical content digest."""

    payload: Mapping[str, Any]
    sha256: str

    @property
    def expected_tokens(self) -> int:
        return int(self.payload["expected_tokens"])

    @property
    def run_kind(self) -> str:
        return str(self.payload["run_kind"])

    @property
    def validation_predictions(self) -> int:
        return int(self.payload["validation_predictions"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash a finite JSON object with one stable, whitespace-free encoding."""

    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PlanError(f"plan must contain only finite JSON values: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def validate_recipe_plan(value: Mapping[str, Any]) -> RecipePlan:
    """Validate the v3 plan protocol and return an immutable JSON copy."""

    if not isinstance(value, Mapping):
        raise PlanError("recipe plan must be a JSON object")
    try:
        payload = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise PlanError(
            f"recipe plan must contain only finite JSON values: {exc}"
        ) from exc

    required = {
        "schema_version",
        "config_schema_version",
        "config_sha256",
        "profile",
        "context_preset",
        "document_masking",
        "tier",
        "run_kind",
        "parameterization",
        "weight_decay_policy",
        "declared_parameters",
        "batch_size",
        "sequence_length",
        "tokens_per_step",
        "target_tokens_per_parameter",
        "achieved_tokens_per_parameter",
        "schedule_steps",
        "stop_after_step",
        "planned_tokens",
        "expected_tokens",
        "validation_predictions",
        "base_learning_rate",
        "batch_ratio",
        "ladder_data_multiplier",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise PlanError("recipe plan is missing: " + ", ".join(missing))
    unknown = sorted(set(payload) - required)
    if unknown:
        raise PlanError("recipe plan has unknown field(s): " + ", ".join(unknown))

    if _integer(payload["schema_version"], "schema_version") != PLAN_SCHEMA_VERSION:
        raise PlanError(f"recipe plan schema_version must be {PLAN_SCHEMA_VERSION}")
    _positive_integer(payload["config_schema_version"], "config_schema_version")
    config_sha256 = _string(payload["config_sha256"], "config_sha256")
    if not _SHA256.fullmatch(config_sha256):
        raise PlanError("recipe plan config_sha256 must be 64 lowercase hex digits")
    profile = _string(payload["profile"], "profile")
    _string(payload["context_preset"], "context_preset")
    _boolean(payload["document_masking"], "document_masking")
    _string(payload["tier"], "tier")
    run_kind = _string(payload["run_kind"], "run_kind")
    if run_kind not in _RUN_KINDS:
        raise PlanError(f"recipe plan run_kind is invalid: {run_kind!r}")
    _string(payload["parameterization"], "parameterization")
    _string(payload["weight_decay_policy"], "weight_decay_policy")

    declared_parameters = _optional_positive_integer(
        payload["declared_parameters"], "declared_parameters"
    )
    batch_size = _positive_integer(payload["batch_size"], "batch_size")
    sequence_length = _positive_integer(payload["sequence_length"], "sequence_length")
    tokens_per_step = _positive_integer(payload["tokens_per_step"], "tokens_per_step")
    target_tpp = _optional_positive_number(
        payload["target_tokens_per_parameter"], "target_tokens_per_parameter"
    )
    achieved_tpp = _optional_positive_number(
        payload["achieved_tokens_per_parameter"], "achieved_tokens_per_parameter"
    )
    schedule_steps = _positive_integer(payload["schedule_steps"], "schedule_steps")
    stop_after_step = _optional_positive_integer(
        payload["stop_after_step"], "stop_after_step"
    )
    planned_tokens = _positive_integer(payload["planned_tokens"], "planned_tokens")
    expected_tokens = _positive_integer(payload["expected_tokens"], "expected_tokens")
    _positive_integer(payload["validation_predictions"], "validation_predictions")
    _positive_number(payload["base_learning_rate"], "base_learning_rate")
    _positive_number(payload["batch_ratio"], "batch_ratio")
    _positive_number(payload["ladder_data_multiplier"], "ladder_data_multiplier")

    if tokens_per_step != batch_size * sequence_length:
        raise PlanError(
            "recipe plan tokens_per_step must equal batch_size * sequence_length"
        )
    if planned_tokens != schedule_steps * tokens_per_step:
        raise PlanError(
            "recipe plan planned_tokens must equal schedule_steps * tokens_per_step"
        )
    if stop_after_step is not None and stop_after_step > schedule_steps:
        raise PlanError("recipe plan stop_after_step cannot exceed schedule_steps")
    final_step = stop_after_step or schedule_steps
    if expected_tokens != final_step * tokens_per_step:
        raise PlanError(
            "recipe plan expected_tokens must equal the executed steps * tokens_per_step"
        )

    if profile == "smoke":
        if run_kind != "smoke":
            raise PlanError("a smoke profile must resolve run_kind='smoke'")
        if stop_after_step is not None:
            raise PlanError("a smoke plan cannot have stop_after_step")
    else:
        expected_kind = "diagnostic" if stop_after_step is not None else "full"
        if run_kind != expected_kind:
            raise PlanError(
                f"a non-smoke plan with this stop policy must be {expected_kind!r}"
            )
        if declared_parameters is None or target_tpp is None or achieved_tpp is None:
            raise PlanError(
                "a fixed-TPP plan requires declared_parameters and target/achieved TPP"
            )
        computed_tpp = planned_tokens / declared_parameters
        if not math.isclose(achieved_tpp, computed_tpp, rel_tol=1e-12, abs_tol=0.0):
            raise PlanError(
                "recipe plan achieved_tokens_per_parameter does not match planned tokens"
            )

    return RecipePlan(payload=payload, sha256=canonical_json_sha256(payload))


def resolve_recipe_plan(
    *,
    python_executable: str | Path,
    trainer: Path,
    arguments: Sequence[str],
    cwd: Path,
) -> RecipePlan:
    """Ask a trusted local recipe to resolve its plan without initializing data."""

    command = [
        str(python_executable),
        str(trainer),
        *[str(argument) for argument in arguments],
        "--print-plan",
    ]
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise PlanError(
            f"recipe plan resolution exited with status {completed.returncode}{suffix}"
        )
    output = completed.stdout.strip()
    if not output:
        raise PlanError("recipe plan resolution produced no JSON")
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise PlanError(
            f"recipe plan resolution did not produce one JSON object: {exc}"
        ) from exc
    return validate_recipe_plan(payload)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PlanError(f"recipe plan {name} must be a non-empty trimmed string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"recipe plan {name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PlanError(f"recipe plan {name} must be a boolean")
    return value


def _positive_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise PlanError(f"recipe plan {name} must be positive")
    return result


def _optional_positive_integer(value: Any, name: str) -> int | None:
    return None if value is None else _positive_integer(value, name)


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError(f"recipe plan {name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PlanError(f"recipe plan {name} must be finite and positive")
    return result


def _optional_positive_number(value: Any, name: str) -> float | None:
    return None if value is None else _positive_number(value, name)


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "PlanError",
    "RecipePlan",
    "canonical_json_sha256",
    "resolve_recipe_plan",
    "validate_recipe_plan",
]
