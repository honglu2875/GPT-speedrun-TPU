"""Versioned local IsoFLOP experiments outside the competition harness.

The official open-track contract intentionally pins one model and token budget.
This module generates immutable diagnostic configs beside a byte-exact trainer,
runs them as ``open/dev`` research artifacts, and fits a deliberately local
compute-allocation law only when every measured optimum is bracketed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform as host_platform
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from speedrun.data import (
    DataError,
    FRESH10_DOMAINS,
    load_manifest,
    load_fresh10_manifest,
    manifest_digest,
    sha256_file,
    verify_dataset,
    verify_fresh10,
)
from speedrun.fineweb_builder import (
    BUILDER_VERSION,
    DEFAULT_SOURCE_DATE_CUTOFF,
    EOT_TOKEN,
    ExclusionPolicy,
    FineWebBuildError,
    PYARROW_VERSION,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    TIKTOKEN_VERSION,
    UPSTREAM_GLOBAL_SHUFFLE_SEED,
    VOCAB_SIZE,
    canonical_json_sha256,
    load_fresh10_exclusion_policy,
    source_inventory_from_dict,
)


DEFAULT_SUITE = (
    Path(__file__).resolve().parent.parent
    / "sweeps"
    / "current_budget_isoflop_v4"
    / "suite.yaml"
)
DEFAULT_RUNS = Path("runs/scaling/current-budget-isoflop-v4")
_LLMC_MAGIC = 20_240_520
_LLMC_VERSION = 1
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINEWEB4B_TRAIN_NAMES = tuple(
    f"fineweb_train_{index:06d}.bin" for index in range(1, 40)
)
_FINEWEB4B_VALIDATION_NAMES = ("fineweb_val_000000.bin",)
_FINEWEB4B_NAMES = _FINEWEB4B_VALIDATION_NAMES + _FINEWEB4B_TRAIN_NAMES
_ARCHIVED_SUITE_IDS = frozenset(
    {"current_budget_isoflop_v2", "current_budget_isoflop_v3"}
)


class ScalingError(ValueError):
    """A sweep definition, run, or fit is not internally consistent."""


class LearningRateEdgeError(ScalingError):
    """The best measured learning rate is still on a bounded grid edge."""

    def __init__(
        self,
        shape_id: str,
        side: str,
        learning_rate: float,
        *,
        slice_id: str | None = None,
    ) -> None:
        self.shape_id = shape_id
        self.slice_id = slice_id
        self.side = side
        self.learning_rate = learning_rate
        label = shape_id if slice_id is None else f"{slice_id}/{shape_id}"
        super().__init__(
            f"{label}: lowest admitted validation loss is at the {side} learning-rate "
            f"edge ({learning_rate:.8g})"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ScalingError(f"{label} must be a string-keyed mapping")
    return value


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScalingError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScalingError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        suffix = " and positive" if positive else ""
        raise ScalingError(f"{label} must be finite{suffix}")
    return result


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ScalingError(f"{label} must be a lowercase filesystem-safe identifier")
    return value


def _round_ratio(value: int, numerator: int, denominator: int) -> int:
    """Round a positive rational half-up using integer arithmetic."""

    return (value * numerator + denominator // 2) // denominator


def parameter_count(*, layers: int, d_model: int, vocab_size: int, seq_len: int) -> int:
    """Match ``reference/train.py:init_params`` without importing JAX."""

    embeddings = vocab_size * d_model + seq_len * d_model
    blocks = layers * (12 * d_model * d_model + 13 * d_model)
    final_norm = 2 * d_model
    return embeddings + blocks + final_norm


def flops_per_token(
    *, layers: int, d_model: int, parameters: int, seq_len: int
) -> int:
    """Match dense-loss, unpadded-sequence reference FLOP accounting."""

    return 6 * parameters + 12 * layers * d_model * seq_len


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shared_trainer_sources(repo: Path) -> tuple[Path, ...]:
    sources = [repo / "speedrun" / "__init__.py"]
    sources.extend(sorted((repo / "speedrun" / "kernels").glob("*.py")))
    if any(not path.is_file() for path in sources):
        raise ScalingError("shared trainer source snapshot is incomplete")
    return tuple(sources)


def _source_snapshot(repo: Path) -> dict[str, str]:
    paths = [repo / "submissions" / "reference" / "train.py"]
    paths.extend(_shared_trainer_sources(repo))
    paths.extend(
        (
            repo / "speedrun" / "data.py",
            repo / "speedrun" / "fineweb_builder.py",
            repo / "scripts" / "prepare_fineweb.py",
            repo / "data" / "manifests" / "fresh10.json",
            repo / "uv.lock",
            Path(__file__).resolve(),
        )
    )
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ScalingError("execution fingerprint source set is incomplete or symlinked")
    return {
        path.relative_to(repo).as_posix(): _sha256(path)
        for path in paths
    }


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScalingError(f"invalid {label} YAML: {exc}") from exc
    return _mapping(payload, label)


def _shape(raw: Any, label: str, *, vocab_size: int, seq_len: int) -> dict[str, Any]:
    source = _mapping(raw, label)
    identifier = _name(source.get("id"), f"{label}.id")
    layers = _integer(source.get("layers"), f"{identifier}.layers")
    heads = _integer(source.get("heads"), f"{identifier}.heads")
    d_model = _integer(source.get("d_model"), f"{identifier}.d_model")
    if d_model % heads:
        raise ScalingError(f"{identifier}: d_model must be divisible by heads")
    if d_model // heads != 64:
        raise ScalingError(f"{identifier}: the suite requires 64-wide attention heads")
    parameters = parameter_count(
        layers=layers,
        d_model=d_model,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )
    declared = source.get("parameters")
    if declared is not None and _integer(declared, f"{identifier}.parameters") != parameters:
        raise ScalingError(f"{identifier}: declared parameter count is incorrect")
    per_token = flops_per_token(
        layers=layers,
        d_model=d_model,
        parameters=parameters,
        seq_len=seq_len,
    )
    return {
        "shape_id": identifier,
        "layers": layers,
        "heads": heads,
        "d_model": d_model,
        "parameters": parameters,
        "flops_per_token": per_token,
    }


def _point(
    *,
    shape: Mapping[str, Any],
    compute_slice: Mapping[str, Any],
    batch_size: int,
    seq_len: int,
    schedule: Mapping[str, int],
    role: str,
    identifier: str | None = None,
    learning_rate: float | None = None,
    learning_rate_source: str | None = None,
) -> dict[str, Any]:
    target = int(compute_slice["target_total_flops"])
    per_step = int(shape["flops_per_token"]) * batch_size * seq_len
    steps = max(1, (target + per_step // 2) // per_step)
    train_tokens = steps * batch_size * seq_len
    total_flops = int(shape["flops_per_token"]) * train_tokens
    point_id = identifier or f"{compute_slice['id']}_{shape['shape_id']}"
    warmup_steps = max(
        1,
        _round_ratio(
            steps,
            schedule["reference_warmup_steps"],
            schedule["reference_steps"],
        ),
    )
    val_every = max(
        1,
        _round_ratio(
            steps,
            schedule["reference_val_every"],
            schedule["reference_steps"],
        ),
    )
    log_every = max(
        1,
        _round_ratio(
            steps,
            schedule["reference_log_every"],
            schedule["reference_steps"],
        ),
    )
    result = {
        "id": _name(point_id, "point.id"),
        "slice": compute_slice["id"],
        "role": role,
        "shape_id": shape["shape_id"],
        "layers": int(shape["layers"]),
        "heads": int(shape["heads"]),
        "d_model": int(shape["d_model"]),
        "parameters": int(shape["parameters"]),
        "flops_per_token": int(shape["flops_per_token"]),
        "steps": steps,
        "train_tokens": train_tokens,
        "total_flops": total_flops,
        "relative_flop_error": total_flops / target - 1.0,
        "tokens_per_parameter": train_tokens / int(shape["parameters"]),
        "warmup_steps": warmup_steps,
        "val_every": val_every,
        "diagnostics_every": val_every,
        "log_every": log_every,
    }
    if learning_rate is not None:
        result["learning_rate"] = float(learning_rate)
    if learning_rate_source is not None:
        result["learning_rate_source"] = learning_rate_source
    return result


def load_suite(path: Path = DEFAULT_SUITE) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    root = _load_yaml(path, "suite")
    if root.get("schema_version") != 1:
        raise ScalingError("suite schema_version must be 1")
    suite_id = _name(root.get("suite_id"), "suite_id")
    seq_len = _integer(root.get("sequence_length"), "sequence_length")
    batch_size = _integer(root.get("batch_size"), "batch_size")
    vocab_size = _integer(root.get("vocab_size"), "vocab_size")
    seed = _integer(root.get("seed"), "seed", minimum=0)
    runtime = dict(_mapping(root.get("runtime"), "runtime"))
    expected_runtime_fields = {
        "python_major_minor",
        "jax_version",
        "jaxlib_version",
        "libtpu_version",
        "platform",
        "device_kind",
        "device_count",
        "local_device_count",
        "process_count",
    }
    if set(runtime) != expected_runtime_fields:
        raise ScalingError("runtime must define the exact version and v4-8 topology contract")
    for field in (
        "python_major_minor",
        "jax_version",
        "jaxlib_version",
        "libtpu_version",
        "platform",
        "device_kind",
    ):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise ScalingError(f"runtime.{field} must be a nonempty string")
    for field in ("device_count", "local_device_count", "process_count"):
        runtime[field] = _integer(runtime[field], f"runtime.{field}")
    if (
        runtime["platform"] != "tpu"
        or runtime["device_kind"] != "TPU v4"
        or runtime["device_count"] != 4
        or runtime["local_device_count"] != 4
        or runtime["process_count"] != 1
    ):
        raise ScalingError("this suite requires exactly one-process TPU v4-8")
    validation_tokens = _integer(root.get("validation_tokens"), "validation_tokens")
    tokens_per_step = batch_size * seq_len
    if validation_tokens % tokens_per_step:
        raise ScalingError("validation_tokens must be divisible by batch_size * sequence_length")

    template_name = root.get("config_template")
    if not isinstance(template_name, str) or not template_name:
        raise ScalingError("config_template must be a relative path")
    template_path = (path.parent / template_name).resolve(strict=True)
    try:
        template_path.relative_to(path.parent)
    except ValueError as exc:
        raise ScalingError("config_template escapes the suite directory") from exc
    template = _load_yaml(template_path, "config template")
    profiles = _mapping(template.get("profiles"), "config template profiles")
    if template.get("schema_version") != 1 or set(profiles) != {"smoke", "dev", "official"}:
        raise ScalingError("config template must contain schema 1 smoke/dev/official profiles")

    anchor = dict(_mapping(root.get("anchor"), "anchor"))
    anchor_parameters = _integer(anchor.get("parameters"), "anchor.parameters")
    anchor_train_tokens = _integer(anchor.get("train_tokens"), "anchor.train_tokens")
    anchor_flops_per_token = _integer(
        anchor.get("flops_per_token"), "anchor.flops_per_token"
    )
    anchor_total_flops = _integer(anchor.get("total_flops"), "anchor.total_flops")
    if anchor_flops_per_token * anchor_train_tokens != anchor_total_flops:
        raise ScalingError("anchor total FLOPs do not equal tokens × FLOPs/token")

    schedule_raw = _mapping(root.get("schedule"), "schedule")
    schedule = {
        key: _integer(schedule_raw.get(key), f"schedule.{key}")
        for key in (
            "reference_steps",
            "reference_warmup_steps",
            "reference_val_every",
            "reference_log_every",
        )
    }
    optimizer = dict(_mapping(root.get("optimizer"), "optimizer"))
    expected_optimizer = {
        "min_lr_ratio", "weight_decay", "beta1", "beta2", "grad_clip"
    }
    if set(optimizer) != expected_optimizer:
        raise ScalingError("optimizer must define exactly the fixed suite hyperparameters")
    for key in optimizer:
        optimizer[key] = _finite(optimizer[key], f"optimizer.{key}")
    learning_rate_scope = root.get(
        "learning_rate_scope", "first_compute_slice_per_shape"
    )
    if learning_rate_scope not in {
        "first_compute_slice_per_shape",
        "per_compute_slice_and_shape",
    }:
        raise ScalingError(
            "learning_rate_scope must be first_compute_slice_per_shape or "
            "per_compute_slice_and_shape"
        )
    stability_raw = root.get("stability_admission")
    stability_admission: dict[str, Any] | None = None
    if stability_raw is not None:
        stability_source = dict(_mapping(stability_raw, "stability_admission"))
        expected_stability_fields = {
            "schema_version",
            "probe_regression_threshold",
            "probe_regression_consecutive",
            "trailing_window_steps",
            "gradient_norm_threshold",
            "maximum_large_gradient_fraction",
            "train_loss_window_regression_threshold",
            "maximum_final_block_param_l2_to_median",
        }
        if set(stability_source) != expected_stability_fields:
            raise ScalingError(
                "stability_admission has unexpected or missing fields"
            )
        if stability_source["schema_version"] != 1:
            raise ScalingError("stability_admission.schema_version must be 1")
        maximum_large_gradient_fraction = _finite(
            stability_source["maximum_large_gradient_fraction"],
            "stability_admission.maximum_large_gradient_fraction",
        )
        if not 0.0 <= maximum_large_gradient_fraction <= 1.0:
            raise ScalingError(
                "stability_admission.maximum_large_gradient_fraction must be in [0, 1]"
            )
        stability_admission = {
            "schema_version": 1,
            "probe_regression_threshold": _finite(
                stability_source["probe_regression_threshold"],
                "stability_admission.probe_regression_threshold",
                positive=True,
            ),
            "probe_regression_consecutive": _integer(
                stability_source["probe_regression_consecutive"],
                "stability_admission.probe_regression_consecutive",
            ),
            "trailing_window_steps": _integer(
                stability_source["trailing_window_steps"],
                "stability_admission.trailing_window_steps",
            ),
            "gradient_norm_threshold": _finite(
                stability_source["gradient_norm_threshold"],
                "stability_admission.gradient_norm_threshold",
                positive=True,
            ),
            "maximum_large_gradient_fraction": maximum_large_gradient_fraction,
            "train_loss_window_regression_threshold": _finite(
                stability_source["train_loss_window_regression_threshold"],
                "stability_admission.train_loss_window_regression_threshold",
                positive=True,
            ),
            "maximum_final_block_param_l2_to_median": _finite(
                stability_source["maximum_final_block_param_l2_to_median"],
                "stability_admission.maximum_final_block_param_l2_to_median",
                positive=True,
            ),
        }
    if (
        learning_rate_scope == "per_compute_slice_and_shape"
        and stability_admission is None
    ):
        raise ScalingError(
            "per-compute-slice learning-rate selection requires stability_admission"
        )
    raw_learning_rates = root.get("learning_rate_candidates")
    if not isinstance(raw_learning_rates, list) or len(raw_learning_rates) < 2:
        raise ScalingError("learning_rate_candidates must contain at least two values")
    learning_rates: list[dict[str, Any]] = []
    seen_learning_rates: set[str] = set()
    for index, raw in enumerate(raw_learning_rates):
        candidate = _mapping(raw, f"learning_rate_candidates[{index}]")
        identifier = _name(candidate.get("id"), f"learning_rate_candidates[{index}].id")
        if identifier in seen_learning_rates:
            raise ScalingError(f"duplicate learning-rate candidate: {identifier}")
        seen_learning_rates.add(identifier)
        learning_rates.append(
            {
                "id": identifier,
                "value": _finite(
                    candidate.get("value"), f"{identifier}.value", positive=True
                ),
            }
        )
    if [item["value"] for item in learning_rates] != sorted(
        item["value"] for item in learning_rates
    ):
        raise ScalingError("learning_rate_candidates must be ordered low to high")
    learning_rate_search_raw = _mapping(
        root.get("learning_rate_search"), "learning_rate_search"
    )
    if set(learning_rate_search_raw) != {"geometric_factor", "lower", "upper"}:
        raise ScalingError(
            "learning_rate_search must define geometric_factor, lower, and upper"
        )
    geometric_factor = _finite(
        learning_rate_search_raw["geometric_factor"],
        "learning_rate_search.geometric_factor",
        positive=True,
    )
    if geometric_factor <= 1.0:
        raise ScalingError("learning-rate geometric factor must exceed one")

    def rate_extensions(side: str) -> list[dict[str, Any]]:
        raw_extensions = learning_rate_search_raw[side]
        if not isinstance(raw_extensions, list) or not raw_extensions:
            raise ScalingError(f"learning_rate_search.{side} must be a nonempty list")
        parsed: list[dict[str, Any]] = []
        previous = (
            float(learning_rates[0]["value"])
            if side == "lower"
            else float(learning_rates[-1]["value"])
        )
        for index, raw_extension in enumerate(raw_extensions):
            item = _mapping(
                raw_extension, f"learning_rate_search.{side}[{index}]"
            )
            if set(item) != {"id", "value"}:
                raise ScalingError(
                    f"learning_rate_search.{side}[{index}] must define id and value"
                )
            identifier = _name(item["id"], f"learning_rate_search.{side}[{index}].id")
            if identifier in seen_learning_rates:
                raise ScalingError(f"duplicate learning-rate candidate: {identifier}")
            seen_learning_rates.add(identifier)
            value = _finite(
                item["value"],
                f"learning_rate_search.{side}[{index}].value",
                positive=True,
            )
            expected = previous / geometric_factor if side == "lower" else previous * geometric_factor
            if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=0.0):
                raise ScalingError(
                    f"learning_rate_search.{side}[{index}] is not the next "
                    "geometric candidate"
                )
            parsed.append({"id": identifier, "value": value, "side": side})
            previous = value
        return parsed

    lower_learning_rates = rate_extensions("lower")
    upper_learning_rates = rate_extensions("upper")
    all_learning_rates = (
        list(reversed(lower_learning_rates))
        + learning_rates
        + upper_learning_rates
    )

    raw_slices = root.get("compute_slices")
    if not isinstance(raw_slices, list) or len(raw_slices) < 3:
        raise ScalingError("compute_slices must contain at least three budgets")
    compute_slices: list[dict[str, Any]] = []
    seen_slices: set[str] = set()
    for index, raw in enumerate(raw_slices):
        source = _mapping(raw, f"compute_slices[{index}]")
        identifier = _name(source.get("id"), f"compute_slices[{index}].id")
        if identifier in seen_slices:
            raise ScalingError(f"duplicate compute slice: {identifier}")
        seen_slices.add(identifier)
        numerator = _integer(source.get("numerator"), f"{identifier}.numerator")
        denominator = _integer(source.get("denominator"), f"{identifier}.denominator")
        scaled = anchor_total_flops * numerator
        if scaled % denominator:
            raise ScalingError(f"{identifier}: anchor compute is not exactly divisible")
        compute_slices.append(
            {
                "id": identifier,
                "numerator": numerator,
                "denominator": denominator,
                "multiplier": numerator / denominator,
                "target_total_flops": scaled // denominator,
            }
        )
    if [item["multiplier"] for item in compute_slices] != sorted(
        item["multiplier"] for item in compute_slices
    ):
        raise ScalingError("compute_slices must be ordered from least to most compute")
    slices_by_id = {str(item["id"]): item for item in compute_slices}

    raw_shapes = root.get("fit_shapes")
    if not isinstance(raw_shapes, list) or len(raw_shapes) < 5:
        raise ScalingError("fit_shapes must contain at least five model sizes")
    shapes = [
        _shape(item, f"fit_shapes[{index}]", vocab_size=vocab_size, seq_len=seq_len)
        for index, item in enumerate(raw_shapes)
    ]
    if len({item["shape_id"] for item in shapes}) != len(shapes):
        raise ScalingError("fit shape IDs must be unique")
    if [item["parameters"] for item in shapes] != sorted(
        item["parameters"] for item in shapes
    ):
        raise ScalingError("fit_shapes must be ordered by parameter count")

    fit_geometry = [
        _point(
            shape=shape,
            compute_slice=compute_slice,
            batch_size=batch_size,
            seq_len=seq_len,
            schedule=schedule,
            role="fit",
        )
        for compute_slice in compute_slices
        for shape in shapes
    ]
    calibration_slices = (
        compute_slices
        if learning_rate_scope == "per_compute_slice_and_shape"
        else compute_slices[:1]
    )
    calibration_slice = compute_slices[0]
    calibrations = [
        _point(
            shape=shape,
            compute_slice=compute_slice,
            batch_size=batch_size,
            seq_len=seq_len,
            schedule=schedule,
            role="learning_rate_calibration",
            identifier=f"{compute_slice['id']}_{shape['shape_id']}_{candidate['id']}",
            learning_rate=float(candidate["value"]),
        )
        for compute_slice in calibration_slices
        for shape in shapes
        for candidate in learning_rates
    ]
    variants = (
        []
        if learning_rate_scope == "per_compute_slice_and_shape"
        else [
            {
                **point,
                "learning_rate_source": point["shape_id"],
            }
            for point in fit_geometry
            if point["slice"] != calibration_slice["id"]
        ]
    )

    def extra_points(field: str, role: str) -> list[dict[str, Any]]:
        raw_points = root.get(field, [])
        if not isinstance(raw_points, list):
            raise ScalingError(f"{field} must be a list")
        result: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_points):
            source = _mapping(raw, f"{field}[{index}]")
            slice_id = _name(source.get("slice"), f"{field}[{index}].slice")
            if slice_id not in slices_by_id:
                raise ScalingError(f"{field}[{index}] names an unknown compute slice")
            shape = _shape(
                source,
                f"{field}[{index}]",
                vocab_size=vocab_size,
                seq_len=seq_len,
            )
            result.append(
                _point(
                    shape=shape,
                    compute_slice=slices_by_id[slice_id],
                    batch_size=batch_size,
                    seq_len=seq_len,
                    schedule=schedule,
                    role=role,
                    identifier=source.get("point_id"),
                    learning_rate=(
                        _finite(
                            source.get("learning_rate"),
                            f"{field}[{index}].learning_rate",
                            positive=True,
                        )
                        if "learning_rate" in source
                        else None
                    ),
                    learning_rate_source=(
                        None if "learning_rate" in source else shape["shape_id"]
                    ),
                )
            )
        return result

    controls = extra_points("controls", "control")
    raw_extension_shapes = root.get("optional_extension_shapes", [])
    if not isinstance(raw_extension_shapes, list):
        raise ScalingError("optional_extension_shapes must be a list")
    extension_shapes = [
        _shape(
            raw_shape,
            f"optional_extension_shapes[{index}]",
            vocab_size=vocab_size,
            seq_len=seq_len,
        )
        for index, raw_shape in enumerate(raw_extension_shapes)
    ]
    if [item["parameters"] for item in extension_shapes] != sorted(
        item["parameters"] for item in extension_shapes
    ):
        raise ScalingError("optional extension shapes must be ordered by parameter count")
    if extension_shapes and extension_shapes[0]["parameters"] <= shapes[-1]["parameters"]:
        raise ScalingError("optional extension shapes must be larger than the base grid")
    # Under the legacy policy, c025 reuses the selected extension calibration
    # and only later slices need a dependent point.  Under the fail-closed v4
    # policy every slice gets its own selected extension calibration instead.
    extensions = (
        []
        if learning_rate_scope == "per_compute_slice_and_shape"
        else [
            _point(
                shape=shape,
                compute_slice=compute_slice,
                batch_size=batch_size,
                seq_len=seq_len,
                schedule=schedule,
                role="extension",
                identifier=f"{compute_slice['id']}_{shape['shape_id']}_extension",
                learning_rate_source=str(shape["shape_id"]),
            )
            for compute_slice in compute_slices[1:]
            for shape in extension_shapes
        ]
    )
    extension_calibrations = [
        _point(
            shape=extension,
            compute_slice=compute_slice,
            batch_size=batch_size,
            seq_len=seq_len,
            schedule=schedule,
            role="extension_learning_rate_calibration",
            identifier=(
                f"{compute_slice['id']}_{extension['shape_id']}_{candidate['id']}_ext"
            ),
            learning_rate=float(candidate["value"]),
        )
        for compute_slice in calibration_slices
        for extension in extension_shapes
        for candidate in learning_rates
    ]
    adaptive_calibrations = [
        _point(
            shape=shape,
            compute_slice=compute_slice,
            batch_size=batch_size,
            seq_len=seq_len,
            schedule=schedule,
            role=(
                "extension_learning_rate_search"
                if shape["shape_id"] in {item["shape_id"] for item in extension_shapes}
                else "learning_rate_search"
            ),
            identifier=(
                f"{compute_slice['id']}_{shape['shape_id']}_{candidate['id']}_"
                "adaptive"
            ),
            learning_rate=float(candidate["value"]),
        )
        for compute_slice in calibration_slices
        for shape in shapes + extension_shapes
        for candidate in lower_learning_rates + upper_learning_rates
    ]
    all_variants = (
        calibrations
        + adaptive_calibrations
        + variants
        + controls
        + extension_calibrations
        + extensions
    )
    identifiers = [str(item["id"]) for item in all_variants]
    if len(set(identifiers)) != len(identifiers):
        duplicates = sorted(
            {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
        )
        raise ScalingError(f"all point IDs must be unique: {', '.join(duplicates)}")
    for point in all_variants:
        target = slices_by_id[str(point["slice"])]["target_total_flops"]
        per_step = point["flops_per_token"] * tokens_per_step
        if abs(point["total_flops"] - target) > per_step / 2 + 1:
            raise ScalingError(f"{point['id']}: point is not nearest-step IsoFLOP")
        if abs(point["relative_flop_error"]) > 1.0e-4:
            raise ScalingError(f"{point['id']}: compute mismatch exceeds 0.01%")

    dataset = dict(_mapping(root.get("dataset"), "dataset"))
    expected_dataset_fields = {
        "id",
        "source_repository",
        "source_revision",
        "tokenizer_version",
        "source_inventory_sha256",
        "exclusion_policy_sha256",
        "preparation_core_sha256",
        "minimum_usable_train_tokens",
        "requested_train_tokens",
        "requested_validation_tokens",
    }
    if set(dataset) != expected_dataset_fields:
        raise ScalingError(
            "dataset must define exactly id, source_repository, source_revision, "
            "tokenizer_version, minimum_usable_train_tokens, "
            "requested_train_tokens, and requested_validation_tokens"
        )
    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ScalingError("dataset.id must be a nonempty string")
    for field in ("source_repository", "source_revision", "tokenizer_version"):
        if not isinstance(dataset.get(field), str) or not dataset[field].strip():
            raise ScalingError(f"dataset.{field} must be a nonempty string")
    for field in (
        "source_inventory_sha256",
        "exclusion_policy_sha256",
        "preparation_core_sha256",
    ):
        digest = dataset.get(field)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ScalingError(f"dataset.{field} must be a lowercase SHA-256")
    minimum_train_tokens = _integer(
        dataset.get("minimum_usable_train_tokens"),
        "dataset.minimum_usable_train_tokens",
    )
    requested_train_tokens = _integer(
        dataset.get("requested_train_tokens"), "dataset.requested_train_tokens"
    )
    requested_validation_tokens = _integer(
        dataset.get("requested_validation_tokens"),
        "dataset.requested_validation_tokens",
    )
    required_train_tokens = max(int(item["train_tokens"]) for item in all_variants)
    if minimum_train_tokens < required_train_tokens:
        raise ScalingError(
            "dataset minimum is smaller than the largest no-replacement point"
        )
    if requested_train_tokens < minimum_train_tokens:
        raise ScalingError("dataset requested train tokens are below its usable minimum")
    if requested_validation_tokens < validation_tokens:
        raise ScalingError(
            "dataset requested validation tokens are below the scored validation budget"
        )

    fresh10 = dict(_mapping(root.get("fresh10"), "fresh10"))
    expected_fresh10_fields = {
        "manifest",
        "name",
        "manifest_raw_sha256",
        "manifest_canonical_sha256",
        "repository",
        "revision",
        "publication_not_before",
        "domains",
        "scored_tokens_per_domain",
        "scored_tokens",
    }
    if set(fresh10) != expected_fresh10_fields:
        raise ScalingError("fresh10 suite contract has unexpected or missing fields")
    manifest_name = fresh10["manifest"]
    if not isinstance(manifest_name, str) or not manifest_name:
        raise ScalingError("fresh10.manifest must be a relative path")
    fresh10_manifest_path = (path.parent / manifest_name).resolve(strict=True)
    repo = Path(__file__).resolve().parent.parent
    expected_fresh10_manifest = (repo / "data" / "manifests" / "fresh10.json").resolve()
    if fresh10_manifest_path != expected_fresh10_manifest:
        raise ScalingError("suite Fresh10 manifest must be the canonical checked-in file")
    for field in ("manifest_raw_sha256", "manifest_canonical_sha256"):
        digest = fresh10[field]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ScalingError(f"fresh10.{field} must be a lowercase SHA-256")
    try:
        fresh10_payload, _ = load_fresh10_manifest(fresh10_manifest_path)
    except DataError as exc:
        raise ScalingError(f"invalid canonical Fresh10 manifest: {exc}") from exc
    if (
        _sha256(fresh10_manifest_path) != fresh10["manifest_raw_sha256"]
        or manifest_digest(fresh10_payload)
        != fresh10["manifest_canonical_sha256"]
    ):
        raise ScalingError("canonical Fresh10 manifest hash differs from suite pin")
    prepared_source = _mapping(
        fresh10_payload.get("prepared_source"), "Fresh10 prepared_source"
    )
    if (
        fresh10_payload.get("name") != fresh10["name"]
        or prepared_source.get("repository") != fresh10["repository"]
        or prepared_source.get("revision") != fresh10["revision"]
        or fresh10_payload.get("publication_not_before")
        != fresh10["publication_not_before"]
        or len(fresh10_payload["domains"]) != fresh10["domains"]
        or fresh10["domains"] != len(FRESH10_DOMAINS)
        or fresh10["scored_tokens_per_domain"] != 8_192
        or fresh10["scored_tokens"] != 81_920
        or any(
            domain["scored_tokens"] != fresh10["scored_tokens_per_domain"]
            for domain in fresh10_payload["domains"]
        )
    ):
        raise ScalingError("Fresh10 manifest identity/count contract differs from suite")
    fresh10["manifest_path"] = fresh10_manifest_path
    fresh10["payload"] = fresh10_payload

    suite = {
        **dict(root),
        "path": path,
        "suite_sha256": _sha256(path),
        "suite_id": suite_id,
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "seed": seed,
        "runtime": runtime,
        "validation_tokens": validation_tokens,
        "validation_batches": validation_tokens // tokens_per_step,
        "template_path": template_path,
        "template_sha256": _sha256(template_path),
        "template": dict(template),
        "anchor": anchor,
        "schedule": schedule,
        "optimizer": optimizer,
        "learning_rate_scope": learning_rate_scope,
        "stability_admission": stability_admission,
        "learning_rate_candidates": learning_rates,
        "learning_rate_search": {
            "geometric_factor": geometric_factor,
            "lower": lower_learning_rates,
            "upper": upper_learning_rates,
        },
        "all_learning_rate_candidates": all_learning_rates,
        "compute_slices": compute_slices,
        "slices_by_id": slices_by_id,
        "fit_shapes": shapes,
        "fit_geometry": fit_geometry,
        "calibrations": calibrations,
        "variants": variants,
        "controls": controls,
        "adaptive_calibrations": adaptive_calibrations,
        "extension_calibrations": extension_calibrations,
        "optional_extension_shapes": extension_shapes,
        "optional_extensions": extensions,
        "all_variants": all_variants,
        "dataset": dataset,
        "fresh10": fresh10,
        "required_train_tokens": required_train_tokens,
    }
    for point in all_variants:
        if "learning_rate" in point:
            config_bytes = variant_config_bytes(suite, point)
            point["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    lineage_declaration = root.get("lineage")
    suite["lineage"] = (
        None
        if lineage_declaration is None
        else _load_lineage_contract(
            suite,
            declaration=lineage_declaration,
            suite_directory=path.parent,
        )
    )
    repo = Path(__file__).resolve().parent.parent
    source_snapshot = _source_snapshot(repo)
    fingerprint_payload = {
        "suite_sha256": suite["suite_sha256"],
        "template_sha256": suite["template_sha256"],
        "source_snapshot": source_snapshot,
    }
    suite["source_snapshot"] = source_snapshot
    suite["execution_fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return suite


def variant_config_bytes(suite: Mapping[str, Any], point: Mapping[str, Any]) -> bytes:
    """Generate the exact sibling config saved with one diagnostic point."""

    if "learning_rate" not in point:
        raise ScalingError(
            f"{point['id']}: learning rate has not been selected for this point"
        )
    template = suite["template"]
    profiles = _mapping(template["profiles"], "config template profiles")
    config = {
        "schema_version": 1,
        "profiles": {
            "smoke": profiles["smoke"],
            "dev": {
                "training": {
                    "train_tokens": int(point["train_tokens"]),
                    "batch_size": int(suite["batch_size"]),
                    "seq_len": int(suite["sequence_length"]),
                    "sampling": "shuffled_epochs",
                    "dtype": "bfloat16",
                },
                "model": {
                    "layers": int(point["layers"]),
                    "heads": int(point["heads"]),
                    "d_model": int(point["d_model"]),
                    "mlp_mult": 4,
                    "vocab_size": int(suite["vocab_size"]),
                    "semantic_vocab_size": int(suite["vocab_size"]),
                },
                "kernels": {
                    "attention_backend": "tpu_flash",
                    "loss_backend": "dense",
                    "vocab_tile_size": 2048,
                },
                "optimizer": {
                    "learning_rate": float(point["learning_rate"]),
                    "min_lr_ratio": suite["optimizer"]["min_lr_ratio"],
                    "warmup_steps": int(point["warmup_steps"]),
                    "weight_decay": suite["optimizer"]["weight_decay"],
                    "beta1": suite["optimizer"]["beta1"],
                    "beta2": suite["optimizer"]["beta2"],
                    "grad_clip": suite["optimizer"]["grad_clip"],
                },
                "evaluation": {
                    "eval_batches": int(suite["validation_batches"]),
                    "val_every": int(point["val_every"]),
                    "val_probe_batches": 8,
                },
                "logging": {
                    "diagnostics_every": int(point["diagnostics_every"]),
                    "log_every": int(point["log_every"]),
                },
            },
            "official": profiles["official"],
        },
    }
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=False).encode("utf-8")


def _public_point(point: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "slice",
        "role",
        "shape_id",
        "layers",
        "heads",
        "d_model",
        "parameters",
        "flops_per_token",
        "steps",
        "train_tokens",
        "total_flops",
        "relative_flop_error",
        "tokens_per_parameter",
        "warmup_steps",
        "val_every",
        "diagnostics_every",
        "log_every",
    )
    result = {key: point[key] for key in keys}
    for key in ("learning_rate", "learning_rate_source", "config_sha256"):
        if key in point:
            result[key] = point[key]
    return result


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_lineage_contract(
    suite: Mapping[str, Any], *, declaration: Any, suite_directory: Path
) -> dict[str, Any]:
    """Load a pinned prior-study allowlist without reading mutable run results.

    Planning remains possible when the ignored ``runs/`` tree is unavailable.
    A result is admitted only later, when :func:`_read_run` verifies both exact
    artifact hashes and the complete historical run/source/data/runtime contract.
    """

    declared = dict(_mapping(declaration, "lineage"))
    if set(declared) != {"manifest", "sha256"}:
        raise ScalingError("lineage must define exactly manifest and sha256")
    manifest_name = declared["manifest"]
    manifest_digest = declared["sha256"]
    if (
        not isinstance(manifest_name, str)
        or not manifest_name
        or Path(manifest_name).is_absolute()
        or not isinstance(manifest_digest, str)
        or _SHA256.fullmatch(manifest_digest) is None
    ):
        raise ScalingError("lineage manifest/path digest declaration is invalid")
    unresolved_manifest_path = suite_directory / manifest_name
    if unresolved_manifest_path.is_symlink():
        raise ScalingError("lineage manifest must not be a symlink")
    manifest_path = unresolved_manifest_path.resolve()
    try:
        manifest_path.relative_to(suite_directory.resolve())
    except ValueError as exc:
        raise ScalingError("lineage manifest escapes the suite directory") from exc
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or _sha256(manifest_path) != manifest_digest
    ):
        raise ScalingError("lineage manifest is missing, symlinked, or differs from its pin")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScalingError(f"invalid lineage JSON: {exc}") from exc
    payload = dict(_mapping(payload, "lineage manifest"))
    if set(payload) != {"schema_version", "lineage_id", "origin", "artifacts"}:
        raise ScalingError("lineage manifest has unexpected or missing fields")
    if payload["schema_version"] != 1:
        raise ScalingError("lineage manifest schema_version must be 1")
    lineage_id = _name(payload["lineage_id"], "lineage_id")

    origin = dict(_mapping(payload["origin"], "lineage origin"))
    expected_origin_fields = {
        "runs_root",
        "suite_path",
        "template_path",
        "git_commit",
        "suite_id",
        "suite_sha256",
        "execution_fingerprint",
        "template_sha256",
        "source_snapshot",
    }
    if set(origin) != expected_origin_fields:
        raise ScalingError("lineage origin has unexpected or missing fields")
    origin_suite_id = _name(origin["suite_id"], "lineage origin suite_id")
    if origin_suite_id == suite["suite_id"]:
        raise ScalingError("lineage origin must be a distinct prior suite")
    for field in ("suite_sha256", "execution_fingerprint", "template_sha256"):
        if not isinstance(origin[field], str) or _SHA256.fullmatch(origin[field]) is None:
            raise ScalingError(f"lineage origin {field} must be a lowercase SHA-256")
    git_commit = origin["git_commit"]
    if not isinstance(git_commit, str) or re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ScalingError("lineage origin git_commit must be a full lowercase commit ID")

    repo = Path(__file__).resolve().parent.parent

    def resolve_repo_path(field: str, *, regular_file: bool) -> Path:
        raw = origin[field]
        if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
            raise ScalingError(f"lineage origin {field} must be a relative path")
        unresolved = manifest_path.parent / raw
        if unresolved.is_symlink():
            raise ScalingError(f"lineage origin {field} must not be a symlink")
        resolved = unresolved.resolve()
        try:
            resolved.relative_to(repo)
        except ValueError as exc:
            raise ScalingError(f"lineage origin {field} escapes the repository") from exc
        if regular_file and (
            not resolved.is_file() or resolved.is_symlink()
        ):
            raise ScalingError(f"lineage origin {field} must be a regular file")
        if not regular_file and resolved.exists() and not resolved.is_dir():
            raise ScalingError("lineage origin runs_root must be a directory")
        return resolved

    origin_suite_path = resolve_repo_path("suite_path", regular_file=True)
    origin_template_path = resolve_repo_path("template_path", regular_file=True)
    origin_runs_root = resolve_repo_path("runs_root", regular_file=False)
    if _sha256(origin_suite_path) != origin["suite_sha256"]:
        raise ScalingError("lineage origin suite bytes differ from their pin")
    if _sha256(origin_template_path) != origin["template_sha256"]:
        raise ScalingError("lineage origin template bytes differ from their pin")
    origin_suite_yaml = _load_yaml(origin_suite_path, "lineage origin suite")
    if origin_suite_yaml.get("suite_id") != origin_suite_id:
        raise ScalingError("lineage origin suite ID differs from the pinned suite bytes")

    commit_paths = {
        origin_suite_path.relative_to(repo).as_posix(): origin["suite_sha256"],
        origin_template_path.relative_to(repo).as_posix(): origin["template_sha256"],
    }

    source_snapshot = dict(
        _mapping(origin["source_snapshot"], "lineage origin source_snapshot")
    )
    current_snapshot = _source_snapshot(repo)
    commit_paths.update(source_snapshot)
    for relative, expected_digest in commit_paths.items():
        try:
            completed = subprocess.run(
                ["git", "cat-file", "blob", f"{git_commit}:{relative}"],
                cwd=repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise ScalingError("cannot validate lineage origin Git commit") from exc
        if (
            completed.returncode != 0
            or hashlib.sha256(completed.stdout).hexdigest() != expected_digest
        ):
            raise ScalingError(
                f"lineage origin Git commit does not pin expected bytes: {relative}"
            )
    if set(source_snapshot) != set(current_snapshot):
        raise ScalingError("lineage origin source snapshot has an incomplete source set")
    for relative, digest in source_snapshot.items():
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ScalingError(f"lineage origin source digest is invalid: {relative}")
        # The lineage-aware scaling runner is necessarily newer. Every source
        # that can affect trainer execution, data identity, or evaluation must
        # remain byte-identical to v2; only this orchestration module may differ.
        if relative != "speedrun/scaling.py" and current_snapshot[relative] != digest:
            raise ScalingError(
                f"current execution source differs from lineage origin: {relative}"
            )
    fingerprint_payload = {
        "suite_sha256": origin["suite_sha256"],
        "template_sha256": origin["template_sha256"],
        "source_snapshot": source_snapshot,
    }
    if _canonical_mapping_sha256(fingerprint_payload) != origin["execution_fingerprint"]:
        raise ScalingError("lineage origin execution fingerprint does not recompute")

    raw_artifacts = payload["artifacts"]
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ScalingError("lineage artifacts must be a nonempty list")
    points_by_id = {str(point["id"]): point for point in suite["all_variants"]}
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()
    calibration_slice = str(suite["compute_slices"][0]["id"])
    for index, raw in enumerate(raw_artifacts):
        entry = dict(_mapping(raw, f"lineage artifacts[{index}]"))
        if set(entry) != {"point_id", "run_manifest_sha256", "result_sha256"}:
            raise ScalingError(f"lineage artifacts[{index}] has unexpected fields")
        point_id = _name(entry["point_id"], f"lineage artifacts[{index}].point_id")
        if point_id in seen:
            raise ScalingError(f"duplicate lineage point: {point_id}")
        point = points_by_id.get(point_id)
        if (
            point is None
            or point["slice"] != calibration_slice
            or point["role"]
            not in {
                "learning_rate_calibration",
                "learning_rate_search",
                "extension_learning_rate_calibration",
                "extension_learning_rate_search",
            }
            or "learning_rate" not in point
            or "config_sha256" not in point
        ):
            raise ScalingError(
                f"lineage point is not an explicit calibration in this suite: {point_id}"
            )
        for field in ("run_manifest_sha256", "result_sha256"):
            digest = entry[field]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ScalingError(f"lineage {point_id} {field} is invalid")
        artifacts.append(entry)
        seen.add(point_id)

    normalized_origin = {
        **origin,
        "runs_root": str(origin_runs_root),
        "suite_path": str(origin_suite_path),
        "template_path": str(origin_template_path),
        "source_snapshot": source_snapshot,
        "source_snapshot_sha256": _canonical_mapping_sha256(source_snapshot),
    }
    return {
        "schema_version": 1,
        "lineage_id": lineage_id,
        "manifest_path": str(manifest_path),
        "manifest_repository_path": manifest_path.relative_to(repo).as_posix(),
        "manifest_sha256": manifest_digest,
        "origin": normalized_origin,
        "artifacts": artifacts,
        "artifacts_by_point": {entry["point_id"]: entry for entry in artifacts},
    }


def _lineage_summary(suite: Mapping[str, Any]) -> dict[str, Any] | None:
    lineage = suite.get("lineage")
    if not isinstance(lineage, Mapping):
        return None
    origin = _mapping(lineage["origin"], "lineage origin")
    artifacts = lineage["artifacts"]
    return {
        "lineage_id": lineage["lineage_id"],
        "manifest": lineage["manifest_repository_path"],
        "manifest_sha256": lineage["manifest_sha256"],
        "origin_suite_id": origin["suite_id"],
        "origin_git_commit": origin["git_commit"],
        "origin_suite_sha256": origin["suite_sha256"],
        "origin_execution_fingerprint": origin["execution_fingerprint"],
        "origin_source_snapshot_sha256": origin["source_snapshot_sha256"],
        "allowlisted_artifact_count": len(artifacts),
        "allowlisted_point_ids": [entry["point_id"] for entry in artifacts],
        "allowlisted_artifacts": [dict(entry) for entry in artifacts],
    }


def _lineage_entry_for_point(
    suite: Mapping[str, Any], point: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    lineage = suite.get("lineage")
    if not isinstance(lineage, Mapping):
        return None
    entries = _mapping(lineage["artifacts_by_point"], "lineage artifact index")
    entry = entries.get(str(point["id"]))
    return None if entry is None else _mapping(entry, "lineage artifact")


def _point_has_result(
    suite: Mapping[str, Any], point: Mapping[str, Any], runs_root: Path
) -> bool:
    if _lineage_entry_for_point(suite, point) is not None:
        return True
    return (runs_root / str(point["id"]) / "artifacts" / "result.json").is_file()


def _validate_lineage_output_root(
    suite: Mapping[str, Any], path: Path, *, label: str
) -> Path:
    """Reject equal, nested, and parent-confused output roots around v2."""

    resolved = path.expanduser().resolve()
    lineage = suite.get("lineage")
    if not isinstance(lineage, Mapping):
        return resolved
    origin = _mapping(lineage["origin"], "lineage origin")
    origin_root = Path(str(origin["runs_root"])).resolve()

    def contains(parent: Path, child: Path) -> bool:
        try:
            child.relative_to(parent)
        except ValueError:
            return False
        return True

    if contains(origin_root, resolved) or contains(resolved, origin_root):
        raise ScalingError(
            f"{label} must be disjoint from immutable lineage root {origin_root}"
        )
    return resolved


def print_plan(suite: Mapping[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "suite_id": suite["suite_id"],
                    "execution_fingerprint": suite["execution_fingerprint"],
                    "lineage": _lineage_summary(suite),
                    "anchor": suite["anchor"],
                    "seed": suite["seed"],
                    "dataset": suite["dataset"],
                    "validation_tokens": suite["validation_tokens"],
                    "compute_slices": suite["compute_slices"],
                    "fit_geometry": [
                        _public_point(item) for item in suite["fit_geometry"]
                    ],
                    "learning_rate_candidates": suite["learning_rate_candidates"],
                    "learning_rate_search": suite["learning_rate_search"],
                    "adaptive_calibration_runs": [
                        _public_point(item)
                        for item in suite["adaptive_calibrations"]
                    ],
                    "calibration_runs": [
                        _public_point(item) for item in suite["calibrations"]
                    ],
                    "dependent_fit_runs": [
                        _public_point(item) for item in suite["variants"]
                    ],
                    "controls": [_public_point(item) for item in suite["controls"]],
                    "optional_extensions": [
                        _public_point(item) for item in suite["optional_extensions"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(f"IsoFLOP suite: {suite['suite_id']}")
    print(f"execution fingerprint: {suite['execution_fingerprint']}")
    lineage = _lineage_summary(suite)
    if lineage is not None:
        print(
            f"immutable lineage: {lineage['allowlisted_artifact_count']} exact artifacts "
            f"from {lineage['origin_suite_id']} "
            f"({lineage['manifest_sha256']})"
        )
    print(f"baseline compute: {int(suite['anchor']['total_flops']):,} FLOPs")
    print(f"fixed seed: {int(suite['seed'])}")
    print(
        f"data requirement: {int(suite['required_train_tokens']):,} usable train + "
        f"{int(suite['validation_tokens']):,} validation tokens"
    )
    for compute_slice in suite["compute_slices"]:
        print(
            f"\n{compute_slice['id']} · {compute_slice['multiplier']:.2f}× · "
            f"{int(compute_slice['target_total_flops']):,} FLOPs"
        )
        print("point       params       steps       tokens          tok/param   FLOP error")
        for point in suite["fit_geometry"]:
            if point["slice"] != compute_slice["id"]:
                continue
            print(
                f"{point['id']:<11} {int(point['parameters']):>11,} "
                f"{int(point['steps']):>10,} {int(point['train_tokens']):>15,} "
                f"{float(point['tokens_per_parameter']):>11.3f} "
                f"{100.0 * float(point['relative_flop_error']):>10.5f}%"
            )
    if suite["controls"]:
        print("\ncontrols: " + ", ".join(item["id"] for item in suite["controls"]))
    if suite["optional_extensions"]:
        print(
            "optional edge extensions: "
            + ", ".join(item["id"] for item in suite["optional_extensions"])
        )
    calibration_multiplier = sum(
        float(suite["slices_by_id"][item["slice"]]["multiplier"])
        for item in suite["calibrations"]
    )
    fit_multiplier = sum(
        float(suite["slices_by_id"][item["slice"]]["multiplier"])
        for item in suite["variants"]
    )
    total_multiplier = calibration_multiplier + fit_multiplier + sum(
        float(suite["slices_by_id"][item["slice"]]["multiplier"])
        for item in suite["controls"]
    )
    rates = ", ".join(
        f"{float(item['value']):.2e}" for item in suite["learning_rate_candidates"]
    )
    upper_rates = ", ".join(
        f"{float(item['value']):.8g}"
        for item in suite["learning_rate_search"]["upper"]
    )
    if suite["learning_rate_scope"] == "per_compute_slice_and_shape":
        print(
            f"\nlearning-rate calibration: {rates} independently at every "
            "(compute slice, model shape); only stable interior winners are admitted"
        )
    else:
        print(
            f"\nlearning-rate calibration: {rates} at {suite['compute_slices'][0]['id']}; "
            "the lowest 100M-token validation loss per shape continues to later slices"
        )
    print(f"bounded geometric upper LR schedule: {upper_rates}")
    print(f"\nplanned base cost: {total_multiplier:.2f} completed-baseline equivalents")
    print(
        "The exponent fit is emitted only if all three one-seed slice minima are "
        "interior. It is local to this tokenizer, data, architecture family, "
        "initialization, fixed batch, and optimizer schedule."
    )


def materialize_configs(
    suite: Mapping[str, Any], destination: Path, names: Sequence[str]
) -> tuple[Path, ...]:
    root = _validate_lineage_output_root(
        suite, destination, label="materialized-config output"
    )
    available = {str(item["id"]): item for item in suite["all_variants"]}
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ScalingError(f"unknown point(s): {', '.join(unknown)}")
    written: list[Path] = []
    for name in names:
        target = root / name / "config.yaml"
        payload = variant_config_bytes(suite, available[name])
        _write_immutable_bytes(target, payload)
        written.append(target)
    return tuple(written)


def _manifest_shard_contract(
    payload: Mapping[str, Any], suite: Mapping[str, Any]
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Validate the exact builder identity and 4B shard inventory without I/O."""

    dataset_contract = suite["dataset"]
    if payload.get("name") != dataset_contract["id"]:
        raise ScalingError(
            f"dataset manifest name must be {dataset_contract['id']!r}; got "
            f"{payload.get('name')!r}"
        )
    source = _mapping(payload.get("source"), "dataset manifest source")
    tokenizer = _mapping(payload.get("tokenizer"), "dataset manifest tokenizer")
    if (
        source.get("dataset") != dataset_contract["source_repository"]
        or source.get("revision") != dataset_contract["source_revision"]
    ):
        raise ScalingError(
            "dataset manifest source repository/revision differs from suite"
        )
    if (
        tokenizer.get("name") != "gpt2"
        or tokenizer.get("implementation") != "tiktoken"
        or tokenizer.get("implementation_version")
        != dataset_contract["tokenizer_version"]
        or tokenizer.get("document_prefix_token") != 50_256
        or tokenizer.get("vocab_size") != 50_257
    ):
        raise ScalingError("dataset manifest tokenizer contract differs from suite")
    format_info = _mapping(payload.get("format"), "dataset manifest format")
    if (
        format_info.get("name") != "llm.c-gpt2-v1"
        or format_info.get("header_bytes") != 1_024
        or format_info.get("header_dtype") != "little-endian int32"
        or format_info.get("magic") != _LLMC_MAGIC
        or format_info.get("version") != _LLMC_VERSION
        or format_info.get("token_dtype") != "little-endian uint16"
    ):
        raise ScalingError("dataset manifest format differs from llm.c GPT-2 v1")
    entries = payload.get("files")
    if not isinstance(entries, list):  # Also gives this helper a standalone contract.
        raise ScalingError("dataset manifest files must be a list")
    if tuple(entry.get("path") for entry in entries) != _FINEWEB4B_NAMES:
        raise ScalingError(
            "4B dataset must list validation shard 000000 followed by exactly "
            "train shards 000001..000039"
        )
    train_entries = tuple(entry for entry in entries if entry.get("split") == "train")
    validation_entries = tuple(
        entry for entry in entries if entry.get("split") == "validation"
    )
    if (
        tuple(entry.get("path") for entry in train_entries)
        != _FINEWEB4B_TRAIN_NAMES
        or tuple(entry.get("path") for entry in validation_entries)
        != _FINEWEB4B_VALIDATION_NAMES
    ):
        raise ScalingError("4B dataset split assignment differs from its 39+1 contract")
    if payload.get("default_train_shards") != 39:
        raise ScalingError("4B manifest default_train_shards must equal 39")
    if payload.get("validation_prefix_tokens") != 100_000_000:
        raise ScalingError("4B manifest validation prefix must equal 100,000,000")
    for entry in entries:
        if (
            entry.get("tokens") != 100_000_000
            or entry.get("bytes") != 200_001_024
        ):
            raise ScalingError(
                "every 4B shard must contain exactly 100,000,000 uint16 tokens"
            )
        digest = entry.get("sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest.lower()) is None:
            raise ScalingError(f"{entry.get('path')}: manifest SHA-256 is required")
    return train_entries, validation_entries


def _read_regular_json(path: Path, label: str) -> dict[str, Any]:
    payload, _ = _read_regular_json_and_sha256(path, label)
    return payload


def _read_regular_json_and_sha256(
    path: Path, label: str
) -> tuple[dict[str, Any], str]:
    """Read once so validation hashes and parsed JSON cover identical bytes."""

    if not path.is_file() or path.is_symlink():
        raise ScalingError(f"{label} must be a regular, non-symlink file: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScalingError(f"cannot read {label} {path}: {exc}") from exc
    return dict(_mapping(payload, label)), hashlib.sha256(raw).hexdigest()


def _validate_production_provenance(
    root: Path, payload: Mapping[str, Any], suite: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and recompute the builder's source/exclusion/core relationships."""

    dataset = suite["dataset"]
    source_path = root / "source.json"
    exclusions_path = root / "exclusions.json"
    plan_path = root / "BUILD_PLAN.json"
    source_payload = _read_regular_json(source_path, "source inventory")
    exclusions_payload = _read_regular_json(exclusions_path, "exclusion policy")
    plan = _read_regular_json(plan_path, "4B build plan")
    try:
        inventory = source_inventory_from_dict(source_payload)
    except FineWebBuildError as exc:
        raise ScalingError(f"invalid pinned source inventory: {exc}") from exc
    if inventory.digest != dataset["source_inventory_sha256"]:
        raise ScalingError("source inventory canonical digest differs from suite pin")

    source_date = _mapping(exclusions_payload.get("source_date"), "exclusion source_date")
    fresh = _mapping(exclusions_payload.get("fresh10"), "exclusion fresh10")
    if set(exclusions_payload) != {"schema_version", "source_date", "fresh10"}:
        raise ScalingError("exclusion policy has unexpected or missing fields")
    if exclusions_payload.get("schema_version") != 1 or dict(source_date) != {
        "operator": "<",
        "cutoff": DEFAULT_SOURCE_DATE_CUTOFF,
        "missing_or_invalid": "exclude",
    }:
        raise ScalingError("exclusion policy does not enforce the pre-2024 cutoff")
    required_fresh_fields = {
        "normalized_urls",
        "exact_canonical_text_sha256",
        "raw_source_sha256",
        "notes",
    }
    if set(fresh) != required_fresh_fields:
        raise ScalingError("Fresh10 exclusion policy has unexpected or missing fields")
    try:
        policy = ExclusionPolicy(
            DEFAULT_SOURCE_DATE_CUTOFF,
            frozenset(str(value) for value in fresh["normalized_urls"]),
            frozenset(
                str(value).lower()
                for value in fresh["exact_canonical_text_sha256"]
            ),
            frozenset(str(value).lower() for value in fresh["raw_source_sha256"]),
        )
        policy.validate()
    except (FineWebBuildError, TypeError) as exc:
        raise ScalingError(f"invalid production exclusion policy: {exc}") from exc
    if policy.as_dict() != exclusions_payload:
        raise ScalingError("exclusion policy is not the canonical builder policy")
    exclusion_digest = canonical_json_sha256(exclusions_payload)
    if exclusion_digest != dataset["exclusion_policy_sha256"]:
        raise ScalingError("exclusion policy canonical digest differs from suite pin")

    repo = Path(__file__).resolve().parent.parent
    builder_sha = _sha256(repo / "speedrun" / "fineweb_builder.py")
    entrypoint_sha = _sha256(repo / "scripts" / "prepare_fineweb.py")
    core = {
        "builder_version": BUILDER_VERSION,
        "builder_module_sha256": builder_sha,
        "entrypoint_sha256": entrypoint_sha,
        "pyarrow_version": PYARROW_VERSION,
        "source_inventory_sha256": inventory.digest,
        "source_order": "upstream-global-shuffle-order",
        "exclusion_policy_sha256": exclusion_digest,
        "exclusion_policy": exclusions_payload,
        "tokenizer": {
            "name": "gpt2",
            "implementation": "tiktoken",
            "version": TIKTOKEN_VERSION,
            "document_prefix_token": EOT_TOKEN,
            "vocab_size": VOCAB_SIZE,
        },
        "format": {
            "name": "llm.c-gpt2-v1",
            "header_bytes": 1_024,
            "magic": _LLMC_MAGIC,
            "version": _LLMC_VERSION,
            "token_dtype": "little-endian uint16",
        },
        "shard_tokens": 100_000_000,
        "validation_tokens": 100_000_000,
        "max_document_bytes": 16 * 1_024**2,
        "split_policy": (
            "first validation_tokens form validation; discard the remainder of its "
            "boundary document; subsequent whole-document stream forms train"
        ),
    }
    core_digest = canonical_json_sha256(core)
    if core_digest != dataset["preparation_core_sha256"]:
        raise ScalingError("recomputed preparation core differs from suite pin")

    if plan != {
        "schema_version": 1,
        "status": "manifest.json appears only after this prefix is complete",
        "directory": "4B",
        "total_tokens": 4_000_000_000,
        "validation_tokens": 100_000_000,
        "training_tokens": 3_900_000_000,
        "source_inventory_sha256": inventory.digest,
        "exclusion_policy_sha256": exclusion_digest,
        "core_sha256": core_digest,
    }:
        raise ScalingError("4B BUILD_PLAN does not match the production core")

    manifest_source = _mapping(payload.get("source"), "dataset manifest source")
    expected_source_fields = {
        "dataset",
        "revision",
        "global_shuffle_seed",
        "global_shuffle_provenance",
        "inventory_path",
        "inventory_sha256",
        "selection",
        "source_dataset_card",
        "source_date_before",
        "exclusion_policy_path",
        "exclusion_policy_sha256",
        "excluded_documents_at_prefix_end",
    }
    if set(manifest_source) != expected_source_fields:
        raise ScalingError("dataset manifest source provenance is incomplete")
    excluded = _mapping(
        manifest_source["excluded_documents_at_prefix_end"],
        "dataset exclusion counts",
    )
    allowed_reasons = {
        "missing_or_invalid_date",
        "date_cutoff",
        "fresh10_url",
        "fresh10_text_sha256",
        "fresh10_raw_sha256",
    }
    if any(
        key not in allowed_reasons
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in excluded.items()
    ):
        raise ScalingError("dataset manifest has invalid exclusion counters")
    expected_source = {
        "dataset": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "global_shuffle_seed": UPSTREAM_GLOBAL_SHUFFLE_SEED,
        "global_shuffle_provenance": "upstream dataset card claim",
        "inventory_path": "source.json",
        "inventory_sha256": inventory.digest,
        "selection": "first 4,000,000,000 prepared tokens",
        "source_dataset_card": (
            "https://huggingface.co/datasets/"
            "HuggingFaceFW/fineweb_100BT-shuffled"
        ),
        "source_date_before": DEFAULT_SOURCE_DATE_CUTOFF,
        "exclusion_policy_path": "exclusions.json",
        "exclusion_policy_sha256": exclusion_digest,
    }
    for field, expected in expected_source.items():
        if manifest_source[field] != expected:
            raise ScalingError(f"dataset manifest source.{field} differs from production")

    preparation = _mapping(payload.get("preparation"), "dataset preparation")
    required_preparation_fields = {
        "builder",
        "builder_version",
        "builder_module_sha256",
        "entrypoint_sha256",
        "pyarrow_version",
        "core_sha256",
        "shard_tokens",
        "split_policy",
        "validation_train_document_disjoint",
        "validation_boundary_discarded_tokens",
        "validation_boundary_document_id_sha256",
        "nested_prefix",
    }
    if set(preparation) != required_preparation_fields:
        raise ScalingError("dataset preparation provenance is incomplete")
    boundary_tokens = preparation["validation_boundary_discarded_tokens"]
    boundary_digest = preparation["validation_boundary_document_id_sha256"]
    if (
        preparation["builder"] != "gpt-tpu-speedrun fineweb builder"
        or preparation["builder_version"] != BUILDER_VERSION
        or preparation["builder_module_sha256"] != builder_sha
        or preparation["entrypoint_sha256"] != entrypoint_sha
        or preparation["pyarrow_version"] != PYARROW_VERSION
        or preparation["core_sha256"] != core_digest
        or preparation["shard_tokens"] != 100_000_000
        or preparation["split_policy"]
        != (
            "first 100,000,000 tokens validation; discard the rest of its "
            "boundary document; remaining documents train"
        )
        or preparation["validation_train_document_disjoint"] is not True
        or preparation["nested_prefix"] is not True
        or isinstance(boundary_tokens, bool)
        or not isinstance(boundary_tokens, int)
        or boundary_tokens <= 0
        or not isinstance(boundary_digest, str)
        or _SHA256.fullmatch(boundary_digest) is None
    ):
        raise ScalingError("dataset preparation/core/disjoint-split contract differs")
    return {
        "source_inventory_sha256": inventory.digest,
        "source_inventory_raw_sha256": _sha256(source_path),
        "exclusion_policy_sha256": exclusion_digest,
        "exclusion_policy_raw_sha256": _sha256(exclusions_path),
        "preparation_core_sha256": core_digest,
        "build_plan_raw_sha256": _sha256(plan_path),
        "builder_module_sha256": builder_sha,
        "entrypoint_sha256": entrypoint_sha,
        "source_date_before": DEFAULT_SOURCE_DATE_CUTOFF,
        "validation_train_document_disjoint": True,
        "validation_boundary_discarded_tokens": boundary_tokens,
        "validation_boundary_document_id_sha256": boundary_digest,
    }


def _dataset_provenance(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON-safe, immutable data identity recorded beside every run."""

    return {
        "name": str(inventory["dataset_name"]),
        "root": str(inventory["root"]),
        "manifest_path": str(inventory["manifest_path"]),
        "manifest_raw_sha256": str(inventory["manifest_raw_sha256"]),
        "manifest_canonical_sha256": str(
            inventory["manifest_canonical_sha256"]
        ),
        "usable_train_tokens": int(inventory["usable_train_tokens"]),
        "usable_validation_tokens": int(inventory["usable_validation_tokens"]),
        "production": dict(inventory["production"]),
        "shards": [dict(item) for item in inventory["shards"]],
    }


def validate_data_directory(data_path: Path, suite: Mapping[str, Any]) -> dict[str, Any]:
    """Perform the suite's one full, release-gating manifest and shard check."""

    root = data_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ScalingError(f"data path is not a directory: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ScalingError(
            f"completed sweep data requires a regular sibling manifest: {manifest_path}"
        )
    try:
        payload, loaded_path = load_manifest(manifest_path)
    except DataError as exc:
        raise ScalingError(f"invalid dataset manifest: {exc}") from exc
    train_entries, validation_entries = _manifest_shard_contract(payload, suite)
    production = _validate_production_provenance(root, payload, suite)
    actual_binary_paths = tuple(sorted(root.glob("*.bin")))
    actual_binary_names = {path.name for path in actual_binary_paths}
    expected_names = set(_FINEWEB4B_NAMES)
    if actual_binary_names != expected_names:
        missing = sorted(expected_names - actual_binary_names)
        extra = sorted(actual_binary_names - expected_names)
        raise ScalingError(
            "dataset .bin inventory differs from its exact 39+1 contract; "
            f"missing={missing}, extra={extra}"
        )
    symlinks = [path.name for path in actual_binary_paths if path.is_symlink()]
    if symlinks:
        raise ScalingError(f"dataset shards must not be symlinks: {symlinks}")
    print(
        f"verifying full SHA-256 integrity for {len(payload['files'])} dataset shards...",
        flush=True,
    )
    try:
        prepared = verify_dataset(loaded_path, root, train_shards=39, verify_hash=True)
    except DataError as exc:
        raise ScalingError(f"dataset shard verification failed: {exc}") from exc
    seq_len = int(suite["sequence_length"])
    usable_train = sum(
        ((int(entry["tokens"]) - 1) // seq_len) * seq_len
        for entry in train_entries
    )
    usable_validation = sum(
        ((int(entry["tokens"]) - 1) // seq_len) * seq_len
        for entry in validation_entries
    )
    required_train = int(suite["required_train_tokens"])
    required_validation = int(suite["validation_tokens"])
    if usable_train < required_train:
        raise ScalingError(
            f"dataset has {usable_train:,} usable train targets; suite needs "
            f"{required_train:,} without replacement"
        )
    if usable_validation < required_validation:
        raise ScalingError(
            f"dataset has {usable_validation:,} usable validation targets; suite "
            f"needs {required_validation:,}"
        )
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest_raw_sha256": sha256_file(manifest_path),
        "manifest_canonical_sha256": manifest_digest(payload),
        "dataset_name": prepared.name,
        "train_files": list(prepared.train_files),
        "validation_files": list(prepared.validation_files),
        "shards": [
            {
                "path": str(entry["path"]),
                "split": str(entry["split"]),
                "tokens": int(entry["tokens"]),
                "bytes": int(entry["bytes"]),
                "sha256": str(entry["sha256"]).lower(),
            }
            for entry in payload["files"]
        ],
        "production": production,
        "usable_train_tokens": usable_train,
        "usable_validation_tokens": usable_validation,
    }


def validate_fresh10_directory(
    manifest_path: Path, root_path: Path, suite: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the exact canonical Fresh10 manifest and every pinned domain byte."""

    expected_manifest = Path(suite["fresh10"]["manifest_path"])
    resolved_manifest = manifest_path.expanduser().resolve(strict=True)
    if (
        resolved_manifest != expected_manifest
        or not resolved_manifest.is_file()
        or resolved_manifest.is_symlink()
    ):
        raise ScalingError("--downstream-manifest must be the pinned canonical Fresh10 file")
    root = root_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ScalingError(f"Fresh10 root is not a directory: {root}")
    try:
        prepared = verify_fresh10(
            root, manifest=resolved_manifest, verify_hash=True
        )
    except DataError as exc:
        raise ScalingError(f"Fresh10 verification failed: {exc}") from exc
    expected_domains = tuple(FRESH10_DOMAINS)
    if (
        prepared.name != suite["fresh10"]["name"]
        or tuple(domain.name for domain in prepared.domains) != expected_domains
        or len(prepared.domains) != suite["fresh10"]["domains"]
        or prepared.scored_tokens != suite["fresh10"]["scored_tokens"]
    ):
        raise ScalingError("verified Fresh10 identity/domain contract differs from suite")
    payload = suite["fresh10"]["payload"]
    payload_by_name = {str(item["name"]): item for item in payload["domains"]}
    domains: list[dict[str, Any]] = []
    for domain in prepared.domains:
        expected = payload_by_name[domain.name]
        if (
            domain.scored_tokens != suite["fresh10"]["scored_tokens_per_domain"]
            or domain.token_count != expected["tokens"]
            or domain.sha256 != expected["sha256"]
            or len(domain.documents) != 4
            or domain.path.is_symlink()
            or not domain.path.is_file()
        ):
            raise ScalingError(f"Fresh10 {domain.name} differs from pinned contract")
        domains.append(
            {
                "name": domain.name,
                "path": str(expected["path"]),
                "tokens": int(domain.token_count),
                "scored_tokens": int(domain.scored_tokens),
                "bytes": int(expected["bytes"]),
                "sha256": domain.sha256,
                "documents": len(domain.documents),
            }
        )
    return {
        "name": prepared.name,
        "root": root,
        "manifest_path": resolved_manifest,
        "manifest_raw_sha256": _sha256(resolved_manifest),
        "manifest_canonical_sha256": prepared.manifest_sha256,
        "repository": suite["fresh10"]["repository"],
        "revision": suite["fresh10"]["revision"],
        "publication_not_before": suite["fresh10"]["publication_not_before"],
        "scored_tokens": prepared.scored_tokens,
        "domains": domains,
    }


def _fresh10_provenance(inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(inventory["name"]),
        "root": str(inventory["root"]),
        "manifest_path": str(inventory["manifest_path"]),
        "manifest_raw_sha256": str(inventory["manifest_raw_sha256"]),
        "manifest_canonical_sha256": str(inventory["manifest_canonical_sha256"]),
        "repository": str(inventory["repository"]),
        "revision": str(inventory["revision"]),
        "publication_not_before": str(inventory["publication_not_before"]),
        "scored_tokens": int(inventory["scored_tokens"]),
        "domains": [dict(item) for item in inventory["domains"]],
    }


def _runtime_inventory_in_current_process(
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Discover the locked TPU runtime inside a short-lived probe process."""

    expected = suite["runtime"]
    python_version = host_platform.python_version()
    if ".".join(python_version.split(".")[:2]) != expected["python_major_minor"]:
        raise ScalingError(
            f"Python {expected['python_major_minor']}.x is required; got {python_version}"
        )
    versions: dict[str, str] = {}
    for package, field in (
        ("jax", "jax_version"),
        ("jaxlib", "jaxlib_version"),
        ("libtpu", "libtpu_version"),
    ):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError as exc:
            raise ScalingError(f"required runtime package is missing: {package}") from exc
        if versions[package] != expected[field]:
            raise ScalingError(
                f"{package} version must be {expected[field]}; got {versions[package]}"
            )
    try:
        import jax

        devices = tuple(jax.devices())
        local_devices = tuple(jax.local_devices())
        process_count = int(jax.process_count())
        device_count = int(jax.device_count())
        local_device_count = int(jax.local_device_count())
    except Exception as exc:
        raise ScalingError(f"JAX TPU discovery failed: {exc}") from exc
    kinds = sorted({str(device.device_kind) for device in devices})
    platforms = sorted({str(device.platform) for device in devices})
    process_indices = [int(getattr(device, "process_index", -1)) for device in devices]
    device_ids = [int(getattr(device, "id", -1)) for device in devices]
    if (
        platforms != [expected["platform"]]
        or kinds != [expected["device_kind"]]
        or process_count != expected["process_count"]
        or device_count != expected["device_count"]
        or local_device_count != expected["local_device_count"]
        or len(devices) != expected["device_count"]
        or len(local_devices) != expected["local_device_count"]
        or process_indices != [0] * expected["device_count"]
        or sorted(device_ids) != list(range(expected["device_count"]))
    ):
        raise ScalingError(
            "scaling suite requires exactly one-process TPU v4-8; detected "
            f"platforms={platforms}, kinds={kinds}, processes={process_count}, "
            f"devices={device_count}, local_devices={local_device_count}, "
            f"device_ids={device_ids}, process_indices={process_indices}"
        )
    return {
        "python_version": python_version,
        "jax_version": versions["jax"],
        "jaxlib_version": versions["jaxlib"],
        "libtpu_version": versions["libtpu"],
        "platform": expected["platform"],
        "device_count": device_count,
        "local_device_count": local_device_count,
        "process_count": process_count,
        "device_kinds": kinds,
        "device_ids": device_ids,
        "process_indices": process_indices,
    }


def validate_runtime_environment(suite: Mapping[str, Any]) -> dict[str, Any]:
    """Probe v4-8 in an isolated process so the trainer can acquire libtpu.

    Importing JAX and calling ``jax.devices()`` permanently initializes the TPU
    client for the life of that process. The sweep runner must therefore never
    perform discovery in the parent that later launches trainers.
    """

    repo = Path(__file__).resolve().parent.parent
    command = [
        sys.executable,
        "-m",
        "speedrun.scaling",
        "--suite",
        str(suite["path"]),
        "--internal-runtime-probe",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScalingError(
            f"isolated TPU runtime probe failed: {type(exc).__name__}"
        ) from exc
    if completed.returncode != 0:
        detail = next(
            (
                line.strip()
                for line in reversed(completed.stderr.splitlines())
                if line.strip()
            ),
            f"exit status {completed.returncode}",
        )
        raise ScalingError(f"isolated TPU runtime probe failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ScalingError("isolated TPU runtime probe returned invalid JSON") from exc
    return _validated_runtime_provenance(
        payload, suite, label="isolated TPU runtime probe"
    )


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ScalingError(f"existing immutable file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _copy_immutable(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != source.read_bytes()
        ):
            raise ScalingError(f"existing immutable file differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def trainer_command(
    *,
    python_executable: str,
    trainer: Path,
    config: Path,
    output: Path,
    seed: int,
    data_path: Path,
    dataset_id: str,
    color: str,
    downstream_manifest: Path,
    downstream_root: Path,
    attention_tuning_cache: Path | None = None,
    autotune_attention: bool = False,
) -> list[str]:
    """Build the direct trainer argv once, with every option/value paired."""

    if autotune_attention and attention_tuning_cache is None:
        raise ScalingError("--autotune-attention requires --attention-tuning-cache")
    command = [
        python_executable,
        str(trainer),
        "--config",
        str(config),
        "--output-dir",
        str(output),
        "--seed",
        str(seed),
        "--track",
        "open",
        "--profile",
        "dev",
        "--data-path",
        str(data_path),
        "--data-format",
        "llmc",
        "--dataset-id",
        dataset_id,
        "--tokenizer-id",
        "gpt2",
        "--omit-checkpoint",
        "--color",
        color,
    ]
    command.extend(("--downstream-manifest", str(downstream_manifest.resolve())))
    command.extend(("--downstream-root", str(downstream_root.resolve())))
    if attention_tuning_cache is not None:
        command.extend(
            ("--attention-tuning-cache", str(attention_tuning_cache.resolve()))
        )
    if autotune_attention:
        command.append("--autotune-attention")
    return command


def _per_slice_learning_rates(suite: Mapping[str, Any]) -> bool:
    return suite.get("learning_rate_scope") == "per_compute_slice_and_shape"


def _calibration_slice_id(
    suite: Mapping[str, Any], slice_id: str | None
) -> str:
    if _per_slice_learning_rates(suite):
        if slice_id is None:
            raise ScalingError(
                "this suite requires a compute slice for learning-rate selection"
            )
        if slice_id not in suite["slices_by_id"]:
            raise ScalingError(f"unknown learning-rate calibration slice: {slice_id}")
        return slice_id
    canonical = str(suite["compute_slices"][0]["id"])
    if slice_id is not None and slice_id != canonical:
        raise ScalingError(
            f"legacy learning-rate calibration is defined only at {canonical}"
        )
    return canonical


def _learning_rate_selection_path(
    runs_root: Path,
    shape_id: str,
    *,
    slice_id: str | None = None,
) -> Path:
    if slice_id is None:
        return runs_root / "learning-rate-selections" / f"{shape_id}.json"
    return runs_root / "learning-rate-selections" / slice_id / f"{shape_id}.json"


def _resolve_point_learning_rate(
    suite: Mapping[str, Any], point: Mapping[str, Any], runs_root: Path
) -> dict[str, Any]:
    resolved = dict(point)
    if "learning_rate" in resolved:
        if "config_sha256" not in resolved:
            payload = variant_config_bytes(suite, resolved)
            resolved["config_sha256"] = hashlib.sha256(payload).hexdigest()
        return resolved
    shape_id = str(resolved.get("learning_rate_source", resolved["shape_id"]))
    selection_slice = (
        str(resolved["slice"]) if _per_slice_learning_rates(suite) else None
    )
    selection_path = _learning_rate_selection_path(
        runs_root, shape_id, slice_id=selection_slice
    )
    if not selection_path.is_file():
        raise ScalingError(
            f"{point['id']}: missing learning-rate selection for {shape_id}; "
            "run its matching calibration stage first"
        )
    # Recompute from the immutable calibration results every time a dependent
    # point resolves its LR. This catches edited/stale selection JSON, newly
    # added edge trials, and mixed provenance before a launch or fit.
    selection = select_learning_rate(
        suite,
        shape_id=shape_id,
        slice_id=selection_slice,
        runs_path=runs_root,
    )
    if (
        selection.get("schema_version")
        != (
            4
            if _per_slice_learning_rates(suite)
            else 3
            if _lineage_summary(suite) is not None
            else 2
        )
        or selection.get("suite_id") != suite["suite_id"]
        or selection.get("suite_sha256") != suite["suite_sha256"]
        or selection.get("execution_fingerprint") != suite["execution_fingerprint"]
        or selection.get("shape_id") != shape_id
        or (
            _per_slice_learning_rates(suite)
            and selection.get("slice")
            != _calibration_slice_id(suite, selection_slice)
        )
    ):
        raise ScalingError(f"learning-rate selection identity mismatch: {selection_path}")
    selected_rate = _finite(
        selection.get("selected_learning_rate"),
        f"{shape_id}.selected_learning_rate",
        positive=True,
    )
    selected_id = selection.get("selected_point_id")
    selected_candidate = next(
        (
            candidate
            for candidate in (
                suite["calibrations"]
                + suite["extension_calibrations"]
                + suite["adaptive_calibrations"]
            )
            if candidate["shape_id"] == shape_id
            and candidate["slice"] == _calibration_slice_id(suite, selection_slice)
            and candidate["id"] == selected_id
        ),
        None,
    )
    if selected_candidate is None or not math.isclose(
        selected_rate,
        float(selected_candidate["learning_rate"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ScalingError(f"learning-rate selection candidate mismatch: {selection_path}")
    resolved["learning_rate"] = selected_rate
    resolved["learning_rate_selection"] = str(selection_path)
    payload = variant_config_bytes(suite, resolved)
    resolved["config_sha256"] = hashlib.sha256(payload).hexdigest()
    return resolved


def run_variants(
    suite: Mapping[str, Any],
    *,
    names: Sequence[str],
    data_path: Path,
    runs_path: Path,
    seed: int,
    color: str,
    downstream_manifest: Path,
    downstream_root: Path,
    attention_tuning_cache: Path | None,
    autotune_attention: bool,
    resume: bool,
    data_inventory: Mapping[str, Any] | None = None,
    fresh10_inventory: Mapping[str, Any] | None = None,
    runtime_inventory: Mapping[str, Any] | None = None,
    allow_adaptive: bool = False,
) -> None:
    if seed != suite["seed"]:
        raise ScalingError(f"sweep seed must be exactly {suite['seed']}; got {seed}")
    runs_root = _validate_lineage_output_root(suite, runs_path, label="--runs")
    available = {str(item["id"]): item for item in suite["all_variants"]}
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ScalingError(f"unknown point(s): {', '.join(unknown)}")
    if _per_slice_learning_rates(suite) and not allow_adaptive:
        v4_calibration_roles = {
            "learning_rate_calibration",
            "learning_rate_search",
            "extension_learning_rate_calibration",
            "extension_learning_rate_search",
        }
        if any(available[name]["role"] in v4_calibration_roles for name in names):
            raise ScalingError(
                "v4 learning-rate trials may only be launched by --staged to "
                "preserve the low-to-high instability frontier"
            )
    adaptive_roles = {
        "learning_rate_search",
        "extension_learning_rate_search",
    }
    adaptive = [available[name] for name in names if available[name]["role"] in adaptive_roles]
    if adaptive and not allow_adaptive:
        raise ScalingError(
            "adaptive learning-rate variants may only be launched by --staged; "
            "resume the staged study to preserve geometric ordering"
        )
    adaptive_groups = sorted(
        {
            (str(point["slice"]), str(point["shape_id"]))
            for point in adaptive
        }
    )
    for slice_id, shape_id in adaptive_groups:
        _validate_adaptive_completion_prefix(
            suite,
            shape_id,
            runs_root,
            slice_id=(slice_id if _per_slice_learning_rates(suite) else None),
        )
    initial_rates = [float(item["value"]) for item in suite["learning_rate_candidates"]]
    for point in adaptive:
        side = (
            "lower"
            if float(point["learning_rate"]) < min(initial_rates)
            else "upper"
        )
        next_point = _next_adaptive_calibration(
            suite,
            str(point["shape_id"]),
            side,
            runs_root,
            slice_id=(
                str(point["slice"])
                if _per_slice_learning_rates(suite)
                else None
            ),
        )
        if next_point is None or next_point["id"] != point["id"]:
            expected = "none" if next_point is None else str(next_point["id"])
            raise ScalingError(
                f"{point['id']}: adaptive LR launch is out of order; expected {expected}"
            )
    inventory = (
        validate_data_directory(data_path, suite)
        if data_inventory is None
        else dict(data_inventory)
    )
    if Path(inventory["root"]) != data_path.expanduser().resolve(strict=True):
        raise ScalingError("reused data inventory does not match --data-path")
    _validated_dataset_provenance(
        _dataset_provenance(inventory), suite, label="prelaunch dataset provenance"
    )
    downstream = (
        validate_fresh10_directory(downstream_manifest, downstream_root, suite)
        if fresh10_inventory is None
        else dict(fresh10_inventory)
    )
    if (
        Path(downstream["manifest_path"])
        != downstream_manifest.expanduser().resolve(strict=True)
        or Path(downstream["root"])
        != downstream_root.expanduser().resolve(strict=True)
    ):
        raise ScalingError("reused Fresh10 inventory does not match CLI paths")
    _validated_fresh10_provenance(
        _fresh10_provenance(downstream), suite, label="prelaunch Fresh10 provenance"
    )
    runtime = (
        validate_runtime_environment(suite)
        if runtime_inventory is None
        else dict(runtime_inventory)
    )
    runtime = _validated_runtime_provenance(
        runtime, suite, label="prelaunch runtime provenance"
    )
    _validate_prelaunch_lineage_coherence(
        suite,
        runs_root=runs_root,
        dataset=_dataset_provenance(inventory),
        fresh10=_fresh10_provenance(downstream),
        runtime=runtime,
    )
    repo = Path(__file__).resolve().parent.parent
    trainer_source = repo / "submissions" / "reference" / "train.py"
    runs_root.mkdir(parents=True, exist_ok=True)
    dataset_id = str(suite["dataset"]["id"])

    for name in names:
        point = _resolve_point_learning_rate(
            suite, available[name], runs_root
        )
        point_root = runs_root / name
        lineage_entry = _lineage_entry_for_point(suite, point)
        if lineage_entry is not None:
            if point_root.exists() or point_root.is_symlink():
                raise ScalingError(
                    f"{name}: local artifacts must not shadow an immutable lineage input"
                )
            measurement = _read_run(suite, point, runs_root)
            print(
                f"reuse {name}: exact {measurement['lineage']['lineage_id']} lineage "
                f"({measurement['result_sha256']})",
                flush=True,
            )
            continue
        work = point_root / "work"
        output = point_root / "artifacts"
        result_path = output / "result.json"
        metrics_path = output / "metrics.json"
        if result_path.is_file():
            if resume:
                _read_run(suite, point, runs_root)
                print(f"skip {name}: complete result already exists", flush=True)
                continue
            raise ScalingError(f"{name} already has a result; pass --resume to skip it")
        if resume and metrics_path.is_file() and not metrics_path.is_symlink():
            _write_immutable_bytes(result_path, metrics_path.read_bytes())
            _read_run(suite, point, runs_root)
            print(f"recover {name}: finalized existing metrics.json", flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise ScalingError(
                f"{name} has incomplete artifacts; preserve them and choose a new --runs path"
            )
        output.mkdir(parents=True, exist_ok=True)
        trainer = work / "train.py"
        config = work / "config.yaml"
        _copy_immutable(trainer_source, trainer)
        for shared_source in _shared_trainer_sources(repo):
            relative = shared_source.relative_to(repo / "speedrun")
            _copy_immutable(shared_source, work / "speedrun" / relative)
        _write_immutable_bytes(config, variant_config_bytes(suite, point))
        work_files = (trainer, config) + tuple(
            work / "speedrun" / source.relative_to(repo / "speedrun")
            for source in _shared_trainer_sources(repo)
        )
        work_snapshot = {
            path.relative_to(work).as_posix(): _sha256(path) for path in work_files
        }
        study_lineage = _lineage_summary(suite)
        manifest = {
            "schema_version": 4 if study_lineage is not None else 3,
            "classification": "diagnostic_noncompetition_isoflop",
            "suite_id": suite["suite_id"],
            "suite_sha256": suite["suite_sha256"],
            "execution_fingerprint": suite["execution_fingerprint"],
            "template_sha256": suite["template_sha256"],
            "source_snapshot": suite["source_snapshot"],
            "work_snapshot": work_snapshot,
            "point": _public_point(point),
            "trainer_sha256": _sha256(trainer),
            "config_sha256": _sha256(config),
            "checkpoint_policy": "omit_research_checkpoint",
            "dataset": _dataset_provenance(inventory),
            "fresh10": _fresh10_provenance(downstream),
            "runtime": runtime,
            "seed": seed,
        }
        if study_lineage is not None:
            manifest["study_lineage"] = study_lineage
        manifest_path = point_root / "run-manifest.json"
        _write_immutable_bytes(
            manifest_path,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        command = trainer_command(
            python_executable=sys.executable,
            trainer=trainer,
            config=config,
            output=output,
            seed=seed,
            data_path=Path(inventory["root"]),
            dataset_id=dataset_id,
            color=color,
            downstream_manifest=downstream_manifest,
            downstream_root=downstream_root,
            attention_tuning_cache=attention_tuning_cache,
            autotune_attention=autotune_attention,
        )
        print(f"\nlaunch {name}: {' '.join(command)}\n", flush=True)
        environment = dict(os.environ)
        completed = subprocess.run(command, cwd=repo, env=environment, check=False)
        if completed.returncode:
            raise ScalingError(f"{name} trainer exited with status {completed.returncode}")
        if not metrics_path.is_file() or metrics_path.is_symlink():
            raise ScalingError(f"{name} trainer did not write regular metrics.json")
        _write_immutable_bytes(result_path, metrics_path.read_bytes())
        _read_run(suite, point, runs_root)


def _validated_dataset_provenance(
    value: Any, suite: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    provenance = dict(_mapping(value, label))
    expected_fields = {
        "name",
        "root",
        "manifest_path",
        "manifest_raw_sha256",
        "manifest_canonical_sha256",
        "usable_train_tokens",
        "usable_validation_tokens",
        "production",
        "shards",
    }
    if set(provenance) != expected_fields:
        raise ScalingError(f"{label} has unexpected or missing fields")
    if provenance["name"] != suite["dataset"]["id"]:
        raise ScalingError(f"{label}.name differs from the suite")
    for field in ("manifest_raw_sha256", "manifest_canonical_sha256"):
        digest = provenance[field]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ScalingError(f"{label}.{field} is not a lowercase SHA-256")
    root = Path(provenance["root"]) if isinstance(provenance["root"], str) else None
    manifest_path = (
        Path(provenance["manifest_path"])
        if isinstance(provenance["manifest_path"], str)
        else None
    )
    if (
        root is None
        or not root.is_absolute()
        or manifest_path is None
        or manifest_path != root / "manifest.json"
    ):
        raise ScalingError(f"{label} has an invalid root/manifest path identity")
    shards = provenance["shards"]
    if not isinstance(shards, list) or len(shards) != len(_FINEWEB4B_NAMES):
        raise ScalingError(f"{label}.shards must contain the exact 39+1 selection")
    normalized_shards: list[dict[str, Any]] = []
    for index, (raw, expected_name) in enumerate(zip(shards, _FINEWEB4B_NAMES, strict=True)):
        shard = dict(_mapping(raw, f"{label}.shards[{index}]"))
        if set(shard) != {"path", "split", "tokens", "bytes", "sha256"}:
            raise ScalingError(f"{label}.shards[{index}] has unexpected fields")
        expected_split = "validation" if index == 0 else "train"
        if (
            shard["path"] != expected_name
            or shard["split"] != expected_split
            or shard["tokens"] != 100_000_000
            or shard["bytes"] != 200_001_024
            or not isinstance(shard["sha256"], str)
            or _SHA256.fullmatch(shard["sha256"]) is None
        ):
            raise ScalingError(f"{label}.shards[{index}] differs from the 4B contract")
        normalized_shards.append(shard)
    seq_len = int(suite["sequence_length"])
    usable_per_shard = ((100_000_000 - 1) // seq_len) * seq_len
    expected_usable_train = 39 * usable_per_shard
    expected_usable_validation = usable_per_shard
    if (
        provenance["usable_train_tokens"] != expected_usable_train
        or provenance["usable_validation_tokens"] != expected_usable_validation
    ):
        raise ScalingError(f"{label} has incorrect usable target counts")
    production = dict(_mapping(provenance["production"], f"{label}.production"))
    expected_production_fields = {
        "source_inventory_sha256",
        "source_inventory_raw_sha256",
        "exclusion_policy_sha256",
        "exclusion_policy_raw_sha256",
        "preparation_core_sha256",
        "build_plan_raw_sha256",
        "builder_module_sha256",
        "entrypoint_sha256",
        "source_date_before",
        "validation_train_document_disjoint",
        "validation_boundary_discarded_tokens",
        "validation_boundary_document_id_sha256",
    }
    if set(production) != expected_production_fields:
        raise ScalingError(f"{label}.production has unexpected or missing fields")
    for field in (
        "source_inventory_sha256",
        "source_inventory_raw_sha256",
        "exclusion_policy_sha256",
        "exclusion_policy_raw_sha256",
        "preparation_core_sha256",
        "build_plan_raw_sha256",
        "builder_module_sha256",
        "entrypoint_sha256",
        "validation_boundary_document_id_sha256",
    ):
        if not isinstance(production[field], str) or _SHA256.fullmatch(production[field]) is None:
            raise ScalingError(f"{label}.production.{field} is invalid")
    if (
        production["source_inventory_sha256"]
        != suite["dataset"]["source_inventory_sha256"]
        or production["exclusion_policy_sha256"]
        != suite["dataset"]["exclusion_policy_sha256"]
        or production["preparation_core_sha256"]
        != suite["dataset"]["preparation_core_sha256"]
        or production["source_date_before"] != DEFAULT_SOURCE_DATE_CUTOFF
        or production["validation_train_document_disjoint"] is not True
        or isinstance(production["validation_boundary_discarded_tokens"], bool)
        or not isinstance(production["validation_boundary_discarded_tokens"], int)
        or production["validation_boundary_discarded_tokens"] <= 0
    ):
        raise ScalingError(f"{label}.production differs from the pinned contract")
    provenance["production"] = production
    provenance["shards"] = normalized_shards
    return provenance


def _validated_fresh10_provenance(
    value: Any, suite: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    provenance = dict(_mapping(value, label))
    expected_fields = {
        "name",
        "root",
        "manifest_path",
        "manifest_raw_sha256",
        "manifest_canonical_sha256",
        "repository",
        "revision",
        "publication_not_before",
        "scored_tokens",
        "domains",
    }
    if set(provenance) != expected_fields:
        raise ScalingError(f"{label} has unexpected or missing fields")
    manifest_path = provenance["manifest_path"]
    if (
        provenance["name"] != suite["fresh10"]["name"]
        or not isinstance(manifest_path, str)
        or not Path(manifest_path).is_absolute()
        or provenance["manifest_raw_sha256"]
        != suite["fresh10"]["manifest_raw_sha256"]
        or provenance["manifest_canonical_sha256"]
        != suite["fresh10"]["manifest_canonical_sha256"]
        or provenance["repository"] != suite["fresh10"]["repository"]
        or provenance["revision"] != suite["fresh10"]["revision"]
        or provenance["publication_not_before"]
        != suite["fresh10"]["publication_not_before"]
        or provenance["scored_tokens"] != suite["fresh10"]["scored_tokens"]
        or not isinstance(provenance["root"], str)
        or not Path(provenance["root"]).is_absolute()
    ):
        raise ScalingError(f"{label} differs from the pinned Fresh10 contract")
    domains = provenance["domains"]
    if not isinstance(domains, list) or len(domains) != len(FRESH10_DOMAINS):
        raise ScalingError(f"{label}.domains must contain exactly ten rows")
    manifest_domains = suite["fresh10"]["payload"]["domains"]
    normalized: list[dict[str, Any]] = []
    for index, (raw, expected_name, expected) in enumerate(
        zip(domains, FRESH10_DOMAINS, manifest_domains, strict=True)
    ):
        domain = dict(_mapping(raw, f"{label}.domains[{index}]"))
        if set(domain) != {
            "name",
            "path",
            "tokens",
            "scored_tokens",
            "bytes",
            "sha256",
            "documents",
        } or domain != {
            "name": expected_name,
            "path": expected["path"],
            "tokens": expected["tokens"],
            "scored_tokens": 8_192,
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
            "documents": 4,
        }:
            raise ScalingError(f"{label}.domains[{index}] differs from manifest")
        normalized.append(domain)
    provenance["domains"] = normalized
    return provenance


def _validated_runtime_provenance(
    value: Any, suite: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    runtime = dict(_mapping(value, label))
    expected_fields = {
        "python_version",
        "jax_version",
        "jaxlib_version",
        "libtpu_version",
        "platform",
        "device_count",
        "local_device_count",
        "process_count",
        "device_kinds",
        "device_ids",
        "process_indices",
    }
    if set(runtime) != expected_fields:
        raise ScalingError(f"{label} has unexpected or missing fields")
    expected = suite["runtime"]
    python_version = runtime["python_version"]
    if (
        not isinstance(python_version, str)
        or ".".join(python_version.split(".")[:2]) != expected["python_major_minor"]
        or runtime["jax_version"] != expected["jax_version"]
        or runtime["jaxlib_version"] != expected["jaxlib_version"]
        or runtime["libtpu_version"] != expected["libtpu_version"]
        or runtime["platform"] != expected["platform"]
        or runtime["device_count"] != 4
        or runtime["local_device_count"] != 4
        or runtime["process_count"] != 1
        or runtime["device_kinds"] != ["TPU v4"]
        or runtime["device_ids"] != [0, 1, 2, 3]
        or runtime["process_indices"] != [0, 0, 0, 0]
    ):
        raise ScalingError(f"{label} differs from the exact locked v4-8 runtime")
    return runtime


def _load_lineage_run_manifest(
    suite: Mapping[str, Any], point: Mapping[str, Any], entry: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, Path]:
    """Validate one historical manifest against its byte allowlist and v2 pins."""

    lineage = _mapping(suite.get("lineage"), "suite lineage")
    origin = _mapping(lineage["origin"], "lineage origin")
    name = str(point["id"])
    if entry.get("point_id") != name:
        raise ScalingError(f"{name}: lineage artifact index mismatch")
    # Recheck the immutable checked-in origin bytes at admission time, not only
    # at suite loading, so a long-lived runner cannot observe later path drift.
    origin_suite_path = Path(str(origin["suite_path"]))
    origin_template_path = Path(str(origin["template_path"]))
    if (
        not origin_suite_path.is_file()
        or origin_suite_path.is_symlink()
        or _sha256(origin_suite_path) != origin["suite_sha256"]
        or not origin_template_path.is_file()
        or origin_template_path.is_symlink()
        or _sha256(origin_template_path) != origin["template_sha256"]
    ):
        raise ScalingError("lineage origin suite/template changed after suite loading")

    root = Path(str(origin["runs_root"]))
    point_root = root / name
    artifacts = point_root / "artifacts"
    if (
        not root.is_dir()
        or root.is_symlink()
        or not point_root.is_dir()
        or point_root.is_symlink()
        or (artifacts.exists() and (not artifacts.is_dir() or artifacts.is_symlink()))
    ):
        raise ScalingError(f"{name}: lineage run directory structure is not regular")
    path = point_root / "run-manifest.json"
    expected_manifest_digest = entry.get("run_manifest_sha256")
    try:
        manifest, actual_manifest_digest = _read_regular_json_and_sha256(
            path, f"{name}.lineage_run_manifest"
        )
    except ScalingError as exc:
        raise ScalingError(
            f"{name}: lineage run manifest is missing, symlinked, or not allowlisted"
        ) from exc
    if actual_manifest_digest != expected_manifest_digest:
        raise ScalingError(f"{name}: lineage run manifest is not allowlisted")
    expected_fields = {
        "schema_version",
        "classification",
        "suite_id",
        "suite_sha256",
        "execution_fingerprint",
        "template_sha256",
        "source_snapshot",
        "work_snapshot",
        "point",
        "trainer_sha256",
        "config_sha256",
        "checkpoint_policy",
        "dataset",
        "fresh10",
        "runtime",
        "seed",
    }
    if set(manifest) != expected_fields:
        raise ScalingError(f"{name}: lineage run manifest has unexpected fields")
    expected_identity = {
        "schema_version": 3,
        "classification": "diagnostic_noncompetition_isoflop",
        "suite_id": origin["suite_id"],
        "suite_sha256": origin["suite_sha256"],
        "execution_fingerprint": origin["execution_fingerprint"],
        "template_sha256": origin["template_sha256"],
        "source_snapshot": origin["source_snapshot"],
        "point": _public_point(point),
        "config_sha256": point["config_sha256"],
        "checkpoint_policy": "omit_research_checkpoint",
    }
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise ScalingError(
                f"{name}: lineage run manifest {field} differs from its pinned origin"
            )
    if manifest.get("seed") != suite["seed"]:
        raise ScalingError(f"{name}: lineage run seed differs from the suite")
    trainer_digest = manifest.get("trainer_sha256")
    if not isinstance(trainer_digest, str) or _SHA256.fullmatch(trainer_digest) is None:
        raise ScalingError(f"{name}: lineage trainer_sha256 is invalid")
    source_snapshot = _mapping(origin["source_snapshot"], "lineage source snapshot")
    expected_work_snapshot = {
        "train.py": source_snapshot["submissions/reference/train.py"],
        "config.yaml": point["config_sha256"],
        **{
            source_path: digest
            for source_path, digest in source_snapshot.items()
            if source_path == "speedrun/__init__.py"
            or source_path.startswith("speedrun/kernels/")
        },
    }
    if manifest.get("work_snapshot") != expected_work_snapshot:
        raise ScalingError(f"{name}: lineage copied work snapshot differs from origin")
    repo = Path(__file__).resolve().parent.parent
    for relative, expected_digest in expected_work_snapshot.items():
        if relative == "config.yaml":
            actual_digest = hashlib.sha256(
                variant_config_bytes(suite, point)
            ).hexdigest()
        else:
            source_relative = (
                "submissions/reference/train.py" if relative == "train.py" else relative
            )
            source = repo / source_relative
            if not source.is_file() or source.is_symlink():
                raise ScalingError(
                    f"{name}: tracked lineage source is missing: {source_relative}"
                )
            actual_digest = _sha256(source)
        if actual_digest != expected_digest:
            raise ScalingError(f"{name}: lineage work source differs: {relative}")
    if trainer_digest != expected_work_snapshot["train.py"]:
        raise ScalingError(f"{name}: lineage trainer hash differs from copied source")
    manifest["dataset"] = _validated_dataset_provenance(
        manifest.get("dataset"), suite, label=f"{name}.lineage_run_manifest.dataset"
    )
    manifest["fresh10"] = _validated_fresh10_provenance(
        manifest.get("fresh10"), suite, label=f"{name}.lineage_run_manifest.fresh10"
    )
    manifest["runtime"] = _validated_runtime_provenance(
        manifest.get("runtime"), suite, label=f"{name}.lineage_run_manifest.runtime"
    )
    return manifest, path, root


def _load_run_manifest(
    suite: Mapping[str, Any], point: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], Path]:
    name = str(point["id"])
    path = root / name / "run-manifest.json"
    manifest, _ = _read_regular_json_and_sha256(path, f"{name}.run_manifest")
    study_lineage = _lineage_summary(suite)
    expected_fields = {
        "schema_version",
        "classification",
        "suite_id",
        "suite_sha256",
        "execution_fingerprint",
        "template_sha256",
        "source_snapshot",
        "work_snapshot",
        "point",
        "trainer_sha256",
        "config_sha256",
        "checkpoint_policy",
        "dataset",
        "fresh10",
        "runtime",
        "seed",
    }
    if study_lineage is not None:
        expected_fields.add("study_lineage")
    if set(manifest) != expected_fields:
        raise ScalingError(f"{name}: run manifest has unexpected or missing fields")
    expected_identity = {
        "schema_version": 4 if study_lineage is not None else 3,
        "classification": "diagnostic_noncompetition_isoflop",
        "suite_id": suite["suite_id"],
        "suite_sha256": suite["suite_sha256"],
        "execution_fingerprint": suite["execution_fingerprint"],
        "template_sha256": suite["template_sha256"],
        "source_snapshot": suite["source_snapshot"],
        "point": _public_point(point),
        "config_sha256": point["config_sha256"],
        "checkpoint_policy": "omit_research_checkpoint",
    }
    if study_lineage is not None:
        expected_identity["study_lineage"] = study_lineage
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise ScalingError(f"{name}: run manifest {field} differs from the suite")
    trainer_digest = manifest.get("trainer_sha256")
    if not isinstance(trainer_digest, str) or _SHA256.fullmatch(trainer_digest) is None:
        raise ScalingError(f"{name}: run manifest trainer_sha256 is invalid")
    if manifest.get("seed") != suite["seed"]:
        raise ScalingError(f"{name}: run manifest seed differs from the suite")
    work = root / name / "work"
    expected_work_snapshot = {
        "train.py": suite["source_snapshot"]["submissions/reference/train.py"],
        "config.yaml": point["config_sha256"],
        **{
            source_path: digest
            for source_path, digest in suite["source_snapshot"].items()
            if source_path == "speedrun/__init__.py"
            or source_path.startswith("speedrun/kernels/")
        },
    }
    if manifest.get("work_snapshot") != expected_work_snapshot:
        raise ScalingError(f"{name}: copied work snapshot differs from suite sources")
    for relative, expected_digest in expected_work_snapshot.items():
        copied = work / relative
        if (
            not copied.is_file()
            or copied.is_symlink()
            or _sha256(copied) != expected_digest
        ):
            raise ScalingError(f"{name}: copied work file differs: {relative}")
    if trainer_digest != expected_work_snapshot["train.py"]:
        raise ScalingError(f"{name}: trainer hash differs from copied source")
    manifest["dataset"] = _validated_dataset_provenance(
        manifest.get("dataset"), suite, label=f"{name}.run_manifest.dataset"
    )
    manifest["fresh10"] = _validated_fresh10_provenance(
        manifest.get("fresh10"), suite, label=f"{name}.run_manifest.fresh10"
    )
    manifest["runtime"] = _validated_runtime_provenance(
        manifest.get("runtime"), suite, label=f"{name}.run_manifest.runtime"
    )
    return manifest, path


def _validated_fresh10_result(
    value: Any, suite: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    fresh = dict(_mapping(value, label))
    if set(fresh) != {
        "domains",
        "macro_loss",
        "macro_perplexity",
        "scored_tokens",
        "seconds",
    }:
        raise ScalingError(f"{label} has unexpected or missing fields")
    domains = _mapping(fresh["domains"], f"{label}.domains")
    if set(domains) != set(FRESH10_DOMAINS):
        raise ScalingError(f"{label} must contain exactly the ten canonical domains")
    losses: dict[str, float] = {}
    seconds: list[float] = []
    scored_total = 0
    for name in FRESH10_DOMAINS:
        row = _mapping(domains[name], f"{label}.domains.{name}")
        if set(row) != {"loss", "perplexity", "scored_tokens", "seconds"}:
            raise ScalingError(f"{label}.domains.{name} has unexpected fields")
        loss = _finite(row["loss"], f"{label}.{name}.loss")
        perplexity = _finite(
            row["perplexity"], f"{label}.{name}.perplexity", positive=True
        )
        if not math.isclose(
            math.log(perplexity), loss, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ScalingError(f"{label}.{name} perplexity disagrees with loss")
        scored = _integer(row["scored_tokens"], f"{label}.{name}.scored_tokens")
        if scored != suite["fresh10"]["scored_tokens_per_domain"]:
            raise ScalingError(f"{label}.{name} scored token count differs")
        elapsed = _finite(
            row["seconds"], f"{label}.{name}.seconds", positive=True
        )
        losses[name] = loss
        seconds.append(elapsed)
        scored_total += scored
    macro_loss = _finite(fresh["macro_loss"], f"{label}.macro_loss")
    expected_macro = math.fsum(losses.values()) / len(FRESH10_DOMAINS)
    if not math.isclose(macro_loss, expected_macro, rel_tol=1e-9, abs_tol=1e-12):
        raise ScalingError(f"{label}.macro_loss is not the ten-domain mean")
    macro_perplexity = _finite(
        fresh["macro_perplexity"], f"{label}.macro_perplexity", positive=True
    )
    if not math.isclose(
        math.log(macro_perplexity), macro_loss, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ScalingError(f"{label}.macro_perplexity disagrees with macro loss")
    if fresh["scored_tokens"] != scored_total or scored_total != 81_920:
        raise ScalingError(f"{label}.scored_tokens must equal 81,920")
    total_seconds = _finite(fresh["seconds"], f"{label}.seconds", positive=True)
    if not math.isclose(total_seconds, math.fsum(seconds), rel_tol=1e-9, abs_tol=1e-12):
        raise ScalingError(f"{label}.seconds is not the domain sum")
    return {
        "macro_loss": macro_loss,
        "scored_tokens": scored_total,
        "domain_losses": losses,
    }


_TRAINING_CSV_FIELDS = (
    "step",
    "tokens_processed",
    "cumulative_estimated_flops",
    "train_loss",
    "learning_rate",
    "grad_norm",
)
_VALIDATION_CSV_FIELDS = (
    "step",
    "tokens_processed",
    "kind",
    "domain",
    "validation_tokens",
    "validation_loss",
    "perplexity",
    "validation_seconds",
    "canonical",
)
_DIAGNOSTICS_CSV_FIELDS = (
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
_DIAGNOSTIC_FAMILIES = ("param", "grad", "update")
_DIAGNOSTIC_STATS = (
    "l1_norm",
    "l2_norm",
    "mean",
    "std",
    "third_moment",
    "fourth_moment",
)


def _csv_rows(path: Path, expected_fields: Sequence[str], label: str) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise ScalingError(f"{label} must be a regular, non-symlink file")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_fields):
                raise ScalingError(f"{label} header differs from the emitted schema")
            rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ScalingError(f"{label} is not valid UTF-8 CSV: {exc}") from exc
    if not rows:
        raise ScalingError(f"{label} has no data rows")
    if any(None in row for row in rows):
        raise ScalingError(f"{label} contains a row wider than its header")
    return rows


def _csv_int(text: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(text, str) or re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise ScalingError(f"{label} must be a canonical non-negative integer")
    value = int(text)
    if value < minimum:
        raise ScalingError(f"{label} must be >= {minimum}")
    return value


def _csv_float(text: Any, label: str, *, minimum: float | None = None) -> float:
    if not isinstance(text, str) or not text:
        raise ScalingError(f"{label} must be a finite number")
    try:
        value = float(text)
    except ValueError as exc:
        raise ScalingError(f"{label} must be a finite number") from exc
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ScalingError(f"{label} must be finite and >= {minimum}")
    return value


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ScalingError("cannot compute a stability percentile without values")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ScalingError("cannot compute a stability median without values")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (float(ordered[midpoint - 1]) + float(ordered[midpoint])) / 2.0


def _expected_learning_rate(
    step: int, point: Mapping[str, Any], suite: Mapping[str, Any]
) -> float:
    """Reproduce the trainer's JAX float32 warmup/cosine schedule exactly."""

    step32 = np.asarray(step, dtype=np.float32)
    warmup_steps = int(point["warmup_steps"])
    warmup = (
        np.minimum(np.float32(1.0), step32 / np.float32(warmup_steps))
        if warmup_steps
        else np.asarray(1.0, dtype=np.float32)
    )
    decay_span = max(1, int(point["steps"]) - warmup_steps)
    progress = np.clip(
        (step32 - np.float32(warmup_steps)) / np.float32(decay_span),
        np.float32(0.0),
        np.float32(1.0),
    )
    cosine = np.float32(0.5) * (
        np.float32(1.0) + np.cos(np.float32(np.pi) * progress)
    )
    min_ratio = np.float32(suite["optimizer"]["min_lr_ratio"])
    multiplier = min_ratio + (np.float32(1.0) - min_ratio) * cosine
    return float(np.float32(point["learning_rate"]) * warmup * multiplier)


def _stability_artifact_path(
    artifact_root: Path, point_name: str, filename: str
) -> Path:
    path = artifact_root / point_name / "artifacts" / filename
    if not path.is_file() or path.is_symlink():
        raise ScalingError(
            f"{point_name}: stability evidence is missing or symlinked: {filename}"
        )
    return path


def _classify_stability_series(
    policy: Mapping[str, Any],
    *,
    probe_steps: Sequence[int],
    probe_losses: Sequence[float],
    gradients: Sequence[float],
    training_losses: Sequence[float],
    final_block_parameter_l2: Sequence[float],
) -> dict[str, Any]:
    """Apply the frozen temporal gates to already-validated numeric series."""

    consecutive = int(policy["probe_regression_consecutive"])
    probe_threshold = float(policy["probe_regression_threshold"])
    probe_flags: list[bool] = []
    best_preceding = float(probe_losses[0])
    for index, loss in enumerate(probe_losses):
        regression = float(loss) - best_preceding
        probe_flags.append(
            index > 0
            and (
                regression > probe_threshold
                or math.isclose(
                    regression, probe_threshold, rel_tol=0.0, abs_tol=1e-12
                )
            )
        )
        best_preceding = min(best_preceding, float(loss))
    probe_triggers = [
        int(probe_steps[index])
        for index in range(consecutive - 1, len(probe_flags))
        if all(probe_flags[index - offset] for offset in range(consecutive))
    ]
    maximum_probe_regression = max(
        (
            float(probe_losses[index]) - min(probe_losses[:index])
            for index in range(1, len(probe_losses))
        ),
        default=0.0,
    )

    window = int(policy["trailing_window_steps"])
    if len(gradients) < window * 2 or len(training_losses) != len(gradients):
        raise ScalingError("training schedule is too short for two stability windows")
    gradient_threshold = float(policy["gradient_norm_threshold"])
    fraction_limit = float(policy["maximum_large_gradient_fraction"])
    large_count = sum(value > gradient_threshold for value in gradients[:window])
    maximum_gradient_fraction = large_count / window
    gradient_first: int | None = window if maximum_gradient_fraction >= fraction_limit else None
    gradient_count = 1 if gradient_first is not None else 0
    for end in range(window, len(gradients)):
        large_count += gradients[end] > gradient_threshold
        large_count -= gradients[end - window] > gradient_threshold
        fraction = large_count / window
        maximum_gradient_fraction = max(maximum_gradient_fraction, fraction)
        if fraction >= fraction_limit:
            if gradient_first is None:
                gradient_first = end + 1
            gradient_count += 1

    cumulative = [0.0]
    for loss in training_losses:
        cumulative.append(cumulative[-1] + float(loss))
    window_means = [
        (cumulative[end] - cumulative[end - window]) / window
        for end in range(window, len(training_losses) + 1)
    ]
    loss_threshold = float(policy["train_loss_window_regression_threshold"])
    loss_first: int | None = None
    loss_count = 0
    maximum_loss_regression = 0.0
    best_prior = window_means[0]
    for index, mean in enumerate(window_means[1:], 1):
        regression = mean - best_prior
        maximum_loss_regression = max(maximum_loss_regression, regression)
        if regression > loss_threshold or math.isclose(
            regression, loss_threshold, rel_tol=0.0, abs_tol=1e-12
        ):
            if loss_first is None:
                loss_first = window + index
            loss_count += 1
        best_prior = min(best_prior, mean)

    block_median = _median(final_block_parameter_l2)
    block_ratio = max(final_block_parameter_l2) / block_median
    hard_reasons: list[str] = []
    witnesses: dict[str, dict[str, Any]] = {}
    if probe_triggers:
        hard_reasons.append("consecutive_fixed_prefix_probe_regression")
        witnesses[hard_reasons[-1]] = {
            "first_trigger_step": probe_triggers[0],
            "trigger_count": len(probe_triggers),
            "maximum_regression": maximum_probe_regression,
        }
    if gradient_first is not None:
        hard_reasons.append("trailing_window_large_gradient_fraction")
        witnesses[hard_reasons[-1]] = {
            "first_trigger_step": gradient_first,
            "trigger_count": gradient_count,
            "maximum_fraction": maximum_gradient_fraction,
        }
    if loss_first is not None:
        hard_reasons.append("trailing_window_train_loss_regression")
        witnesses[hard_reasons[-1]] = {
            "first_trigger_step": loss_first,
            "trigger_count": loss_count,
            "maximum_regression": maximum_loss_regression,
        }
    suspect_reasons = (
        ["final_block_parameter_l2_imbalance"]
        if block_ratio
        > float(policy["maximum_final_block_param_l2_to_median"])
        else []
    )
    classification = (
        "rejected" if hard_reasons else "suspect" if suspect_reasons else "stable"
    )
    return {
        "classification": classification,
        "reasons": hard_reasons + suspect_reasons,
        "witnesses": witnesses,
        "summaries": {
            "best_probe_loss": min(probe_losses),
            "maximum_probe_regression_from_best_preceding": maximum_probe_regression,
            "gradient_median": _median(gradients),
            "gradient_maximum": max(gradients),
            "large_gradient_count": sum(
                value > gradient_threshold for value in gradients
            ),
            "maximum_trailing_large_gradient_fraction": maximum_gradient_fraction,
            "best_trailing_train_loss_window_mean": min(window_means),
            "final_trailing_train_loss_window_mean": window_means[-1],
            "maximum_train_loss_window_regression_from_best_prior": maximum_loss_regression,
            "final_block_parameter_l2": list(final_block_parameter_l2),
            "final_max_block_parameter_l2_to_median": block_ratio,
        },
    }


def _validate_stability_evidence(
    suite: Mapping[str, Any],
    point: Mapping[str, Any],
    *,
    artifact_root: Path,
    result: Mapping[str, Any],
    metrics: Mapping[str, Any],
    validation_loss: float,
    fresh10_result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate every emitted curve row and apply the frozen v4 gate.

    Structural corruption is an error.  A complete, finite run that crossed a
    frozen instability threshold remains valid raw evidence but is
    explicitly rejected from LR selection and all downstream fitting.
    """

    policy = suite.get("stability_admission")
    if not isinstance(policy, Mapping):
        return None
    name = str(point["id"])
    expected_artifacts = {
        "training_curve": "training.csv",
        "validation_curve": "validation.csv",
        "diagnostics": "diagnostics.csv",
    }
    if dict(_mapping(result.get("artifacts"), f"{name}.artifacts")) != expected_artifacts:
        raise ScalingError(
            f"{name}: stability admission requires all three canonical curve artifacts"
        )
    paths = {
        key: _stability_artifact_path(artifact_root, name, filename)
        for key, filename in expected_artifacts.items()
    }
    steps = int(point["steps"])
    tokens_per_step = int(suite["batch_size"]) * int(suite["sequence_length"])
    flops_per_step = int(point["flops_per_token"]) * tokens_per_step

    training_rows = _csv_rows(
        paths["training_curve"], _TRAINING_CSV_FIELDS, f"{name}.training.csv"
    )
    if len(training_rows) != steps:
        raise ScalingError(
            f"{name}: training.csv must contain exactly one row per optimizer step"
        )
    gradients: list[float] = []
    training_losses: list[float] = []
    for expected_step, row in enumerate(training_rows, 1):
        label = f"{name}.training.csv:{expected_step + 1}"
        step = _csv_int(row["step"], f"{label}.step", minimum=1)
        tokens = _csv_int(row["tokens_processed"], f"{label}.tokens", minimum=1)
        flops = _csv_int(
            row["cumulative_estimated_flops"], f"{label}.flops", minimum=1
        )
        if (
            step != expected_step
            or tokens != expected_step * tokens_per_step
            or flops != expected_step * flops_per_step
        ):
            raise ScalingError(
                f"{label} step/token/FLOP coordinates differ from the suite"
            )
        training_losses.append(
            _csv_float(row["train_loss"], f"{label}.train_loss", minimum=0.0)
        )
        observed_lr = _csv_float(
            row["learning_rate"], f"{label}.learning_rate", minimum=0.0
        )
        expected_lr = _expected_learning_rate(expected_step, point, suite)
        if not math.isclose(
            observed_lr, expected_lr, rel_tol=2e-6, abs_tol=1e-12
        ):
            raise ScalingError(
                f"{label}.learning_rate differs from the configured float32 "
                "schedule within the pinned tolerance"
            )
        if observed_lr < 0.0 or observed_lr > float(point["learning_rate"]) * (1.0 + 2e-6):
            raise ScalingError(f"{label}.learning_rate is outside the configured range")
        gradients.append(
            _csv_float(row["grad_norm"], f"{label}.grad_norm", minimum=0.0)
        )
    if not math.isclose(
        training_losses[-1],
        _finite(metrics.get("train_loss"), f"{name}.metrics.train_loss"),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ScalingError(f"{name}: training.csv final loss differs from result.json")

    validation_rows = _csv_rows(
        paths["validation_curve"],
        _VALIDATION_CSV_FIELDS,
        f"{name}.validation.csv",
    )
    probe_steps = list(range(int(point["val_every"]), steps, int(point["val_every"])))
    if len(validation_rows) != len(probe_steps) + 12:
        raise ScalingError(f"{name}: validation.csv row count differs from the schedule")
    probe_losses: list[float] = []
    for index, expected_step in enumerate(probe_steps):
        row = validation_rows[index]
        label = f"{name}.validation.csv:{index + 2}"
        loss = _csv_float(
            row["validation_loss"], f"{label}.validation_loss", minimum=0.0
        )
        perplexity = _csv_float(
            row["perplexity"], f"{label}.perplexity", minimum=0.0
        )
        if (
            _csv_int(row["step"], f"{label}.step", minimum=1) != expected_step
            or _csv_int(row["tokens_processed"], f"{label}.tokens", minimum=1)
            != expected_step * tokens_per_step
            or row["kind"] != "fineweb_probe"
            or row["domain"] != "fineweb"
            or _csv_int(
                row["validation_tokens"], f"{label}.validation_tokens", minimum=1
            )
            != 8 * tokens_per_step
            or row["canonical"] != "false"
            or not math.isclose(math.log(perplexity), loss, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ScalingError(f"{label} differs from the fixed-prefix probe contract")
        _csv_float(
            row["validation_seconds"], f"{label}.validation_seconds", minimum=0.0
        )
        probe_losses.append(loss)
    consecutive_probes = int(policy["probe_regression_consecutive"])
    if len(probe_losses) <= consecutive_probes:
        raise ScalingError(
            f"{name}: too few fixed-prefix probes for stability admission"
        )

    final_rows = validation_rows[len(probe_steps) :]

    def validate_final_row(
        row: Mapping[str, str],
        *,
        offset: int,
        kind: str,
        domain: str,
        validation_tokens: int,
        expected_loss: float,
        canonical: str,
    ) -> None:
        label = f"{name}.validation.csv:{len(probe_steps) + offset + 2}"
        loss = _csv_float(
            row["validation_loss"], f"{label}.validation_loss", minimum=0.0
        )
        perplexity = _csv_float(
            row["perplexity"], f"{label}.perplexity", minimum=0.0
        )
        if (
            _csv_int(row["step"], f"{label}.step", minimum=1) != steps
            or _csv_int(row["tokens_processed"], f"{label}.tokens", minimum=1)
            != steps * tokens_per_step
            or row["kind"] != kind
            or row["domain"] != domain
            or _csv_int(
                row["validation_tokens"], f"{label}.validation_tokens", minimum=1
            )
            != validation_tokens
            or row["canonical"] != canonical
            or not math.isclose(loss, expected_loss, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(math.log(perplexity), loss, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ScalingError(f"{label} differs from the final evaluation contract")
        _csv_float(
            row["validation_seconds"], f"{label}.validation_seconds", minimum=0.0
        )

    validate_final_row(
        final_rows[0],
        offset=0,
        kind="fineweb",
        domain="fineweb",
        validation_tokens=int(suite["validation_tokens"]),
        expected_loss=validation_loss,
        canonical="true",
    )
    domain_losses = _mapping(
        fresh10_result["domain_losses"], f"{name}.fresh10_domain_losses"
    )
    for offset, domain in enumerate(FRESH10_DOMAINS, 1):
        validate_final_row(
            final_rows[offset],
            offset=offset,
            kind="downstream",
            domain=domain,
            validation_tokens=int(suite["fresh10"]["scored_tokens_per_domain"]),
            expected_loss=float(domain_losses[domain]),
            canonical="false",
        )
    validate_final_row(
        final_rows[-1],
        offset=11,
        kind="downstream_macro",
        domain="fresh10_macro",
        validation_tokens=int(suite["fresh10"]["scored_tokens"]),
        expected_loss=float(fresh10_result["macro_loss"]),
        canonical="false",
    )
    if metrics.get("validation_probe_count") != len(probe_steps):
        raise ScalingError(f"{name}: result validation probe count differs from CSV")

    diagnostics_rows = _csv_rows(
        paths["diagnostics"],
        _DIAGNOSTICS_CSV_FIELDS,
        f"{name}.diagnostics.csv",
    )
    diagnostic_steps = sorted(
        {1, steps, *range(int(point["diagnostics_every"]), steps + 1, int(point["diagnostics_every"]))}
    )
    scopes = (
        (("overall", None), ("embeddings", None))
        + tuple(("block", layer) for layer in range(int(point["layers"])))
        + (("final_norm", None),)
    )
    expected_grid = {
        (scope, layer, family, statistic)
        for scope, layer in scopes
        for family in _DIAGNOSTIC_FAMILIES
        for statistic in _DIAGNOSTIC_STATS
    }
    expected_row_count = len(diagnostic_steps) * len(expected_grid)
    if len(diagnostics_rows) != expected_row_count:
        raise ScalingError(f"{name}: diagnostics.csv row count differs from schedule")
    identities_by_step: dict[int, set[tuple[str, int | None, str, str]]] = {}
    element_counts: dict[tuple[str, int | None], int] = {}
    overall_parameter_l2: dict[int, float] = {}
    overall_gradient_l2: dict[int, float] = {}
    final_block_parameter_l2: dict[int, float] = {}
    allowed_steps = set(diagnostic_steps)
    allowed_scopes = set(scopes)
    for index, row in enumerate(diagnostics_rows, 2):
        label = f"{name}.diagnostics.csv:{index}"
        step = _csv_int(row["step"], f"{label}.step", minimum=1)
        if step not in allowed_steps:
            raise ScalingError(f"{label}.step is outside the diagnostic schedule")
        if (
            _csv_int(row["tokens_processed"], f"{label}.tokens", minimum=1)
            != step * tokens_per_step
            or _csv_int(
                row["cumulative_estimated_flops"], f"{label}.flops", minimum=1
            )
            != step * flops_per_step
        ):
            raise ScalingError(f"{label} token/FLOP coordinates differ from training")
        scope = row["scope"]
        layer_text = row["layer"]
        layer = (
            _csv_int(layer_text, f"{label}.layer")
            if scope == "block"
            else None
        )
        if (scope != "block" and layer_text) or (scope, layer) not in allowed_scopes:
            raise ScalingError(f"{label} scope/layer is invalid")
        family = row["family"]
        statistic = row["stat"]
        identity = (scope, layer, family, statistic)
        if family not in _DIAGNOSTIC_FAMILIES or statistic not in _DIAGNOSTIC_STATS:
            raise ScalingError(f"{label} family/stat is invalid")
        step_identities = identities_by_step.setdefault(step, set())
        if identity in step_identities:
            raise ScalingError(f"{label} duplicates a diagnostic identity")
        step_identities.add(identity)
        count = _csv_int(row["element_count"], f"{label}.element_count", minimum=1)
        prior_count = element_counts.setdefault((scope, layer), count)
        if prior_count != count:
            raise ScalingError(f"{label} element count changes across diagnostic steps")
        value = _csv_float(row["value"], f"{label}.value")
        if statistic in {"l1_norm", "l2_norm", "std", "fourth_moment"} and value < 0.0:
            raise ScalingError(f"{label} non-negative diagnostic is negative")
        if identity == ("overall", None, "param", "l2_norm"):
            overall_parameter_l2[step] = value
        if identity == ("overall", None, "grad", "l2_norm"):
            overall_gradient_l2[step] = value
        if (
            step == steps
            and scope == "block"
            and family == "param"
            and statistic == "l2_norm"
            and layer is not None
        ):
            final_block_parameter_l2[layer] = value
    if set(identities_by_step) != allowed_steps or any(
        identities != expected_grid for identities in identities_by_step.values()
    ):
        raise ScalingError(f"{name}: diagnostics.csv has an incomplete diagnostic grid")
    expected_counts = {
        ("overall", None): int(point["parameters"]),
        ("embeddings", None): (
            int(suite["vocab_size"]) + int(suite["sequence_length"])
        )
        * int(point["d_model"]),
        ("final_norm", None): 2 * int(point["d_model"]),
        **{
            ("block", layer): 12 * int(point["d_model"]) ** 2
            + 13 * int(point["d_model"])
            for layer in range(int(point["layers"]))
        },
    }
    if element_counts != expected_counts:
        raise ScalingError(f"{name}: diagnostic scope element counts differ from model")
    for step in diagnostic_steps:
        if overall_gradient_l2[step] != gradients[step - 1]:
            raise ScalingError(
                f"{name}: diagnostics gradient norm differs from training.csv at step {step}"
            )
    if metrics.get("diagnostic_point_count") != len(diagnostic_steps):
        raise ScalingError(f"{name}: result diagnostic point count differs from CSV")

    block_l2_values = [
        final_block_parameter_l2[layer] for layer in sorted(final_block_parameter_l2)
    ]
    if len(block_l2_values) != int(point["layers"]):
        raise ScalingError(f"{name}: final block parameter L2 diagnostics are incomplete")
    decision = _classify_stability_series(
        policy,
        probe_steps=probe_steps,
        probe_losses=probe_losses,
        gradients=gradients,
        training_losses=training_losses,
        final_block_parameter_l2=block_l2_values,
    )
    classification = str(decision["classification"])
    reasons = list(decision["reasons"])
    witnesses = dict(decision["witnesses"])
    payload = {
        "schema_version": 1,
        "method": "full_curve_temporal_stability_v1",
        "suite_id": suite["suite_id"],
        "suite_sha256": suite["suite_sha256"],
        "execution_fingerprint": suite["execution_fingerprint"],
        "point": _public_point(point),
        "config_sha256": point["config_sha256"],
        "policy_sha256": _canonical_mapping_sha256(dict(policy)),
        "validation_tolerances": {
            "learning_rate_relative": 2e-6,
            "learning_rate_absolute": 1e-12,
            "inclusive_threshold_absolute": 1e-12,
        },
        "classification": classification,
        "admitted": classification == "stable",
        "reasons": reasons,
        "witnesses": witnesses,
        "policy": dict(policy),
        "evidence": {
            key: {
                "path": f"{name}/artifacts/{expected_artifacts[key]}",
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(
                    training_rows
                    if key == "training_curve"
                    else validation_rows
                    if key == "validation_curve"
                    else diagnostics_rows
                ),
                "header": list(
                    _TRAINING_CSV_FIELDS
                    if key == "training_curve"
                    else _VALIDATION_CSV_FIELDS
                    if key == "validation_curve"
                    else _DIAGNOSTICS_CSV_FIELDS
                ),
            }
            for key, path in paths.items()
        },
        "counts": {
            "training_steps": len(training_rows),
            "fixed_prefix_probes": len(probe_losses),
            "diagnostic_points": len(diagnostic_steps),
            "diagnostic_rows": len(diagnostics_rows),
        },
        "summaries": {
            **dict(decision["summaries"]),
            "learning_rate_first": float(training_rows[0]["learning_rate"]),
            "learning_rate_peak": max(
                float(row["learning_rate"]) for row in training_rows
            ),
            "learning_rate_final": float(training_rows[-1]["learning_rate"]),
            "initial_parameter_l2": overall_parameter_l2[1],
            "final_parameter_l2": overall_parameter_l2[steps],
        },
    }
    admission_path = artifact_root / name / "artifacts" / "stability-admission.json"
    _write_immutable_bytes(
        admission_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    payload["artifact"] = {
        "path": f"{name}/artifacts/stability-admission.json",
        "sha256": _sha256(admission_path),
        "bytes": admission_path.stat().st_size,
    }
    return payload


def _require_stable_measurement(measurement: Mapping[str, Any]) -> None:
    stability = measurement.get("stability_admission")
    if isinstance(stability, Mapping) and stability.get("classification") != "stable":
        reasons = stability.get("reasons")
        detail = ", ".join(str(item) for item in reasons) if isinstance(reasons, list) else "unknown"
        raise ScalingError(
            f"{measurement.get('id')}: rejected by stability admission ({detail})"
        )


def _read_run(
    suite: Mapping[str, Any], point: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    point = _resolve_point_learning_rate(suite, point, root)
    name = str(point["id"])
    lineage_entry = _lineage_entry_for_point(suite, point)
    artifact_root = root
    lineage_public: dict[str, Any] | None = None
    if lineage_entry is None:
        run_manifest, run_manifest_path = _load_run_manifest(suite, point, root)
    else:
        if (root / name).exists() or (root / name).is_symlink():
            raise ScalingError(
                f"{name}: local artifacts must not shadow an immutable lineage input"
            )
        run_manifest, run_manifest_path, artifact_root = _load_lineage_run_manifest(
            suite, point, lineage_entry
        )
        lineage = _mapping(suite["lineage"], "suite lineage")
        origin = _mapping(lineage["origin"], "lineage origin")
        lineage_public = {
            "kind": "exact_immutable_prior_study_artifact",
            "lineage_id": lineage["lineage_id"],
            "lineage_manifest_sha256": lineage["manifest_sha256"],
            "origin_suite_id": origin["suite_id"],
            "origin_suite_sha256": origin["suite_sha256"],
            "origin_execution_fingerprint": origin["execution_fingerprint"],
            "point_id": name,
            "run_manifest_sha256": lineage_entry["run_manifest_sha256"],
            "result_sha256": lineage_entry["result_sha256"],
        }
    result_path = artifact_root / name / "artifacts" / "result.json"
    result, result_sha256 = _read_regular_json_and_sha256(
        result_path, f"{name}.result"
    )
    if (
        lineage_entry is not None
        and result_sha256 != lineage_entry["result_sha256"]
    ):
        raise ScalingError(f"{name}: lineage result is not the exact allowlisted bytes")
    metrics = _mapping(result.get("metrics"), f"{name}.metrics")
    if (
        result.get("schema_version") != 1
        or result.get("status") != "ok"
        or result.get("seed") != suite["seed"]
    ):
        raise ScalingError(f"{name}: result schema/status/seed differs from suite")
    result_system = dict(_mapping(result.get("system"), f"{name}.system"))
    if result_system.pop("controller_process_index", None) != 0:
        raise ScalingError(f"{name}: result must be written by process zero")
    validated_system = _validated_runtime_provenance(
        result_system, suite, label=f"{name}.system"
    )
    if validated_system != run_manifest["runtime"]:
        raise ScalingError(f"{name}: result runtime differs from prelaunch runtime")
    contract = _mapping(result.get("contract"), f"{name}.contract")
    implementation = _mapping(result.get("implementation"), f"{name}.implementation")
    configuration = _mapping(
        implementation.get("configuration"), f"{name}.configuration"
    )
    resolved = _mapping(configuration.get("resolved"), f"{name}.resolved")
    resolved_training = _mapping(
        resolved.get("training"), f"{name}.resolved.training"
    )
    resolved_model = _mapping(resolved.get("model"), f"{name}.resolved.model")
    resolved_kernels = _mapping(
        resolved.get("kernels"), f"{name}.resolved.kernels"
    )
    resolved_optimizer = _mapping(
        resolved.get("optimizer"), f"{name}.resolved.optimizer"
    )
    resolved_evaluation = _mapping(
        resolved.get("evaluation"), f"{name}.resolved.evaluation"
    )
    resolved_logging = _mapping(
        resolved.get("logging"), f"{name}.resolved.logging"
    )
    expected = {
        "tokens_processed": int(point["train_tokens"]),
        "training_token_budget": int(point["train_tokens"]),
        "training_steps": int(point["steps"]),
        "training_usable_tokens_per_epoch": int(
            run_manifest["dataset"]["usable_train_tokens"]
        ),
        "parameters": int(point["parameters"]),
        "flops_per_token": int(point["flops_per_token"]),
        "estimated_total_flops": int(point["total_flops"]),
        "training_sampling": "shuffled_epochs",
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ScalingError(
                f"{name}: metrics.{key} does not match suite ({metrics.get(key)!r})"
            )
    epochs = _finite(metrics.get("training_data_epochs"), f"{name}.training_data_epochs")
    expected_epochs = int(point["train_tokens"]) / int(
        run_manifest["dataset"]["usable_train_tokens"]
    )
    if (
        epochs > 1.0 + 1e-12
        or not math.isclose(epochs, expected_epochs, rel_tol=1e-15, abs_tol=1e-15)
    ):
        raise ScalingError(f"{name}: exact no-replacement epoch ratio differs")
    if result.get("profile") != "dev" or result.get("track") != "open":
        raise ScalingError(f"{name}: sweep runs must be open/dev diagnostics")
    if result.get("checkpoint") is not None:
        raise ScalingError(f"{name}: diagnostic sweep unexpectedly retained a checkpoint")
    if configuration.get("sha256") != point["config_sha256"]:
        raise ScalingError(f"{name}: result config hash differs from generated config")
    if (
        configuration.get("schema_version") != 1
        or configuration.get("path") != "config.yaml"
        or configuration.get("profile") != "dev"
        or configuration.get("overrides") != {}
    ):
        raise ScalingError(f"{name}: result configuration provenance differs")
    expected_model = {
        "layers": int(point["layers"]),
        "heads": int(point["heads"]),
        "d_model": int(point["d_model"]),
        "mlp_mult": 4,
        "vocab_size": int(suite["vocab_size"]),
        "semantic_vocab_size": int(suite["vocab_size"]),
        "tied_embeddings": True,
    }
    if dict(contract) != {
        "model_id": "reference-gpt-v1",
        "dataset_id": suite["dataset"]["id"],
        "tokenizer_id": "gpt2",
        "sequence_length": int(suite["sequence_length"]),
        "model": expected_model,
    }:
        raise ScalingError(f"{name}: result model/data/tokenizer contract differs")
    expected_training = {
        "steps": int(point["steps"]),
        "train_tokens": int(point["train_tokens"]),
        "batch_size": int(suite["batch_size"]),
        "seq_len": int(suite["sequence_length"]),
        "sampling": "shuffled_epochs",
        "dtype": "bfloat16",
    }
    if dict(resolved_training) != expected_training:
        raise ScalingError(f"{name}: resolved training/no-replacement contract differs")
    expected_kernels = {
        "attention_backend": "tpu_flash",
        "loss_backend": "dense",
        "vocab_tile_size": 2_048,
    }
    expected_optimizer = {
        "learning_rate": float(point["learning_rate"]),
        "min_lr_ratio": float(suite["optimizer"]["min_lr_ratio"]),
        "warmup_steps": int(point["warmup_steps"]),
        "weight_decay": float(suite["optimizer"]["weight_decay"]),
        "beta1": float(suite["optimizer"]["beta1"]),
        "beta2": float(suite["optimizer"]["beta2"]),
        "grad_clip": float(suite["optimizer"]["grad_clip"]),
    }
    expected_evaluation = {
        "eval_batches": int(suite["validation_batches"]),
        "val_every": int(point["val_every"]),
        "val_probe_batches": 8,
    }
    expected_logging = {
        "diagnostics_every": int(point["diagnostics_every"]),
        "log_every": int(point["log_every"]),
    }
    if dict(resolved_model) != expected_model:
        raise ScalingError(f"{name}: resolved model contract differs from suite")
    if dict(resolved_kernels) != expected_kernels:
        raise ScalingError(f"{name}: resolved kernel contract differs from suite")
    if (
        implementation.get("attention_backend") != "tpu_flash"
        or implementation.get("loss_backend") != "dense"
        or implementation.get("vocab_tile_size") != 2_048
    ):
        raise ScalingError(f"{name}: reported kernel implementation differs")
    if set(resolved_optimizer) != set(expected_optimizer) or any(
        not math.isclose(
            _finite(resolved_optimizer[key], f"{name}.optimizer.{key}"),
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for key, expected in expected_optimizer.items()
    ):
        raise ScalingError(f"{name}: resolved optimizer contract differs from suite")
    if dict(resolved_evaluation) != expected_evaluation:
        raise ScalingError(f"{name}: resolved evaluation contract differs from suite")
    if dict(resolved_logging) != expected_logging:
        raise ScalingError(f"{name}: resolved logging contract differs from suite")
    validation_tokens = _integer(
        metrics.get("validation_tokens"), f"{name}.validation_tokens"
    )
    if validation_tokens != int(suite["validation_tokens"]):
        raise ScalingError(f"{name}: validation token count differs from suite")
    evaluations = _mapping(result.get("evaluations"), f"{name}.evaluations")
    if set(evaluations) != {"fineweb", "fresh10"}:
        raise ScalingError(f"{name}: FineWeb and Fresh10 evaluations are mandatory")
    fineweb = _mapping(evaluations["fineweb"], f"{name}.evaluations.fineweb")
    validation_loss = _finite(metrics.get("validation_loss"), f"{name}.validation_loss")
    if (
        fineweb.get("scored_tokens") != int(suite["validation_tokens"])
        or fineweb.get("canonical") is not True
        or not math.isclose(
            _finite(fineweb.get("loss"), f"{name}.fineweb.loss"),
            validation_loss,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ScalingError(f"{name}: FineWeb evaluation differs from canonical metric")
    fresh10_result = _validated_fresh10_result(
        evaluations["fresh10"], suite, label=f"{name}.evaluations.fresh10"
    )
    stability = _validate_stability_evidence(
        suite,
        point,
        artifact_root=artifact_root,
        result=result,
        metrics=metrics,
        validation_loss=validation_loss,
        fresh10_result=fresh10_result,
    )
    measurement = {
        "id": name,
        "slice": point["slice"],
        "role": point["role"],
        "parameters": int(point["parameters"]),
        "train_tokens": int(point["train_tokens"]),
        "total_flops": int(point["total_flops"]),
        "tokens_per_parameter": float(point["tokens_per_parameter"]),
        "learning_rate": float(point["learning_rate"]),
        "validation_loss": validation_loss,
        "fresh10_macro_loss": fresh10_result["macro_loss"],
        "fresh10_domain_losses": fresh10_result["domain_losses"],
        "fresh10_scored_tokens": fresh10_result["scored_tokens"],
        "train_seconds": _finite(
            metrics.get("train_seconds"), f"{name}.train_seconds", positive=True
        ),
        "training_data_epochs": epochs,
        "dataset_manifest_sha256": run_manifest["dataset"][
            "manifest_canonical_sha256"
        ],
        "run_manifest": f"{name}/run-manifest.json",
        "run_manifest_sha256": _sha256(run_manifest_path),
        "result": f"{name}/artifacts/result.json",
        "result_sha256": result_sha256,
        "_dataset_provenance": run_manifest["dataset"],
        "_fresh10_provenance": run_manifest["fresh10"],
        "_runtime_provenance": run_manifest["runtime"],
    }
    if stability is not None:
        measurement["stability_admission"] = stability
        measurement["stability_admission_path"] = stability["artifact"]["path"]
        measurement["stability_admission_sha256"] = stability["artifact"]["sha256"]
    if lineage_public is not None:
        measurement["lineage"] = lineage_public
    return measurement


def select_learning_rate(
    suite: Mapping[str, Any],
    *,
    shape_id: str,
    runs_path: Path,
    slice_id: str | None = None,
) -> dict[str, Any]:
    """Select a stable, bracketed LR for one slice/shape calibration."""

    root = runs_path.expanduser().resolve()
    _validate_lineage_output_root(suite, root, label="learning-rate selection root")
    calibration_slice = _calibration_slice_id(suite, slice_id)
    group_label = (
        f"{calibration_slice}/{shape_id}"
        if _per_slice_learning_rates(suite)
        else shape_id
    )
    _validate_adaptive_completion_prefix(
        suite, shape_id, root, slice_id=slice_id
    )
    initial_candidates = [
        point
        for point in suite["calibrations"] + suite["extension_calibrations"]
        if point["shape_id"] == shape_id and point["slice"] == calibration_slice
    ]
    if len(initial_candidates) != len(suite["learning_rate_candidates"]):
        raise ScalingError(f"{group_label}: incomplete learning-rate candidate definition")
    adaptive_candidates = [
        point
        for point in suite["adaptive_calibrations"]
        if point["shape_id"] == shape_id and point["slice"] == calibration_slice
    ]
    candidates = sorted(
        initial_candidates + adaptive_candidates,
        key=lambda point: float(point["learning_rate"]),
    )
    completed = [
        point
        for point in candidates
        if _point_has_result(suite, point, root)
    ]
    initial_completed = [point for point in initial_candidates if point in completed]
    if not _per_slice_learning_rates(suite) and len(initial_completed) != len(initial_candidates):
        raise ScalingError(f"{group_label}: initial learning-rate grid is incomplete")
    if _per_slice_learning_rates(suite) and len(completed) < 2:
        raise ScalingError(
            f"{group_label}: at least two contiguous learning-rate trials are required"
        )
    measured = [_read_run(suite, point, root) for point in completed]
    dataset_identity = _coherent_dataset_identity(measured)
    fresh10_identity = _coherent_fresh10_identity(measured)
    runtime_identity = _coherent_auxiliary_identity(
        measured, "_runtime_provenance", "runtime identity"
    )
    stable = [
        item
        for item in measured
        if not isinstance(item.get("stability_admission"), Mapping)
        or item["stability_admission"].get("classification") == "stable"
    ]
    if not stable:
        raise ScalingError(f"{group_label}: no stable learning-rate candidates")
    selected = min(
        stable,
        key=lambda item: (float(item["validation_loss"]), float(item["learning_rate"])),
    )
    selected_index = measured.index(selected)
    lower_neighbor = measured[selected_index - 1] if selected_index else None
    lower_stability = (
        lower_neighbor.get("stability_admission")
        if isinstance(lower_neighbor, Mapping)
        else None
    )
    lower_is_stable = lower_neighbor is not None and (
        not isinstance(lower_stability, Mapping)
        or lower_stability.get("classification") == "stable"
    )
    if not lower_is_stable:
        raise LearningRateEdgeError(
            shape_id,
            "lower",
            float(selected["learning_rate"]),
            slice_id=(calibration_slice if _per_slice_learning_rates(suite) else None),
        )
    if selected_index == len(measured) - 1:
        raise LearningRateEdgeError(
            shape_id,
            "upper",
            float(selected["learning_rate"]),
            slice_id=(calibration_slice if _per_slice_learning_rates(suite) else None),
        )
    upper_neighbor = measured[selected_index + 1]
    upper_stability = upper_neighbor.get("stability_admission")
    upper_is_stable = (
        not isinstance(upper_stability, Mapping)
        or upper_stability.get("classification") == "stable"
    )
    if (
        upper_is_stable
        and float(upper_neighbor["validation_loss"])
        < float(selected["validation_loss"])
    ):
        raise ScalingError(
            f"{group_label}: selected LR is not the lowest-loss stable candidate"
        )
    payload = {
        "schema_version": (
            4
            if _per_slice_learning_rates(suite)
            else 3
            if _lineage_summary(suite) is not None
            else 2
        ),
        "suite_id": suite["suite_id"],
        "suite_sha256": suite["suite_sha256"],
        "execution_fingerprint": suite["execution_fingerprint"],
        "dataset_manifest_sha256": dataset_identity[
            "manifest_canonical_sha256"
        ],
        "fresh10_manifest_sha256": fresh10_identity[
            "manifest_canonical_sha256"
        ],
        "runtime": runtime_identity,
        "slice": calibration_slice,
        "shape_id": shape_id,
        "criterion": (
            "lowest canonical validation loss among stable candidates, with a "
            "stable lower-LR neighbor and an adjacent completed upper boundary"
        ),
        "edge_policy": (
            "never cross the first suspect/rejected high-LR frontier; an adjacent "
            "ineligible trial closes the upper bracket"
        ),
        "seed_count": 1,
        "study_lineage": _lineage_summary(suite),
        "candidates": [
            {
                **{
                    key: item[key]
                    for key in (
                        "id",
                        "learning_rate",
                        "validation_loss",
                        "result",
                        "result_sha256",
                        "run_manifest_sha256",
                        "stability_admission_path",
                        "stability_admission_sha256",
                    )
                    if key in item
                },
                **(
                    {"stability_admission": item["stability_admission"]}
                    if "stability_admission" in item
                    else {}
                ),
                **({"lineage": item["lineage"]} if "lineage" in item else {}),
            }
            for item in measured
        ],
        "selected_point_id": selected["id"],
        "selected_learning_rate": selected["learning_rate"],
        "selected_validation_loss": selected["validation_loss"],
        "lower_boundary_point_id": lower_neighbor["id"],
        "upper_boundary_point_id": upper_neighbor["id"],
        "upper_boundary_classification": (
            upper_stability.get("classification")
            if isinstance(upper_stability, Mapping)
            else "legacy_unclassified"
        ),
    }
    selection_path = _learning_rate_selection_path(
        root,
        shape_id,
        slice_id=(calibration_slice if _per_slice_learning_rates(suite) else None),
    )
    if selection_path.is_symlink():
        raise ScalingError(
            f"learning-rate selection must not be a symlink: {selection_path}"
        )
    _write_immutable_bytes(
        selection_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def _selected_calibration_run(
    suite: Mapping[str, Any],
    shape_id: str,
    runs_root: Path,
    *,
    slice_id: str | None = None,
) -> dict[str, Any]:
    # Recompute the selection from all completed immutable calibration runs.
    # `_write_immutable_bytes` makes any edited/stale selection fail closed.
    selection = select_learning_rate(
        suite, shape_id=shape_id, slice_id=slice_id, runs_path=runs_root
    )
    selected_id = selection.get("selected_point_id")
    candidate = next(
        (
            point
            for point in (
                suite["calibrations"]
                + suite["extension_calibrations"]
                + suite["adaptive_calibrations"]
            )
            if point["id"] == selected_id
            and point["shape_id"] == shape_id
            and point["slice"] == _calibration_slice_id(suite, slice_id)
        ),
        None,
    )
    if candidate is None:
        raise ScalingError(f"{shape_id}: selected calibration point is not in the suite")
    measurement = _read_run(suite, candidate, runs_root)
    _require_stable_measurement(measurement)
    return measurement


def _coherent_dataset_identity(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not measurements:
        raise ScalingError("cannot establish dataset identity without measurements")
    provenances = [
        _mapping(item.get("_dataset_provenance"), "measurement dataset provenance")
        for item in measurements
    ]

    def content_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "name": provenance["name"],
            "manifest_canonical_sha256": provenance[
                "manifest_canonical_sha256"
            ],
            "usable_train_tokens": provenance["usable_train_tokens"],
            "usable_validation_tokens": provenance["usable_validation_tokens"],
            "production": provenance["production"],
            "selected_shards": provenance["shards"],
        }

    identity = content_identity(provenances[0])
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    for provenance in provenances[1:]:
        candidate = json.dumps(
            content_identity(provenance), sort_keys=True, separators=(",", ":")
        )
        if candidate != canonical:
            raise ScalingError(
                "completed sweep measurements do not share one canonical dataset"
            )
    identity["manifest_raw_sha256s"] = sorted(
        {str(item["manifest_raw_sha256"]) for item in provenances}
    )
    identity["observed_roots"] = sorted({str(item["root"]) for item in provenances})
    return identity


def _coherent_auxiliary_identity(
    measurements: Sequence[Mapping[str, Any]], key: str, label: str
) -> dict[str, Any]:
    if not measurements:
        raise ScalingError(f"cannot establish {label} identity without measurements")
    values = [dict(_mapping(item.get(key), label)) for item in measurements]
    canonical = json.dumps(values[0], sort_keys=True, separators=(",", ":"))
    for value in values[1:]:
        if json.dumps(value, sort_keys=True, separators=(",", ":")) != canonical:
            raise ScalingError(f"completed measurements do not share one {label}")
    return values[0]


def _coherent_fresh10_identity(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare Fresh10 content while treating absolute locations as observations."""

    if not measurements:
        raise ScalingError("cannot establish Fresh10 identity without measurements")
    provenances = [
        dict(
            _mapping(
                item.get("_fresh10_provenance"), "measurement Fresh10 provenance"
            )
        )
        for item in measurements
    ]

    def content_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in provenance.items()
            if key not in {"root", "manifest_path"}
        }

    identity = content_identity(provenances[0])
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    for provenance in provenances[1:]:
        candidate = json.dumps(
            content_identity(provenance), sort_keys=True, separators=(",", ":")
        )
        if candidate != canonical:
            raise ScalingError(
                "completed measurements do not share one Fresh10 content identity"
            )
    identity["observed_roots"] = sorted(
        {str(provenance["root"]) for provenance in provenances}
    )
    identity["observed_manifest_paths"] = sorted(
        {str(provenance["manifest_path"]) for provenance in provenances}
    )
    return identity


def _validate_prelaunch_lineage_coherence(
    suite: Mapping[str, Any],
    *,
    runs_root: Path,
    dataset: Mapping[str, Any],
    fresh10: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    """Prove current inputs match historical measurements before TPU launch."""

    lineage = suite.get("lineage")
    if not isinstance(lineage, Mapping):
        return
    entries = lineage["artifacts"]
    if not entries:
        raise ScalingError("lineage study has no historical inputs to compare")
    first_id = str(entries[0]["point_id"])
    point = next(
        item for item in suite["all_variants"] if str(item["id"]) == first_id
    )
    historical = _read_run(suite, point, runs_root)
    current = {
        "_dataset_provenance": _validated_dataset_provenance(
            dataset, suite, label="prelaunch lineage dataset"
        ),
        "_fresh10_provenance": _validated_fresh10_provenance(
            fresh10, suite, label="prelaunch lineage Fresh10"
        ),
        "_runtime_provenance": _validated_runtime_provenance(
            runtime, suite, label="prelaunch lineage runtime"
        ),
    }
    _coherent_dataset_identity((historical, current))
    _coherent_fresh10_identity((historical, current))
    _coherent_auxiliary_identity(
        (historical, current), "_runtime_provenance", "runtime identity"
    )


def _public_measurement(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _fit_slice(
    points: Sequence[Mapping[str, Any]],
    *,
    slice_id: str,
    target_total_flops: int,
    random_seed: int = 20_260_813,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    for point in points:
        _require_stable_measurement(point)
    ordered = sorted(points, key=lambda item: int(item["parameters"]))
    if len(ordered) < 5:
        raise ScalingError(f"{slice_id}: at least five measured points are required")
    parameters = np.asarray([item["parameters"] for item in ordered], dtype=np.float64)
    tokens = np.asarray([item["train_tokens"] for item in ordered], dtype=np.float64)
    losses = np.asarray([item["validation_loss"] for item in ordered], dtype=np.float64)
    anchor = float(parameters[len(parameters) // 2])
    x = np.log(parameters / anchor)
    design = np.column_stack((x * x, x, np.ones_like(x)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, losses, rcond=None)
    if rank != 3:
        raise ScalingError(f"{slice_id}: model-size grid is rank deficient")
    curvature, slope, intercept = (float(value) for value in coefficients)
    fitted = design @ coefficients
    residuals = losses - fitted
    observed_index = int(np.argmin(losses))
    observed_edge = observed_index in (0, len(ordered) - 1)
    optimum_x = -slope / (2.0 * curvature) if curvature > 0.0 else math.nan
    fitted_interior = curvature > 0.0 and float(x.min()) < optimum_x < float(x.max())
    bracketed = fitted_interior and not observed_edge
    reasons: list[str] = []
    if curvature <= 0.0:
        reasons.append("quadratic curvature is non-positive")
    elif not fitted_interior:
        reasons.append("quadratic minimum lies outside the measured range")
    if observed_edge:
        reasons.append("lowest observed loss is at a grid endpoint")

    optimum: dict[str, float] | None = None
    confidence: dict[str, float] | None = None
    samples_payload: dict[str, np.ndarray] | None = None
    token_slope, token_intercept = np.polyfit(x, np.log(tokens), 1)
    if bracketed:
        optimum_parameters = anchor * math.exp(optimum_x)
        optimum_tokens = math.exp(float(token_intercept + token_slope * optimum_x))
        optimum = {
            "parameters": optimum_parameters,
            "train_tokens": optimum_tokens,
            "tokens_per_parameter": optimum_tokens / optimum_parameters,
            "validation_loss": curvature * optimum_x**2 + slope * optimum_x + intercept,
        }
        degrees_freedom = len(ordered) - 3
        variance = float(np.dot(residuals, residuals) / degrees_freedom)
        covariance = variance * np.linalg.inv(design.T @ design)
        rng = np.random.default_rng(random_seed)
        coefficient_samples = rng.multivariate_normal(
            coefficients, covariance, size=40_000
        )
        sampled_x = -coefficient_samples[:, 1] / (2.0 * coefficient_samples[:, 0])
        accepted = sampled_x[
            (coefficient_samples[:, 0] > 0.0)
            & (sampled_x > x.min())
            & (sampled_x < x.max())
        ]
        if len(accepted) >= 400:
            sampled_parameters = anchor * np.exp(accepted)
            sampled_tokens = np.exp(token_intercept + token_slope * accepted)
            sampled_ratios = sampled_tokens / sampled_parameters
            confidence = {
                "parameters_p05": float(np.quantile(sampled_parameters, 0.05)),
                "parameters_p95": float(np.quantile(sampled_parameters, 0.95)),
                "train_tokens_p05": float(np.quantile(sampled_tokens, 0.05)),
                "train_tokens_p95": float(np.quantile(sampled_tokens, 0.95)),
                "tokens_per_parameter_p05": float(np.quantile(sampled_ratios, 0.05)),
                "tokens_per_parameter_p95": float(np.quantile(sampled_ratios, 0.95)),
                "accepted_draw_fraction": len(accepted) / len(coefficient_samples),
            }
            samples_payload = {
                "parameters": sampled_parameters,
                "train_tokens": sampled_tokens,
            }

    return (
        {
            "slice": slice_id,
            "target_total_flops": target_total_flops,
            "points": [_public_measurement(item) for item in ordered],
            "quadratic": {
                "coordinate": "log(parameters / anchor_parameters)",
                "anchor_parameters": anchor,
                "curvature": curvature,
                "slope": slope,
                "intercept": intercept,
                "residual_sum_squares": float(np.dot(residuals, residuals)),
            },
            "observed_best": {
                key: ordered[observed_index][key]
                for key in ("id", "parameters", "train_tokens", "validation_loss")
            },
            "observed_best_at_endpoint": observed_edge,
            "bracketed": bracketed,
            "unbracketed_reasons": reasons,
            "interpolated_optimum": optimum,
            "fit_only_uncertainty": confidence,
        },
        samples_payload,
    )


def _warrants_high_side_extension(fitted: Mapping[str, Any]) -> bool:
    """Return whether adding a larger model can address this failed bracket."""

    if fitted.get("bracketed"):
        return False
    points = fitted.get("points")
    observed = fitted.get("observed_best")
    if not isinstance(points, list) or not points or not isinstance(observed, Mapping):
        raise ScalingError("malformed slice fit while deciding grid extension")
    largest_parameters = max(int(item["parameters"]) for item in points)
    if observed.get("parameters") == largest_parameters:
        return True
    quadratic = _mapping(fitted.get("quadratic"), "slice quadratic")
    curvature = _finite(quadratic.get("curvature"), "slice curvature")
    slope = _finite(quadratic.get("slope"), "slice slope")
    anchor = _finite(
        quadratic.get("anchor_parameters"), "slice anchor_parameters", positive=True
    )
    if curvature <= 0.0:
        return False
    optimum_x = -slope / (2.0 * curvature)
    return optimum_x >= math.log(largest_parameters / anchor)


def _completed_v4_extensions_for_fit(
    suite: Mapping[str, Any],
    root: Path,
    measured_fit: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and read the prospective optional model-size frontier."""

    completed: list[dict[str, Any]] = []
    for slice_index, compute_slice in enumerate(suite["compute_slices"]):
        slice_id = str(compute_slice["id"])
        extension_states: list[tuple[str, bool]] = []
        for shape in suite["optional_extension_shapes"]:
            shape_id = str(shape["shape_id"])
            selection_path = _learning_rate_selection_path(
                root, shape_id, slice_id=slice_id
            )
            candidates = [
                point
                for point in (
                    suite["extension_calibrations"]
                    + suite["adaptive_calibrations"]
                )
                if point["slice"] == slice_id
                and point["shape_id"] == shape_id
            ]
            raw_evidence = any(
                (
                    root
                    / str(point["id"])
                    / "artifacts"
                    / "result.json"
                ).exists()
                or (
                    root
                    / str(point["id"])
                    / "artifacts"
                    / "result.json"
                ).is_symlink()
                for point in candidates
            )
            derived_evidence = selection_path.exists() or selection_path.is_symlink()
            extension_states.append((shape_id, raw_evidence or derived_evidence))

        gap_shape: str | None = None
        for shape_id, begun in extension_states:
            if not begun:
                if gap_shape is None:
                    gap_shape = shape_id
                continue
            if gap_shape is not None:
                raise ScalingError(
                    f"{slice_id}: optional extension evidence is not a "
                    f"model-size prefix: {shape_id} exists before {gap_shape}"
                )

        prior = [item for item in measured_fit if item["slice"] == slice_id]
        first_missing: str | None = None
        for extension_index, (shape_id, begun) in enumerate(extension_states):
            if not begun:
                first_missing = shape_id
                break
            # Recompute the selection from raw curves even when a derived JSON
            # exists. A crash after any extension result but before selection
            # therefore fails closed if its LR bracket is not complete.
            measurement = _selected_calibration_run(
                suite, shape_id, root, slice_id=slice_id
            )
            prior_fit, _ = _fit_slice(
                prior,
                slice_id=slice_id,
                target_total_flops=int(compute_slice["target_total_flops"]),
                random_seed=20_260_900 + slice_index * 10 + extension_index,
            )
            if prior_fit["bracketed"] or not _warrants_high_side_extension(prior_fit):
                reason = (
                    "the preceding model-size grid is already bracketed"
                    if prior_fit["bracketed"]
                    else "the preceding model-size grid does not warrant a high-side extension"
                )
                raise ScalingError(
                    f"{slice_id}/{shape_id}: optional extension evidence lies "
                    f"beyond the prospective model-size stopping frontier: {reason}"
                )
            completed.append(measurement)
            prior.append(measurement)
        if first_missing is not None:
            current_fit, _ = _fit_slice(
                prior,
                slice_id=slice_id,
                target_total_flops=int(compute_slice["target_total_flops"]),
                random_seed=20_260_950 + slice_index,
            )
            if (
                not current_fit["bracketed"]
                and _warrants_high_side_extension(current_fit)
            ):
                raise ScalingError(
                    f"{slice_id}: optional extension frontier is incomplete; "
                    f"the current model-size grid still warrants {first_missing}"
                )
    return completed


def fit_results(suite: Mapping[str, Any], runs_path: Path) -> dict[str, Any]:
    root = _validate_lineage_output_root(suite, runs_path, label="fit --runs")
    if not root.is_dir() or root.is_symlink():
        raise ScalingError(f"fit --runs must be a regular directory: {root}")
    if _per_slice_learning_rates(suite):
        measured_fit = [
            _selected_calibration_run(
                suite,
                str(shape["shape_id"]),
                root,
                slice_id=str(compute_slice["id"]),
            )
            for compute_slice in suite["compute_slices"]
            for shape in suite["fit_shapes"]
        ]
    else:
        measured_fit = [
            _selected_calibration_run(suite, shape["shape_id"], root)
            for shape in suite["fit_shapes"]
        ]
        measured_fit.extend(
            _read_run(suite, point, root) for point in suite["variants"]
        )
    completed_extensions: list[dict[str, Any]] = []
    if _per_slice_learning_rates(suite):
        completed_extensions = _completed_v4_extensions_for_fit(
            suite, root, measured_fit
        )
    else:
        # A selected c025 calibration is the extension shape's c025 IsoFLOP
        # measurement; do not spend compute rerunning an identical point.
        for shape in suite["optional_extension_shapes"]:
            selection_path = _learning_rate_selection_path(root, shape["shape_id"])
            if selection_path.is_file():
                completed_extensions.append(
                    _selected_calibration_run(suite, shape["shape_id"], root)
                )
        for point in suite["optional_extensions"]:
            result_path = root / point["id"] / "artifacts" / "result.json"
            if result_path.is_file():
                completed_extensions.append(_read_run(suite, point, root))
    completed_controls: list[dict[str, Any]] = []
    for point in suite["controls"]:
        result_path = root / point["id"] / "artifacts" / "result.json"
        if result_path.is_file():
            completed_controls.append(_read_run(suite, point, root))
    all_measurements = measured_fit + completed_extensions + completed_controls
    dataset_identity = _coherent_dataset_identity(all_measurements)
    fresh10_identity = _coherent_fresh10_identity(all_measurements)
    runtime_identity = _coherent_auxiliary_identity(
        all_measurements, "_runtime_provenance", "runtime identity"
    )

    slice_fits: list[dict[str, Any]] = []
    slice_samples: list[dict[str, np.ndarray] | None] = []
    for index, compute_slice in enumerate(suite["compute_slices"]):
        slice_id = str(compute_slice["id"])
        points = [item for item in measured_fit if item["slice"] == slice_id]
        points.extend(item for item in completed_extensions if item["slice"] == slice_id)
        fitted, samples = _fit_slice(
            points,
            slice_id=slice_id,
            target_total_flops=int(compute_slice["target_total_flops"]),
            random_seed=20_260_813 + index,
        )
        slice_fits.append(fitted)
        slice_samples.append(samples)

    all_bracketed = all(item["bracketed"] for item in slice_fits)
    scaling_law: dict[str, Any] | None = None
    if all_bracketed:
        compute = np.asarray(
            [item["target_total_flops"] for item in slice_fits], dtype=np.float64
        )
        optima = [item["interpolated_optimum"] for item in slice_fits]
        optimum_parameters = np.asarray(
            [item["parameters"] for item in optima], dtype=np.float64
        )
        optimum_tokens = np.asarray(
            [item["train_tokens"] for item in optima], dtype=np.float64
        )
        anchor_compute = float(suite["anchor"]["total_flops"])
        log_compute = np.log(compute / anchor_compute)
        parameter_exponent, log_parameters_at_anchor = np.polyfit(
            log_compute, np.log(optimum_parameters), 1
        )
        token_exponent, log_tokens_at_anchor = np.polyfit(
            log_compute, np.log(optimum_tokens), 1
        )
        uncertainty: dict[str, float] | None = None
        if all(samples is not None for samples in slice_samples):
            sample_count = min(len(samples["parameters"]) for samples in slice_samples)
            rng = np.random.default_rng(20_260_816)
            parameter_slopes = np.empty(sample_count, dtype=np.float64)
            token_slopes = np.empty(sample_count, dtype=np.float64)
            for draw in range(sample_count):
                # A slice draw is one jointly fitted optimum. Preserve the
                # parameter/token pairing instead of independently resampling
                # its two coordinates and inventing impossible allocations.
                paired_draws = [
                    rng.integers(len(samples["parameters"]))
                    for samples in slice_samples
                ]
                sampled_parameters = np.asarray(
                    [
                        samples["parameters"][index]
                        for samples, index in zip(
                            slice_samples, paired_draws, strict=True
                        )
                    ]
                )
                sampled_tokens = np.asarray(
                    [
                        samples["train_tokens"][index]
                        for samples, index in zip(
                            slice_samples, paired_draws, strict=True
                        )
                    ]
                )
                parameter_slopes[draw] = np.polyfit(
                    log_compute, np.log(sampled_parameters), 1
                )[0]
                token_slopes[draw] = np.polyfit(
                    log_compute, np.log(sampled_tokens), 1
                )[0]
            uncertainty = {
                "parameter_exponent_p05": float(np.quantile(parameter_slopes, 0.05)),
                "parameter_exponent_p95": float(np.quantile(parameter_slopes, 0.95)),
                "token_exponent_p05": float(np.quantile(token_slopes, 0.05)),
                "token_exponent_p95": float(np.quantile(token_slopes, 0.95)),
            }
        scaling_law = {
            "form": "N_opt(C)=N0*(C/C0)^a; D_opt(C)=D0*(C/C0)^b",
            "C0": int(suite["anchor"]["total_flops"]),
            "N0": math.exp(float(log_parameters_at_anchor)),
            "D0": math.exp(float(log_tokens_at_anchor)),
            "parameter_exponent_a": float(parameter_exponent),
            "token_exponent_b": float(token_exponent),
            "exponent_sum": float(parameter_exponent + token_exponent),
            "fit_only_uncertainty": uncertainty,
        }

    return {
        "schema_version": 1,
        "suite_id": suite["suite_id"],
        "suite_sha256": suite["suite_sha256"],
        "study_lineage": _lineage_summary(suite),
        "fit_kind": "three-slice-local-isoflop",
        "dataset": dataset_identity,
        "fresh10": fresh10_identity,
        "runtime": runtime_identity,
        "seed": int(suite["seed"]),
        "seed_count": 1,
        "slices": slice_fits,
        "controls": [_public_measurement(item) for item in completed_controls],
        "completed_optional_extensions": [
            _public_measurement(item) for item in completed_extensions
        ],
        "can_estimate_scaling_exponent": all_bracketed,
        "scaling_law": scaling_law,
        "limitation": (
            "This is a one-seed local empirical law for this tokenizer, FineWeb "
            "build, dense-transformer aspect family, initialization, global batch, "
            "and fixed optimizer schedule. Fit-only intervals come from quadratic "
            "residuals and exclude run-to-run noise, learning-rate uncertainty, data "
            "quality shifts, and alternative architectures. "
            + (
                "Learning rates are independently selected with a fail-closed "
                "stability gate for every compute-slice/model-shape pair. The "
                "same canonical FineWeb validation set is reused across many LR "
                "candidates for selection and again for the model-size fit; the "
                "intervals exclude selection and multiple-comparison optimism. "
                if _per_slice_learning_rates(suite)
                else "Learning rates are selected at 0.25 C0 and may not remain "
                "optimal at longer horizons. "
            )
            + "It is not a universal Chinchilla law."
        ),
    }


def write_fit(result: Mapping[str, Any], output: Path) -> tuple[Path, Path]:
    json_path = output.expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(json_path)
    markdown_path = json_path.with_suffix(".md")
    lines = [f"# IsoFLOP fit: {result['suite_id']}", ""]
    for slice_fit in result["slices"]:
        lines.extend(
            (
                f"## {slice_fit['slice']}",
                "",
                "| Point | Parameters | Train tokens | Tokens/parameter | Validation loss |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for point in slice_fit["points"]:
            lines.append(
                f"| {point['id']} | {int(point['parameters']):,} | "
                f"{int(point['train_tokens']):,} | "
                f"{float(point['tokens_per_parameter']):.3f} | "
                f"{float(point['validation_loss']):.6f} |"
            )
        optimum = slice_fit["interpolated_optimum"]
        if optimum is None:
            reasons = "; ".join(slice_fit["unbracketed_reasons"])
            lines.extend(("", f"No interpolated claim: {reasons}.", ""))
        else:
            lines.extend(
                (
                    "",
                    f"Interpolated optimum: {optimum['parameters']:,.0f} parameters, "
                    f"{optimum['train_tokens']:,.0f} tokens, "
                    f"{optimum['tokens_per_parameter']:.3f} tokens/parameter.",
                    "",
                )
            )
    law = result.get("scaling_law")
    if isinstance(law, Mapping):
        lines.extend(
            (
                "## Local scaling fit",
                "",
                f"`N_opt(C) = {law['N0']:,.0f} * (C / {int(law['C0']):,})^"
                f"{law['parameter_exponent_a']:.4f}`",
                "",
                f"`D_opt(C) = {law['D0']:,.0f} * (C / {int(law['C0']):,})^"
                f"{law['token_exponent_b']:.4f}`",
                "",
            )
        )
    else:
        lines.extend(
            (
                "## Local scaling fit",
                "",
                "No exponent is reported because at least one slice minimum was not "
                "bracketed by measured points.",
                "",
            )
        )
    lines.extend((f"> {result['limitation']}", ""))
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def _write_derived_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a derived fit; immutable raw run inputs stay untouched."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_kwargs(
    args: argparse.Namespace,
    *,
    data_inventory: Mapping[str, Any] | None = None,
    fresh10_inventory: Mapping[str, Any] | None = None,
    runtime_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    options = {
        "data_path": args.data_path,
        "runs_path": args.runs,
        "seed": args.seed,
        "color": args.color,
        "downstream_manifest": args.downstream_manifest,
        "downstream_root": args.downstream_root,
        "attention_tuning_cache": args.attention_tuning_cache,
        "autotune_attention": args.autotune_attention,
        "resume": args.resume,
    }
    if data_inventory is not None:
        options["data_inventory"] = data_inventory
    if fresh10_inventory is not None:
        options["fresh10_inventory"] = fresh10_inventory
    if runtime_inventory is not None:
        options["runtime_inventory"] = runtime_inventory
    return options


def _initial_calibrations(
    suite: Mapping[str, Any], shape_id: str, *, slice_id: str | None = None
) -> list[dict[str, Any]]:
    calibration_slice = _calibration_slice_id(suite, slice_id)
    return [
        item
        for item in suite["calibrations"] + suite["extension_calibrations"]
        if item["shape_id"] == shape_id and item["slice"] == calibration_slice
    ]


def _validate_adaptive_completion_prefix(
    suite: Mapping[str, Any],
    shape_id: str,
    runs_root: Path,
    *,
    slice_id: str | None = None,
) -> None:
    """Fail on completion holes or any run beyond an ineligible LR frontier."""

    calibration_slice = _calibration_slice_id(suite, slice_id)
    points = [
        item
        for item in suite["adaptive_calibrations"]
        if item["shape_id"] == shape_id and item["slice"] == calibration_slice
    ]
    by_rate = {float(item["learning_rate"]): item for item in points}
    for side in ("lower", "upper"):
        gap = False
        for candidate in suite["learning_rate_search"][side]:
            rate = float(candidate["value"])
            point = by_rate.get(rate)
            if point is None:
                raise ScalingError(
                    f"{shape_id}: missing declared {side} adaptive LR point {rate:.8g}"
                )
            complete = _point_has_result(suite, point, runs_root)
            if complete and gap:
                raise ScalingError(
                    f"{shape_id}: noncontiguous {side} learning-rate completion at "
                    f"{rate:.8g}; refusing to skip a mandatory geometric point"
                )
            if not complete:
                gap = True
    if not _per_slice_learning_rates(suite):
        return

    initial = _initial_calibrations(suite, shape_id, slice_id=slice_id)
    all_points = sorted(
        initial + points, key=lambda item: float(item["learning_rate"])
    )
    completed_indices = [
        index
        for index, point in enumerate(all_points)
        if _point_has_result(suite, point, runs_root)
    ]
    if completed_indices and completed_indices != list(
        range(min(completed_indices), max(completed_indices) + 1)
    ):
        raise ScalingError(
            f"{calibration_slice}/{shape_id}: learning-rate completions have a "
            "missing middle point"
        )
    initial_complete = [
        _point_has_result(suite, point, runs_root) for point in initial
    ]
    if any(initial_complete[index] and not all(initial_complete[:index]) for index in range(len(initial))):
        raise ScalingError(
            f"{calibration_slice}/{shape_id}: initial learning-rate completion "
            "is not a low-to-high prefix"
        )
    measured: list[dict[str, Any]] = []
    for point in all_points:
        if _point_has_result(suite, point, runs_root):
            measured.append(_read_run(suite, point, runs_root))
    by_id = {str(item["id"]): item for item in measured}
    lower_rates = {
        float(item["value"]) for item in suite["learning_rate_search"]["lower"]
    }
    for point in points:
        measurement = by_id.get(str(point["id"]))
        if (
            float(point["learning_rate"]) in lower_rates
            and measurement is not None
            and measurement["stability_admission"]["classification"] != "stable"
        ):
            raise ScalingError(
                f"{calibration_slice}/{shape_id}: lower LR expansion is ineligible; "
                "refusing to search beyond it"
            )
    initial_and_upper = [
        point
        for point in all_points
        if float(point["learning_rate"]) >= float(initial[0]["learning_rate"])
        and point["id"] in by_id
    ]
    frontier_index = next(
        (
            index
            for index, point in enumerate(initial_and_upper)
            if by_id[point["id"]]["stability_admission"]["classification"]
            != "stable"
        ),
        None,
    )
    if frontier_index is not None and frontier_index != len(initial_and_upper) - 1:
        frontier = initial_and_upper[frontier_index]
        raise ScalingError(
            f"{calibration_slice}/{shape_id}: completed LR "
            f"{float(frontier['learning_rate']):.8g} is an ineligible frontier, "
            "but a higher LR artifact also exists"
        )


def _next_adaptive_calibration(
    suite: Mapping[str, Any],
    shape_id: str,
    side: str,
    runs_root: Path,
    *,
    slice_id: str | None = None,
) -> dict[str, Any] | None:
    calibration_slice = _calibration_slice_id(suite, slice_id)
    _validate_adaptive_completion_prefix(
        suite, shape_id, runs_root, slice_id=slice_id
    )
    all_points = _initial_calibrations(suite, shape_id, slice_id=slice_id) + [
        item
        for item in suite["adaptive_calibrations"]
        if item["shape_id"] == shape_id and item["slice"] == calibration_slice
    ]
    completed = [
        item
        for item in all_points
        if _point_has_result(suite, item, runs_root)
    ]
    if not completed:
        raise ScalingError(f"{shape_id}: cannot expand an empty learning-rate grid")
    completed_rates = [float(item["learning_rate"]) for item in completed]
    pending = [
        item
        for item in suite["adaptive_calibrations"]
        if item["shape_id"] == shape_id
        and item["slice"] == calibration_slice
        and not _point_has_result(suite, item, runs_root)
    ]
    if side == "lower":
        candidates = [
            item
            for item in pending
            if float(item["learning_rate"]) < min(completed_rates)
        ]
        return max(candidates, key=lambda item: float(item["learning_rate"]), default=None)
    if side == "upper":
        if _per_slice_learning_rates(suite):
            highest = max(
                completed, key=lambda item: float(item["learning_rate"])
            )
            highest_measurement = _read_run(suite, highest, runs_root)
            if (
                highest_measurement["stability_admission"]["classification"]
                != "stable"
            ):
                return None
        candidates = [
            item
            for item in pending
            if float(item["learning_rate"]) > max(completed_rates)
        ]
        return min(candidates, key=lambda item: float(item["learning_rate"]), default=None)
    raise ScalingError(f"unknown learning-rate expansion side: {side}")


def _calibrate_shape(
    suite: Mapping[str, Any],
    shape_id: str,
    runs_root: Path,
    run_options: Mapping[str, Any],
    *,
    slice_id: str | None = None,
) -> dict[str, Any]:
    calibration_slice = _calibration_slice_id(suite, slice_id)
    group_label = (
        f"{calibration_slice}/{shape_id}"
        if _per_slice_learning_rates(suite)
        else shape_id
    )
    selection_path = _learning_rate_selection_path(
        runs_root,
        shape_id,
        slice_id=(calibration_slice if _per_slice_learning_rates(suite) else None),
    )
    if selection_path.is_file():
        # Fully validate the immutable selection and selected run before reuse.
        _selected_calibration_run(
            suite, shape_id, runs_root, slice_id=slice_id
        )
        return dict(_read_regular_json(selection_path, "learning-rate selection"))
    initial = _initial_calibrations(suite, shape_id, slice_id=slice_id)
    if len(initial) != len(suite["learning_rate_candidates"]):
        raise ScalingError(f"{group_label}: initial learning-rate grid is incomplete")
    _validate_adaptive_completion_prefix(
        suite, shape_id, runs_root, slice_id=slice_id
    )
    # V4 launches low-to-high, validates each completed curve immediately, and
    # never crosses the first ineligible high-LR frontier.  Legacy suites retain
    # their original all-at-once initial grid behavior.
    if _per_slice_learning_rates(suite):
        frontier = False
        first_initial_classification: str | None = None
        for point in initial:
            if not _point_has_result(suite, point, runs_root):
                if frontier:
                    break
                run_variants(
                    suite, names=[str(point["id"])], **run_options
                )
            measurement = _read_run(suite, point, runs_root)
            stability = _mapping(
                measurement.get("stability_admission"),
                f"{point['id']}.stability_admission",
            )
            if stability.get("classification") != "stable":
                frontier = True
                if point is initial[0]:
                    first_initial_classification = str(stability["classification"])
                break
        # If even the lowest initial LR is ineligible, move only toward safer
        # (lower) declared rates until a stable point appears or the bounded
        # table is exhausted. Never try lr300/lr450 beyond the lr200 frontier.
        if first_initial_classification is not None:
            lower_completed = [
                point
                for point in suite["adaptive_calibrations"]
                if point["slice"] == calibration_slice
                and point["shape_id"] == shape_id
                and float(point["learning_rate"])
                < float(initial[0]["learning_rate"])
                and _point_has_result(suite, point, runs_root)
            ]
            stable_lower_count = sum(
                _read_run(suite, point, runs_root)["stability_admission"][
                    "classification"
                ]
                == "stable"
                for point in lower_completed
            )
            while True:
                if stable_lower_count >= 2:
                    break
                next_lower = _next_adaptive_calibration(
                    suite,
                    shape_id,
                    "lower",
                    runs_root,
                    slice_id=slice_id,
                )
                if next_lower is None:
                    raise ScalingError(
                        f"{group_label}: bounded lower LR recovery exhausted "
                        "without a stable candidate"
                    )
                run_variants(
                    suite, names=[str(next_lower["id"])], **run_options
                )
                recovery = _read_run(suite, next_lower, runs_root)
                if (
                    recovery["stability_admission"]["classification"]
                    == "stable"
                ):
                    stable_lower_count += 1
    else:
        run_variants(
            suite,
            names=[str(item["id"]) for item in initial],
            **run_options,
        )
    while True:
        try:
            return select_learning_rate(
                suite,
                shape_id=shape_id,
                slice_id=slice_id,
                runs_path=runs_root,
            )
        except LearningRateEdgeError as exc:
            next_point = _next_adaptive_calibration(
                suite,
                shape_id,
                exc.side,
                runs_root,
                slice_id=slice_id,
            )
            if next_point is None:
                raise ScalingError(
                    f"{group_label}: bounded geometric learning-rate search is "
                    f"exhausted on the {exc.side} side; refusing to select an "
                    "edge winner or launch dependent runs"
                ) from exc
            print(
                f"{group_label}: expanding {exc.side} LR edge with "
                f"{float(next_point['learning_rate']):.8g}",
                flush=True,
            )
            run_variants(
                suite,
                names=[str(next_point["id"])],
                **run_options,
            )


def _completed_slice_measurements(
    suite: Mapping[str, Any], slice_id: str, runs_root: Path
) -> list[dict[str, Any]]:
    """Read every completed fit/extension measurement for one compute slice."""

    calibration_slice_id = str(suite["compute_slices"][0]["id"])
    if _per_slice_learning_rates(suite):
        measured = [
            _selected_calibration_run(
                suite,
                shape["shape_id"],
                runs_root,
                slice_id=slice_id,
            )
            for shape in suite["fit_shapes"]
        ]
        for shape in suite["optional_extension_shapes"]:
            selection = _learning_rate_selection_path(
                runs_root, str(shape["shape_id"]), slice_id=slice_id
            )
            if selection.is_file():
                measured.append(
                    _selected_calibration_run(
                        suite,
                        str(shape["shape_id"]),
                        runs_root,
                        slice_id=slice_id,
                    )
                )
    elif slice_id == calibration_slice_id:
        measured = [
            _selected_calibration_run(suite, shape["shape_id"], runs_root)
            for shape in suite["fit_shapes"]
        ]
        for shape in suite["optional_extension_shapes"]:
            if _learning_rate_selection_path(runs_root, shape["shape_id"]).is_file():
                measured.append(
                    _selected_calibration_run(suite, shape["shape_id"], runs_root)
                )
    else:
        base_points = [
            point for point in suite["variants"] if point["slice"] == slice_id
        ]
        measured = [_read_run(suite, point, runs_root) for point in base_points]
        for point in suite["optional_extensions"]:
            result = runs_root / point["id"] / "artifacts" / "result.json"
            if point["slice"] == slice_id and result.is_file():
                measured.append(_read_run(suite, point, runs_root))

    by_parameters: dict[int, dict[str, Any]] = {}
    for measurement in measured:
        parameters = int(measurement["parameters"])
        if parameters in by_parameters:
            raise ScalingError(
                f"{slice_id}: duplicate completed measurement at {parameters:,} parameters"
            )
        by_parameters[parameters] = measurement
    return list(by_parameters.values())


def _ensure_slice_extension(
    suite: Mapping[str, Any],
    *,
    compute_slice: Mapping[str, Any],
    shape: Mapping[str, Any],
    runs_root: Path,
    run_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize one missing extension, reusing any completed immutable run."""

    slice_id = str(compute_slice["id"])
    shape_id = str(shape["shape_id"])
    _calibrate_shape(
        suite,
        shape_id,
        runs_root,
        run_options,
        slice_id=(slice_id if _per_slice_learning_rates(suite) else None),
    )
    if _per_slice_learning_rates(suite):
        return _selected_calibration_run(
            suite, shape_id, runs_root, slice_id=slice_id
        )
    if slice_id == str(suite["compute_slices"][0]["id"]):
        return _selected_calibration_run(suite, shape_id, runs_root)

    extension = next(
        point
        for point in suite["optional_extensions"]
        if point["slice"] == slice_id and point["shape_id"] == shape_id
    )
    result = runs_root / extension["id"] / "artifacts" / "result.json"
    if not result.is_file():
        run_variants(suite, names=[str(extension["id"])], **run_options)
    return _read_run(suite, extension, runs_root)


def _revisit_extension_fixed_point(
    suite: Mapping[str, Any],
    *,
    runs_root: Path,
    run_options: Mapping[str, Any],
) -> None:
    """Close cross-slice extension dependencies over the finite declared grid.

    Calibrating an extension first requested by c050/c100 also creates its c025
    IsoFLOP measurement. That new endpoint can invalidate c025's earlier
    bracket. Restarting from the lowest slice after every newly materialized
    (slice, shape) point reaches the fixed point without skipping that case.
    """

    shapes = list(suite["optional_extension_shapes"])
    max_actions = len(suite["compute_slices"]) * len(shapes)
    actions = 0
    while True:
        restarted = False
        for compute_slice in suite["compute_slices"]:
            slice_id = str(compute_slice["id"])
            measured = _completed_slice_measurements(suite, slice_id, runs_root)
            fitted, _ = _fit_slice(
                measured,
                slice_id=slice_id,
                target_total_flops=int(compute_slice["target_total_flops"]),
            )
            if fitted["bracketed"] or not _warrants_high_side_extension(fitted):
                _write_derived_json(
                    runs_root / "fits" / f"{slice_id}.json", fitted
                )
                continue

            measured_parameters = {int(item["parameters"]) for item in measured}
            missing = [
                shape
                for shape in shapes
                if int(shape["parameters"]) not in measured_parameters
            ]
            if not missing:
                print(
                    f"{slice_id}: bounded model-size extension grid exhausted; "
                    "recording no-law instead of extrapolating",
                    flush=True,
                )
                _write_derived_json(
                    runs_root / "fits" / f"{slice_id}.json", fitted
                )
                continue
            if actions >= max_actions:
                raise ScalingError(
                    "model-grid fixed point exceeded its finite slice/shape bound"
                )

            shape = missing[0]
            measurement = _ensure_slice_extension(
                suite,
                compute_slice=compute_slice,
                shape=shape,
                runs_root=runs_root,
                run_options=run_options,
            )
            if int(measurement["parameters"]) in measured_parameters:
                raise ScalingError(
                    f"{slice_id}: extension did not add a new model-size measurement"
                )
            actions += 1
            restarted = True
            break
        if not restarted:
            return


def run_staged(suite: Mapping[str, Any], args: argparse.Namespace) -> None:
    """Run low-to-high compute with bounded LR and model-grid adaptation."""

    _validate_lineage_output_root(suite, args.runs, label="--runs")
    # These expensive/TPU-sensitive gates happen exactly once before any run;
    # their byte-exact identities are then copied into every run manifest.
    runtime_inventory = validate_runtime_environment(suite)
    data_inventory = validate_data_directory(args.data_path, suite)
    fresh10_inventory = validate_fresh10_directory(
        args.downstream_manifest, args.downstream_root, suite
    )
    run_options = _run_kwargs(
        args,
        data_inventory=data_inventory,
        fresh10_inventory=fresh10_inventory,
        runtime_inventory=runtime_inventory,
    )
    run_options["allow_adaptive"] = True
    runs_root = args.runs.expanduser().resolve()
    if _per_slice_learning_rates(suite):
        for compute_slice in suite["compute_slices"]:
            slice_id = str(compute_slice["id"])
            for shape in suite["fit_shapes"]:
                shape_id = str(shape["shape_id"])
                selection = _calibrate_shape(
                    suite,
                    shape_id,
                    runs_root,
                    run_options,
                    slice_id=slice_id,
                )
                print(
                    f"selected {slice_id}/{shape_id}: "
                    f"lr={selection['selected_learning_rate']:.2e} "
                    f"at loss {selection['selected_validation_loss']:.6f}",
                    flush=True,
                )
            measured = [
                _selected_calibration_run(
                    suite,
                    str(shape["shape_id"]),
                    runs_root,
                    slice_id=slice_id,
                )
                for shape in suite["fit_shapes"]
            ]
            fitted, _ = _fit_slice(
                measured,
                slice_id=slice_id,
                target_total_flops=int(compute_slice["target_total_flops"]),
            )
            for shape in suite["optional_extension_shapes"]:
                if fitted["bracketed"]:
                    break
                if not _warrants_high_side_extension(fitted):
                    print(
                        f"{slice_id}: minimum remains unbracketed, but not on the "
                        "high-model-size side; recording no-law instead of extrapolating",
                        flush=True,
                    )
                    break
                measurement = _ensure_slice_extension(
                    suite,
                    compute_slice=compute_slice,
                    shape=shape,
                    runs_root=runs_root,
                    run_options=run_options,
                )
                measured.append(measurement)
                fitted, _ = _fit_slice(
                    measured,
                    slice_id=slice_id,
                    target_total_flops=int(compute_slice["target_total_flops"]),
                )
            _write_derived_json(runs_root / "fits" / f"{slice_id}.json", fitted)
    else:
        for shape in suite["fit_shapes"]:
            shape_id = str(shape["shape_id"])
            selection = _calibrate_shape(
                suite, shape_id, runs_root, run_options
            )
            print(
                f"selected {shape_id}: lr={selection['selected_learning_rate']:.2e} "
                f"at loss {selection['selected_validation_loss']:.6f}",
                flush=True,
            )

        for compute_slice in suite["compute_slices"]:
            slice_id = str(compute_slice["id"])
            if compute_slice is suite["compute_slices"][0]:
                measured = [
                    _selected_calibration_run(
                        suite, shape["shape_id"], runs_root
                    )
                    for shape in suite["fit_shapes"]
                ]
            else:
                points = [
                    item for item in suite["variants"] if item["slice"] == slice_id
                ]
                run_variants(
                    suite, names=[str(item["id"]) for item in points], **run_options
                )
                measured = [_read_run(suite, item, runs_root) for item in points]
            fitted, _ = _fit_slice(
                measured,
                slice_id=slice_id,
                target_total_flops=int(compute_slice["target_total_flops"]),
            )
            for shape in suite["optional_extension_shapes"]:
                if fitted["bracketed"]:
                    break
                if not _warrants_high_side_extension(fitted):
                    print(
                        f"{slice_id}: minimum remains unbracketed, but not on the "
                        "high-model-size side; recording no-law instead of extrapolating",
                        flush=True,
                    )
                    break
                shape_id = str(shape["shape_id"])
                _calibrate_shape(suite, shape_id, runs_root, run_options)
                if slice_id == suite["compute_slices"][0]["id"]:
                    measurement = _selected_calibration_run(
                        suite, shape_id, runs_root
                    )
                else:
                    extension = next(
                        item
                        for item in suite["optional_extensions"]
                        if item["slice"] == slice_id
                        and item["shape_id"] == shape_id
                    )
                    run_variants(
                        suite, names=[str(extension["id"])], **run_options
                    )
                    measurement = _read_run(suite, extension, runs_root)
                if all(
                    item["parameters"] != measurement["parameters"]
                    for item in measured
                ):
                    measured.append(measurement)
                fitted, _ = _fit_slice(
                    measured,
                    slice_id=slice_id,
                    target_total_flops=int(compute_slice["target_total_flops"]),
                )
            _write_derived_json(runs_root / "fits" / f"{slice_id}.json", fitted)

        _revisit_extension_fixed_point(
            suite, runs_root=runs_root, run_options=run_options
        )

    control_names = [str(item["id"]) for item in suite["controls"]]
    if control_names:
        run_variants(suite, names=control_names, **run_options)
    result = fit_results(suite, args.runs)
    # Re-emit slice summaries because an extension first needed by a later
    # slice also contributes its already-paid c025 calibration measurement.
    for fitted in result["slices"]:
        _write_derived_json(
            runs_root / "fits" / f"{fitted['slice']}.json", fitted
        )
    json_path, markdown_path = write_fit(result, args.runs / "fit.json")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m speedrun.scaling",
        description="Plan, run, and fit the versioned diagnostic IsoFLOP suite.",
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--internal-runtime-probe",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    commands = parser.add_subparsers(dest="command")
    plan = commands.add_parser("plan")
    plan.add_argument("--json", action="store_true")

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--include-extensions", action="store_true")

    run = commands.add_parser("run")
    run.add_argument("--data-path", type=Path, required=True)
    run.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    run.add_argument(
        "--confirm-execution-fingerprint",
        required=True,
        help="exact digest printed by `plan`; freezes code/config review before TPU use",
    )
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--variant", action="append", default=[])
    selection.add_argument(
        "--staged",
        action="store_true",
        help="calibrate LR, run c025→c050→c100, and extend endpoint minima",
    )
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    run.add_argument("--downstream-manifest", type=Path, required=True)
    run.add_argument("--downstream-root", type=Path, required=True)
    run.add_argument("--attention-tuning-cache", type=Path)
    run.add_argument("--autotune-attention", action="store_true")
    run.add_argument("--resume", action="store_true")

    fit = commands.add_parser("fit")
    fit.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    fit.add_argument("--output", type=Path, default=DEFAULT_RUNS / "fit.json")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        suite = load_suite(args.suite)
        if args.internal_runtime_probe:
            if args.command is not None:
                raise ScalingError("internal runtime probe cannot be combined with a command")
            print(
                json.dumps(
                    _runtime_inventory_in_current_process(suite),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        elif args.command == "plan":
            print_plan(suite, as_json=args.json)
        elif args.command == "materialize":
            points = list(suite["calibrations"]) + list(suite["controls"])
            if args.include_extensions:
                points.extend(suite["extension_calibrations"])
            for path in materialize_configs(
                suite, args.output, [item["id"] for item in points]
            ):
                print(path)
        elif args.command == "run":
            if suite["suite_id"] in _ARCHIVED_SUITE_IDS:
                raise ScalingError(
                    f"{suite['suite_id']} is an immutable archived study; use the "
                    "versioned continuation suite"
                )
            if args.seed != suite["seed"]:
                raise ScalingError(
                    f"this one-seed suite pins --seed {suite['seed']}; got {args.seed}"
                )
            if args.confirm_execution_fingerprint != suite["execution_fingerprint"]:
                raise ScalingError(
                    "execution fingerprint does not match the current suite, trainer, "
                    "kernel sources, and scaling runner; inspect `plan` again"
                )
            if args.staged:
                run_staged(suite, args)
            else:
                names = args.variant
                run_variants(suite, names=names, **_run_kwargs(args))
        elif args.command == "fit":
            # ``fit_results`` may materialize derived learning-rate selections,
            # so reject a hostile explicit output before doing any fit work.
            _validate_lineage_output_root(suite, args.output, label="fit --output")
            result = fit_results(suite, args.runs)
            json_path, markdown_path = write_fit(result, args.output)
            print(f"wrote {json_path}")
            print(f"wrote {markdown_path}")
        else:  # pragma: no cover - argparse owns choices
            raise ScalingError("a scaling command is required")
    except (OSError, ScalingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
