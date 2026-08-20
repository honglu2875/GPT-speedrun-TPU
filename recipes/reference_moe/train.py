#!/usr/bin/env python3
"""A compact, dependency-light GPT trainer for the GPT TPU rig.

Everything involved in training lives in this file.  The default model is sized
for a TPU v4-8 and uses pure JAX: model state is replicated while the global
batch is sharded over every visible device.  ``--smoke`` selects a tiny CPU-
friendly configuration and the built-in byte corpus means the script never
requires a download.

Prepared data can be supplied as a directory of llm.c-style FineWeb shards,
individual NumPy/token/text files, or repeatable explicit shard paths. The
final stdout line of a competition run is a machine-readable result and is
intentionally never colorized. Diagnostic XProf runs deliberately omit it.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import functools
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform as host_platform
import re
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.experimental import multihost_utils
import yaml
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax.experimental.pallas.ops.tpu import megablox

from rig import logpack
from rig.attention import (
    AttentionCallable,
    AttentionRuntime,
    attention_console_rows,
    attention_runtime_metadata,
    attention_softmax_scale,
    document_segments,
    make_mesh_attention,
    resolve_attention_runtime,
)
from rig.metrics import DIAGNOSTIC_FAMILIES, DIAGNOSTIC_STATS
from rig.configfile import (
    config_bool,
    config_choice,
    config_float,
    config_int,
    config_keys,
    config_mapping,
    read_config_document,
    resolve_sibling_config_path,
)
from rig.runlog import (
    CHECKPOINT_NAME,
    DIAGNOSTICS_LOG_NAME,
    DIAGNOSTIC_FLUSH_POINTS,
    RESULT_PREFIX,
    TRAINING_LOG_NAME,
    VALIDATION_CSV_NAME,
    DiagnosticPoint,
    ValidationRow,
    append_log_row,
    close_log,
    diagnostic_log_columns,
    open_log,
    profiler_options,
    save_checkpoint,
    ROUTER_SUMMARY_METRICS,
    training_log_columns,
    write_diagnostics_log,
    write_result,
    write_training_log,
    write_validation_csv,
    xprof_step_window,
)
from rig.evaluation import (
    evaluate_downstream_domain,
    evaluate_validation_prefix,
    perplexity_from_loss,
    should_run_diagnostics,
    should_run_validation_probe,
)
from rig.nn import (
    apply_rotary,
    flatten_arrays,
    linear,
    normal,
    parameter_count,
    rms_norm,
)
from rig.tokens import (
    DownstreamDomain,
    ShardedTokens,
    ShuffledEpochBatchStream,
    TokenDataset,
    downstream_batches,
    file_sha256,
    load_dataset,
    load_downstream_domains,
)
from rig.console import Console, device_label, format_count, format_rate
from rig.mesh import (
    finite_metric,
    initialize_distributed_runtime,
    inferred_peak_tflops,
    is_controller_process,
    local_batch_size,
    local_device_get,
    put_host_local_array,
    put_replicated_tree,
    rank_local_slice,
    sync_tree,
    system_metadata,
    validate_official_topology,
)
from rig.flops import (
    FlopBreakdown,
    count_training_flops,
    default_rules,
    describe,
)
from rig.kernels import (
    AttentionConfig,
    make_causal_attention,
    select_attention_tiles,
    tiled_tied_cross_entropy,
    tiled_tied_cross_entropy_losses,
)


SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 4
CONFIG_FILENAME = "config.yaml"
CONFIG_PATH = Path(__file__).resolve().with_name(CONFIG_FILENAME)
_VALID_PROFILES = ("smoke", "dev", "official")
_DOMAIN_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_TIER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_CONTEXT_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


# A deliberately small, original corpus for offline and smoke-test use.  The
# repeated motifs make it possible for tiny models to show measurable progress,
# while the shuffled clauses prevent every training window from being identical.


@dataclass(frozen=True)
class Config:
    steps: int
    batch_size: int
    seq_len: int
    sampling: str
    layers: int
    heads: int
    d_model: int
    mlp_mult: int
    normalization: str
    position_encoding: str
    mlp_activation: str
    tier: str
    declared_parameters: int | None
    base_parameters: int
    parameterization: str
    base_width: int
    base_depth: int
    depth_alpha: float
    init_std: float
    attention_scale: str
    embeddings: str
    width_multiplier: float
    depth_multiplier: float
    data_multiplier: float
    batch_multiplier: float
    target_tokens_per_parameter: float | None
    tokens_per_parameter: float | None
    learning_rate: float
    min_lr_ratio: float
    warmup_steps: int
    weight_decay: float
    adam_epsilon: float
    beta1: float
    beta2: float
    grad_clip: float
    eval_batches: int
    val_every: int
    val_probe_batches: int
    diagnostics_every: int
    log_every: int
    vocab_size: int
    semantic_vocab_size: int
    attention_backend: str
    loss_backend: str
    vocab_tile_size: int
    compute_dtype: Any
    dtype_name: str
    config_schema_version: int
    config_sha256: str
    config_profile: str
    context_preset: str
    config_overrides: tuple[tuple[str, int | str], ...]
    # Optimizer step after which to stop. steps, warmup, and m_D still resolve
    # from the full horizon, so the trajectory matches the untruncated run up
    # to this point. None runs to completion.
    stop_after_step: int | None = None
    # Block-diagonal attention over documents. The selected context preset owns
    # this policy together with sequence length and the recipe-local batch anchor.
    document_masking: bool = False
    document_boundary_token: int = 50256
    # Routing. experts=0 keeps the dense MLP, so this file still describes the
    # baseline exactly when the keys are absent.
    experts: int = 0
    expert_top_k: int = 2
    expert_mult: int = 2
    router_aux_coefficient: float = 0.01

    @property
    def final_step(self) -> int:
        """Last optimizer step this run takes; steps stays the schedule horizon."""

        return self.stop_after_step or self.steps


@dataclass(frozen=True)
class ExperimentProfile:
    """One fully explicit, versioned profile loaded from sibling config.yaml."""

    schema_version: int
    source_sha256: str
    name: str
    context_preset: str
    steps: int | None
    tokens_per_parameter: float | None
    batch_size: int
    seq_len: int
    sampling: str
    dtype_name: str
    layers: int
    heads: int
    d_model: int
    mlp_mult: int
    normalization: str
    position_encoding: str
    mlp_activation: str
    tier: str
    declared_parameters: int | None
    base_parameters: int
    parameterization: str
    base_width: int
    base_depth: int
    depth_alpha: float
    init_std: float
    attention_scale: str
    embeddings: str
    vocab_size: int
    semantic_vocab_size: int
    attention_backend: str
    loss_backend: str
    vocab_tile_size: int
    learning_rate: float
    min_lr_ratio: float
    warmup_ratio: float
    weight_decay: float
    adam_epsilon: float
    beta1: float
    beta2: float
    grad_clip: float
    eval_batches: int
    val_every: int
    val_probe_batches: int
    diagnostics_every: int
    log_every: int
    document_masking: bool = False
    experts: int = 0
    expert_top_k: int = 2
    expert_mult: int = 2
    router_aux_coefficient: float = 0.01


_UINT64_MASK = (1 << 64) - 1


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def _parse_model(value: Any, label: str) -> dict[str, Any]:
    model = config_keys(
        value,
        label,
        {
            "layers",
            "heads",
            "d_model",
            "mlp_mult",
            "normalization",
            "position_encoding",
            "mlp_activation",
            "vocab_size",
            "semantic_vocab_size",
        },
        optional={"experts", "expert_top_k", "expert_mult", "router_aux_coefficient"},
    )
    parsed = {
        "layers": config_int(model["layers"], f"{label}.layers", minimum=1),
        "heads": config_int(model["heads"], f"{label}.heads", minimum=1),
        "d_model": config_int(model["d_model"], f"{label}.d_model", minimum=1),
        "mlp_mult": config_int(model["mlp_mult"], f"{label}.mlp_mult", minimum=1),
        "experts": config_int(model.get("experts", 0), f"{label}.experts", minimum=0)
        if "experts" in model
        else 0,
        "expert_top_k": config_int(
            model.get("expert_top_k", 2), f"{label}.expert_top_k", minimum=1
        )
        if "expert_top_k" in model
        else 2,
        "expert_mult": config_int(
            model.get("expert_mult", 2), f"{label}.expert_mult", minimum=1
        )
        if "expert_mult" in model
        else 2,
        "router_aux_coefficient": config_float(
            model.get("router_aux_coefficient", 0.01),
            f"{label}.router_aux_coefficient",
        )
        if "router_aux_coefficient" in model
        else 0.01,
        "normalization": config_choice(
            model["normalization"], f"{label}.normalization", ("rms_norm",)
        ),
        "position_encoding": config_choice(
            model["position_encoding"],
            f"{label}.position_encoding",
            ("rope_base_10000",),
        ),
        "mlp_activation": config_choice(
            model["mlp_activation"], f"{label}.mlp_activation", ("gelu",)
        ),
        "vocab_size": config_int(model["vocab_size"], f"{label}.vocab_size", minimum=1),
        "semantic_vocab_size": config_int(
            model["semantic_vocab_size"],
            f"{label}.semantic_vocab_size",
            minimum=1,
        ),
    }
    if parsed["semantic_vocab_size"] > parsed["vocab_size"]:
        raise ValueError(
            f"config.yaml {label}.semantic_vocab_size must not exceed vocab_size"
        )
    if parsed["d_model"] % parsed["heads"]:
        raise ValueError(f"config.yaml {label}.d_model must be divisible by heads")
    if (parsed["d_model"] // parsed["heads"]) % 2:
        raise ValueError(f"config.yaml {label} head dimension must be even for RoPE")
    if parsed["router_aux_coefficient"] < 0.0:
        raise ValueError(
            f"config.yaml {label}.router_aux_coefficient must be nonnegative"
        )
    if parsed["experts"]:
        if parsed["expert_top_k"] > parsed["experts"]:
            raise ValueError(
                f"config.yaml {label}.expert_top_k must not exceed experts"
            )
        if parsed["expert_top_k"] * parsed["expert_mult"] != parsed["mlp_mult"]:
            raise ValueError(
                f"config.yaml {label} must satisfy expert_top_k * expert_mult "
                "== mlp_mult so active MLP FLOPs match the dense tier"
            )
    return parsed


def _declared_family_parameter_count(model: Mapping[str, Any]) -> int:
    """Count this trainer's untied, position-free RMSNorm transformer exactly."""

    width = int(model["d_model"])
    return (
        2 * int(model["vocab_size"]) * width
        + int(model["layers"]) * (12 * width * width + 11 * width)
        + width
    )


def _parse_experiment_profile(
    payload: Mapping[str, Any],
    profile: str,
    source_sha256: str,
    tier: str | None,
    context: str | None,
) -> ExperimentProfile:
    top = config_keys(payload, "document", {"schema_version", "family", "profiles"})
    schema_version = config_int(top["schema_version"], "schema_version", minimum=1)
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "unsupported config.yaml schema_version "
            f"{schema_version}; expected {CONFIG_SCHEMA_VERSION}"
        )
    family = config_keys(
        top["family"],
        "family",
        {
            "default_tier",
            "default_context",
            "contexts",
            "parameterization",
            "tiers",
        },
    )
    parameterization = config_keys(
        family["parameterization"],
        "family.parameterization",
        {
            "name",
            "base_tier",
            "base_width",
            "base_depth",
            "depth_alpha",
            "init_std",
            "attention_scale",
            "embeddings",
        },
    )
    parameterization_name = config_choice(
        parameterization["name"],
        "family.parameterization.name",
        ("completep_fixed_tpp_v1",),
    )
    base_tier = parameterization["base_tier"]
    if not isinstance(base_tier, str) or not _TIER_NAME.fullmatch(base_tier):
        raise ValueError("config.yaml family.parameterization.base_tier is invalid")
    base_width = config_int(
        parameterization["base_width"],
        "family.parameterization.base_width",
        minimum=1,
    )
    base_depth = config_int(
        parameterization["base_depth"],
        "family.parameterization.base_depth",
        minimum=1,
    )
    depth_alpha = config_float(
        parameterization["depth_alpha"], "family.parameterization.depth_alpha"
    )
    init_std = config_float(
        parameterization["init_std"], "family.parameterization.init_std"
    )
    if not 0.5 <= depth_alpha <= 1.0:
        raise ValueError(
            "config.yaml family.parameterization.depth_alpha must be in [0.5, 1]"
        )
    if init_std <= 0.0:
        raise ValueError(
            "config.yaml family.parameterization.init_std must be positive"
        )
    attention_scale = config_choice(
        parameterization["attention_scale"],
        "family.parameterization.attention_scale",
        ("inverse_head_dim",),
    )
    embeddings = config_choice(
        parameterization["embeddings"],
        "family.parameterization.embeddings",
        ("untied",),
    )
    raw_contexts = config_mapping(family["contexts"], "family.contexts")
    if not raw_contexts:
        raise ValueError("config.yaml family.contexts must define at least one preset")
    invalid_contexts = sorted(
        name for name in raw_contexts if not _CONTEXT_NAME.fullmatch(name)
    )
    if invalid_contexts:
        raise ValueError(
            "config.yaml family.contexts contains invalid name(s): "
            + ", ".join(invalid_contexts)
        )
    default_context = family["default_context"]
    if not isinstance(default_context, str) or default_context not in raw_contexts:
        raise ValueError(
            "config.yaml family.default_context must name a defined context preset"
        )
    selected_context = context or default_context
    if selected_context not in raw_contexts:
        raise ValueError(
            f"unknown context preset {selected_context!r}; expected "
            + ", ".join(sorted(raw_contexts))
        )
    parsed_contexts: dict[str, tuple[int, int, bool]] = {}
    for context_name in sorted(raw_contexts):
        context_value = config_keys(
            raw_contexts[context_name],
            f"family.contexts.{context_name}",
            {"seq_len", "reference_batch_size", "document_masking"},
        )
        parsed_contexts[context_name] = (
            config_int(
                context_value["seq_len"],
                f"family.contexts.{context_name}.seq_len",
                minimum=1,
            ),
            config_int(
                context_value["reference_batch_size"],
                f"family.contexts.{context_name}.reference_batch_size",
                minimum=1,
            ),
            config_bool(
                context_value["document_masking"],
                f"family.contexts.{context_name}.document_masking",
            ),
        )
    raw_tiers = config_mapping(family["tiers"], "family.tiers")
    if not raw_tiers:
        raise ValueError("config.yaml family.tiers must define at least one tier")
    invalid_tiers = sorted(name for name in raw_tiers if not _TIER_NAME.fullmatch(name))
    if invalid_tiers:
        raise ValueError(
            "config.yaml family.tiers contains invalid name(s): "
            + ", ".join(invalid_tiers)
        )
    default_tier = family["default_tier"]
    if not isinstance(default_tier, str) or default_tier not in raw_tiers:
        raise ValueError("config.yaml family.default_tier must name a defined tier")
    if base_tier not in raw_tiers:
        raise ValueError(
            "config.yaml family.parameterization.base_tier must name a defined tier"
        )
    selected_family_tier = tier or default_tier
    if selected_family_tier not in raw_tiers:
        raise ValueError(
            f"unknown model tier {selected_family_tier!r}; expected "
            + ", ".join(sorted(raw_tiers))
        )
    parsed_tiers: dict[str, tuple[int, dict[str, Any]]] = {}
    for tier_name in sorted(raw_tiers):
        tier_value = config_keys(
            raw_tiers[tier_name],
            f"family.tiers.{tier_name}",
            {"parameters", "model"},
        )
        declared = config_int(
            tier_value["parameters"],
            f"family.tiers.{tier_name}.parameters",
            minimum=1,
        )
        tier_model = _parse_model(
            tier_value["model"], f"family.tiers.{tier_name}.model"
        )
        if tier_model["d_model"] // tier_model["heads"] != 64:
            raise ValueError(
                f"config.yaml family.tiers.{tier_name} must use 64-wide heads"
            )
        counted = _declared_family_parameter_count(tier_model)
        if declared != counted:
            raise ValueError(
                f"config.yaml family.tiers.{tier_name}.parameters is {declared:,}, "
                f"but this trainer counts {counted:,}"
            )
        parsed_tiers[tier_name] = (declared, tier_model)
    # The explicit width/depth anchor normally matches the smallest tier, but
    # controlled aspect-ratio candidates may retain the reference anchor while
    # changing that tier. The data-horizon anchor remains the candidate's own
    # 60m parameter count below, keeping fixed-TPP comparisons well defined.

    profiles = config_keys(top["profiles"], "profiles", set(_VALID_PROFILES))
    selected = config_keys(
        profiles[profile],
        f"profiles.{profile}",
        {"training", "model", "kernels", "optimizer", "evaluation", "logging"},
    )
    training = config_keys(
        selected["training"],
        f"profiles.{profile}.training",
        (
            {"batch_size", "seq_len", "sampling", "dtype", "steps"}
            if profile == "smoke"
            else {"tokens_per_parameter", "sampling", "dtype"}
        ),
    )
    if profile == "smoke":
        model = _parse_model(selected["model"], f"profiles.{profile}.model")
        selected_tier = "smoke"
        declared_parameters = None
        selected_parameterization = "standard"
        selected_base_width = model["d_model"]
        selected_base_depth = model["layers"]
        selected_depth_alpha = 0.0
        selected_init_std = 0.02
        selected_attention_scale = "inverse_sqrt_head_dim"
        selected_embeddings = "tied"
        selected_context_name = "smoke"
        selected_seq_len = config_int(
            training["seq_len"], f"profiles.{profile}.training.seq_len", minimum=1
        )
        selected_batch_size = config_int(
            training["batch_size"],
            f"profiles.{profile}.training.batch_size",
            minimum=1,
        )
        selected_document_masking = False
    else:
        if selected["model"] != "family_tier":
            raise ValueError(
                f"config.yaml profiles.{profile}.model must be 'family_tier'"
            )
        declared_parameters, model = parsed_tiers[selected_family_tier]
        selected_tier = selected_family_tier
        selected_parameterization = parameterization_name
        selected_base_width = base_width
        selected_base_depth = base_depth
        selected_depth_alpha = depth_alpha
        selected_init_std = init_std
        selected_attention_scale = attention_scale
        selected_embeddings = embeddings
        selected_context_name = selected_context
        (
            selected_seq_len,
            selected_batch_size,
            selected_document_masking,
        ) = parsed_contexts[selected_context]
    kernels = config_keys(
        selected["kernels"],
        f"profiles.{profile}.kernels",
        {"attention_backend", "loss_backend", "vocab_tile_size"},
    )
    optimizer = config_keys(
        selected["optimizer"],
        f"profiles.{profile}.optimizer",
        {
            "learning_rate",
            "min_lr_ratio",
            "warmup_ratio",
            "weight_decay",
            "adam_epsilon",
            "beta1",
            "beta2",
            "grad_clip",
        },
    )
    evaluation = config_keys(
        selected["evaluation"],
        f"profiles.{profile}.evaluation",
        {"eval_batches", "val_every", "val_probe_batches"},
    )
    logging = config_keys(
        selected["logging"],
        f"profiles.{profile}.logging",
        {"diagnostics_every", "log_every"},
    )
    prefix = f"profiles.{profile}"
    learning_rate = config_float(
        optimizer["learning_rate"], f"{prefix}.optimizer.learning_rate"
    )
    min_lr_ratio = config_float(
        optimizer["min_lr_ratio"], f"{prefix}.optimizer.min_lr_ratio"
    )
    weight_decay = config_float(
        optimizer["weight_decay"], f"{prefix}.optimizer.weight_decay"
    )
    adam_epsilon = config_float(
        optimizer["adam_epsilon"], f"{prefix}.optimizer.adam_epsilon"
    )
    warmup_ratio = config_float(
        optimizer["warmup_ratio"], f"{prefix}.optimizer.warmup_ratio"
    )
    beta1 = config_float(optimizer["beta1"], f"{prefix}.optimizer.beta1")
    beta2 = config_float(optimizer["beta2"], f"{prefix}.optimizer.beta2")
    grad_clip = config_float(optimizer["grad_clip"], f"{prefix}.optimizer.grad_clip")
    if learning_rate <= 0.0:
        raise ValueError(
            f"config.yaml {prefix}.optimizer.learning_rate must be positive"
        )
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(
            f"config.yaml {prefix}.optimizer.min_lr_ratio must be in [0, 1]"
        )
    if weight_decay < 0.0:
        raise ValueError(
            f"config.yaml {prefix}.optimizer.weight_decay must be nonnegative"
        )
    if adam_epsilon <= 0.0:
        raise ValueError(
            f"config.yaml {prefix}.optimizer.adam_epsilon must be positive"
        )
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(
            f"config.yaml {prefix}.optimizer.warmup_ratio must be in [0, 1)"
        )
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError(
            f"config.yaml {prefix}.optimizer beta values must be in [0, 1)"
        )
    if grad_clip < 0.0:
        raise ValueError(
            f"config.yaml {prefix}.optimizer.grad_clip must be nonnegative"
        )
    result = ExperimentProfile(
        schema_version=schema_version,
        source_sha256=source_sha256,
        name=profile,
        context_preset=selected_context_name,
        steps=(
            config_int(training["steps"], f"{prefix}.training.steps", minimum=1)
            if profile == "smoke"
            else None
        ),
        tokens_per_parameter=(
            config_float(
                training["tokens_per_parameter"],
                f"{prefix}.training.tokens_per_parameter",
            )
            if profile != "smoke"
            else None
        ),
        batch_size=selected_batch_size,
        seq_len=selected_seq_len,
        sampling=config_choice(
            training["sampling"],
            f"{prefix}.training.sampling",
            ("random_windows", "shuffled_epochs"),
        ),
        dtype_name=config_choice(
            training["dtype"], f"{prefix}.training.dtype", ("bfloat16", "float32")
        ),
        layers=int(model["layers"]),
        heads=int(model["heads"]),
        d_model=int(model["d_model"]),
        mlp_mult=int(model["mlp_mult"]),
        experts=int(model["experts"]),
        expert_top_k=int(model["expert_top_k"]),
        expert_mult=int(model["expert_mult"]),
        router_aux_coefficient=float(model["router_aux_coefficient"]),
        normalization=str(model["normalization"]),
        position_encoding=str(model["position_encoding"]),
        mlp_activation=str(model["mlp_activation"]),
        tier=selected_tier,
        declared_parameters=declared_parameters,
        base_parameters=parsed_tiers[base_tier][0],
        parameterization=selected_parameterization,
        base_width=selected_base_width,
        base_depth=selected_base_depth,
        depth_alpha=selected_depth_alpha,
        init_std=selected_init_std,
        attention_scale=selected_attention_scale,
        embeddings=selected_embeddings,
        vocab_size=int(model["vocab_size"]),
        semantic_vocab_size=int(model["semantic_vocab_size"]),
        attention_backend=config_choice(
            kernels["attention_backend"],
            f"{prefix}.kernels.attention_backend",
            ("dense", "jax_flash", "tpu_flash"),
        ),
        loss_backend=config_choice(
            kernels["loss_backend"],
            f"{prefix}.kernels.loss_backend",
            ("dense", "tiled"),
        ),
        vocab_tile_size=config_int(
            kernels["vocab_tile_size"], f"{prefix}.kernels.vocab_tile_size", minimum=1
        ),
        document_masking=selected_document_masking,
        learning_rate=learning_rate,
        min_lr_ratio=min_lr_ratio,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        adam_epsilon=adam_epsilon,
        beta1=beta1,
        beta2=beta2,
        grad_clip=grad_clip,
        eval_batches=config_int(
            evaluation["eval_batches"], f"{prefix}.evaluation.eval_batches", minimum=1
        ),
        val_every=config_int(
            evaluation["val_every"], f"{prefix}.evaluation.val_every", minimum=0
        ),
        val_probe_batches=config_int(
            evaluation["val_probe_batches"],
            f"{prefix}.evaluation.val_probe_batches",
            minimum=1,
        ),
        diagnostics_every=config_int(
            logging["diagnostics_every"],
            f"{prefix}.logging.diagnostics_every",
            minimum=0,
        ),
        log_every=config_int(
            logging["log_every"], f"{prefix}.logging.log_every", minimum=1
        ),
    )
    if result.tokens_per_parameter is not None and result.tokens_per_parameter <= 0.0:
        raise ValueError(
            f"config.yaml {prefix}.training.tokens_per_parameter must be positive"
        )
    if result.attention_backend != "dense" and result.dtype_name != "bfloat16":
        raise ValueError(
            f"config.yaml {prefix}.kernels.attention_backend "
            f"{result.attention_backend} requires training.dtype bfloat16"
        )
    tokens_per_step = result.batch_size * result.seq_len
    if profile == "official":
        validation_tokens = 10_485_760
        if validation_tokens % tokens_per_step:
            raise ValueError(
                f"config.yaml {prefix} batch_size * seq_len must divide the "
                f"official {validation_tokens:,}-prediction validation prefix"
            )
        required_eval_batches = validation_tokens // tokens_per_step
        if result.eval_batches != required_eval_batches:
            raise ValueError(
                f"config.yaml {prefix}.evaluation.eval_batches must be "
                f"{required_eval_batches} for the official validation prefix"
            )
    if result.val_every and result.val_probe_batches > result.eval_batches:
        raise ValueError(
            f"config.yaml {prefix}.evaluation.val_probe_batches must not exceed eval_batches"
        )
    return result


def load_experiment_profile(
    profile: str,
    requested_path: Path | None = None,
    *,
    tier: str | None = None,
    context: str | None = None,
) -> ExperimentProfile:
    if profile not in _VALID_PROFILES:
        raise ValueError(f"unknown experiment profile: {profile!r}")
    path = resolve_sibling_config_path(requested_path, CONFIG_PATH)
    mapping, source_sha256 = read_config_document(path)
    parsed = {
        name: _parse_experiment_profile(mapping, name, source_sha256, tier, context)
        for name in _VALID_PROFILES
    }
    return parsed[profile]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a decoder-only GPT with JAX. Static experiment settings come "
            "from config.yaml beside this entry script."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    run = parser.add_argument_group("run")
    run.add_argument(
        "--config",
        type=Path,
        default=None,
        help="experiment definition (must resolve to the config.yaml beside train.py)",
    )
    run.add_argument("--output-dir", type=Path, default=Path("runs/reference"))
    run.add_argument("--seed", type=int, default=1337)
    environment_tier = os.environ.get("RIG_TIER")
    run.add_argument(
        "--tier",
        default=environment_tier,
        help="model-family size tier; defaults to family.default_tier",
    )
    run.add_argument(
        "--context",
        default=None,
        help="named context preset; defaults to family.default_context",
    )
    run.add_argument(
        "--stop-after-step",
        type=positive_int,
        default=None,
        help=(
            "stop after this optimizer step while keeping the full schedule; "
            "requires --tokens-per-parameter so the horizon is unchanged"
        ),
    )
    run.add_argument(
        "--tokens-per-parameter",
        type=float,
        default=None,
        help="research budget rounded to the nearest complete global step",
    )
    environment_profile = os.environ.get("RIG_PROFILE")
    if environment_profile not in _VALID_PROFILES:
        environment_profile = None
    run.add_argument("--profile", choices=_VALID_PROFILES, default=environment_profile)
    run.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    run.add_argument("--print-plan", action="store_true", help=argparse.SUPPRESS)

    profiling = parser.add_argument_group("profiling")
    profiling.add_argument(
        "--xprof-dir",
        type=Path,
        default=None,
        help="write an XProf trace for a bounded training-step window",
    )
    profiling.add_argument(
        "--xprof-start-step",
        type=positive_int,
        default=None,
        help="first 1-based step to capture; required with --xprof-dir",
    )
    profiling.add_argument(
        "--xprof-steps",
        type=positive_int,
        default=None,
        help="number of consecutive steps to capture; required with --xprof-dir",
    )
    profiling.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help="XProf-only execution without evaluation, diagnostics, checkpoint, or result",
    )
    profiling.add_argument(
        "--omit-checkpoint",
        action="store_true",
        help=(
            "non-official research only: retain final validation and metrics but omit "
            "the parameter checkpoint"
        ),
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data",
        "--data-path",
        dest="data_path",
        type=Path,
        default=None,
        help="train file or directory containing discovered train/val shards",
    )
    data.add_argument(
        "--train-data",
        type=Path,
        action="append",
        default=[],
        help="explicit training shard; repeat for multiple shards",
    )
    data.add_argument(
        "--val-data",
        type=Path,
        action="append",
        default=[],
        help="explicit validation shard; repeat for multiple shards",
    )
    data.add_argument(
        "--data-dtype",
        choices=("uint8", "uint16", "uint32", "int32"),
        default="uint16",
        help="dtype for raw .bin token files",
    )
    data.add_argument("--val-fraction", type=float, default=0.05)
    data.add_argument(
        "--vocab-size", type=positive_int, default=None, help=argparse.SUPPRESS
    )
    data.add_argument(
        "--dataset-id", default=None, help="stable dataset identifier for records"
    )
    data.add_argument(
        "--tokenizer-id", default=None, help="stable tokenizer identifier for records"
    )
    data.add_argument(
        "--data-format",
        choices=("auto", "raw", "llmc"),
        default="auto",
        help="raw binaries or llm.c 256-int-header shards",
    )
    data.add_argument(
        "--downstream-manifest",
        type=Path,
        default=None,
        help="fresh10 manifest containing domain shard paths and document spans",
    )
    data.add_argument(
        "--downstream-root",
        type=Path,
        default=None,
        help="directory containing shards named by --downstream-manifest",
    )
    data.add_argument(
        "--downstream-data",
        action="append",
        default=[],
        metavar="DOMAIN=PATH",
        help="standalone downstream document; repeat paths and domains as needed",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument(
        "--base-learning-rate",
        type=float,
        default=None,
        help="research override for the transferable base learning rate",
    )
    optim.add_argument(
        "--batch-size",
        type=positive_int,
        default=None,
        help="research override for the global sequence batch",
    )
    optim.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="hardware bf16 peak for the whole mesh; enables an MFU estimate",
    )
    return parser


def validate_args(args: argparse.Namespace) -> ExperimentProfile:
    experiment = load_experiment_profile(
        selected_profile(args), args.config, tier=args.tier, context=args.context
    )
    if selected_profile(args) == "smoke" and args.context is not None:
        raise ValueError("--context is not applicable to the smoke profile")
    if args.tokens_per_parameter is not None and (
        not math.isfinite(args.tokens_per_parameter) or args.tokens_per_parameter <= 0.0
    ):
        raise ValueError("--tokens-per-parameter must be finite and positive")
    if args.base_learning_rate is not None and (
        not math.isfinite(args.base_learning_rate) or args.base_learning_rate <= 0.0
    ):
        raise ValueError("--base-learning-rate must be finite and positive")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.peak_tflops is not None and (
        not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0.0
    ):
        raise ValueError("--peak-tflops must be positive")
    if args.downstream_root is not None and args.downstream_manifest is None:
        raise ValueError("--downstream-root requires --downstream-manifest")
    if args.downstream_manifest is not None and args.downstream_data:
        raise ValueError(
            "--downstream-manifest and --downstream-data are mutually exclusive"
        )
    xprof_window_args = (args.xprof_start_step, args.xprof_steps)
    if args.xprof_dir is None:
        if any(value is not None for value in xprof_window_args):
            raise ValueError("--xprof-start-step and --xprof-steps require --xprof-dir")
        if args.diagnostic_mode:
            raise ValueError("--diagnostic-mode requires --xprof-dir")
    elif any(value is None for value in xprof_window_args):
        raise ValueError(
            "--xprof-dir requires both --xprof-start-step and --xprof-steps"
        )
    if args.omit_checkpoint and args.diagnostic_mode:
        raise ValueError(
            "--omit-checkpoint and --diagnostic-mode are mutually exclusive"
        )
    if args.omit_checkpoint and selected_profile(args) != "dev":
        raise ValueError("--omit-checkpoint is restricted to development research runs")
    if args.diagnostic_mode and (
        args.downstream_manifest is not None or args.downstream_data
    ):
        raise ValueError(
            "--diagnostic-mode cannot be combined with downstream evaluation data"
        )
    return experiment


def should_compile_evaluation(
    args: argparse.Namespace,
    config: Config,
    downstream_domains: Sequence[DownstreamDomain],
) -> bool:
    """Return whether this invocation can execute any validation workload."""

    return not args.diagnostic_mode or config.val_every > 0 or bool(downstream_domains)


def selected_profile(args: argparse.Namespace) -> str:
    return args.profile or "dev"


def resolve_config(
    args: argparse.Namespace,
    platform: str,
    vocab_size: int,
    experiment: ExperimentProfile | None = None,
) -> Config:
    profile = selected_profile(args)
    experiment = experiment or load_experiment_profile(
        profile, args.config, tier=args.tier, context=args.context
    )
    if experiment.name != profile:
        raise ValueError(
            f"resolved config profile {experiment.name!r} does not match {profile!r}"
        )
    if vocab_size != experiment.vocab_size:
        raise ValueError(
            "loaded dataset vocabulary does not match config.yaml: "
            f"dataset={vocab_size}, configured={experiment.vocab_size}"
        )
    batch_size = args.batch_size or experiment.batch_size
    seq_len = experiment.seq_len
    tokens_per_step = batch_size * seq_len
    requested_tpp = args.tokens_per_parameter or experiment.tokens_per_parameter
    early_stop = getattr(args, "stop_after_step", None)
    if profile == "smoke":
        if args.tokens_per_parameter is not None:
            raise ValueError("--tokens-per-parameter cannot override the smoke profile")
        if early_stop is not None:
            raise ValueError("--stop-after-step requires a fixed-TPP profile")
        if experiment.steps is None:
            raise AssertionError("smoke profile did not resolve a step count")
        steps = experiment.steps
    else:
        if requested_tpp is None:
            raise AssertionError("non-smoke profile did not resolve a TPP horizon")
        if experiment.declared_parameters is None:
            raise AssertionError("fixed-TPP profile has no declared parameter count")
        ideal_tokens = float(experiment.declared_parameters) * requested_tpp
        steps = max(1, int(math.floor(ideal_tokens / tokens_per_step + 0.5)))
    if early_stop is not None and requested_tpp is None:
        raise ValueError("--stop-after-step requires a fixed-TPP profile")
    if early_stop is not None and early_stop > steps:
        raise ValueError(
            f"--stop-after-step {early_stop:,} is past the {steps:,}-step "
            "horizon this configuration resolves to"
        )
    if profile == "official":
        validation_tokens = 10_485_760
        predictions_per_batch = batch_size * seq_len
        if validation_tokens % predictions_per_batch:
            raise ValueError(
                "official validation requires batch_size * seq_len to divide "
                f"{validation_tokens:,} exactly; got {predictions_per_batch:,}"
            )
        required_eval_batches = validation_tokens // predictions_per_batch
        if experiment.eval_batches != required_eval_batches:
            raise ValueError(
                "official config.yaml validation must cover exactly 10,485,760 "
                f"predictions; set eval_batches to {required_eval_batches}"
            )
        eval_batches = required_eval_batches
    else:
        eval_batches = experiment.eval_batches
    val_every = 0 if args.diagnostic_mode else experiment.val_every
    val_probe_batches = experiment.val_probe_batches
    if val_every > 0 and val_probe_batches > eval_batches:
        raise ValueError(
            "config.yaml val_probe_batches must not exceed the canonical evaluation batch "
            f"count ({eval_batches}); got {val_probe_batches}"
        )
    log_every = steps if args.diagnostic_mode else experiment.log_every
    diagnostics_every = 0 if args.diagnostic_mode else experiment.diagnostics_every
    dtype_name = experiment.dtype_name
    compute_dtype = jnp.bfloat16 if dtype_name == "bfloat16" else jnp.float32
    attention_backend = experiment.attention_backend
    if attention_backend != "dense" and platform != "tpu":
        raise ValueError(
            f"config.yaml attention_backend {attention_backend} requires a TPU runtime"
        )
    if attention_backend != "dense" and compute_dtype != jnp.bfloat16:
        raise ValueError(
            f"config.yaml attention_backend {attention_backend} currently requires "
            "dtype bfloat16"
        )
    override_values: list[tuple[str, int | str]] = []
    for name, value in (
        (
            "tokens_per_parameter_micros",
            (
                round(args.tokens_per_parameter * 1_000_000)
                if args.tokens_per_parameter is not None
                else None
            ),
        ),
        ("batch_size", args.batch_size),
        ("diagnostic_mode", 1 if args.diagnostic_mode else None),
    ):
        if value is not None:
            override_values.append((name, int(value)))
    if args.context is not None:
        override_values.append(("context", args.context))
    overrides = tuple(override_values)
    width_multiplier = experiment.d_model / float(experiment.base_width)
    depth_multiplier = experiment.layers / float(experiment.base_depth)
    batch_multiplier = batch_size / float(experiment.batch_size)
    achieved_tpp = (
        steps * tokens_per_step / float(experiment.declared_parameters)
        if experiment.declared_parameters is not None
        else None
    )
    # This project reanchors every TPP ladder. The multiplier captures only the
    # model-size-induced data growth within one fixed-TPP ladder; it deliberately
    # omits any cross-horizon TPP / TPP_0 factor.
    data_multiplier = (
        experiment.declared_parameters / float(experiment.base_parameters)
        if experiment.declared_parameters is not None
        else 1.0
    )
    base_learning_rate = (
        args.base_learning_rate
        if args.base_learning_rate is not None
        else experiment.learning_rate
    )
    warmup_steps = int(math.floor(steps * experiment.warmup_ratio + 0.5))
    if steps > 1:
        warmup_steps = min(warmup_steps, steps - 1)
    else:
        warmup_steps = 0
    return Config(
        steps=steps,
        stop_after_step=early_stop,
        document_masking=experiment.document_masking,
        experts=experiment.experts,
        expert_top_k=experiment.expert_top_k,
        expert_mult=experiment.expert_mult,
        router_aux_coefficient=experiment.router_aux_coefficient,
        batch_size=batch_size,
        seq_len=seq_len,
        sampling=experiment.sampling,
        layers=experiment.layers,
        heads=experiment.heads,
        d_model=experiment.d_model,
        mlp_mult=experiment.mlp_mult,
        normalization=experiment.normalization,
        position_encoding=experiment.position_encoding,
        mlp_activation=experiment.mlp_activation,
        tier=experiment.tier,
        declared_parameters=experiment.declared_parameters,
        base_parameters=experiment.base_parameters,
        parameterization=experiment.parameterization,
        base_width=experiment.base_width,
        base_depth=experiment.base_depth,
        depth_alpha=experiment.depth_alpha,
        init_std=experiment.init_std,
        attention_scale=experiment.attention_scale,
        embeddings=experiment.embeddings,
        width_multiplier=width_multiplier,
        depth_multiplier=depth_multiplier,
        data_multiplier=data_multiplier,
        batch_multiplier=batch_multiplier,
        target_tokens_per_parameter=requested_tpp,
        tokens_per_parameter=achieved_tpp,
        learning_rate=base_learning_rate,
        min_lr_ratio=experiment.min_lr_ratio,
        warmup_steps=warmup_steps,
        weight_decay=experiment.weight_decay,
        adam_epsilon=experiment.adam_epsilon,
        beta1=experiment.beta1,
        beta2=experiment.beta2,
        grad_clip=experiment.grad_clip,
        eval_batches=eval_batches,
        val_every=val_every,
        val_probe_batches=val_probe_batches,
        diagnostics_every=diagnostics_every,
        log_every=log_every,
        vocab_size=vocab_size,
        semantic_vocab_size=experiment.semantic_vocab_size,
        attention_backend=attention_backend,
        loss_backend=experiment.loss_backend,
        vocab_tile_size=experiment.vocab_tile_size,
        compute_dtype=compute_dtype,
        dtype_name=dtype_name,
        config_schema_version=experiment.schema_version,
        config_sha256=experiment.source_sha256,
        config_profile=experiment.name,
        context_preset=experiment.context_preset,
        config_overrides=overrides,
    )


def init_params(config: Config, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    d_model = config.d_model
    hidden = config.mlp_mult * d_model
    # Each expert is expert_mult wide; top_k of them fire, so active MLP FLOPs
    # match the dense mlp_mult when expert_mult * top_k == mlp_mult.
    expert_hidden = config.expert_mult * d_model
    hidden_scale = (
        config.init_std / math.sqrt(config.width_multiplier)
        if config.parameterization == "completep_fixed_tpp_v1"
        else 0.02
    )
    blocks: list[dict[str, np.ndarray]] = []
    for _ in range(config.layers):
        blocks.append(
            {
                "ln1_scale": np.ones((d_model,), dtype=np.float32),
                "qkv_w": normal(rng, (d_model, 3 * d_model), hidden_scale),
                "qkv_b": np.zeros((3 * d_model,), dtype=np.float32),
                "attn_w": normal(rng, (d_model, d_model), hidden_scale),
                "attn_b": np.zeros((d_model,), dtype=np.float32),
                "ln2_scale": np.ones((d_model,), dtype=np.float32),
                **(
                    {
                        # Router logits are a readout: E does not scale with
                        # width, so it follows the unembedding's rules. Small
                        # init keeps routing near-uniform at step one.
                        "router_w": normal(
                            rng, (d_model, config.experts), hidden_scale
                        ),
                        "expert_up_w": normal(
                            rng, (config.experts, d_model, expert_hidden), hidden_scale
                        ),
                        "expert_up_b": np.zeros(
                            (config.experts, expert_hidden), dtype=np.float32
                        ),
                        "expert_down_w": normal(
                            rng, (config.experts, expert_hidden, d_model), hidden_scale
                        ),
                        "expert_down_b": np.zeros(
                            (config.experts, d_model), dtype=np.float32
                        ),
                    }
                    if config.experts
                    else {
                        "mlp_up_w": normal(rng, (d_model, hidden), hidden_scale),
                        "mlp_up_b": np.zeros((hidden,), dtype=np.float32),
                        "mlp_down_w": normal(rng, (hidden, d_model), hidden_scale),
                        "mlp_down_b": np.zeros((d_model,), dtype=np.float32),
                    }
                ),
            }
        )
    result = {
        "token_embedding": normal(rng, (config.vocab_size, d_model), config.init_std),
        "blocks": blocks,
        "final_ln_scale": np.ones((d_model,), dtype=np.float32),
    }
    if config.embeddings == "untied":
        result["output_embedding"] = normal(
            rng,
            (config.vocab_size, d_model),
            config.init_std / config.width_multiplier,
        )
    return result


def contract_model_metadata(config: Config) -> dict[str, Any]:
    """Return the resolved model architecture metadata."""

    return {
        "layers": config.layers,
        "heads": config.heads,
        "d_model": config.d_model,
        "mlp_mult": config.mlp_mult,
        "normalization": config.normalization,
        "position_encoding": config.position_encoding,
        "mlp_activation": config.mlp_activation,
        "vocab_size": config.vocab_size,
        "semantic_vocab_size": config.semantic_vocab_size,
        "tied_embeddings": config.embeddings == "tied",
        "tier": config.tier,
        "parameterization": config.parameterization,
    }


def experiment_config_metadata(config: Config) -> dict[str, Any]:
    """Return stable source identity and the fully resolved experiment values."""

    return {
        "schema_version": config.config_schema_version,
        "path": CONFIG_FILENAME,
        "sha256": config.config_sha256,
        "profile": config.config_profile,
        "context_preset": config.context_preset,
        "overrides": dict(config.config_overrides),
        "resolved": {
            "training": {
                "steps": config.steps,
                "train_tokens": config.steps * config.batch_size * config.seq_len,
                "batch_size": config.batch_size,
                "seq_len": config.seq_len,
                "sampling": config.sampling,
                "dtype": config.dtype_name,
                "tokens_per_parameter": config.tokens_per_parameter,
                "target_tokens_per_parameter": config.target_tokens_per_parameter,
            },
            "model": contract_model_metadata(config),
            "kernels": {
                "attention_backend": config.attention_backend,
                "loss_backend": config.loss_backend,
                "vocab_tile_size": config.vocab_tile_size,
                "document_masking": config.document_masking,
            },
            "optimizer": {
                "learning_rate": config.learning_rate,
                "effective": effective_optimizer_metadata(config),
                "min_lr_ratio": config.min_lr_ratio,
                "warmup_steps": config.warmup_steps,
                "weight_decay": config.weight_decay,
                "adam_epsilon": config.adam_epsilon,
                "beta1": config.beta1,
                "beta2": config.beta2,
                "grad_clip": config.grad_clip,
            },
            "parameterization": {
                "name": config.parameterization,
                "base_width": config.base_width,
                "base_depth": config.base_depth,
                "width_multiplier": config.width_multiplier,
                "depth_multiplier": config.depth_multiplier,
                "ladder_data_multiplier": config.data_multiplier,
                "batch_ratio": config.batch_multiplier,
                "depth_alpha": config.depth_alpha,
                "init_std": config.init_std,
                "attention_scale": config.attention_scale,
                "embeddings": config.embeddings,
            },
            "evaluation": {
                "eval_batches": config.eval_batches,
                "val_every": config.val_every,
                "val_probe_batches": config.val_probe_batches,
            },
            "logging": {
                "diagnostics_every": config.diagnostics_every,
                "log_every": config.log_every,
            },
        },
    }


def resolved_plan_metadata(config: Config) -> dict[str, Any]:
    """Return the deterministic, data-independent execution contract."""

    tokens_per_step = config.batch_size * config.seq_len
    return {
        "schema_version": 2,
        "config_schema_version": config.config_schema_version,
        "config_sha256": config.config_sha256,
        "profile": config.config_profile,
        "context_preset": config.context_preset,
        "document_masking": config.document_masking,
        "tier": config.tier,
        "run_kind": (
            "smoke"
            if config.config_profile == "smoke"
            else ("diagnostic" if config.stop_after_step is not None else "full")
        ),
        "parameterization": config.parameterization,
        "weight_decay_policy": "weights_and_embeddings_only_v2",
        "declared_parameters": config.declared_parameters,
        "batch_size": config.batch_size,
        "sequence_length": config.seq_len,
        "tokens_per_step": tokens_per_step,
        "target_tokens_per_parameter": config.target_tokens_per_parameter,
        "achieved_tokens_per_parameter": config.tokens_per_parameter,
        "schedule_steps": config.steps,
        "stop_after_step": config.stop_after_step,
        "planned_tokens": config.steps * tokens_per_step,
        "expected_tokens": config.final_step * tokens_per_step,
        "base_learning_rate": config.learning_rate,
        "batch_ratio": config.batch_multiplier,
        "ladder_data_multiplier": config.data_multiplier,
    }


def checkpoint_metadata(
    config: Config, seed: int, attention_runtime: AttentionRuntime
) -> dict[str, Any]:
    """Describe this model well enough to reconstruct it from the weights.

    Travels inside checkpoint.npz. It stays here rather than in rig/ because
    naming a model's layers, activation, and parameterization is exactly what
    a recipe is for.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "configuration": experiment_config_metadata(config),
        "model": {
            "vocab_size": config.vocab_size,
            "semantic_vocab_size": config.semantic_vocab_size,
            "seq_len": config.seq_len,
            "layers": config.layers,
            "heads": config.heads,
            "d_model": config.d_model,
            "mlp_mult": config.mlp_mult,
            "normalization": config.normalization,
            "position_encoding": config.position_encoding,
            "mlp_activation": config.mlp_activation,
            "dtype": config.dtype_name,
            "attention_backend": config.attention_backend,
            "attention_tuning": attention_runtime_metadata(attention_runtime),
            "loss_backend": config.loss_backend,
            "vocab_tile_size": config.vocab_tile_size,
            "tied_embeddings": config.embeddings == "tied",
            "tier": config.tier,
            "parameterization": config.parameterization,
        },
    }


def implementation_metadata(
    config: Config, runtime: AttentionRuntime
) -> dict[str, Any]:
    """Return systems/kernel provenance that may vary in either track."""

    return {
        "attention_backend": config.attention_backend,
        "attention_tuning": attention_runtime_metadata(runtime),
        "loss_backend": config.loss_backend,
        "vocab_tile_size": config.vocab_tile_size,
        "weight_decay_policy": "weights_and_embeddings_only_v2",
        "context_preset": config.context_preset,
        "document_masking": config.document_masking,
        "configuration": experiment_config_metadata(config),
    }


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RouterStats:
    """What the routed blocks report back for logging and the auxiliary loss.

    Registered as a pytree because it is returned as the aux output of
    ``jax.value_and_grad(..., has_aux=True)``: an unregistered dataclass is an
    opaque leaf to JAX, so the tracers inside its fields would not be
    recognized as part of the traced computation and would dangle once the
    transformation exits -- an ``UnexpectedTracerError`` at first use, not at
    construction.
    """

    balance_loss: jax.Array
    # [layers, experts]: the fraction of assignments each expert received.
    load: jax.Array
    # [layers, 3]: entropy, top-1 gate, logit RMS, in ROUTER_SUMMARY_METRICS order.
    summary: jax.Array


def active_parameter_count(params: Any, config: Config) -> int:
    """Parameters a single token actually visits.

    A routed model stores ``experts`` copies of the MLP and visits
    ``expert_top_k`` of them, so its total is not what the tier declares. The
    ladder is defined by *active* parameters -- that is what makes a sparse
    tier comparable with the dense tier of the same name, and what makes the
    two equi-FLOP.
    """

    total = parameter_count(params)
    if not config.experts:
        return total
    width, hidden = config.d_model, config.expert_mult * config.d_model
    per_expert = width * hidden + hidden + hidden * width + width
    unvisited = config.experts - config.expert_top_k
    return total - config.layers * unvisited * per_expert


def expected_active_parameters(config: Config) -> int:
    """What the declared tier size becomes once the MLP is routed.

    Routing preserves the dense MLP's active *width* exactly, because
    ``expert_top_k * expert_mult == mlp_mult``. It adds two things and only
    two: the router projection, and one extra set of expert biases per
    additional expert a token visits. Both are named here rather than absorbed
    into a tolerance, so the check stays a check.
    """

    declared = config.declared_parameters
    if declared is None or not config.experts:
        return declared
    router = config.d_model * config.experts
    extra_biases = (config.expert_top_k - 1) * config.d_model
    return declared + config.layers * (router + extra_biases)


def routed_mlp_local(
    x: jax.Array,
    router_w: jax.Array,
    up_w: jax.Array,
    up_b: jax.Array,
    down_w: jax.Array,
    down_b: jax.Array,
    *,
    experts: int,
    top_k: int,
    dtype: Any,
    axis_name: str | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Top-k routed experts via a grouped matmul, on one device's tokens.

    Returns the block output, the mean router probability per expert, and the
    fraction of assignments each expert received -- the two statistics the
    balance loss is built from. Both are averaged across the data axis when one
    is given, so the loss sees global load rather than one shard's view.

    Dropless by construction: ``group_sizes`` is data while the total
    ``tokens * top_k`` is static, so every assignment is served and no capacity
    factor exists. Nothing couples one token's routing to another's, which is
    what keeps the model causal.
    """

    batch, length, width = x.shape
    flat = x.reshape(batch * length, width)

    logits = jnp.einsum(
        "md,de->me",
        flat,
        router_w.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    probabilities = jax.nn.softmax(logits, axis=-1)
    chosen_logits, chosen = jax.lax.top_k(logits, top_k)
    gate = jax.nn.softmax(chosen_logits, axis=-1)

    assignments = chosen.reshape(-1)
    order = jnp.argsort(assignments, stable=True)
    sorted_assignments = assignments[order]
    counts = jax.nn.one_hot(assignments, experts, dtype=jnp.int32).sum(0)
    rows = jnp.repeat(jnp.arange(batch * length), top_k)[order]

    # Pallas lowers only in interpret mode off TPU, which is how the routed
    # path stays checkable against a dense reference on CPU.
    interpret = jax.default_backend() != "tpu"
    grouped = flat[rows].astype(dtype)
    hidden = megablox.gmm(grouped, up_w.astype(dtype), counts, interpret=interpret)
    hidden = hidden + up_b[sorted_assignments].astype(hidden.dtype)
    hidden = jax.nn.gelu(hidden, approximate=True)
    out = megablox.gmm(
        hidden.astype(dtype), down_w.astype(dtype), counts, interpret=interpret
    )
    out = out + down_b[sorted_assignments].astype(out.dtype)

    weighted = out * gate.reshape(-1)[order][:, None].astype(out.dtype)
    combined = jnp.zeros((batch * length, width), out.dtype).at[rows].add(weighted)

    mean_probability = probabilities.mean(axis=0)
    load = counts.astype(jnp.float32) / jnp.float32(batch * length * top_k)
    # Standard routing diagnostics, all reduced to one number per layer. Every
    # one is a by-product of tensors this function already has, so the cost is
    # a handful of reductions rather than a second pass.
    entropy = -jnp.sum(probabilities * jnp.log(probabilities + 1.0e-9), axis=-1).mean()
    top1_gate = gate.max(axis=-1).mean()
    # The mean square crosses the collective, not the root of it: averaging
    # per-device roots is not the root of the global average, which would make
    # this number depend on how many devices the run happened to use.
    logit_mean_square = jnp.mean(jnp.square(logits))
    summary = jnp.stack((entropy, top1_gate, logit_mean_square))
    if axis_name is not None:
        mean_probability = jax.lax.pmean(mean_probability, axis_name)
        load = jax.lax.pmean(load, axis_name)
        summary = jax.lax.pmean(summary, axis_name)
    summary = summary.at[2].set(jnp.sqrt(summary[2]))
    return combined.reshape(batch, length, width), mean_probability, load, summary


def make_mesh_routed_mlp(config: Config, mesh: Mesh) -> Any:
    """Wrap the routed MLP in an explicit data-sharded boundary.

    The grouped matmul is a Mosaic kernel, so an outer jit cannot partition it
    automatically -- the same constraint that forces make_mesh_attention to
    exist. Experts stay replicated (plan phase 1): each device routes its own
    tokens among all of them, so there are no expert collectives, only the two
    tiny mean-reductions that give the balance loss a global view.
    """

    if not config.experts:
        return None
    batch_partition = P("data", None, None)
    replicated = P()
    local = functools.partial(
        routed_mlp_local,
        experts=config.experts,
        top_k=config.expert_top_k,
        dtype=config.compute_dtype,
        axis_name="data",
    )
    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            batch_partition,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
        ),
        out_specs=(batch_partition, replicated, replicated, replicated),
        check_vma=False,
    )


def load_balance_loss(mean_probability: jax.Array, load: jax.Array) -> jax.Array:
    """Switch-style auxiliary loss: E * sum_i (f_i * P_i).

    Both arguments are per-expert vectors of length E: ``load`` is the realized
    fraction of assignments an expert received and ``mean_probability`` the
    router's mean probability for it, each already averaged over every token on
    every device. Minimized at 1.0 when both are uniform, and E when one expert
    takes everything.

    This *encourages* balance; it never enforces it, because any rule that
    equalized loads would have to couple one token's routing to another's and
    break causality.

    Both must arrive already reduced over tokens. Passing the raw ``[tokens, E]``
    probability matrix here would average it to a scalar, which makes the whole
    term collapse to the constant 1.0 with no gradient to the router at all --
    a failure that trains happily and simply never balances.
    """

    if mean_probability.ndim != 1 or load.ndim != 1:
        raise ValueError(
            "load_balance_loss takes per-expert vectors already reduced over "
            f"tokens, got shapes {mean_probability.shape} and {load.shape}"
        )
    return jnp.float32(load.shape[0]) * jnp.sum(load * mean_probability)


def gpt_hidden(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[jax.Array, "RouterStats | None"]:
    """Return final token representations, and router statistics when routed."""

    dtype = config.compute_dtype
    batch, length = tokens.shape
    del batch
    x = params["token_embedding"][tokens].astype(dtype)
    head_dim = config.d_model // config.heads
    if config.attention_backend != "dense":
        # Direct construction keeps this function convenient for single-device
        # tests. Multi-device training supplies an explicit shard_map wrapper;
        # Mosaic kernels cannot be partitioned automatically by an outer jit.
        attention = attention_fn or make_causal_attention(
            AttentionConfig(
                backend=config.attention_backend,
                tiles=select_attention_tiles(
                    sequence=length, head_dim=head_dim, training=True
                ),
                softmax_scale=attention_softmax_scale(config.attention_scale, head_dim),
            )
        )
        causal = None
    else:
        attention = None
        causal = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))[None, None, :, :]

    segments = (
        document_segments(tokens, config.document_boundary_token)
        if config.document_masking
        else None
    )
    router_losses: list[jax.Array] = []
    router_loads: list[jax.Array] = []
    router_summaries: list[jax.Array] = []

    for block in params["blocks"]:
        residual = x
        x_norm = rms_norm(x, block["ln1_scale"], dtype)
        qkv = linear(x_norm, block["qkv_w"], block["qkv_b"], dtype)
        query, key, value = jnp.split(qkv, 3, axis=-1)
        query = query.reshape(tokens.shape[0], length, config.heads, head_dim)
        key = key.reshape(tokens.shape[0], length, config.heads, head_dim)
        value = value.reshape(tokens.shape[0], length, config.heads, head_dim)
        query = apply_rotary(query)
        key = apply_rotary(key)
        if attention is not None:
            attended = attention(
                jnp.transpose(query, (0, 2, 1, 3)),
                jnp.transpose(key, (0, 2, 1, 3)),
                jnp.transpose(value, (0, 2, 1, 3)),
                *((segments,) if segments is not None else ()),
            )
            attended = jnp.transpose(attended, (0, 2, 1, 3))
        else:
            scores = jnp.einsum("bthd,bshd->bhts", query, key)
            scores = scores.astype(jnp.float32) * attention_softmax_scale(
                config.attention_scale, head_dim
            )
            visible = causal
            if segments is not None:
                same = segments[:, None, :, None] == segments[:, None, None, :]
                visible = jnp.logical_and(visible, same)
            scores = jnp.where(visible, scores, jnp.finfo(jnp.float32).min)
            probabilities = jax.nn.softmax(scores, axis=-1).astype(dtype)
            attended = jnp.einsum("bhts,bshd->bthd", probabilities, value)
        attended = attended.reshape(tokens.shape[0], length, config.d_model)
        x = residual + (config.depth_multiplier ** (-config.depth_alpha)) * linear(
            attended, block["attn_w"], block["attn_b"], dtype
        )

        residual = x
        x_norm = rms_norm(x, block["ln2_scale"], dtype)
        if config.experts:
            routed = routed_fn or functools.partial(
                routed_mlp_local,
                experts=config.experts,
                top_k=config.expert_top_k,
                dtype=dtype,
                axis_name=None,
            )
            mlp_out, mean_probability, load, summary = routed(
                x_norm,
                block["router_w"],
                block["expert_up_w"],
                block["expert_up_b"],
                block["expert_down_w"],
                block["expert_down_b"],
            )
            router_losses.append(load_balance_loss(mean_probability, load))
            router_loads.append(load)
            router_summaries.append(summary)
        else:
            hidden = linear(x_norm, block["mlp_up_w"], block["mlp_up_b"], dtype)
            hidden = jax.nn.gelu(hidden, approximate=True)
            mlp_out = linear(hidden, block["mlp_down_w"], block["mlp_down_b"], dtype)
        x = residual + (config.depth_multiplier ** (-config.depth_alpha)) * mlp_out

    hidden_state = rms_norm(x, params["final_ln_scale"], dtype)
    router = (
        RouterStats(
            balance_loss=jnp.mean(jnp.stack(router_losses)),
            load=jnp.stack(router_loads),
            summary=jnp.stack(router_summaries),
        )
        if config.experts
        else None
    )
    return hidden_state, router


def gpt_logits(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> jax.Array:
    return gpt_logits_and_router(params, tokens, config, attention_fn, routed_fn)[0]


def gpt_logits_and_router(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[jax.Array, "RouterStats | None"]:
    """Logits, and the router statistics the balance loss needs.

    Exists because discarding the statistics here is invisible: the model still
    trains, the loss still falls, and the auxiliary term is simply never
    applied. The dense loss backend did exactly that.
    """

    x, router = gpt_hidden(params, tokens, config, attention_fn, routed_fn)
    output_embedding = params.get("output_embedding", params["token_embedding"])
    logits = jnp.einsum(
        "btd,vd->btv",
        x,
        output_embedding.astype(config.compute_dtype),
    ).astype(jnp.float32)
    return logits, router


def cross_entropy(
    params: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> jax.Array:
    """Training objective. For routed models this includes the balance loss.

    The auxiliary term is deliberately *not* part of any reported loss: a run
    that balances well and models badly must not look like the reverse, so
    ``router.load_balance_loss`` is logged as its own metric.
    """

    if config.loss_backend == "tiled":
        hidden, router = gpt_hidden(params, x, config, attention_fn, routed_fn)
        loss = tiled_tied_cross_entropy(
            hidden,
            params.get("output_embedding", params["token_embedding"]),
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    else:
        logits, router = gpt_logits_and_router(
            params, x, config, attention_fn, routed_fn
        )
        logits = logits[..., : config.semantic_vocab_size]
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(log_probabilities, y[..., None], axis=-1)
        loss = -jnp.mean(selected, dtype=jnp.float32)
    if router is not None and config.router_aux_coefficient:
        loss = loss + config.router_aux_coefficient * router.balance_loss
    return loss


def cross_entropy_and_router(
    params: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[jax.Array, "RouterStats | None"]:
    """The training objective, plus the balance loss and per-layer expert load.

    Carried out of the update as ``value_and_grad`` aux so logging costs an
    already-computed array rather than a second forward pass. Dense models
    return ``None`` and log nothing.
    """

    if not config.experts:
        return cross_entropy(params, x, y, config, attention_fn, routed_fn), None

    hidden, router = gpt_hidden(params, x, config, attention_fn, routed_fn)
    if config.loss_backend == "tiled":
        loss = tiled_tied_cross_entropy(
            hidden,
            params.get("output_embedding", params["token_embedding"]),
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    else:
        output_embedding = params.get("output_embedding", params["token_embedding"])
        logits = jnp.einsum(
            "btd,vd->btv", hidden, output_embedding.astype(config.compute_dtype)
        ).astype(jnp.float32)[..., : config.semantic_vocab_size]
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(log_probabilities, y[..., None], axis=-1)
        loss = -jnp.mean(selected, dtype=jnp.float32)

    assert router is not None
    if config.router_aux_coefficient:
        loss = loss + config.router_aux_coefficient * router.balance_loss
    return loss, router


def learning_rate(step: jax.Array, config: Config) -> jax.Array:
    step_float = step.astype(jnp.float32)
    if config.warmup_steps:
        warmup = jnp.minimum(1.0, step_float / float(config.warmup_steps))
    else:
        warmup = jnp.asarray(1.0, dtype=jnp.float32)
    decay_span = max(1, config.steps - config.warmup_steps)
    progress = jnp.clip(
        (step_float - float(config.warmup_steps)) / float(decay_span), 0.0, 1.0
    )
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    multiplier = config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine
    horizon_scale = math.sqrt(config.batch_multiplier / config.data_multiplier)
    return (
        jnp.asarray(config.learning_rate * horizon_scale, jnp.float32)
        * warmup
        * multiplier
    )


def init_optimizer(params: Any, config: Config) -> dict[str, Any]:
    zeros = jax.tree_util.tree_map(lambda value: np.zeros_like(value), params)
    # Keeping the small scalar history on-device avoids a host synchronization
    # on every step. It is copied once, after the synchronized timing boundary.
    # A routed run widens it by the routing columns so those are recorded every
    # step too -- the end-of-run rewrite supersedes the sampled rows, so
    # anything absent here is discarded no matter how often it was appended.
    #
    # The width is derived from config rather than taken as a parameter so
    # there is exactly one place that can get it wrong: a caller that forgot
    # to pass router_row_width(config) here once shipped a run whose history
    # buffer was too narrow for the row train_step tried to write into it,
    # and the run never reached the first optimizer step.
    history = np.zeros((config.steps, 3 + router_row_width(config)), dtype=np.float32)
    return {
        "step": np.asarray(0, dtype=np.int32),
        "m": zeros,
        "v": zeros,
        "history": history,
    }


def weight_decay_mask(params: Any) -> Any:
    """Select AdamW decay from parameter roles, never from array rank.

    Expert-stacked biases are rank two, so shape is not a reliable indication
    that a leaf is a weight.  Parameter names are part of this recipe's
    optimizer contract; failing closed also makes a newly introduced role ask
    for an explicit decay decision.
    """

    def decay_for_path(path: tuple[Any, ...], _value: Any) -> bool:
        name = getattr(path[-1], "key", None) if path else None
        if not isinstance(name, str):
            raise ValueError(f"cannot classify unnamed parameter leaf at {path!r}")
        if name in {"token_embedding", "output_embedding"} or name.endswith("_w"):
            return True
        if name.endswith(("_b", "_bias", "_scale")):
            return False
        raise ValueError(f"weight-decay policy has no rule for parameter {name!r}")

    return jax.tree_util.tree_map_with_path(decay_for_path, params)


def optimizer_hyperparameter_trees(
    params: Mapping[str, Any], config: Config
) -> tuple[Any, Any, Any]:
    """Return Complete(d)P-inspired LR, epsilon, and decay multipliers per tensor.

    Input/output layers and the residual backbone are intentionally distinct.
    Complete(d)P corrects CompleteP's input-embedding epsilon to ``1 / m_N``;
    the unembedding epsilon remains unscaled after its forward multiplier is
    absorbed into initialization and learning rate.
    """

    if config.parameterization != "completep_fixed_tpp_v1":
        ones = jax.tree_util.tree_map(lambda _: 1.0, params)
        return ones, ones, ones

    width = config.width_multiplier
    depth = config.depth_multiplier
    alpha = config.depth_alpha
    hidden_lr = width**-1 * depth ** (alpha - 1.0)
    hidden_vector_lr = depth ** (alpha - 1.0)
    hidden_epsilon = width**-1 * depth**-alpha

    lr_blocks: list[dict[str, float]] = []
    epsilon_blocks: list[dict[str, float]] = []
    decay_blocks: list[dict[str, float]] = []
    for block in params["blocks"]:
        lr_blocks.append(
            {
                name: (hidden_lr if name.endswith("_w") else hidden_vector_lr)
                for name in block
            }
        )
        epsilon_blocks.append({name: hidden_epsilon for name in block})
        decay_blocks.append(
            {name: (width if name.endswith("_w") else 1.0) for name in block}
        )

    lr_tree: dict[str, Any] = {
        "token_embedding": 1.0,
        "blocks": lr_blocks,
        "final_ln_scale": 1.0,
    }
    epsilon_tree: dict[str, Any] = {
        "token_embedding": width**-1,
        "blocks": epsilon_blocks,
        "final_ln_scale": 1.0,
    }
    decay_tree: dict[str, Any] = {
        "token_embedding": 1.0,
        "blocks": decay_blocks,
        "final_ln_scale": 1.0,
    }
    if "output_embedding" in params:
        # Complete(d)P absorbs the old 1/m_N output multiplier into output
        # initialization and learning rate. Its width-scaled decay keeps the
        # actual AdamW shrink invariant.
        lr_tree["output_embedding"] = width**-1
        epsilon_tree["output_embedding"] = 1.0
        decay_tree["output_embedding"] = width
    return lr_tree, epsilon_tree, decay_tree


def effective_adam_betas(config: Config) -> tuple[float, float]:
    ratio = config.batch_multiplier / config.data_multiplier
    beta1 = 1.0 - (1.0 - config.beta1) * ratio
    beta2 = 1.0 - (1.0 - config.beta2) * ratio
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError(
            "fixed-TPP batch/data scaling produced invalid Adam momenta; "
            "use a closer transfer base"
        )
    return beta1, beta2


def effective_optimizer_metadata(config: Config) -> dict[str, float]:
    """Return the fixed-TPP hybrid's global horizon/batch scalars."""

    beta1, beta2 = effective_adam_betas(config)
    ratio = config.batch_multiplier / config.data_multiplier
    return {
        "global_peak_learning_rate": config.learning_rate * math.sqrt(ratio),
        "adam_epsilon_horizon_multiplier": math.sqrt(1.0 / ratio),
        "weight_decay_horizon_multiplier": math.sqrt(ratio),
        "beta1": beta1,
        "beta2": beta2,
    }


def _apply_training_update(
    params: Any,
    optimizer: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    decay_mask: Any | None = None,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, jax.Array], Any]:
    """Apply one ordinary update and also return the raw, pre-clip gradient.

    Both the ordinary and sparse-diagnostic executables use this exact function.
    Diagnostics therefore do not substitute a different optimizer formula.
    """

    if decay_mask is None:
        decay_mask = weight_decay_mask(params)
    lr_multipliers, epsilon_multipliers, decay_multipliers = (
        optimizer_hyperparameter_trees(params, config)
    )
    beta1, beta2 = effective_adam_betas(config)
    (loss, router_aux), gradients = jax.value_and_grad(
        lambda candidate: cross_entropy_and_router(
            candidate, x, y, config, attention_fn, routed_fn
        ),
        has_aux=True,
    )(params)
    gradients = jax.tree_util.tree_map(lambda grad: grad.astype(jnp.float32), gradients)
    raw_gradients = gradients
    squared_norms = [
        jnp.sum(jnp.square(grad)) for grad in jax.tree_util.tree_leaves(gradients)
    ]
    grad_norm = jnp.sqrt(sum(squared_norms))
    clip_scale = (
        jnp.minimum(1.0, config.grad_clip / (grad_norm + 1.0e-6))
        if config.grad_clip > 0.0
        else jnp.asarray(1.0, dtype=jnp.float32)
    )
    gradients = jax.tree_util.tree_map(lambda grad: grad * clip_scale, gradients)

    step = optimizer["step"] + jnp.asarray(1, dtype=jnp.int32)
    lr = learning_rate(step, config)
    m = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        optimizer["m"],
        gradients,
    )
    v = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        optimizer["v"],
        gradients,
    )
    bias_correction1 = 1.0 - beta1 ** step.astype(jnp.float32)
    bias_correction2 = 1.0 - beta2 ** step.astype(jnp.float32)
    epsilon_horizon_scale = math.sqrt(config.data_multiplier / config.batch_multiplier)
    decay_horizon_scale = math.sqrt(config.batch_multiplier / config.data_multiplier)

    def update(
        parameter: jax.Array,
        first: jax.Array,
        second: jax.Array,
        should_decay: bool,
        lr_multiplier: float,
        epsilon_multiplier: float,
        decay_multiplier: float,
    ) -> jax.Array:
        epsilon = config.adam_epsilon * epsilon_horizon_scale * epsilon_multiplier
        adam = (first / bias_correction1) / (
            jnp.sqrt(second / bias_correction2) + epsilon
        )
        decay = (
            config.weight_decay * decay_horizon_scale * decay_multiplier * parameter
            if should_decay
            else 0.0
        )
        return parameter - lr * lr_multiplier * (adam + decay)

    params = jax.tree_util.tree_map(
        update,
        params,
        m,
        v,
        decay_mask,
        lr_multipliers,
        epsilon_multipliers,
        decay_multipliers,
    )
    routing = router_row(router_aux)
    history_row = jnp.concatenate(
        (jnp.stack((loss, lr, grad_norm)).astype(jnp.float32), routing)
    )
    history = optimizer["history"].at[step - 1].set(history_row)
    return (
        params,
        {"step": step, "m": m, "v": v, "history": history},
        {
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": lr,
            # Empty for a dense model. The shape is static either way, so
            # this does not vary the executable.
            "router_row": routing,
        },
        raw_gradients,
    )


def train_step(
    params: Any,
    optimizer: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    decay_mask: Any | None = None,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, jax.Array]]:
    params, optimizer, metrics, _ = _apply_training_update(
        params, optimizer, x, y, config, decay_mask, attention_fn, routed_fn
    )
    return params, optimizer, metrics


def diagnostic_scopes(
    tree: Mapping[str, Any],
) -> tuple[tuple[str, int | None, tuple[Any, ...]], ...]:
    """Group a parameter-shaped tree into stable logical report scopes."""

    embeddings = tuple(jax.tree_util.tree_leaves(tree["token_embedding"]))
    blocks = tuple(
        (
            "block",
            layer,
            tuple(jax.tree_util.tree_leaves(block)),
        )
        for layer, block in enumerate(tree["blocks"])
    )
    final_norm = tuple(jax.tree_util.tree_leaves(tree["final_ln_scale"]))
    output = (
        ("overall", None, tuple(jax.tree_util.tree_leaves(tree))),
        ("embeddings", None, embeddings),
        *blocks,
        ("final_norm", None, final_norm),
    )
    if "output_embedding" in tree:
        output = (
            output[0],
            output[1],
            (
                "unembedding",
                None,
                tuple(jax.tree_util.tree_leaves(tree["output_embedding"])),
            ),
            *output[2:],
        )
    return output


def router_row(router: "RouterStats | None") -> jax.Array:
    """Flatten the router statistics into the order training_log_columns names.

    Built on device and stored in the optimizer history, so every step is
    recorded rather than only the sampled ones, and so the live rows and the
    authoritative end-of-run rewrite are necessarily the same numbers in the
    same order. Empty for a dense model.
    """

    if router is None:
        return jnp.zeros((0,), jnp.float32)
    load, summary = router.load, router.summary
    per_layer = jnp.concatenate(
        [
            jnp.concatenate((summary[layer], load[layer]))
            for layer in range(load.shape[0])
        ]
    )
    return jnp.concatenate(
        (
            jnp.stack((router.balance_loss, load.max(), load.min())),
            summary.mean(axis=0),
            # Per layer: the three summary statistics, then the whole load
            # vector. Per-expert load is the exact distribution, so max, min,
            # and any histogram of it are derivable and none are stored.
            per_layer,
        )
    ).astype(jnp.float32)


def router_row_width(config: Config) -> int:
    """How many columns router_row emits, for sizing the history buffer."""

    if not config.experts:
        return 0
    return 6 + config.layers * (len(ROUTER_SUMMARY_METRICS) + config.experts)


def diagnostic_scope_metadata(
    params: Mapping[str, Any],
) -> tuple[tuple[str, int | None, int], ...]:
    """Return scope labels and exact element counts without device work."""

    return tuple(
        (scope, layer, sum(int(value.size) for value in leaves))
        for scope, layer, leaves in diagnostic_scopes(params)
    )


def _diagnostic_stat_vector(values: Sequence[jax.Array]) -> jax.Array:
    """Return norms and stable two-pass centered moments for several arrays."""

    values32 = tuple(value.astype(jnp.float32) for value in values)
    count = sum(int(value.size) for value in values32)
    if count <= 0:  # pragma: no cover - model scopes are statically nonempty
        raise ValueError("diagnostic scope cannot be empty")
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    total = sum((jnp.sum(value) for value in values32), zero)
    mean = total / float(count)

    # The mean is completed before the centered reduction, rather than deriving
    # variance and higher moments from cancellation-prone raw power sums.
    l1_sum = sum((jnp.sum(jnp.abs(value)) for value in values32), zero)
    square_sum = sum((jnp.sum(jnp.square(value)) for value in values32), zero)
    variance_sum = sum((jnp.sum(jnp.square(value - mean)) for value in values32), zero)
    third_sum = sum((jnp.sum(jnp.power(value - mean, 3)) for value in values32), zero)
    fourth_sum = sum((jnp.sum(jnp.power(value - mean, 4)) for value in values32), zero)
    return jnp.stack(
        (
            l1_sum,
            jnp.sqrt(jnp.maximum(square_sum, zero)),
            mean,
            jnp.sqrt(jnp.maximum(variance_sum / float(count), zero)),
            third_sum / float(count),
            fourth_sum / float(count),
        )
    ).astype(jnp.float32)


def diagnostic_values(
    params_before: Mapping[str, Any],
    raw_gradients: Mapping[str, Any],
    params_after: Mapping[str, Any],
) -> jax.Array:
    """Return ``[scope, family, stat]`` sparse diagnostic values.

    ``param`` observes the parameter after this step, so the final point exactly
    matches the checkpoint. ``grad`` is the raw gradient before global clipping.
    ``update`` is the signed actual delta ``params_after - params_before``,
    including clipping, AdamW, and decay.
    """

    updates = jax.tree_util.tree_map(
        lambda after, before: after - before, params_after, params_before
    )
    family_scopes = tuple(
        diagnostic_scopes(tree) for tree in (params_after, raw_gradients, updates)
    )
    scope_count = len(family_scopes[0])
    return jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    _diagnostic_stat_vector(family_scopes[family][scope][2])
                    for family in range(len(DIAGNOSTIC_FAMILIES))
                )
            )
            for scope in range(scope_count)
        )
    )


def diagnostic_train_step(
    params: Any,
    optimizer: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    decay_mask: Any | None = None,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, jax.Array], jax.Array]:
    """Run the same update as :func:`train_step` and emit sparse statistics."""

    params_before = params
    params, optimizer, metrics, raw_gradients = _apply_training_update(
        params, optimizer, x, y, config, decay_mask, attention_fn, routed_fn
    )
    values = diagnostic_values(params_before, raw_gradients, params)
    return params, optimizer, metrics, values


def eval_step(
    params: Any,
    x: jax.Array,
    y: jax.Array,
    mask: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[jax.Array, jax.Array]:
    """Return a loss sum and exact target count for fixed-shape masked eval."""

    if config.loss_backend == "tiled":
        hidden, _ = gpt_hidden(params, x, config, attention_fn, routed_fn)
        losses = tiled_tied_cross_entropy_losses(
            hidden,
            params.get("output_embedding", params["token_embedding"]),
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    else:
        logits = gpt_logits(params, x, config, attention_fn, routed_fn)[
            ..., : config.semantic_vocab_size
        ]
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(log_probabilities, y[..., None], axis=-1)[..., 0]
        losses = -selected
    mask = mask.astype(jnp.float32)
    return (
        jnp.sum(losses * mask, dtype=jnp.float32),
        jnp.sum(mask, dtype=jnp.float32),
    )


def traced_flops(config: Config, params: Mapping[str, Any]) -> FlopBreakdown:
    """Count one training step's algorithmic FLOPs by tracing the model.

    Nothing executes and nothing is allocated: ``jax.make_jaxpr`` builds the
    graph from shapes alone. The count therefore follows the architecture
    automatically -- change the depth, width, head count, or the shape of a
    block and this number moves with it, with no formula to maintain.

    A single sequence is traced and the result divided by ``seq_len``. Every
    term is linear in the batch dimension (attention included, which is
    quadratic in sequence but linear in batch), so one sequence determines
    the per-token cost; ``test_flops_are_linear_in_batch`` pins that down.

    ADDING A COMPONENT
    ------------------
    Ordinary blocks built from matmuls need nothing: they are counted from
    their traced shapes. Two cases do need attention, and both announce
    themselves in ``breakdown.warnings`` rather than failing quietly:

    * A new opaque kernel (anything built with ``pallas_call``) is invisible
      to the tracer. Register its cost with
      ``rules.with_kernel("<kernel name>", rule)`` in ``rig.flops``.
    * A component whose real cost differs from its traced cost -- sparsity
      being the usual reason -- must say so. A mixture-of-experts written as
      "compute every expert, then mask to top-k" contains the full dense work
      in its graph and will be billed for all of it, because the tracer sees
      real multiplications and cannot know a mask discards them. Wrap the
      component in a named ``jax.jit`` and register
      ``rules.with_scope("<name>", rule)``; the walker then bills the rule
      and does not descend. This is the one case no warning can catch, since
      nothing about the graph looks unusual.

    See ``docs/FLOPS.md`` for the full checklist.
    """

    tokens = jnp.zeros((1, config.seq_len), jnp.int32)
    targets = jnp.zeros((1, config.seq_len), jnp.int32)

    def loss(trainable: Mapping[str, Any]) -> jax.Array:
        return cross_entropy(trainable, tokens, targets, config)

    return count_training_flops(loss, params, rules=default_rules())


# How many diagnostic captures may sit on the accelerator before being pulled
# to the host. Bounded on purpose: without a cap this list grows with the run,
# holding one small device allocation per capture until the very end, and a
# preempted job loses every one of them.


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    experiment = validate_args(args)
    process_index, process_count = initialize_distributed_runtime()
    is_controller = is_controller_process(process_index)
    console = Console(args.color, active=is_controller)
    console.banner()
    profile = selected_profile(args)
    using_builtin_data = (
        args.data_path is None and not args.train_data and not args.val_data
    )

    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices")
    validate_official_topology(profile, devices)
    platform = devices[0].platform
    if profile == "smoke" and platform != "cpu":
        # Smoke remains tiny on accelerators too; this is informational only.
        console.phase("Smoke configuration", f"running on {platform.upper()}")

    dataset = load_dataset(
        data_path=args.data_path,
        train_data=args.train_data,
        val_data=args.val_data,
        data_dtype=args.data_dtype,
        data_format=args.data_format,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    vocab_size = experiment.vocab_size
    config = resolve_config(args, platform, vocab_size, experiment)
    capture_window = xprof_step_window(args, config.final_step)
    downstream_domains = load_downstream_domains(
        manifest=args.downstream_manifest,
        root=args.downstream_root,
        documents=args.downstream_data,
        vocab_size=config.semantic_vocab_size,
    )
    diagnostic_mode = args.diagnostic_mode
    needs_evaluation = should_compile_evaluation(args, config, downstream_domains)
    if config.batch_size % len(devices):
        raise ValueError(
            f"global batch size {config.batch_size} must be divisible by "
            f"visible device count {len(devices)}"
        )
    local_batch = local_batch_size(config.batch_size, process_count)
    shuffled_train_stream = (
        ShuffledEpochBatchStream(
            dataset.train,
            global_batch_size=config.batch_size,
            seq_len=config.seq_len,
            vocab_size=config.semantic_vocab_size,
            seed=args.seed + 1,
            process_index=process_index,
            process_count=process_count,
        )
        if config.sampling == "shuffled_epochs"
        else None
    )
    if config.attention_backend != "dense":
        console.phase(
            "Attention tile preflight",
            "resolving the shipped lookup or shape heuristic",
        )
    attention_runtime = resolve_attention_runtime(
        backend=config.attention_backend,
        dtype=config.compute_dtype,
        global_batch_size=config.batch_size,
        heads=config.heads,
        sequence=config.seq_len,
        head_dim=config.d_model // config.heads,
        devices=devices,
    )
    if (
        max(map(len, dataset.train.shards)) < config.seq_len + 1
        or max(map(len, dataset.validation.shards)) < config.seq_len + 1
    ):
        raise ValueError(
            "both data splits need a shard with at least seq_len + 1 tokens; "
            f"got train={len(dataset.train):,}, validation={len(dataset.validation):,}, "
            f"seq_len={config.seq_len}"
        )

    host_params = init_params(config, args.seed)
    host_optimizer = init_optimizer(host_params, config)
    decay_mask = weight_decay_mask(host_params)
    diagnostic_metadata = diagnostic_scope_metadata(host_params)
    params_total = parameter_count(host_params)
    params_active = active_parameter_count(host_params, config)
    expected_active = expected_active_parameters(config)
    if expected_active is not None and params_active != expected_active:
        raise ValueError(
            f"tier {config.tier} should have {expected_active:,} active "
            f"parameters, but initialized {params_active:,} "
            f"(total {params_total:,})"
        )
    flop_breakdown = traced_flops(config, host_params)
    flops_per_token = flop_breakdown.per_token(config.seq_len)
    for warning in flop_breakdown.warnings:
        console.warn(f"FLOP accounting: {warning}")
    tokens_processed = config.final_step * config.batch_size * config.seq_len

    console.table(
        "run configuration",
        (
            (
                "experiment config",
                f"{CONFIG_FILENAME} · {config.config_profile} · "
                f"sha256:{config.config_sha256[:12]}",
            ),
            ("devices", f"{len(devices)} × {device_label(devices)}"),
            ("JAX processes", f"{process_count} (this rank {process_index})"),
            ("mesh", f"data={len(devices)} (replicated model)"),
            ("dataset", dataset.source),
            (
                "train / val tokens",
                f"{len(dataset.train):,} / {len(dataset.validation):,}",
            ),
            (
                "downstream",
                (
                    f"{len(downstream_domains)} domains / "
                    f"{sum(domain.scored_tokens for domain in downstream_domains):,} scored"
                    if downstream_domains
                    else "not requested"
                ),
            ),
            (
                "model",
                f"{config.tier} · L{config.layers} D{config.d_model} H{config.heads} "
                f"RoPE RMSNorm GELU MLP×{config.mlp_mult}",
            ),
            ("parameters", format_count(params_total)),
            (
                "parameterization",
                f"{config.parameterization} · mN={config.width_multiplier:.4g} · "
                f"mL={config.depth_multiplier:.4g} · mD={config.data_multiplier:.4g}",
            ),
            ("global batch", f"{config.batch_size} × {config.seq_len} tokens"),
            (
                "train sampling",
                (
                    f"shuffled epochs · "
                    f"{shuffled_train_stream.usable_tokens_per_epoch:,} unique targets/epoch"
                    if shuffled_train_stream is not None
                    else "random windows with replacement"
                ),
            ),
            ("compute", config.dtype_name),
            ("attention", config.attention_backend),
            *attention_console_rows(attention_runtime),
            (
                "output loss",
                (
                    f"tiled CE (semantic {config.semantic_vocab_size:,}, "
                    f"tile {config.vocab_tile_size:,})"
                    if config.loss_backend == "tiled"
                    else f"dense CE ({config.semantic_vocab_size:,} classes)"
                ),
            ),
            (
                "diagnostics",
                (
                    f"step 1 / every {config.diagnostics_every} / final"
                    if config.diagnostics_every
                    else "disabled"
                ),
            ),
            (
                "duration",
                (
                    f"{config.final_step:,} of {config.steps:,} scheduled steps "
                    "(early stop)"
                    if config.stop_after_step is not None
                    else f"{config.steps:,} steps"
                ),
            ),
            ("train tokens", format_count(tokens_processed)),
            ("traced FLOPs", format_count(flops_per_token * tokens_processed)),
            (
                "FLOP breakdown",
                " · ".join(
                    f"{label} {share}" for label, share in describe(flop_breakdown)
                )
                or "none",
            ),
            (
                "XProf",
                (
                    f"steps {capture_window[0]}..{capture_window[1]} → "
                    f"{args.xprof_dir.expanduser().resolve()}"
                    if capture_window is not None
                    else "disabled"
                ),
            ),
        ),
    )

    mesh = Mesh(np.asarray(devices, dtype=object), ("data",))
    replicated = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("data", None))
    attention_fn = make_mesh_attention(
        backend=config.attention_backend,
        mesh=mesh,
        tiles=attention_runtime.tiles,
        softmax_scale=attention_softmax_scale(
            config.attention_scale, config.d_model // config.heads
        ),
        document_masking=config.document_masking,
    )
    routed_fn = make_mesh_routed_mlp(config, mesh)
    params = put_replicated_tree(host_params, mesh, replicated, process_count)
    optimizer = put_replicated_tree(host_optimizer, mesh, replicated, process_count)
    del host_params, host_optimizer

    train_rng = np.random.default_rng(args.seed + 1 + process_index * 1_000_003)
    # Compilation may not inspect real data. Shapes and dtypes are sufficient.
    sample_x_host = np.zeros((local_batch, config.seq_len), dtype=np.int32)
    sample_y_host = np.zeros((local_batch, config.seq_len), dtype=np.int32)
    sample_x = put_host_local_array(
        sample_x_host, mesh, P("data", None), data_sharding, process_count
    )
    sample_y = put_host_local_array(
        sample_y_host, mesh, P("data", None), data_sharding, process_count
    )

    compiled_step = jax.jit(
        lambda p, o, x, y: train_step(
            p, o, x, y, config, decay_mask, attention_fn, routed_fn
        ),
        in_shardings=(replicated, replicated, data_sharding, data_sharding),
        donate_argnums=(0, 1),
    )
    console.phase("Compiling train step", "compilation is outside train_seconds")
    compile_started = time.perf_counter()
    executable = compiled_step.lower(params, optimizer, sample_x, sample_y).compile()
    train_compile_seconds = time.perf_counter() - compile_started

    diagnostic_executable: Any | None = None
    diagnostic_compile_seconds = 0.0
    if config.diagnostics_every:
        console.phase(
            "Compiling sparse diagnostics",
            "separate executable; compilation is outside train_seconds",
        )
        diagnostic_compile_started = time.perf_counter()
        diagnostic_executable = (
            jax.jit(
                lambda p, o, x, y: diagnostic_train_step(
                    p, o, x, y, config, decay_mask, attention_fn, routed_fn
                ),
                in_shardings=(replicated, replicated, data_sharding, data_sharding),
                donate_argnums=(0, 1),
            )
            .lower(params, optimizer, sample_x, sample_y)
            .compile()
        )
        diagnostic_compile_seconds = time.perf_counter() - diagnostic_compile_started

    # Compile evaluation exactly once when it is requested. Diagnostic XProf
    # runs can skip this executable entirely, keeping their setup focused on the
    # training step being inspected.
    compiled_eval: Any | None = None
    sample_mask: jax.Array | None = None
    eval_compile_seconds = 0.0
    if needs_evaluation:
        sample_mask_host = np.ones((local_batch, config.seq_len), dtype=np.float32)
        sample_mask = put_host_local_array(
            sample_mask_host,
            mesh,
            P("data", None),
            data_sharding,
            process_count,
        )
        console.phase("Compiling evaluation", "reused by probes and final validation")
        eval_compile_started = time.perf_counter()
        compiled_eval = (
            jax.jit(
                lambda p, x, y, mask: eval_step(
                    p, x, y, mask, config, attention_fn, routed_fn
                ),
                in_shardings=(replicated, data_sharding, data_sharding, data_sharding),
            )
            .lower(params, sample_x, sample_y, sample_mask)
            .compile()
        )
        eval_compile_seconds = time.perf_counter() - eval_compile_started
    total_compile_seconds = (
        train_compile_seconds + diagnostic_compile_seconds + eval_compile_seconds
    )

    sync_tree((params, optimizer, sample_x, sample_y, sample_mask))
    probe_detail = (
        f"; validation {config.val_probe_batches} batches every {config.val_every} steps"
        if config.val_every
        else "; periodic validation disabled"
    )
    console.phase(
        "Training",
        f"train compiled in {train_compile_seconds:.2f}s, "
        + (
            f"eval in {eval_compile_seconds:.2f}s{probe_detail}"
            if needs_evaluation
            else "evaluation skipped; diagnostic mode"
        ),
    )

    last_metrics: Mapping[str, jax.Array] | None = None
    diagnostic_device_points: list[tuple[int, jax.Array]] = []
    validation_rows: list[ValidationRow] = []
    validation_probe_seconds = 0.0
    if process_count > 1:
        multihost_utils.sync_global_devices("rig-training-start")
    train_started = time.perf_counter()
    xprof_dir = (
        args.xprof_dir.expanduser().resolve() if capture_window is not None else None
    )
    trace_active = False
    # Needed inside the loop for best-effort partial artifacts, not just
    # by the writers that run after it.
    # Captures already pulled off the accelerator. The device list stays
    # bounded by DIAGNOSTIC_FLUSH_POINTS; this keeps the full history so
    # the authoritative writer still sees every point.
    diagnostic_points_host: list[DiagnosticPoint] = []
    output_dir = args.output_dir.expanduser().resolve()
    training_columns = training_log_columns(
        config.layers if config.experts else 0, config.experts
    )
    progress_log: logpack.LogWriter | None = None
    diagnostic_log: logpack.LogWriter | None = None
    if is_controller:
        output_dir.mkdir(parents=True, exist_ok=True)
        # A stale file from a reused directory would be appended to.
        (output_dir / TRAINING_LOG_NAME).unlink(missing_ok=True)
        (output_dir / DIAGNOSTICS_LOG_NAME).unlink(missing_ok=True)
        progress_log = open_log(
            output_dir / TRAINING_LOG_NAME,
            training_columns,
            tokens_per_step=config.batch_size * config.seq_len,
            flops_per_token=flops_per_token,
        )
        diagnostic_log = open_log(
            output_dir / DIAGNOSTICS_LOG_NAME,
            diagnostic_log_columns(diagnostic_metadata),
            tokens_per_step=config.batch_size * config.seq_len,
            flops_per_token=flops_per_token,
        )
    try:
        # final_step is the horizon unless --stop-after-step truncates it.
        # The schedule below still spans config.steps, so a truncated run walks
        # exactly the prefix of the trajectory it samples.
        for step_index in range(1, config.final_step + 1):
            if capture_window is not None and step_index == capture_window[0]:
                # Drain earlier asynchronous work before opening the trace. The
                # capture therefore begins at the requested steady-state step,
                # rather than including a backlog dispatched by preceding steps.
                sync_tree((params, optimizer, last_metrics))
                assert xprof_dir is not None
                if is_controller:
                    # TPU VM filesystems are independent. Capture the controller's
                    # local chips while every process still runs the distributed
                    # step; this gives worker 0 a self-contained trace to serve.
                    xprof_dir.mkdir(parents=True, exist_ok=True)
                    console.phase(
                        "Starting XProf capture",
                        f"steps {capture_window[0]}..{capture_window[1]} → {xprof_dir}",
                    )
                    jax.profiler.start_trace(
                        xprof_dir,
                        profiler_options=profiler_options(
                            platform, int(jax.local_device_count())
                        ),
                    )
                    trace_active = True
                if process_count > 1:
                    multihost_utils.sync_global_devices("rig-xprof-capture-started")

            annotation = (
                jax.profiler.StepTraceAnnotation("train", step_num=step_index)
                if trace_active
                else nullcontext()
            )
            with annotation:
                # Keep the host sampling, transfer, dispatch, and any logging
                # synchronization inside the step annotation. This exposes input
                # gaps alongside TPU execution in the same XProf timeline.
                if shuffled_train_stream is None:
                    batch_x, batch_y = dataset.batch(
                        "train",
                        train_rng,
                        local_batch,
                        config.seq_len,
                        config.semantic_vocab_size,
                    )
                else:
                    batch_x, batch_y = shuffled_train_stream.next_batch()
                batch_x = put_host_local_array(
                    batch_x, mesh, P("data", None), data_sharding, process_count
                )
                batch_y = put_host_local_array(
                    batch_y, mesh, P("data", None), data_sharding, process_count
                )
                if should_run_diagnostics(
                    step_index,
                    every=config.diagnostics_every,
                    final_step=config.final_step,
                ):
                    if diagnostic_executable is None:  # defensive invariant
                        raise AssertionError("diagnostic executable was not compiled")
                    params, optimizer, last_metrics, diagnostic_values_at_step = (
                        diagnostic_executable(params, optimizer, batch_x, batch_y)
                    )
                    diagnostic_device_points.append(
                        (step_index, diagnostic_values_at_step)
                    )
                    if len(diagnostic_device_points) >= DIAGNOSTIC_FLUSH_POINTS:
                        # Pull to the host and drop the device references. This
                        # is what bounds accelerator residency: without it one
                        # small allocation per capture lives until the run ends.
                        flushed = [
                            DiagnosticPoint(
                                step, np.asarray(local_device_get(v), dtype=np.float32)
                            )
                            for step, v in diagnostic_device_points
                        ]
                        diagnostic_device_points.clear()
                        diagnostic_points_host.extend(flushed)
                        for point in flushed:
                            append_log_row(
                                diagnostic_log,
                                point.step,
                                np.asarray(point.values, dtype=np.float32).reshape(-1),
                            )
                else:
                    params, optimizer, last_metrics = executable(
                        params, optimizer, batch_x, batch_y
                    )
                if should_run_validation_probe(
                    step_index,
                    every=config.val_every,
                    final_step=config.final_step,
                ):
                    # Attribute all preceding asynchronous training work to training,
                    # then start the probe's own honest wall clock inside the helper.
                    sync_tree((params, optimizer, last_metrics))
                    if compiled_eval is None:  # defensive configuration invariant
                        raise AssertionError("validation executable was not compiled")
                    probe_loss, probe_seconds = evaluate_validation_prefix(
                        params,
                        dataset,
                        compiled_eval,
                        data_sharding,
                        batch_size=config.batch_size,
                        seq_len=config.seq_len,
                        semantic_vocab_size=config.semantic_vocab_size,
                        batches=config.val_probe_batches,
                        mesh=mesh,
                        process_index=process_index,
                        process_count=process_count,
                    )
                    probe_tokens = (
                        config.val_probe_batches * config.batch_size * config.seq_len
                    )
                    validation_probe_seconds += probe_seconds
                    validation_rows.append(
                        ValidationRow(
                            step=step_index,
                            tokens_processed=(
                                step_index * config.batch_size * config.seq_len
                            ),
                            kind="fineweb_probe",
                            domain="fineweb",
                            validation_tokens=probe_tokens,
                            validation_loss=probe_loss,
                            perplexity=perplexity_from_loss(probe_loss),
                            validation_seconds=probe_seconds,
                            canonical=False,
                        )
                    )
                    console.validation_probe(
                        step_index, probe_loss, config.val_probe_batches, probe_seconds
                    )
                should_log = (
                    step_index == 1
                    or step_index == config.final_step
                    or step_index % config.log_every == 0
                )
                if should_log:
                    host_metrics = local_device_get(last_metrics)
                    elapsed_so_far = max(time.perf_counter() - train_started, 1.0e-12)
                    seen_tokens = step_index * config.batch_size * config.seq_len
                    # host_metrics is already on the host for the progress
                    # line, so this adds no synchronization.
                    append_log_row(
                        progress_log,
                        step_index,
                        (
                            float(host_metrics["loss"]),
                            float(host_metrics["learning_rate"]),
                            float(host_metrics["grad_norm"]),
                            *(float(v) for v in host_metrics["router_row"]),
                        ),
                    )
                    console.step(
                        step_index,
                        config.final_step,
                        float(host_metrics["loss"]),
                        float(host_metrics["learning_rate"]),
                        float(host_metrics["grad_norm"]),
                        seen_tokens / elapsed_so_far,
                    )

                if capture_window is not None and step_index == capture_window[1]:
                    # Include the final synchronization in the trace so all
                    # captured TPU work is exported before profiling stops.
                    sync_tree((params, optimizer, last_metrics))

            if capture_window is not None and step_index == capture_window[1]:
                if process_count > 1:
                    multihost_utils.sync_global_devices("rig-xprof-capture-finished")
                if trace_active:
                    jax.profiler.stop_trace()
                    trace_active = False
                    console.phase("XProf capture saved", str(xprof_dir))
                if process_count > 1:
                    multihost_utils.sync_global_devices("rig-xprof-capture-stopped")
    finally:
        if trace_active:
            # Avoid leaving process-global profiler state active when a sampled
            # batch or training step raises midway through the capture window.
            jax.profiler.stop_trace()
        # Release both handles before the final writers replace these paths, so
        # a salvage append can never land after the authoritative artifact.
        close_log(progress_log)
        close_log(diagnostic_log)

    if last_metrics is None:  # defensive: argparse prevents zero steps
        raise AssertionError("training produced no metrics")
    # Sparse diagnostic reductions are part of benchmark time even if their
    # result branch is otherwise independent of the next optimizer state.
    sync_tree((params, optimizer, last_metrics, diagnostic_device_points))
    if process_count > 1:
        multihost_utils.sync_global_devices("rig-training-finished")
    train_seconds = max(time.perf_counter() - train_started, 1.0e-12)
    final_train = local_device_get(last_metrics)
    training_history = np.asarray(
        local_device_get(optimizer["history"]), dtype=np.float32
    )
    # Points already pulled by an intermediate flush, then whatever is still
    # resident. Dropping the first group here would silently truncate the
    # authoritative file to the last partial buffer.
    diagnostic_points = tuple(
        [
            *diagnostic_points_host,
            *(
                DiagnosticPoint(
                    step, np.asarray(local_device_get(values), dtype=np.float32)
                )
                for step, values in diagnostic_device_points
            ),
        ]
    )
    train_loss = finite_metric("train_loss", float(final_train["loss"]))

    if diagnostic_mode:
        output_dir = args.output_dir.expanduser().resolve()
        if is_controller:
            write_training_log(
                output_dir,
                training_history,
                tokens_per_step=config.batch_size * config.seq_len,
                final_step=config.final_step,
                flops_per_token=flops_per_token,
                columns=training_columns,
            )
            if diagnostic_points:
                write_diagnostics_log(
                    output_dir,
                    diagnostic_points,
                    diagnostic_metadata,
                    tokens_per_step=config.batch_size * config.seq_len,
                    final_step=config.final_step,
                    flops_per_token=flops_per_token,
                )
        diagnostic_rate = finite_metric(
            "tokens_per_second", tokens_processed / train_seconds, positive=True
        )
        assert capture_window is not None and xprof_dir is not None
        console.table(
            "profile complete",
            (
                ("training steps", f"{config.final_step:,}"),
                ("captured steps", f"{capture_window[0]}..{capture_window[1]}"),
                ("train loss", f"{train_loss:.4f}"),
                ("diagnostic rate", f"{format_rate(diagnostic_rate)} tok/s"),
                ("training curve", output_dir / TRAINING_LOG_NAME),
                (
                    "diagnostics",
                    (
                        output_dir / DIAGNOSTICS_LOG_NAME
                        if diagnostic_points
                        else "disabled"
                    ),
                ),
                ("XProf trace", xprof_dir),
            ),
        )
        if process_count > 1:
            multihost_utils.sync_global_devices("rig-profile-artifacts-written")
        return None

    console.phase(
        "Canonical validation",
        f"{config.eval_batches} deterministic batches outside train_seconds",
    )
    if compiled_eval is None:  # defensive configuration invariant
        raise AssertionError("final validation executable was not compiled")
    validation_loss, final_validation_seconds = evaluate_validation_prefix(
        params,
        dataset,
        compiled_eval,
        data_sharding,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        semantic_vocab_size=config.semantic_vocab_size,
        batches=config.eval_batches,
        mesh=mesh,
        process_index=process_index,
        process_count=process_count,
    )
    validation_rows.append(
        ValidationRow(
            step=config.final_step,
            tokens_processed=tokens_processed,
            kind="fineweb",
            domain="fineweb",
            validation_tokens=config.eval_batches * config.batch_size * config.seq_len,
            validation_loss=validation_loss,
            perplexity=perplexity_from_loss(validation_loss),
            validation_seconds=final_validation_seconds,
            canonical=True,
        )
    )

    downstream_results: dict[str, dict[str, float | int]] = {}
    if downstream_domains:
        console.phase(
            "Fresh-domain validation",
            f"{len(downstream_domains)} domains outside train_seconds",
        )
        for domain in downstream_domains:
            domain_result = evaluate_downstream_domain(
                params,
                domain,
                compiled_eval,
                data_sharding,
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                mesh=mesh,
                process_index=process_index,
                process_count=process_count,
            )
            downstream_results[domain.name] = domain_result
            validation_rows.append(
                ValidationRow(
                    step=config.final_step,
                    tokens_processed=tokens_processed,
                    kind="downstream",
                    domain=domain.name,
                    validation_tokens=int(domain_result["scored_tokens"]),
                    validation_loss=float(domain_result["loss"]),
                    perplexity=float(domain_result["perplexity"]),
                    validation_seconds=float(domain_result["seconds"]),
                    canonical=False,
                )
            )
            console.downstream(
                domain.name,
                float(domain_result["loss"]),
                float(domain_result["perplexity"]),
                int(domain_result["scored_tokens"]),
                float(domain_result["seconds"]),
            )
        macro_loss = finite_metric(
            "fresh10 macro loss",
            float(np.mean([float(row["loss"]) for row in downstream_results.values()])),
        )
        macro_perplexity = perplexity_from_loss(macro_loss)
        downstream_seconds = finite_metric(
            "fresh10 seconds",
            sum(float(row["seconds"]) for row in downstream_results.values()),
            positive=True,
        )
        downstream_scored_tokens = sum(
            int(row["scored_tokens"]) for row in downstream_results.values()
        )
        validation_rows.append(
            ValidationRow(
                step=config.final_step,
                tokens_processed=tokens_processed,
                kind="downstream_macro",
                domain="fresh10_macro",
                validation_tokens=downstream_scored_tokens,
                validation_loss=macro_loss,
                perplexity=macro_perplexity,
                validation_seconds=downstream_seconds,
                canonical=False,
            )
        )
        console.downstream(
            "fresh10 macro",
            macro_loss,
            macro_perplexity,
            downstream_scored_tokens,
            downstream_seconds,
        )
    else:
        console.phase("Fresh-domain validation", "skipped; no downstream data supplied")

    output_dir = args.output_dir.expanduser().resolve()
    artifact_names = [TRAINING_LOG_NAME, VALIDATION_CSV_NAME]
    if diagnostic_points:
        artifact_names.append(DIAGNOSTICS_LOG_NAME)
    if not args.omit_checkpoint:
        artifact_names.append(CHECKPOINT_NAME)
    console.phase("Artifacts", " + ".join(artifact_names))
    if is_controller:
        write_training_log(
            output_dir,
            training_history,
            tokens_per_step=config.batch_size * config.seq_len,
            final_step=config.final_step,
            flops_per_token=flops_per_token,
            columns=training_columns,
        )
        if diagnostic_points:
            write_diagnostics_log(
                output_dir,
                diagnostic_points,
                diagnostic_metadata,
                tokens_per_step=config.batch_size * config.seq_len,
                final_step=config.final_step,
                flops_per_token=flops_per_token,
            )
        write_validation_csv(output_dir, validation_rows)
        if not args.omit_checkpoint:
            save_checkpoint(
                output_dir,
                params,
                checkpoint_metadata(config, args.seed, attention_runtime),
            )

    tokens_per_second = finite_metric(
        "tokens_per_second", tokens_processed / train_seconds, positive=True
    )
    total_flops = int(flops_per_token * tokens_processed)
    achieved_tflops = finite_metric(
        "achieved_tflops", total_flops / train_seconds / 1.0e12
    )
    peak_tflops = inferred_peak_tflops(args.peak_tflops, devices)
    mfu = achieved_tflops / peak_tflops if peak_tflops is not None else 0.0
    smoke_contract = profile == "smoke" or using_builtin_data
    dataset_id = args.dataset_id or (
        "builtin-byte-v1" if smoke_contract else "fineweb10b-gpt2"
    )
    tokenizer_id = args.tokenizer_id or ("byte" if smoke_contract else "gpt2")
    fineweb_tokens = config.eval_batches * config.batch_size * config.seq_len
    evaluations: dict[str, Any] = {
        "fineweb": {
            "loss": validation_loss,
            "perplexity": perplexity_from_loss(validation_loss),
            "scored_tokens": int(fineweb_tokens),
            "seconds": finite_metric(
                "final_validation_seconds", final_validation_seconds, positive=True
            ),
            "canonical": True,
        }
    }
    if downstream_results:
        evaluations["fresh10"] = {
            "domains": downstream_results,
            "macro_loss": macro_loss,
            "macro_perplexity": macro_perplexity,
            "scored_tokens": int(downstream_scored_tokens),
            "seconds": downstream_seconds,
        }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "track": "open",
        "profile": profile,
        "seed": int(args.seed),
        "checkpoint": None if args.omit_checkpoint else CHECKPOINT_NAME,
        "artifacts": {
            "training_curve": TRAINING_LOG_NAME,
            "validation_curve": VALIDATION_CSV_NAME,
            **({"diagnostics": DIAGNOSTICS_LOG_NAME} if diagnostic_points else {}),
        },
        "system": {
            **system_metadata(devices),
            "controller_process_index": process_index,
        },
        "contract": {
            "model_id": "reference-gpt-v3-family",
            "dataset_id": dataset_id,
            "tokenizer_id": tokenizer_id,
            "sequence_length": config.seq_len,
            "context_preset": config.context_preset,
            "model": contract_model_metadata(config),
        },
        # Keep kernel choices in implementation provenance so the architecture
        # metadata remains easy to compare across otherwise different recipes.
        "implementation": implementation_metadata(config, attention_runtime),
        "evaluations": evaluations,
        "metrics": {
            "train_seconds": finite_metric(
                "train_seconds", train_seconds, positive=True
            ),
            "tokens_processed": int(tokens_processed),
            "training_token_budget": int(tokens_processed),
            "training_steps": int(config.final_step),
            "schedule_steps": int(config.steps),
            "stop_after_step": (
                int(config.stop_after_step)
                if config.stop_after_step is not None
                else None
            ),
            "model_tier": config.tier,
            "parameter_count": int(params_active),
            "total_parameter_count": int(params_total),
            "experts": int(config.experts),
            "expert_top_k": int(config.expert_top_k),
            "tokens_per_parameter": (
                float(config.tokens_per_parameter)
                if config.tokens_per_parameter is not None
                else None
            ),
            "target_tokens_per_parameter": (
                float(config.target_tokens_per_parameter)
                if config.target_tokens_per_parameter is not None
                else None
            ),
            "base_learning_rate": float(config.learning_rate),
            "training_sampling": config.sampling,
            "training_data_sharding": (
                "rank_disjoint_shuffled_windows"
                if shuffled_train_stream is not None
                else "rank_local_random_windows"
            ),
            "training_usable_tokens_per_epoch": int(
                shuffled_train_stream.usable_tokens_per_epoch
                if shuffled_train_stream is not None
                else len(dataset.train)
            ),
            "training_data_epochs": finite_metric(
                "training_data_epochs",
                tokens_processed
                / (
                    shuffled_train_stream.usable_tokens_per_epoch
                    if shuffled_train_stream is not None
                    else len(dataset.train)
                ),
                positive=True,
            ),
            "validation_loss": validation_loss,
            "validation_tokens": int(fineweb_tokens),
            "validation_probe_count": sum(
                row.kind == "fineweb_probe" for row in validation_rows
            ),
            "diagnostic_point_count": len(diagnostic_points),
            "diagnostics_every": int(config.diagnostics_every),
            "validation_probe_seconds": finite_metric(
                "validation_probe_seconds", validation_probe_seconds
            ),
            "final_validation_seconds": finite_metric(
                "final_validation_seconds", final_validation_seconds, positive=True
            ),
            "train_loss": train_loss,
            "parameters": int(params_total),
            "flops_per_token": int(flops_per_token),
            "estimated_total_flops": total_flops,
            # Traced from the jaxpr, not a maintained formula. The breakdown
            # attributes the count to its sources so a change in architecture
            # is auditable after the fact; warnings record any work the walker
            # could not account for.
            "flop_accounting": {
                "method": "traced-jaxpr",
                "matmul_per_sequence": int(flop_breakdown.matmul),
                "elementwise_per_sequence": int(flop_breakdown.elementwise),
                "by_site": {
                    label: int(value)
                    for label, value in sorted(flop_breakdown.by_site.items())
                },
                "warnings": list(flop_breakdown.warnings),
            },
            "tokens_per_second": tokens_per_second,
            "achieved_tflops": achieved_tflops,
            "mfu_estimate": finite_metric("mfu_estimate", mfu),
            "attention_tune_seconds": finite_metric(
                "attention_tune_seconds", attention_runtime.tune_seconds
            ),
            "train_compile_seconds": finite_metric(
                "train_compile_seconds", train_compile_seconds
            ),
            "eval_compile_seconds": finite_metric(
                "eval_compile_seconds", eval_compile_seconds
            ),
            "diagnostic_compile_seconds": finite_metric(
                "diagnostic_compile_seconds", diagnostic_compile_seconds
            ),
            "total_compile_seconds": finite_metric(
                "total_compile_seconds", total_compile_seconds
            ),
        },
    }
    if is_controller:
        write_result(output_dir, result)
    console.success(validation_loss, train_seconds, final_validation_seconds)
    if process_count > 1:
        multihost_utils.sync_global_devices("rig-final-artifacts-written")
    return result if is_controller else None


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_plan:
        try:
            experiment = validate_args(args)
            planned = resolve_config(
                args,
                "cpu" if selected_profile(args) == "smoke" else "tpu",
                experiment.vocab_size,
                experiment,
            )
        except Exception as error:
            print(f"\nerror: {error}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                resolved_plan_metadata(planned), sort_keys=True, separators=(",", ":")
            )
        )
        return 0
    try:
        result = run(args)
    except Exception as error:
        # A concise colored-ish error is useful interactively; a traceback can be
        # requested naturally via Python's exception chaining during development.
        print(f"\nerror: {error}", file=sys.stderr)
        if os.environ.get("RIG_DEBUG") == "1":
            raise
        return 1
    if result is not None:
        print(
            RESULT_PREFIX
            + json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
