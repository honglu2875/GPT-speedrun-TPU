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
from typing import Annotated, Any, Iterable, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.experimental import multihost_utils
import yaml
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from rig import logpack
from rig.arguments import positive_int
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
from rig.recipe_args import (
    add_standard_config_arguments,
    add_standard_data_arguments,
    add_standard_reporting_arguments,
    add_standard_xprof_arguments,
    new_recipe_parser,
    validate_standard_data_arguments,
    validate_standard_reporting_arguments,
    validate_standard_xprof_arguments,
)
from rig.metrics import DIAGNOSTIC_FAMILIES, DIAGNOSTIC_STATS
from rig.configfile import read_config_document, resolve_sibling_config_path
from rig.configschema import (
    Bounds,
    ConfigSchema,
    Length,
    Matches,
    NonnegativeFloat,
    NonnegativeInt,
    PositiveFloat,
    PositiveInt,
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
RECIPE_DIR = Path(__file__).resolve().parent
RECIPE_NAME = RECIPE_DIR.name
CONFIG_PATH = RECIPE_DIR / CONFIG_FILENAME
_VALID_PROFILES = ("smoke", "dev", "official")
_DOMAIN_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_TIER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_CONTEXT_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


# A deliberately small, original corpus for offline and smoke-test use.  The
# repeated motifs make it possible for tiny models to show measurable progress,
# while the shuffled clauses prevent every training window from being identical.


@dataclass(frozen=True, slots=True)
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
    # Stable run-protocol names; both values are derived from the typed model
    # definitions rather than declared independently in config.yaml.
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

    @property
    def final_step(self) -> int:
        """Last optimizer step this run takes; steps stays the schedule horizon."""

        return self.stop_after_step or self.steps


TierName = Annotated[str, Matches(_TIER_NAME.pattern)]
ContextName = Annotated[str, Matches(_CONTEXT_NAME.pattern)]
Probability = Annotated[float, Bounds(ge=0.0, le=1.0)]
OpenProbability = Annotated[float, Bounds(ge=0.0, lt=1.0)]
DepthAlpha = Annotated[float, Bounds(ge=0.5, le=1.0)]


@dataclass(frozen=True, slots=True)
class ContextPreset:
    """One coupled sequence-length, batch-anchor, and masking preset."""

    seq_len: PositiveInt
    reference_batch_size: PositiveInt
    document_masking: bool

    @property
    def tokens_per_step(self) -> int:
        """Number of tokens in one recipe-default global optimizer step."""

        return self.reference_batch_size * self.seq_len


@dataclass(frozen=True, slots=True)
class ParameterizationDefinition:
    """The fixed-TPP CompleteP contract shared by every family tier."""

    name: Literal["completep_fixed_tpp_v1"]
    base_tier: TierName
    base_width: PositiveInt
    base_depth: PositiveInt
    depth_alpha: DepthAlpha
    init_std: PositiveFloat
    attention_scale: Literal["inverse_head_dim"]
    embeddings: Literal["untied"]


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Architecture fields represented literally in ``config.yaml``."""

    layers: PositiveInt
    heads: PositiveInt
    d_model: PositiveInt
    mlp_mult: PositiveInt
    normalization: Literal["rms_norm"]
    position_encoding: Literal["rope_base_10000"]
    mlp_activation: Literal["gelu"]
    vocab_size: PositiveInt
    semantic_vocab_size: PositiveInt

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""

        return self.d_model // self.heads

    def validate(self, label: str) -> None:
        """Enforce architecture relations that no single annotation can express."""

        if self.semantic_vocab_size > self.vocab_size:
            raise ValueError(
                f"config.yaml {label}.semantic_vocab_size must not exceed vocab_size"
            )
        if self.d_model % self.heads:
            raise ValueError(f"config.yaml {label}.d_model must be divisible by heads")
        if self.head_dim % 2:
            raise ValueError(
                f"config.yaml {label} head dimension must be even for RoPE"
            )


@dataclass(frozen=True, slots=True)
class TierDefinition:
    model: ModelDefinition

    @property
    def tpp_parameters(self) -> int:
        """Parameter denominator used by the fixed-TPP ladder."""

        model = self.model
        width = model.d_model
        return (
            2 * model.vocab_size * width
            + model.layers * (12 * width * width + 11 * width)
            + width
        )


@dataclass(frozen=True, slots=True)
class FamilyDefinition:
    default_tier: TierName
    default_context: ContextName
    contexts: Annotated[dict[ContextName, ContextPreset], Length(ge=1)]
    parameterization: ParameterizationDefinition
    tiers: Annotated[dict[TierName, TierDefinition], Length(ge=1)]


Sampling = Literal["random_windows", "shuffled_epochs"]
ComputeDtype = Literal["bfloat16", "float32"]


@dataclass(frozen=True, slots=True)
class SmokeTraining:
    steps: PositiveInt
    batch_size: PositiveInt
    seq_len: PositiveInt
    sampling: Sampling
    dtype: ComputeDtype


@dataclass(frozen=True, slots=True)
class LadderTraining:
    tokens_per_parameter: PositiveFloat
    sampling: Sampling
    dtype: ComputeDtype


@dataclass(frozen=True, slots=True)
class KernelSettings:
    attention_backend: Literal["dense", "jax_flash", "tpu_flash"]
    loss_backend: Literal["dense", "tiled"]
    vocab_tile_size: PositiveInt


@dataclass(frozen=True, slots=True)
class OptimizerSettings:
    learning_rate: PositiveFloat
    min_lr_ratio: Probability
    warmup_ratio: OpenProbability
    weight_decay: NonnegativeFloat
    adam_epsilon: PositiveFloat
    beta1: OpenProbability
    beta2: OpenProbability
    grad_clip: NonnegativeFloat


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    eval_batches: PositiveInt
    val_every: NonnegativeInt
    val_probe_batches: PositiveInt


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    diagnostics_every: NonnegativeInt
    log_every: PositiveInt


@dataclass(frozen=True, slots=True)
class SmokeProfileDefinition:
    training: SmokeTraining
    model: ModelDefinition
    kernels: KernelSettings
    optimizer: OptimizerSettings
    evaluation: EvaluationSettings
    logging: LoggingSettings


@dataclass(frozen=True, slots=True)
class LadderProfileDefinition:
    training: LadderTraining
    model: Literal["family_tier"]
    kernels: KernelSettings
    optimizer: OptimizerSettings
    evaluation: EvaluationSettings
    logging: LoggingSettings


@dataclass(frozen=True, slots=True)
class ProfileDefinitions:
    smoke: SmokeProfileDefinition
    dev: LadderProfileDefinition
    official: LadderProfileDefinition


@dataclass(frozen=True, slots=True)
class ExperimentConfig(ConfigSchema):
    """Complete typed representation of the recipe's ``config.yaml``."""

    schema_version: Literal[4]
    family: FamilyDefinition
    profiles: ProfileDefinitions

    def validate(self) -> None:
        """Enforce the few scientific contracts involving multiple fields."""

        family = self.family
        if family.default_context not in family.contexts:
            raise ValueError(
                "config.yaml family.default_context must name a defined context preset"
            )
        if family.default_tier not in family.tiers:
            raise ValueError("config.yaml family.default_tier must name a defined tier")
        base_tier = family.parameterization.base_tier
        if base_tier not in family.tiers:
            raise ValueError(
                "config.yaml family.parameterization.base_tier must name a defined tier"
            )

        for tier_name, tier in family.tiers.items():
            label = f"family.tiers.{tier_name}.model"
            tier.model.validate(label)
            if tier.model.head_dim != 64:
                raise ValueError(
                    f"config.yaml family.tiers.{tier_name} must use 64-wide heads"
                )

        self.profiles.smoke.model.validate("profiles.smoke.model")
        for name, profile in (
            ("smoke", self.profiles.smoke),
            ("dev", self.profiles.dev),
            ("official", self.profiles.official),
        ):
            if (
                profile.kernels.attention_backend != "dense"
                and profile.training.dtype != "bfloat16"
            ):
                raise ValueError(
                    f"config.yaml profiles.{name}.kernels.attention_backend "
                    f"{profile.kernels.attention_backend} requires "
                    "training.dtype bfloat16"
                )
            if (
                profile.evaluation.val_every
                and profile.evaluation.val_probe_batches
                > profile.evaluation.eval_batches
            ):
                raise ValueError(
                    f"config.yaml profiles.{name}.evaluation.val_probe_batches "
                    "must not exceed eval_batches"
                )

    def select(
        self,
        profile: str,
        source_sha256: str,
        *,
        tier: str | None = None,
        context: str | None = None,
    ) -> SelectedExperiment:
        """Validate runtime selectors and return a zero-copy view of this document."""

        if profile not in _VALID_PROFILES:
            raise ValueError(f"unknown experiment profile: {profile!r}")
        self.validate()

        tier_name = tier or self.family.default_tier
        if tier_name not in self.family.tiers:
            raise ValueError(
                f"unknown model tier {tier_name!r}; expected "
                + ", ".join(sorted(self.family.tiers))
            )
        context_name = context or self.family.default_context
        if context_name not in self.family.contexts:
            raise ValueError(
                f"unknown context preset {context_name!r}; expected "
                + ", ".join(sorted(self.family.contexts))
            )

        official = self.profiles.official
        tokens_per_step = self.family.contexts[context_name].tokens_per_step
        validation_tokens = 10_485_760
        if validation_tokens % tokens_per_step:
            raise ValueError(
                "config.yaml profiles.official batch_size * seq_len must divide "
                f"the official {validation_tokens:,}-prediction validation prefix"
            )
        required_eval_batches = validation_tokens // tokens_per_step
        if official.evaluation.eval_batches != required_eval_batches:
            raise ValueError(
                "config.yaml profiles.official.evaluation.eval_batches must be "
                f"{required_eval_batches} for the official validation prefix"
            )

        return SelectedExperiment(
            config=self,
            source_sha256=source_sha256,
            name=profile,
            tier_name=tier_name,
            context_name=context_name,
        )


@dataclass(frozen=True, slots=True)
class SelectedExperiment:
    """Runtime profile/tier/context selection over one decoded YAML config."""

    config: ExperimentConfig
    source_sha256: str
    name: str
    tier_name: str
    context_name: str

    @property
    def profile(self) -> SmokeProfileDefinition | LadderProfileDefinition:
        if self.name == "smoke":
            return self.config.profiles.smoke
        if self.name == "dev":
            return self.config.profiles.dev
        if self.name == "official":
            return self.config.profiles.official
        raise AssertionError(f"invalid selected profile: {self.name!r}")

    @property
    def tier(self) -> TierDefinition:
        return self.config.family.tiers[self.tier_name]

    @property
    def context(self) -> ContextPreset:
        return self.config.family.contexts[self.context_name]

    @property
    def model(self) -> ModelDefinition:
        if self.name == "smoke":
            profile = self.profile
            if not isinstance(profile, SmokeProfileDefinition):
                raise AssertionError("smoke selection resolved a ladder profile")
            return profile.model
        return self.tier.model

    @property
    def tpp_parameters(self) -> int | None:
        return None if self.name == "smoke" else self.tier.tpp_parameters

    @property
    def base_tpp_parameters(self) -> int:
        base_tier = self.config.family.parameterization.base_tier
        return self.config.family.tiers[base_tier].tpp_parameters


_UINT64_MASK = (1 << 64) - 1


def load_experiment_profile(
    profile: str,
    requested_path: Path | None = None,
    *,
    tier: str | None = None,
    context: str | None = None,
) -> SelectedExperiment:
    if profile not in _VALID_PROFILES:
        raise ValueError(f"unknown experiment profile: {profile!r}")
    path = resolve_sibling_config_path(requested_path, CONFIG_PATH)
    mapping, source_sha256 = read_config_document(path)
    experiment_config = ExperimentConfig.from_mapping(mapping)
    return experiment_config.select(profile, source_sha256, tier=tier, context=context)


def build_parser() -> argparse.ArgumentParser:
    parser = new_recipe_parser(
        description=(
            "Train a decoder-only GPT with JAX. Static experiment settings come "
            "from config.yaml beside this entry script."
        )
    )
    run = parser.add_argument_group("run")
    add_standard_config_arguments(
        run,
        default_output_dir=Path("runs") / RECIPE_NAME,
        profiles=_VALID_PROFILES,
    )
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
    add_standard_xprof_arguments(parser)

    add_standard_data_arguments(parser)

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
    add_standard_reporting_arguments(optim)
    return parser


def validate_args(args: argparse.Namespace) -> SelectedExperiment:
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
    validate_standard_data_arguments(args)
    validate_standard_reporting_arguments(args)
    validate_standard_xprof_arguments(args, profile=selected_profile(args))
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
    experiment: SelectedExperiment | None = None,
) -> Config:
    profile = selected_profile(args)
    experiment = experiment or load_experiment_profile(
        profile, args.config, tier=args.tier, context=args.context
    )
    if experiment.name != profile:
        raise ValueError(
            f"resolved config profile {experiment.name!r} does not match {profile!r}"
        )

    definition = experiment.profile
    model = experiment.model
    kernels = definition.kernels
    optimizer = definition.optimizer
    evaluation = definition.evaluation
    logging = definition.logging
    if vocab_size != model.vocab_size:
        raise ValueError(
            "loaded dataset vocabulary does not match config.yaml: "
            f"dataset={vocab_size}, configured={model.vocab_size}"
        )

    if profile == "smoke":
        if not isinstance(definition, SmokeProfileDefinition):
            raise AssertionError("smoke selection resolved a ladder profile")
        training = definition.training
        batch_anchor = training.batch_size
        seq_len = training.seq_len
        document_masking = False
        requested_tpp = args.tokens_per_parameter
        tier_name = "smoke"
        parameterization = "standard"
        base_width = model.d_model
        base_depth = model.layers
        depth_alpha = 0.0
        init_std = 0.02
        attention_scale = "inverse_sqrt_head_dim"
        embeddings = "tied"
        context_preset = "smoke"
    else:
        if not isinstance(definition, LadderProfileDefinition):
            raise AssertionError("ladder selection resolved the smoke profile")
        training = definition.training
        context = experiment.context
        family_parameterization = experiment.config.family.parameterization
        batch_anchor = context.reference_batch_size
        seq_len = context.seq_len
        document_masking = context.document_masking
        requested_tpp = args.tokens_per_parameter or training.tokens_per_parameter
        tier_name = experiment.tier_name
        parameterization = family_parameterization.name
        base_width = family_parameterization.base_width
        base_depth = family_parameterization.base_depth
        depth_alpha = family_parameterization.depth_alpha
        init_std = family_parameterization.init_std
        attention_scale = family_parameterization.attention_scale
        embeddings = family_parameterization.embeddings
        context_preset = experiment.context_name

    batch_size = args.batch_size or batch_anchor
    tokens_per_step = batch_size * seq_len
    early_stop = getattr(args, "stop_after_step", None)
    tpp_parameters = experiment.tpp_parameters
    if profile == "smoke":
        if args.tokens_per_parameter is not None:
            raise ValueError("--tokens-per-parameter cannot override the smoke profile")
        if early_stop is not None:
            raise ValueError("--stop-after-step requires a fixed-TPP profile")
        steps = training.steps
    else:
        if requested_tpp is None:
            raise AssertionError("non-smoke profile did not resolve a TPP horizon")
        if tpp_parameters is None:
            raise AssertionError("fixed-TPP profile has no parameter denominator")
        ideal_tokens = float(tpp_parameters) * requested_tpp
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
        if evaluation.eval_batches != required_eval_batches:
            raise ValueError(
                "official config.yaml validation must cover exactly 10,485,760 "
                f"predictions; set eval_batches to {required_eval_batches}"
            )
        eval_batches = required_eval_batches
    else:
        eval_batches = evaluation.eval_batches
    val_every = 0 if args.diagnostic_mode else evaluation.val_every
    val_probe_batches = evaluation.val_probe_batches
    if val_every > 0 and val_probe_batches > eval_batches:
        raise ValueError(
            "config.yaml val_probe_batches must not exceed the canonical evaluation batch "
            f"count ({eval_batches}); got {val_probe_batches}"
        )
    log_every = steps if args.diagnostic_mode else logging.log_every
    diagnostics_every = 0 if args.diagnostic_mode else logging.diagnostics_every

    dtype_name = training.dtype
    compute_dtype = jnp.bfloat16 if dtype_name == "bfloat16" else jnp.float32
    attention_backend = kernels.attention_backend
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

    width_multiplier = model.d_model / float(base_width)
    depth_multiplier = model.layers / float(base_depth)
    batch_multiplier = batch_size / float(batch_anchor)
    achieved_tpp = (
        steps * tokens_per_step / float(tpp_parameters)
        if tpp_parameters is not None
        else None
    )
    # This project reanchors every TPP ladder. The multiplier captures only the
    # model-size-induced data growth within one fixed-TPP ladder; it deliberately
    # omits any cross-horizon TPP / TPP_0 factor.
    data_multiplier = (
        tpp_parameters / float(experiment.base_tpp_parameters)
        if tpp_parameters is not None
        else 1.0
    )
    base_learning_rate = (
        args.base_learning_rate
        if args.base_learning_rate is not None
        else optimizer.learning_rate
    )
    warmup_steps = int(math.floor(steps * optimizer.warmup_ratio + 0.5))
    if steps > 1:
        warmup_steps = min(warmup_steps, steps - 1)
    else:
        warmup_steps = 0

    return Config(
        steps=steps,
        stop_after_step=early_stop,
        document_masking=document_masking,
        batch_size=batch_size,
        seq_len=seq_len,
        sampling=training.sampling,
        layers=model.layers,
        heads=model.heads,
        d_model=model.d_model,
        mlp_mult=model.mlp_mult,
        normalization=model.normalization,
        position_encoding=model.position_encoding,
        mlp_activation=model.mlp_activation,
        tier=tier_name,
        declared_parameters=tpp_parameters,
        base_parameters=experiment.base_tpp_parameters,
        parameterization=parameterization,
        base_width=base_width,
        base_depth=base_depth,
        depth_alpha=depth_alpha,
        init_std=init_std,
        attention_scale=attention_scale,
        embeddings=embeddings,
        width_multiplier=width_multiplier,
        depth_multiplier=depth_multiplier,
        data_multiplier=data_multiplier,
        batch_multiplier=batch_multiplier,
        target_tokens_per_parameter=requested_tpp,
        tokens_per_parameter=achieved_tpp,
        learning_rate=base_learning_rate,
        min_lr_ratio=optimizer.min_lr_ratio,
        warmup_steps=warmup_steps,
        weight_decay=optimizer.weight_decay,
        adam_epsilon=optimizer.adam_epsilon,
        beta1=optimizer.beta1,
        beta2=optimizer.beta2,
        grad_clip=optimizer.grad_clip,
        eval_batches=eval_batches,
        val_every=val_every,
        val_probe_batches=val_probe_batches,
        diagnostics_every=diagnostics_every,
        log_every=log_every,
        vocab_size=vocab_size,
        semantic_vocab_size=model.semantic_vocab_size,
        attention_backend=attention_backend,
        loss_backend=kernels.loss_backend,
        vocab_tile_size=kernels.vocab_tile_size,
        compute_dtype=compute_dtype,
        dtype_name=dtype_name,
        config_schema_version=experiment.config.schema_version,
        config_sha256=experiment.source_sha256,
        config_profile=experiment.name,
        context_preset=context_preset,
        config_overrides=overrides,
    )


def init_params(config: Config, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    d_model = config.d_model
    hidden = config.mlp_mult * d_model
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
                "mlp_up_w": normal(rng, (d_model, hidden), hidden_scale),
                "mlp_up_b": np.zeros((hidden,), dtype=np.float32),
                "mlp_down_w": normal(rng, (hidden, d_model), hidden_scale),
                "mlp_down_b": np.zeros((d_model,), dtype=np.float32),
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


def gpt_hidden(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
) -> jax.Array:
    """Return final normalized token representations before the tied head."""

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
        hidden = linear(x_norm, block["mlp_up_w"], block["mlp_up_b"], dtype)
        hidden = jax.nn.gelu(hidden, approximate=True)
        x = residual + (config.depth_multiplier ** (-config.depth_alpha)) * linear(
            hidden, block["mlp_down_w"], block["mlp_down_b"], dtype
        )

    return rms_norm(x, params["final_ln_scale"], dtype)


def gpt_logits(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
) -> jax.Array:
    x = gpt_hidden(params, tokens, config, attention_fn)
    output_embedding = params.get("output_embedding", params["token_embedding"])
    return jnp.einsum(
        "btd,vd->btv",
        x,
        output_embedding.astype(config.compute_dtype),
    ).astype(jnp.float32)


def cross_entropy(
    params: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
) -> jax.Array:
    if config.loss_backend == "tiled":
        hidden = gpt_hidden(params, x, config, attention_fn)
        return tiled_tied_cross_entropy(
            hidden,
            params.get("output_embedding", params["token_embedding"]),
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    logits = gpt_logits(params, x, config, attention_fn)[
        ..., : config.semantic_vocab_size
    ]
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(log_probabilities, y[..., None], axis=-1)
    return -jnp.mean(selected, dtype=jnp.float32)


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


def init_optimizer(params: Any, steps: int) -> dict[str, Any]:
    zeros = jax.tree_util.tree_map(lambda value: np.zeros_like(value), params)
    # Keeping the small scalar history on-device avoids a host synchronization
    # on every step. It is copied once, after the synchronized timing boundary.
    history = np.zeros((steps, 3), dtype=np.float32)
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
    loss, gradients = jax.value_and_grad(
        lambda candidate: cross_entropy(candidate, x, y, config, attention_fn)
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
    history_row = jnp.stack((loss, lr, grad_norm)).astype(jnp.float32)
    history = optimizer["history"].at[step - 1].set(history_row)
    return (
        params,
        {"step": step, "m": m, "v": v, "history": history},
        {
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": lr,
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
) -> tuple[Any, dict[str, Any], dict[str, jax.Array]]:
    params, optimizer, metrics, _ = _apply_training_update(
        params, optimizer, x, y, config, decay_mask, attention_fn
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
) -> tuple[Any, dict[str, Any], dict[str, jax.Array], jax.Array]:
    """Run the same update as :func:`train_step` and emit sparse statistics."""

    params_before = params
    params, optimizer, metrics, raw_gradients = _apply_training_update(
        params, optimizer, x, y, config, decay_mask, attention_fn
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
) -> tuple[jax.Array, jax.Array]:
    """Return a loss sum and exact target count for fixed-shape masked eval."""

    if config.loss_backend == "tiled":
        hidden = gpt_hidden(params, x, config, attention_fn)
        losses = tiled_tied_cross_entropy_losses(
            hidden,
            params.get("output_embedding", params["token_embedding"]),
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    else:
        logits = gpt_logits(params, x, config, attention_fn)[
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
    vocab_size = experiment.model.vocab_size
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
    host_optimizer = init_optimizer(host_params, config.steps)
    decay_mask = weight_decay_mask(host_params)
    diagnostic_metadata = diagnostic_scope_metadata(host_params)
    params_total = parameter_count(host_params)
    if (
        config.declared_parameters is not None
        and params_total != config.declared_parameters
    ):
        raise ValueError(
            f"tier {config.tier} declares {config.declared_parameters:,} parameters, "
            f"but initialized {params_total:,}"
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
        lambda p, o, x, y: train_step(p, o, x, y, config, decay_mask, attention_fn),
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
                    p, o, x, y, config, decay_mask, attention_fn
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
                lambda p, x, y, mask: eval_step(p, x, y, mask, config, attention_fn),
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
    progress_log: logpack.LogWriter | None = None
    diagnostic_log: logpack.LogWriter | None = None
    if is_controller:
        output_dir.mkdir(parents=True, exist_ok=True)
        # A stale file from a reused directory would be appended to.
        (output_dir / TRAINING_LOG_NAME).unlink(missing_ok=True)
        (output_dir / DIAGNOSTICS_LOG_NAME).unlink(missing_ok=True)
        progress_log = open_log(
            output_dir / TRAINING_LOG_NAME,
            training_log_columns(),
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
            "parameter_count": int(params_total),
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
                experiment.model.vocab_size,
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
