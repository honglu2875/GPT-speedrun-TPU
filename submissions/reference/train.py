#!/usr/bin/env python3
"""A compact, dependency-light GPT trainer for the GPT TPU speedrun.

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
from typing import Any, Callable, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.experimental import multihost_utils
import yaml
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from speedrun.kernels import (
    AttentionConfig,
    AttentionTiles,
    make_causal_attention,
    select_attention_tiles,
    tiled_tied_cross_entropy,
    tiled_tied_cross_entropy_losses,
)
from speedrun.kernels.autotune import (
    autotune_attention,
    make_runtime_key,
    padded_sequence_length,
    resolve_attention_tile_plan,
)


RESULT_PREFIX = "SPEEDRUN_RESULT="
CHECKPOINT_NAME = "checkpoint.npz"
TRAINING_CSV_NAME = "training.csv"
VALIDATION_CSV_NAME = "validation.csv"
DIAGNOSTICS_CSV_NAME = "diagnostics.csv"
SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
CONFIG_FILENAME = "config.yaml"
CONFIG_PATH = Path(__file__).resolve().with_name(CONFIG_FILENAME)
_MAX_CONFIG_BYTES = 256 * 1024
_VALID_TRACKS = ("open", "sample_efficiency")
_VALID_PROFILES = ("smoke", "dev", "official")
_DOMAIN_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_DIAGNOSTIC_FAMILIES = ("param", "grad", "update")
_DIAGNOSTIC_STATS = (
    "l1_norm",
    "l2_norm",
    "mean",
    "std",
    "third_moment",
    "fourth_moment",
)
_DISTRIBUTED_ENV = "SPEEDRUN_DISTRIBUTED"
_PROCESS_COUNT_ENV = "SPEEDRUN_PROCESS_COUNT"
_CONTROLLER_HOST_ENV = "SPEEDRUN_CONTROLLER_HOSTNAME"


# A deliberately small, original corpus for offline and smoke-test use.  The
# repeated motifs make it possible for tiny models to show measurable progress,
# while the shuffled clauses prevent every training window from being identical.
_CORPUS_LINES = (
    "A small model learns one careful prediction at a time.",
    "Fast experiments need honest clocks, fixed seeds, and useful baselines.",
    "The compiler maps arrays across a mesh while the host prepares tokens.",
    "Attention connects earlier symbols to the next symbol in the sequence.",
    "Measure the validation loss after training and preserve the final weights.",
    "Clear code makes an optimization easier to test, compare, and reproduce.",
    "Bright terminals are pleasant, but machine-readable results stay plain.",
    "A byte vocabulary can represent every line without a tokenizer download.",
    "The quick copper fox counts vectors beneath a quiet indigo moon.",
    "Seven tiny robots trade stories about matrices, rivers, and warm bread.",
)


@dataclass(frozen=True)
class Config:
    steps: int
    batch_size: int
    seq_len: int
    layers: int
    heads: int
    d_model: int
    mlp_mult: int
    learning_rate: float
    min_lr_ratio: float
    warmup_steps: int
    weight_decay: float
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
    config_overrides: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ExperimentProfile:
    """One fully explicit, versioned profile loaded from sibling config.yaml."""

    schema_version: int
    source_sha256: str
    name: str
    steps: int | None
    train_tokens: int | None
    batch_size: int
    seq_len: int
    dtype_name: str
    layers: int
    heads: int
    d_model: int
    mlp_mult: int
    vocab_size: int
    semantic_vocab_size: int
    attention_backend: str
    loss_backend: str
    vocab_tile_size: int
    learning_rate: float
    min_lr_ratio: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    eval_batches: int
    val_every: int
    val_probe_batches: int
    diagnostics_every: int
    log_every: int


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _StrictSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError("config.yaml mapping keys must be scalar values") from exc
        if duplicate:
            raise ValueError(f"config.yaml contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


_STATIC_CLI_FIELDS = {
    "eval_batches": "--eval-batches",
    "val_probe_batches": "--val-probe-batches",
    "vocab_size": "--vocab-size",
    "batch_size": "--batch-size",
    "seq_len": "--seq-len",
    "layers": "--layers",
    "heads": "--heads",
    "d_model": "--d-model",
    "mlp_mult": "--mlp-mult",
    "dtype": "--dtype",
    "attention_backend": "--attention-backend",
    "loss_backend": "--loss-backend",
    "semantic_vocab_size": "--semantic-vocab-size",
    "vocab_tile_size": "--vocab-tile-size",
    "learning_rate": "--learning-rate",
    "min_lr_ratio": "--min-lr-ratio",
    "warmup_steps": "--warmup-steps",
    "weight_decay": "--weight-decay",
    "beta1": "--beta1",
    "beta2": "--beta2",
    "grad_clip": "--grad-clip",
}


@dataclass(frozen=True)
class AttentionRuntime:
    """One static attention plan resolved before any real-step compilation."""

    key_digest: str | None
    resolution_source: str
    tiles: AttentionTiles | None
    tune_seconds: float

    def __post_init__(self) -> None:
        if self.resolution_source not in (
            "dense",
            "cache",
            "shipped",
            "heuristic",
            "autotuned",
        ):
            raise ValueError(
                f"invalid attention resolution source: {self.resolution_source!r}"
            )
        if not math.isfinite(self.tune_seconds) or self.tune_seconds < 0.0:
            raise ValueError("attention tune seconds must be finite and nonnegative")
        if self.resolution_source == "dense":
            if self.key_digest is not None or self.tiles is not None:
                raise ValueError("dense attention must not carry a tuning key or tiles")
        elif self.key_digest is None or self.tiles is None:
            raise ValueError("non-dense attention requires a tuning key and tile plan")


@dataclass(frozen=True)
class DocumentSpan:
    """The target interval for one document inside a token shard."""

    token_offset: int
    token_count: int
    score_offset: int
    scored_tokens: int


@dataclass(frozen=True)
class DownstreamDomain:
    name: str
    tokens: np.ndarray
    documents: tuple[DocumentSpan, ...]

    @property
    def scored_tokens(self) -> int:
        return sum(document.scored_tokens for document in self.documents)


@dataclass(frozen=True)
class ValidationRow:
    step: int
    tokens_processed: int
    kind: str
    domain: str
    validation_tokens: int
    validation_loss: float
    perplexity: float
    validation_seconds: float
    canonical: bool


@dataclass(frozen=True)
class DiagnosticPoint:
    """Host-side statistics captured at one optimizer step."""

    step: int
    values: np.ndarray


class Console:
    """Tiny ANSI renderer; avoids adding a UI dependency to the reference."""

    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[38;5;81m",
        "blue": "\033[38;5;75m",
        "green": "\033[38;5;114m",
        "yellow": "\033[38;5;221m",
        "magenta": "\033[38;5;176m",
        "red": "\033[38;5;203m",
        "white": "\033[38;5;255m",
    }

    def __init__(self, mode: str, *, active: bool = True) -> None:
        auto = sys.stderr.isatty() and "NO_COLOR" not in os.environ
        self.enabled = mode == "always" or (mode == "auto" and auto)
        self.active = active

    def paint(self, text: object, *styles: str) -> str:
        raw = str(text)
        if not self.enabled or not styles:
            return raw
        prefix = "".join(self.COLORS[s] for s in styles)
        return f"{prefix}{raw}{self.COLORS['reset']}"

    def banner(self) -> None:
        if not self.active:
            return
        mark = self.paint("◆", "magenta", "bold")
        title = self.paint(" GPT TPU SPEEDRUN ", "white", "bold")
        print(
            f"\n  {mark}{title}{self.paint('reference / jax', 'cyan')}\n",
            file=sys.stderr,
        )

    def table(self, title: str, rows: Sequence[tuple[str, object]]) -> None:
        if not self.active:
            return
        # Keep configuration cards readable in ordinary terminals. Provenance
        # remains complete in result.json/checkpoints; the live card is a
        # compact summary and must never grow to a digest- or JSON-sized width.
        width = min(
            78,
            max(52, *(max(20, len(str(k))) + len(str(v)) + 7 for k, v in rows)),
        )
        inner = width - 2
        heading = f" {title} "
        top_fill = max(0, inner - len(heading) - 1)
        print(
            "  "
            + self.paint("╭─", "blue")
            + self.paint(heading, "white", "bold")
            + self.paint("─" * top_fill + "╮", "blue"),
            file=sys.stderr,
        )
        for key, value in rows:
            raw_key = str(key)
            if len(raw_key) > 20:
                raw_key = raw_key[:19] + "…"
            key_text = f"{raw_key:<20}"
            value_text = str(value)
            value_limit = max(8, inner - 23)
            if len(value_text) > value_limit:
                value_text = value_text[: value_limit - 1] + "…"
            padding = max(1, inner - 2 - len(key_text) - len(value_text))
            print(
                "  "
                + self.paint("│", "blue")
                + " "
                + self.paint(key_text, "dim")
                + " " * padding
                + self.paint(value_text, "cyan", "bold")
                + " "
                + self.paint("│", "blue"),
                file=sys.stderr,
            )
        print(
            "  " + self.paint("╰" + "─" * inner + "╯", "blue"),
            file=sys.stderr,
        )

    def phase(self, label: str, detail: str = "") -> None:
        if not self.active:
            return
        suffix = f" {self.paint(detail, 'dim')}" if detail else ""
        print(
            f"\n  {self.paint('●', 'magenta')} {self.paint(label, 'white', 'bold')}{suffix}",
            file=sys.stderr,
        )

    def step(
        self,
        step: int,
        total: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_per_second: float,
    ) -> None:
        if not self.active:
            return
        fraction = step / total
        slots = 18
        filled = min(slots, int(round(fraction * slots)))
        bar = self.paint("━" * filled, "green") + self.paint("─" * (slots - filled), "dim")
        print(
            f"  {self.paint(f'{step:>4}/{total:<4}', 'white', 'bold')} "
            f"{bar}  loss {self.paint(f'{loss:.4f}', 'yellow', 'bold')}  "
            f"lr {lr:.2e}  |g| {grad_norm:.3f}  "
            f"{format_rate(tokens_per_second)} tok/s",
            file=sys.stderr,
        )

    def success(
        self,
        validation_loss: float,
        train_seconds: float,
        validation_seconds: float,
    ) -> None:
        if not self.active:
            return
        print(
            f"\n  {self.paint('✓', 'green', 'bold')} "
            f"synchronized training "
            f"{self.paint(f'{train_seconds:.3f}s', 'white', 'bold')} "
            f"{self.paint('(compilation excluded)', 'dim')}\n"
            f"    validation loss {self.paint(f'{validation_loss:.4f}', 'green', 'bold')} "
            f"in {self.paint(f'{validation_seconds:.3f}s', 'white', 'bold')}\n",
            file=sys.stderr,
        )

    def validation_probe(
        self, step: int, loss: float, batches: int, elapsed: float
    ) -> None:
        if not self.active:
            return
        print(
            f"  {self.paint('◇', 'cyan')} validation @ {step:,}  "
            f"loss {self.paint(f'{loss:.4f}', 'yellow', 'bold')}  "
            f"{batches} batches in {elapsed:.3f}s",
            file=sys.stderr,
        )

    def downstream(
        self, domain: str, loss: float, perplexity: float, tokens: int, elapsed: float
    ) -> None:
        if not self.active:
            return
        print(
            f"  {self.paint('◇', 'cyan')} {domain:<14} "
            f"loss {self.paint(f'{loss:.4f}', 'yellow', 'bold')}  "
            f"ppl {perplexity:.2f}  {tokens:,} tokens in {elapsed:.3f}s",
            file=sys.stderr,
        )


class ShardedTokens:
    """A zero-copy logical token stream backed by one or more mmap shards."""

    def __init__(self, shards: Sequence[np.ndarray]) -> None:
        self.shards = tuple(shards)
        if not self.shards:
            raise ValueError("a token split must contain at least one shard")

    def __len__(self) -> int:
        return sum(len(shard) for shard in self.shards)

    def sample(
        self, rng: np.random.Generator, batch_size: int, seq_len: int
    ) -> np.ndarray:
        # Selecting shards proportional to their number of valid starts samples
        # uniformly over all within-shard windows without concatenating gigabytes.
        valid_starts = np.asarray(
            [max(0, len(shard) - seq_len) for shard in self.shards], dtype=np.int64
        )
        total_starts = int(valid_starts.sum())
        if total_starts <= 0:
            raise ValueError(
                f"split has {len(self):,} tokens across {len(self.shards)} shard(s), "
                f"but none is long enough for seq_len + 1 ({seq_len + 1:,})"
            )
        choices = rng.choice(
            len(self.shards), size=batch_size, p=valid_starts / total_starts
        )
        windows = np.empty((batch_size, seq_len + 1), dtype=np.int32)
        offsets = np.arange(seq_len + 1)
        for shard_index in np.unique(choices):
            rows = np.flatnonzero(choices == shard_index)
            shard = self.shards[int(shard_index)]
            starts = rng.integers(0, len(shard) - seq_len, size=len(rows))
            windows[rows] = shard[starts[:, None] + offsets]
        return windows

    def sequential(self, batch_index: int, batch_size: int, seq_len: int) -> np.ndarray:
        """Return deterministic non-overlapping windows from the split prefix."""
        window_counts = np.asarray(
            [(len(shard) - 1) // seq_len for shard in self.shards], dtype=np.int64
        )
        total_windows = int(window_counts.sum())
        if total_windows <= 0:
            raise ValueError(f"no shard is long enough for sequence length {seq_len}")
        cumulative = np.cumsum(window_counts)
        windows = np.empty((batch_size, seq_len + 1), dtype=np.int32)
        for row in range(batch_size):
            logical_index = (batch_index * batch_size + row) % total_windows
            shard_index = int(np.searchsorted(cumulative, logical_index, side="right"))
            previous = int(cumulative[shard_index - 1]) if shard_index else 0
            local_index = logical_index - previous
            start = local_index * seq_len
            windows[row] = self.shards[shard_index][start : start + seq_len + 1]
        return windows


class TokenDataset:
    def __init__(self, train: ShardedTokens, validation: ShardedTokens, source: str) -> None:
        self.train = train
        self.validation = validation
        self.source = source

    def batch(
        self,
        split: str,
        rng: np.random.Generator,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.train if split == "train" else self.validation
        window = tokens.sample(rng, batch_size, seq_len)
        observed_min = int(window.min())
        observed_max = int(window.max())
        if observed_min < 0 or observed_max >= vocab_size:
            raise ValueError(
                f"sampled token ids [{observed_min}, {observed_max}] do not fit "
                f"vocab_size={vocab_size}; pass --vocab-size explicitly"
            )
        return np.ascontiguousarray(window[:, :-1]), np.ascontiguousarray(window[:, 1:])

    def validation_batch(
        self, batch_index: int, batch_size: int, seq_len: int, vocab_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        window = self.validation.sequential(batch_index, batch_size, seq_len)
        observed_min = int(window.min())
        observed_max = int(window.max())
        if observed_min < 0 or observed_max >= vocab_size:
            raise ValueError(
                f"validation token ids [{observed_min}, {observed_max}] do not fit "
                f"vocab_size={vocab_size}"
            )
        return np.ascontiguousarray(window[:, :-1]), np.ascontiguousarray(window[:, 1:])


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


def _config_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"config.yaml {label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"config.yaml {label} keys must be strings")
    return value


def _config_keys(
    value: Any,
    label: str,
    required: set[str],
    *,
    optional: set[str] = frozenset(),
) -> Mapping[str, Any]:
    mapping = _config_mapping(value, label)
    keys = set(mapping)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ValueError(
            f"config.yaml {label} is missing required key(s): {', '.join(missing)}"
        )
    if unknown:
        raise ValueError(
            f"config.yaml {label} contains unknown key(s): {', '.join(unknown)}"
        )
    return mapping


def _config_int(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"config.yaml {label} must be an integer >= {minimum}; got {value!r}"
        )
    return value


def _config_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config.yaml {label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"config.yaml {label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"config.yaml {label} must be a finite number")
    return result


def _config_choice(value: Any, label: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"config.yaml {label} must be one of {', '.join(choices)}; got {value!r}"
        )
    return value


def resolve_experiment_config_path(requested: Path | None) -> Path:
    """Resolve the one accepted config path: config.yaml beside this trainer."""

    if CONFIG_PATH.is_symlink() or not CONFIG_PATH.is_file():
        raise ValueError(
            f"required sibling experiment config must be a regular, non-symlink file: {CONFIG_PATH}"
        )
    try:
        expected = CONFIG_PATH.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"required sibling experiment config is unavailable: {CONFIG_PATH}") from exc
    candidate = CONFIG_PATH if requested is None else requested
    if candidate.is_symlink():
        raise ValueError("--config may not be a symlink")
    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"experiment config is unavailable: {candidate}") from exc
    if resolved != expected:
        raise ValueError(
            f"--config must name the config.yaml beside train.py: {CONFIG_PATH}"
        )
    return expected


def _parse_experiment_profile(
    payload: Mapping[str, Any], profile: str, source_sha256: str
) -> ExperimentProfile:
    top = _config_keys(payload, "document", {"schema_version", "profiles"})
    schema_version = _config_int(
        top["schema_version"], "schema_version", minimum=1
    )
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "unsupported config.yaml schema_version "
            f"{schema_version}; expected {CONFIG_SCHEMA_VERSION}"
        )
    profiles = _config_keys(
        top["profiles"], "profiles", set(_VALID_PROFILES)
    )
    selected = _config_keys(
        profiles[profile],
        f"profiles.{profile}",
        {"training", "model", "kernels", "optimizer", "evaluation", "logging"},
    )
    training = _config_keys(
        selected["training"],
        f"profiles.{profile}.training",
        {"batch_size", "seq_len", "dtype"},
        optional={"steps", "train_tokens"},
    )
    has_steps = "steps" in training
    has_train_tokens = "train_tokens" in training
    if has_steps == has_train_tokens:
        raise ValueError(
            f"config.yaml profiles.{profile}.training must define exactly one of "
            "steps or train_tokens"
        )
    model = _config_keys(
        selected["model"],
        f"profiles.{profile}.model",
        {"layers", "heads", "d_model", "mlp_mult", "vocab_size", "semantic_vocab_size"},
    )
    kernels = _config_keys(
        selected["kernels"],
        f"profiles.{profile}.kernels",
        {"attention_backend", "loss_backend", "vocab_tile_size"},
    )
    optimizer = _config_keys(
        selected["optimizer"],
        f"profiles.{profile}.optimizer",
        {
            "learning_rate", "min_lr_ratio", "warmup_steps", "weight_decay",
            "beta1", "beta2", "grad_clip",
        },
    )
    evaluation = _config_keys(
        selected["evaluation"],
        f"profiles.{profile}.evaluation",
        {"eval_batches", "val_every", "val_probe_batches"},
    )
    logging = _config_keys(
        selected["logging"],
        f"profiles.{profile}.logging",
        {"diagnostics_every", "log_every"},
    )
    prefix = f"profiles.{profile}"
    learning_rate = _config_float(
        optimizer["learning_rate"], f"{prefix}.optimizer.learning_rate"
    )
    min_lr_ratio = _config_float(
        optimizer["min_lr_ratio"], f"{prefix}.optimizer.min_lr_ratio"
    )
    weight_decay = _config_float(
        optimizer["weight_decay"], f"{prefix}.optimizer.weight_decay"
    )
    beta1 = _config_float(optimizer["beta1"], f"{prefix}.optimizer.beta1")
    beta2 = _config_float(optimizer["beta2"], f"{prefix}.optimizer.beta2")
    grad_clip = _config_float(
        optimizer["grad_clip"], f"{prefix}.optimizer.grad_clip"
    )
    if learning_rate <= 0.0:
        raise ValueError(f"config.yaml {prefix}.optimizer.learning_rate must be positive")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"config.yaml {prefix}.optimizer.min_lr_ratio must be in [0, 1]")
    if weight_decay < 0.0:
        raise ValueError(f"config.yaml {prefix}.optimizer.weight_decay must be nonnegative")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError(f"config.yaml {prefix}.optimizer beta values must be in [0, 1)")
    if grad_clip <= 0.0:
        raise ValueError(f"config.yaml {prefix}.optimizer.grad_clip must be positive")
    result = ExperimentProfile(
        schema_version=schema_version,
        source_sha256=source_sha256,
        name=profile,
        steps=(
            _config_int(training["steps"], f"{prefix}.training.steps", minimum=1)
            if has_steps else None
        ),
        train_tokens=(
            _config_int(
                training["train_tokens"], f"{prefix}.training.train_tokens", minimum=1
            ) if has_train_tokens else None
        ),
        batch_size=_config_int(
            training["batch_size"], f"{prefix}.training.batch_size", minimum=1
        ),
        seq_len=_config_int(
            training["seq_len"], f"{prefix}.training.seq_len", minimum=1
        ),
        dtype_name=_config_choice(
            training["dtype"], f"{prefix}.training.dtype", ("bfloat16", "float32")
        ),
        layers=_config_int(model["layers"], f"{prefix}.model.layers", minimum=1),
        heads=_config_int(model["heads"], f"{prefix}.model.heads", minimum=1),
        d_model=_config_int(model["d_model"], f"{prefix}.model.d_model", minimum=1),
        mlp_mult=_config_int(model["mlp_mult"], f"{prefix}.model.mlp_mult", minimum=1),
        vocab_size=_config_int(
            model["vocab_size"], f"{prefix}.model.vocab_size", minimum=1
        ),
        semantic_vocab_size=_config_int(
            model["semantic_vocab_size"],
            f"{prefix}.model.semantic_vocab_size",
            minimum=1,
        ),
        attention_backend=_config_choice(
            kernels["attention_backend"],
            f"{prefix}.kernels.attention_backend",
            ("dense", "jax_flash", "tpu_flash"),
        ),
        loss_backend=_config_choice(
            kernels["loss_backend"], f"{prefix}.kernels.loss_backend", ("dense", "tiled")
        ),
        vocab_tile_size=_config_int(
            kernels["vocab_tile_size"], f"{prefix}.kernels.vocab_tile_size", minimum=1
        ),
        learning_rate=learning_rate,
        min_lr_ratio=min_lr_ratio,
        warmup_steps=_config_int(
            optimizer["warmup_steps"], f"{prefix}.optimizer.warmup_steps", minimum=0
        ),
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        grad_clip=grad_clip,
        eval_batches=_config_int(
            evaluation["eval_batches"], f"{prefix}.evaluation.eval_batches", minimum=1
        ),
        val_every=_config_int(
            evaluation["val_every"], f"{prefix}.evaluation.val_every", minimum=0
        ),
        val_probe_batches=_config_int(
            evaluation["val_probe_batches"],
            f"{prefix}.evaluation.val_probe_batches",
            minimum=1,
        ),
        diagnostics_every=_config_int(
            logging["diagnostics_every"], f"{prefix}.logging.diagnostics_every", minimum=0
        ),
        log_every=_config_int(
            logging["log_every"], f"{prefix}.logging.log_every", minimum=1
        ),
    )
    if result.semantic_vocab_size > result.vocab_size:
        raise ValueError(
            f"config.yaml {prefix}.model.semantic_vocab_size must not exceed vocab_size"
        )
    if result.d_model % result.heads:
        raise ValueError(f"config.yaml {prefix}.model.d_model must be divisible by heads")
    if result.attention_backend != "dense" and result.dtype_name != "bfloat16":
        raise ValueError(
            f"config.yaml {prefix}.kernels.attention_backend "
            f"{result.attention_backend} requires training.dtype bfloat16"
        )
    tokens_per_step = result.batch_size * result.seq_len
    if result.train_tokens is not None and result.train_tokens % tokens_per_step:
        raise ValueError(
            f"config.yaml {prefix}.training.train_tokens must be divisible by "
            f"batch_size * seq_len ({tokens_per_step:,})"
        )
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
    profile: str, requested_path: Path | None = None
) -> ExperimentProfile:
    if profile not in _VALID_PROFILES:
        raise ValueError(f"unknown experiment profile: {profile!r}")
    path = resolve_experiment_config_path(requested_path)
    raw = path.read_bytes()
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ValueError(
            f"config.yaml exceeds the {_MAX_CONFIG_BYTES:,}-byte safety limit"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("config.yaml must be UTF-8") from exc
    try:
        forbidden_tokens = (
            yaml.tokens.AliasToken,
            yaml.tokens.AnchorToken,
            yaml.tokens.DirectiveToken,
            yaml.tokens.TagToken,
        )
        for token in yaml.scan(text, Loader=_StrictSafeLoader):
            if isinstance(token, forbidden_tokens):
                kind = type(token).__name__.removesuffix("Token").lower()
                raise ValueError(f"config.yaml may not contain YAML {kind}s")
        documents = list(yaml.load_all(text, Loader=_StrictSafeLoader))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid config.yaml YAML: {exc}") from exc
    if len(documents) != 1:
        raise ValueError("config.yaml must contain exactly one YAML document")
    mapping = _config_mapping(documents[0], "document")
    source_sha256 = hashlib.sha256(raw).hexdigest()
    parsed = {
        name: _parse_experiment_profile(mapping, name, source_sha256)
        for name in _VALID_PROFILES
    }
    return parsed[profile]


def reject_static_cli_overrides(args: argparse.Namespace) -> None:
    for destination, option in _STATIC_CLI_FIELDS.items():
        if getattr(args, destination) is not None:
            raise ValueError(
                f"{option} is defined by sibling config.yaml; edit the selected "
                "profile or clone a new submission variant"
            )


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
    duration = run.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=positive_int, default=None)
    duration.add_argument(
        "--train-tokens",
        type=positive_int,
        default=None,
        help="derive an exact step count from the global batch and sequence length",
    )
    environment_track = os.environ.get("SPEEDRUN_TRACK", "open")
    if environment_track not in _VALID_TRACKS:
        environment_track = "open"
    environment_profile = os.environ.get("SPEEDRUN_PROFILE")
    if environment_profile not in _VALID_PROFILES:
        environment_profile = None
    run.add_argument("--track", choices=_VALID_TRACKS, default=environment_track)
    run.add_argument(
        "--profile", choices=_VALID_PROFILES, default=environment_profile
    )
    run.add_argument(
        "--smoke", action="store_true", help="alias for --profile smoke"
    )
    run.add_argument(
        "--eval-batches", type=positive_int, default=None, help=argparse.SUPPRESS
    )
    run.add_argument(
        "--val-every",
        type=nonnegative_int,
        default=None,
        help="run a deterministic validation probe every N optimizer steps; 0 disables probes",
    )
    run.add_argument(
        "--val-probe-batches",
        type=positive_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    run.add_argument(
        "--diagnostics-every",
        type=nonnegative_int,
        default=None,
        help=(
            "capture sparse parameter, pre-clipping gradient, and actual update "
            "statistics every N steps; 0 disables diagnostics"
        ),
    )
    run.add_argument("--log-every", type=positive_int, default=None)
    run.add_argument("--color", choices=("auto", "always", "never"), default="auto")

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
        "--no-final-validation",
        action="store_true",
        help="diagnostic-only: omit evaluation compilation and final validation",
    )
    profiling.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="diagnostic-only: omit the checkpoint and competition result",
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
    data.add_argument("--dataset-id", default=None, help="stable dataset identifier for records")
    data.add_argument("--tokenizer-id", default=None, help="stable tokenizer identifier for records")
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

    model = parser.add_argument_group("model")
    model.add_argument("--batch-size", type=positive_int, default=None, help=argparse.SUPPRESS)
    model.add_argument("--seq-len", type=positive_int, default=None, help=argparse.SUPPRESS)
    model.add_argument("--layers", type=positive_int, default=None, help=argparse.SUPPRESS)
    model.add_argument("--heads", type=positive_int, default=None, help=argparse.SUPPRESS)
    model.add_argument("--d-model", type=positive_int, default=None, help=argparse.SUPPRESS)
    model.add_argument("--mlp-mult", type=positive_int, default=None, help=argparse.SUPPRESS)
    model.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default=None,
        help=argparse.SUPPRESS,
    )
    model.add_argument(
        "--attention-backend",
        choices=("dense", "jax_flash", "tpu_flash"),
        default=None,
        help=argparse.SUPPRESS,
    )
    model.add_argument(
        "--attention-tuning-cache",
        type=Path,
        default=None,
        help=(
            "JSON cache for exact runtime-fingerprinted attention tile plans; "
            "used only by non-dense attention"
        ),
    )
    model.add_argument(
        "--autotune-attention",
        action="store_true",
        help=(
            "AOT-compile and benchmark synthetic attention candidates before "
            "the real train-step compilation"
        ),
    )
    model.add_argument(
        "--loss-backend",
        choices=("dense", "tiled"),
        default=None,
        help=argparse.SUPPRESS,
    )
    model.add_argument(
        "--semantic-vocab-size",
        type=positive_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    model.add_argument(
        "--vocab-tile-size",
        type=positive_int,
        default=None,
        help=argparse.SUPPRESS,
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--learning-rate", type=float, default=None, help=argparse.SUPPRESS)
    optim.add_argument("--min-lr-ratio", type=float, default=None, help=argparse.SUPPRESS)
    optim.add_argument("--warmup-steps", type=nonnegative_int, default=None, help=argparse.SUPPRESS)
    optim.add_argument("--weight-decay", type=float, default=None, help=argparse.SUPPRESS)
    optim.add_argument("--beta1", type=float, default=None, help=argparse.SUPPRESS)
    optim.add_argument("--beta2", type=float, default=None, help=argparse.SUPPRESS)
    optim.add_argument("--grad-clip", type=float, default=None, help=argparse.SUPPRESS)
    optim.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="hardware bf16 peak for the whole mesh; enables an MFU estimate",
    )
    return parser


def validate_args(args: argparse.Namespace) -> ExperimentProfile:
    if args.smoke and args.profile not in (None, "smoke"):
        raise ValueError("--smoke cannot be combined with a non-smoke --profile")
    reject_static_cli_overrides(args)
    experiment = load_experiment_profile(selected_profile(args), args.config)
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.peak_tflops is not None and (
        not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0.0
    ):
        raise ValueError("--peak-tflops must be positive")
    if args.autotune_attention and args.attention_tuning_cache is None:
        raise ValueError(
            "--autotune-attention requires --attention-tuning-cache PATH"
        )
    if (
        args.attention_tuning_cache is not None
        and experiment.attention_backend == "dense"
    ):
        raise ValueError(
            "--attention-tuning-cache requires a non-dense attention_backend "
            "in config.yaml"
        )
    if args.downstream_root is not None and args.downstream_manifest is None:
        raise ValueError("--downstream-root requires --downstream-manifest")
    if args.downstream_manifest is not None and args.downstream_data:
        raise ValueError(
            "--downstream-manifest and --downstream-data are mutually exclusive"
        )
    xprof_window_args = (args.xprof_start_step, args.xprof_steps)
    if args.xprof_dir is None:
        if any(value is not None for value in xprof_window_args):
            raise ValueError(
                "--xprof-start-step and --xprof-steps require --xprof-dir"
            )
        if args.no_final_validation or args.no_checkpoint:
            raise ValueError(
                "--no-final-validation and --no-checkpoint require --xprof-dir"
            )
    elif any(value is None for value in xprof_window_args):
        raise ValueError(
            "--xprof-dir requires both --xprof-start-step and --xprof-steps"
        )
    if args.no_final_validation != args.no_checkpoint:
        raise ValueError(
            "--no-final-validation and --no-checkpoint must be used together"
        )
    if args.no_final_validation and (
        args.downstream_manifest is not None or args.downstream_data
    ):
        raise ValueError(
            "--no-final-validation cannot be combined with downstream evaluation data"
        )
    effective_val_every = (
        args.val_every if args.val_every is not None else experiment.val_every
    )
    if args.no_final_validation and effective_val_every:
        raise ValueError(
            "--no-final-validation requires --val-every 0 (the official profile "
            "otherwise enables periodic validation by default)"
        )
    return experiment


def xprof_step_window(
    args: argparse.Namespace, total_steps: int
) -> tuple[int, int] | None:
    """Return the inclusive 1-based capture window, validating its bounds."""

    if args.xprof_dir is None:
        return None
    # ``validate_args`` establishes that these are both positive integers.
    start = int(args.xprof_start_step)
    end = start + int(args.xprof_steps) - 1
    if start > total_steps or end > total_steps:
        raise ValueError(
            "XProf capture window must fit inside the training run; "
            f"requested steps {start}..{end} with --steps {total_steps}"
        )
    return start, end


def profiler_options(platform: str, device_count: int) -> Any:
    """Build an XProf configuration with useful TPU compute and sync events."""

    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 2
    if platform == "tpu":
        options.advanced_configuration = {
            "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
            "tpu_num_chips_to_profile_per_task": device_count,
        }
    return options


def should_compile_evaluation(
    args: argparse.Namespace,
    config: Config,
    downstream_domains: Sequence[DownstreamDomain],
) -> bool:
    """Return whether this invocation can execute any validation workload."""

    return (
        not args.no_final_validation
        or config.val_every > 0
        or bool(downstream_domains)
    )


def selected_profile(args: argparse.Namespace) -> str:
    return "smoke" if args.smoke else (args.profile or "dev")


def resolve_config(
    args: argparse.Namespace,
    platform: str,
    vocab_size: int,
    experiment: ExperimentProfile | None = None,
) -> Config:
    profile = selected_profile(args)
    reject_static_cli_overrides(args)
    experiment = experiment or load_experiment_profile(profile, args.config)
    if experiment.name != profile:
        raise ValueError(
            f"resolved config profile {experiment.name!r} does not match {profile!r}"
        )
    if vocab_size != experiment.vocab_size:
        raise ValueError(
            "loaded dataset vocabulary does not match config.yaml: "
            f"dataset={vocab_size}, configured={experiment.vocab_size}"
        )
    batch_size = experiment.batch_size
    seq_len = experiment.seq_len
    tokens_per_step = batch_size * seq_len
    requested_train_tokens = (
        args.train_tokens
        if args.train_tokens is not None
        else (None if args.steps is not None else experiment.train_tokens)
    )
    if requested_train_tokens is not None:
        if requested_train_tokens % tokens_per_step:
            raise ValueError(
                "training token budget must be divisible by batch_size * seq_len "
                f"({tokens_per_step:,}); got {requested_train_tokens:,}"
            )
        steps = requested_train_tokens // tokens_per_step
    else:
        steps = args.steps if args.steps is not None else experiment.steps
        if steps is None:  # schema validation establishes a duration, defensively retain type.
            raise AssertionError("experiment duration did not resolve")
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
    val_every = args.val_every if args.val_every is not None else experiment.val_every
    val_probe_batches = experiment.val_probe_batches
    if val_every > 0 and val_probe_batches > eval_batches:
        raise ValueError(
            "config.yaml val_probe_batches must not exceed the canonical evaluation batch "
            f"count ({eval_batches}); got {val_probe_batches}"
        )
    log_every = args.log_every if args.log_every is not None else experiment.log_every
    diagnostics_every = (
        args.diagnostics_every
        if args.diagnostics_every is not None
        else experiment.diagnostics_every
    )
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
    overrides = tuple(
        (name, int(value))
        for name, value in (
            ("steps", args.steps),
            ("train_tokens", args.train_tokens),
            ("val_every", args.val_every),
            ("diagnostics_every", args.diagnostics_every),
            ("log_every", args.log_every),
        )
        if value is not None
    )
    return Config(
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        layers=experiment.layers,
        heads=experiment.heads,
        d_model=experiment.d_model,
        mlp_mult=experiment.mlp_mult,
        learning_rate=experiment.learning_rate,
        min_lr_ratio=experiment.min_lr_ratio,
        warmup_steps=experiment.warmup_steps,
        weight_decay=experiment.weight_decay,
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
        config_overrides=overrides,
    )


def built_in_dataset(seed: int) -> TokenDataset:
    rng = np.random.default_rng(seed)
    chunks: list[str] = []
    # Enough data for random windows without wasting repository space.
    for epoch in range(1536):
        order = rng.permutation(len(_CORPUS_LINES))
        chunks.append(f"\n[pass {epoch:04d}]\n")
        chunks.extend(_CORPUS_LINES[int(index)] + "\n" for index in order)
    tokens = np.frombuffer("".join(chunks).encode("utf-8"), dtype=np.uint8)
    split = int(len(tokens) * 0.95)
    return TokenDataset(
        ShardedTokens((tokens[:split],)),
        ShardedTokens((tokens[split:],)),
        "built-in byte corpus",
    )


def find_split_files(directory: Path, split: str) -> list[Path]:
    patterns = (
        (
            "fineweb_train_*.bin",
            "*_train_*.bin",
            "train_*.bin",
            "train.bin",
            "train.npy",
            "train.txt",
        )
        if split == "train"
        else (
            "fineweb_val_*.bin",
            "*_val_*.bin",
            "val_*.bin",
            "validation_*.bin",
            "val.bin",
            "validation.bin",
            "val.npy",
            "validation.npy",
            "val.txt",
        )
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in directory.glob(pattern) if path.is_file())
    return sorted(paths)


def load_token_file(path: Path, raw_dtype: str, data_format: str) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"token data not found: {path}")
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 1:
            raise ValueError(f"expected a one-dimensional token array in {path}")
        return values
    if path.suffix in (".txt", ".text"):
        return np.frombuffer(path.read_bytes(), dtype=np.uint8)
    with path.open("rb") as handle:
        header_bytes = handle.read(1024)
    header = (
        np.frombuffer(header_bytes, dtype="<i4", count=256)
        if len(header_bytes) == 1024
        else np.empty((0,), dtype=np.int32)
    )
    is_llmc = len(header) == 256 and int(header[0]) == 20_240_520
    if data_format == "llmc" and not is_llmc:
        raise ValueError(f"{path} does not have an llm.c FineWeb header")
    if data_format != "raw" and is_llmc:
        if int(header[1]) != 1:
            raise ValueError(f"unsupported llm.c data version {int(header[1])} in {path}")
        token_count = int(header[2])
        if token_count <= 0:
            raise ValueError(f"invalid llm.c token count {token_count} in {path}")
        expected_bytes = 1024 + token_count * np.dtype("<u2").itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"bad llm.c shard size for {path}: expected {expected_bytes:,} bytes, "
                f"found {path.stat().st_size:,}"
            )
        return np.memmap(path, dtype="<u2", mode="r", offset=1024, shape=(token_count,))
    dtype = np.dtype(raw_dtype)
    if path.stat().st_size % dtype.itemsize:
        raise ValueError(f"{path} size is not a multiple of dtype {raw_dtype}")
    return np.memmap(path, dtype=dtype, mode="r")


def split_shards(
    shards: Sequence[np.ndarray], validation_fraction: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    total = sum(len(shard) for shard in shards)
    train_limit = int(total * (1.0 - validation_fraction))
    if train_limit <= 0 or train_limit >= total:
        raise ValueError("data is too short to create nonempty train and validation splits")
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    cursor = 0
    for shard in shards:
        shard_end = cursor + len(shard)
        if shard_end <= train_limit:
            train.append(shard)
        elif cursor >= train_limit:
            validation.append(shard)
        else:
            cut = train_limit - cursor
            train.append(shard[:cut])
            validation.append(shard[cut:])
        cursor = shard_end
    return train, validation


def load_dataset(
    args: argparse.Namespace, experiment: ExperimentProfile | None = None
) -> tuple[TokenDataset, int]:
    experiment = experiment or load_experiment_profile(
        selected_profile(args), args.config
    )
    configured_vocab_size = experiment.vocab_size
    if args.data_path is None and not args.train_data and not args.val_data:
        dataset = built_in_dataset(args.seed)
        return dataset, configured_vocab_size

    train_paths = [path.expanduser().resolve() for path in args.train_data]
    validation_paths = [path.expanduser().resolve() for path in args.val_data]
    if args.data_path is not None:
        data_path = args.data_path.expanduser().resolve()
        if data_path.is_dir():
            train_paths.extend(find_split_files(data_path, "train"))
            if not validation_paths:
                validation_paths.extend(find_split_files(data_path, "val"))
        else:
            train_paths.append(data_path)
    # Preserve CLI/discovery order but prevent a path from being sampled twice.
    train_paths = list(dict.fromkeys(train_paths))
    validation_paths = list(dict.fromkeys(validation_paths))
    if not train_paths:
        location = args.data_path if args.data_path is not None else "explicit arguments"
        raise FileNotFoundError(f"no training shards found from {location}")

    train_shards = [
        load_token_file(path, args.data_dtype, args.data_format) for path in train_paths
    ]
    if validation_paths:
        validation_shards = [
            load_token_file(path, args.data_dtype, args.data_format)
            for path in validation_paths
        ]
    else:
        train_shards, validation_shards = split_shards(train_shards, args.val_fraction)

    source = f"{len(train_shards)} train + {len(validation_shards)} val shard(s)"
    return (
        TokenDataset(ShardedTokens(train_shards), ShardedTokens(validation_shards), source),
        configured_vocab_size,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"fresh10 {field} must be an integer >= {minimum}")
    return value


def _validate_domain(
    name: Any,
    tokens: np.ndarray,
    documents_payload: Sequence[Mapping[str, Any]],
    *,
    expected_scored_tokens: Any = None,
    vocab_size: int,
) -> DownstreamDomain:
    if not isinstance(name, str) or not _DOMAIN_NAME.fullmatch(name):
        raise ValueError(f"invalid downstream domain name: {name!r}")
    if not documents_payload:
        raise ValueError(f"downstream domain {name!r} contains no documents")
    documents: list[DocumentSpan] = []
    previous_end = 0
    for index, document in enumerate(documents_payload):
        if not isinstance(document, Mapping):
            raise ValueError(f"fresh10 {name} document {index} must be an object")
        prefix = f"{name}.documents[{index}]"
        token_offset = _manifest_integer(document.get("token_offset"), f"{prefix}.token_offset")
        token_count = _manifest_integer(
            document.get("token_count"), f"{prefix}.token_count", minimum=2
        )
        score_offset = _manifest_integer(
            document.get("score_offset"), f"{prefix}.score_offset", minimum=1
        )
        scored_tokens = _manifest_integer(
            document.get("scored_tokens"), f"{prefix}.scored_tokens", minimum=1
        )
        token_end = token_offset + token_count
        score_end = score_offset + scored_tokens
        if token_offset < previous_end:
            raise ValueError(f"fresh10 {name} document spans overlap or are unsorted")
        if score_offset <= token_offset or score_end > token_end:
            raise ValueError(
                f"fresh10 {prefix} scored interval must follow a context token "
                "and stay inside its document"
            )
        if token_end > len(tokens):
            raise ValueError(
                f"fresh10 {prefix} ends at token {token_end:,}, beyond shard "
                f"length {len(tokens):,}"
            )
        previous_end = token_end
        documents.append(
            DocumentSpan(token_offset, token_count, score_offset, scored_tokens)
        )
    total_scored = sum(document.scored_tokens for document in documents)
    if expected_scored_tokens is not None:
        expected = _manifest_integer(
            expected_scored_tokens, f"{name}.scored_tokens", minimum=1
        )
        if total_scored != expected:
            raise ValueError(
                f"fresh10 {name} spans score {total_scored:,} tokens; expected "
                f"{expected:,}"
            )
    observed = np.asarray(tokens)
    observed_min = int(observed.min())
    observed_max = int(observed.max())
    if observed_min < 0 or observed_max >= vocab_size:
        raise ValueError(
            f"downstream domain {name!r} token ids [{observed_min}, {observed_max}] "
            f"do not fit vocab_size={vocab_size}"
        )
    return DownstreamDomain(name, tokens, tuple(documents))


def load_downstream_domains(
    args: argparse.Namespace, vocab_size: int
) -> tuple[DownstreamDomain, ...]:
    """Load canonical manifest shards or repeatable standalone documents."""

    if args.downstream_manifest is None and not args.downstream_data:
        return ()
    if args.downstream_manifest is not None:
        manifest_path = args.downstream_manifest.expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"downstream manifest not found: {manifest_path}")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid downstream manifest JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("downstream manifest must contain a JSON object")
        if payload.get("schema_version") != 1 or payload.get("kind") != "fresh10":
            raise ValueError(
                "downstream manifest must use schema_version=1 and kind='fresh10'"
            )
        tokenizer = payload.get("tokenizer")
        tokenizer_vocab = tokenizer.get("vocab_size") if isinstance(tokenizer, dict) else None
        if (
            not isinstance(tokenizer, dict)
            or tokenizer.get("name") != "gpt2"
            or isinstance(tokenizer_vocab, bool)
            or not isinstance(tokenizer_vocab, int)
            or tokenizer_vocab != 50_257
            or tokenizer_vocab > vocab_size
        ):
            raise ValueError(
                "downstream manifest must use the 50,257-token GPT-2 tokenizer, "
                f"which must fit the model vocabulary ({vocab_size})"
            )
        root = (
            args.downstream_root.expanduser().resolve()
            if args.downstream_root is not None
            else manifest_path.parent
        )
        domain_payloads = payload.get("domains")
        if not isinstance(domain_payloads, list) or not domain_payloads:
            raise ValueError("downstream manifest domains must be a nonempty list")
        domains: list[DownstreamDomain] = []
        seen: set[str] = set()
        for entry in domain_payloads:
            if not isinstance(entry, dict):
                raise ValueError("each downstream manifest domain must be an object")
            name = entry.get("name")
            if not isinstance(name, str) or not _DOMAIN_NAME.fullmatch(name):
                raise ValueError(f"invalid downstream domain name: {name!r}")
            if name in seen:
                raise ValueError(f"duplicate downstream domain: {name}")
            seen.add(name)
            relative = entry.get("path")
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"fresh10 {name}.path must be a nonempty string")
            unresolved = Path(relative)
            if unresolved.is_absolute() or ".." in unresolved.parts:
                raise ValueError(f"fresh10 {name}.path must stay below the data root")
            shard_path = (root / unresolved).resolve()
            try:
                shard_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"fresh10 {name}.path escapes the data root") from exc
            expected_bytes = entry.get("bytes")
            if expected_bytes is not None and shard_path.stat().st_size != _manifest_integer(
                expected_bytes, f"{name}.bytes", minimum=1
            ):
                raise ValueError(f"fresh10 {name} shard size does not match its manifest")
            expected_hash = entry.get("sha256")
            if expected_hash is not None:
                if not isinstance(expected_hash, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", expected_hash
                ):
                    raise ValueError(f"fresh10 {name}.sha256 is invalid")
                if file_sha256(shard_path) != expected_hash:
                    raise ValueError(f"fresh10 {name} shard SHA-256 does not match")
            tokens = load_token_file(shard_path, "uint16", "llmc")
            expected_tokens = _manifest_integer(
                entry.get("tokens"), f"{name}.tokens", minimum=2
            )
            if len(tokens) != expected_tokens:
                raise ValueError(
                    f"fresh10 {name} has {len(tokens):,} tokens; expected "
                    f"{expected_tokens:,}"
                )
            documents = entry.get("documents")
            if not isinstance(documents, list):
                raise ValueError(f"fresh10 {name}.documents must be a list")
            domains.append(
                _validate_domain(
                    name,
                    tokens,
                    documents,
                    expected_scored_tokens=entry.get("scored_tokens"),
                    vocab_size=vocab_size,
                )
            )
        return tuple(domains)

    grouped: dict[str, list[np.ndarray]] = {}
    for specification in args.downstream_data:
        if not isinstance(specification, str) or "=" not in specification:
            raise ValueError("--downstream-data must use DOMAIN=PATH")
        name, raw_path = specification.split("=", 1)
        if not _DOMAIN_NAME.fullmatch(name) or not raw_path:
            raise ValueError(f"invalid --downstream-data value: {specification!r}")
        grouped.setdefault(name, []).append(
            load_token_file(Path(raw_path), args.data_dtype, "auto")
        )
    domains = []
    for name, shards in grouped.items():
        documents_payload: list[dict[str, int]] = []
        cursor = 0
        for shard in shards:
            if len(shard) < 2:
                raise ValueError(f"downstream document for {name!r} has fewer than 2 tokens")
            documents_payload.append(
                {
                    "token_offset": cursor,
                    "token_count": len(shard),
                    "score_offset": cursor + 1,
                    "scored_tokens": len(shard) - 1,
                }
            )
            cursor += len(shard)
        tokens = np.concatenate(tuple(np.asarray(shard) for shard in shards))
        domains.append(
            _validate_domain(name, tokens, documents_payload, vocab_size=vocab_size)
        )
    return tuple(domains)


def normal(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)


def init_params(config: Config, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    d_model = config.d_model
    hidden = config.mlp_mult * d_model
    residual_scale = 0.02 / math.sqrt(2.0 * config.layers)
    blocks: list[dict[str, np.ndarray]] = []
    for _ in range(config.layers):
        blocks.append(
            {
                "ln1_scale": np.ones((d_model,), dtype=np.float32),
                "ln1_bias": np.zeros((d_model,), dtype=np.float32),
                "qkv_w": normal(rng, (d_model, 3 * d_model), 0.02),
                "qkv_b": np.zeros((3 * d_model,), dtype=np.float32),
                "attn_w": normal(rng, (d_model, d_model), residual_scale),
                "attn_b": np.zeros((d_model,), dtype=np.float32),
                "ln2_scale": np.ones((d_model,), dtype=np.float32),
                "ln2_bias": np.zeros((d_model,), dtype=np.float32),
                "mlp_up_w": normal(rng, (d_model, hidden), 0.02),
                "mlp_up_b": np.zeros((hidden,), dtype=np.float32),
                "mlp_down_w": normal(rng, (hidden, d_model), residual_scale),
                "mlp_down_b": np.zeros((d_model,), dtype=np.float32),
            }
        )
    return {
        "token_embedding": normal(rng, (config.vocab_size, d_model), 0.02),
        "position_embedding": normal(rng, (config.seq_len, d_model), 0.01),
        "blocks": blocks,
        "final_ln_scale": np.ones((d_model,), dtype=np.float32),
        "final_ln_bias": np.zeros((d_model,), dtype=np.float32),
    }


def layer_norm(x: jax.Array, scale: jax.Array, bias: jax.Array, dtype: Any) -> jax.Array:
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x32 - mean), axis=-1, keepdims=True)
    normalized = ((x32 - mean) * jax.lax.rsqrt(variance + 1.0e-5)).astype(dtype)
    return normalized * scale.astype(dtype) + bias.astype(dtype)


def linear(x: jax.Array, weight: jax.Array, bias: jax.Array, dtype: Any) -> jax.Array:
    return jnp.einsum("...d,df->...f", x, weight.astype(dtype)) + bias.astype(dtype)


AttentionCallable = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def attention_runtime_metadata(runtime: AttentionRuntime) -> dict[str, Any]:
    """Return the JSON-safe attention tile provenance shared by all artifacts."""

    return {
        "key_digest": runtime.key_digest,
        "resolution_source": runtime.resolution_source,
        "tune_seconds": float(runtime.tune_seconds),
        "tiles": None if runtime.tiles is None else runtime.tiles.to_dict(),
    }


def attention_console_rows(
    runtime: AttentionRuntime,
) -> tuple[tuple[str, str], ...]:
    """Return compact, terminal-safe attention provenance rows."""

    if runtime.tiles is None:
        return (
            ("attention tuning", "not applicable (dense)"),
            ("attention plan", "not applicable"),
        )
    digest = runtime.key_digest or "unknown"
    timing = (
        f" · {runtime.tune_seconds:.3f}s"
        if runtime.tune_seconds > 0.0
        else ""
    )
    tiles = runtime.tiles
    assert tiles.block_q_dkv is not None
    assert tiles.block_q_dkv_compute is not None
    assert tiles.block_kv_dkv is not None
    assert tiles.block_kv_dkv_compute is not None
    assert tiles.block_q_dq is not None
    assert tiles.block_kv_dq is not None
    assert tiles.block_kv_dq_compute is not None
    return (
        (
            "attention tuning",
            f"{runtime.resolution_source}{timing} · key {digest[:12]}",
        ),
        (
            "attention fwd",
            f"q{tiles.block_q} · kv{tiles.block_kv}/{tiles.block_kv_compute}",
        ),
        (
            "attention dK/dV",
            f"q{tiles.block_q_dkv}/{tiles.block_q_dkv_compute} · "
            f"kv{tiles.block_kv_dkv}/{tiles.block_kv_dkv_compute}",
        ),
        (
            "attention dQ",
            f"q{tiles.block_q_dq} · "
            f"kv{tiles.block_kv_dq}/{tiles.block_kv_dq_compute}",
        ),
    )


def contract_model_metadata(config: Config) -> dict[str, Any]:
    """Return the fixed sample-efficiency architecture contract only."""

    return {
        "layers": config.layers,
        "heads": config.heads,
        "d_model": config.d_model,
        "mlp_mult": config.mlp_mult,
        "vocab_size": config.vocab_size,
        "semantic_vocab_size": config.semantic_vocab_size,
        "tied_embeddings": True,
    }


def experiment_config_metadata(config: Config) -> dict[str, Any]:
    """Return stable source identity and the fully resolved experiment values."""

    return {
        "schema_version": config.config_schema_version,
        "path": CONFIG_FILENAME,
        "sha256": config.config_sha256,
        "profile": config.config_profile,
        "overrides": dict(config.config_overrides),
        "resolved": {
            "training": {
                "steps": config.steps,
                "train_tokens": config.steps * config.batch_size * config.seq_len,
                "batch_size": config.batch_size,
                "seq_len": config.seq_len,
                "dtype": config.dtype_name,
            },
            "model": contract_model_metadata(config),
            "kernels": {
                "attention_backend": config.attention_backend,
                "loss_backend": config.loss_backend,
                "vocab_tile_size": config.vocab_tile_size,
            },
            "optimizer": {
                "learning_rate": config.learning_rate,
                "min_lr_ratio": config.min_lr_ratio,
                "warmup_steps": config.warmup_steps,
                "weight_decay": config.weight_decay,
                "beta1": config.beta1,
                "beta2": config.beta2,
                "grad_clip": config.grad_clip,
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


def implementation_metadata(
    config: Config, runtime: AttentionRuntime
) -> dict[str, Any]:
    """Return systems/kernel provenance that may vary in either track."""

    return {
        "attention_backend": config.attention_backend,
        "attention_tuning": attention_runtime_metadata(runtime),
        "loss_backend": config.loss_backend,
        "vocab_tile_size": config.vocab_tile_size,
        "configuration": experiment_config_metadata(config),
    }


def prepare_attention_runtime(
    args: argparse.Namespace,
    config: Config,
    devices: Sequence[jax.Device],
) -> AttentionRuntime:
    """Resolve or synthetically tune attention before constructing shard_map.

    Only static model shape, runtime identity, and deterministic synthetic BHSD
    tensors reach the tuner. No training or validation dataset is accepted by
    this boundary.
    """

    if config.attention_backend == "dense":
        return AttentionRuntime(None, "dense", None, 0.0)
    if not devices:
        raise ValueError("attention tile resolution requires at least one device")
    if config.batch_size % len(devices):
        raise ValueError(
            "global batch must divide the device count before attention tuning"
        )
    local_batch = config.batch_size // len(devices)
    process_index = int(jax.process_index())
    runtime_devices = tuple(
        device
        for device in devices
        if int(getattr(device, "process_index", process_index)) == process_index
    )
    if not runtime_devices:
        raise RuntimeError("JAX reported no addressable device for this process")
    head_dim = config.d_model // config.heads
    key = make_runtime_key(
        backend=config.attention_backend,
        dtype=config.compute_dtype,
        batch=local_batch,
        heads=config.heads,
        sequence=config.seq_len,
        head_dim=head_dim,
        mode="forward_backward",
        device=runtime_devices[0],
    )
    cache_path: Path | None = None
    if args.attention_tuning_cache is not None:
        cache_argument = args.attention_tuning_cache.expanduser()
        if not cache_argument.is_absolute():
            cache_argument = Path.cwd() / cache_argument
        # Resolve parent-directory aliases (including the user's shm/ symlink)
        # without resolving the cache file itself; the cache layer must still
        # be able to reject a final-component symlink safely.
        cache_path = cache_argument.parent.resolve() / cache_argument.name
    if args.autotune_attention:
        if cache_path is None:  # validate_args establishes this for normal runs.
            raise ValueError(
                "--autotune-attention requires --attention-tuning-cache PATH"
            )

        def factory(tiles: AttentionTiles) -> AttentionCallable:
            return make_causal_attention(
                AttentionConfig(backend=config.attention_backend, tiles=tiles)
            )

        started = time.perf_counter()
        record = autotune_attention(
            key=key,
            attention_factory=factory,
            cache_path=cache_path,
            device=runtime_devices[0],
            force=True,
        )
        tune_seconds = time.perf_counter() - started
        return AttentionRuntime(
            key.digest,
            "autotuned",
            record.winner,
            tune_seconds,
        )

    resolved = resolve_attention_tile_plan(key, cache_path=cache_path)
    return AttentionRuntime(key.digest, resolved.source, resolved.tiles, 0.0)


def make_mesh_attention(
    config: Config,
    mesh: Mesh,
    tiles: AttentionTiles | None,
) -> AttentionCallable | None:
    """Build an explicitly data-sharded Pallas attention boundary.

    Parameters and optimizer state remain replicated, while the leading batch
    axis is partitioned over ``mesh['data']``.  Pallas/Mosaic calls require this
    explicit boundary: automatic SPMD partitioning of a custom kernel is not
    supported by JAX.  Each kernel invocation consequently receives the local
    per-chip batch and performs no attention collectives.
    """

    if config.attention_backend == "dense":
        return None
    if tiles is None:
        raise ValueError("non-dense attention requires a resolved tile plan")
    local_attention = make_causal_attention(
        AttentionConfig(
            backend=config.attention_backend,
            tiles=tiles,
        )
    )
    batch_partition = P("data", None, None, None)
    return jax.shard_map(
        local_attention,
        mesh=mesh,
        in_specs=(batch_partition, batch_partition, batch_partition),
        out_specs=batch_partition,
        check_vma=False,
    )


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
    x = x + params["position_embedding"][:length].astype(dtype)
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
            )
        )
        causal = None
    else:
        attention = None
        causal = jnp.tril(
            jnp.ones((length, length), dtype=jnp.bool_)
        )[None, None, :, :]

    for block in params["blocks"]:
        residual = x
        x_norm = layer_norm(x, block["ln1_scale"], block["ln1_bias"], dtype)
        qkv = linear(x_norm, block["qkv_w"], block["qkv_b"], dtype)
        query, key, value = jnp.split(qkv, 3, axis=-1)
        query = query.reshape(tokens.shape[0], length, config.heads, head_dim)
        key = key.reshape(tokens.shape[0], length, config.heads, head_dim)
        value = value.reshape(tokens.shape[0], length, config.heads, head_dim)
        if attention is not None:
            attended = attention(
                jnp.transpose(query, (0, 2, 1, 3)),
                jnp.transpose(key, (0, 2, 1, 3)),
                jnp.transpose(value, (0, 2, 1, 3)),
            )
            attended = jnp.transpose(attended, (0, 2, 1, 3))
        else:
            scores = jnp.einsum("bthd,bshd->bhts", query, key)
            scores = scores.astype(jnp.float32) * (head_dim**-0.5)
            scores = jnp.where(causal, scores, jnp.finfo(jnp.float32).min)
            probabilities = jax.nn.softmax(scores, axis=-1).astype(dtype)
            attended = jnp.einsum("bhts,bshd->bthd", probabilities, value)
        attended = attended.reshape(tokens.shape[0], length, config.d_model)
        x = residual + linear(attended, block["attn_w"], block["attn_b"], dtype)

        residual = x
        x_norm = layer_norm(x, block["ln2_scale"], block["ln2_bias"], dtype)
        hidden = linear(x_norm, block["mlp_up_w"], block["mlp_up_b"], dtype)
        hidden = jax.nn.gelu(hidden, approximate=True)
        x = residual + linear(hidden, block["mlp_down_w"], block["mlp_down_b"], dtype)

    return layer_norm(x, params["final_ln_scale"], params["final_ln_bias"], dtype)


def gpt_logits(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
) -> jax.Array:
    x = gpt_hidden(params, tokens, config, attention_fn)
    # Weight tying avoids a second vocabulary-sized parameter matrix.
    return jnp.einsum(
        "btd,vd->btv",
        x,
        params["token_embedding"].astype(config.compute_dtype),
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
            params["token_embedding"],
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    logits = gpt_logits(params, x, config, attention_fn)[..., : config.semantic_vocab_size]
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
    return jnp.asarray(config.learning_rate, jnp.float32) * warmup * multiplier


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
    """Match the parameter tree, selecting only matrices for AdamW decay.

    Every learned projection and embedding in this model is rank two. Biases
    and layer-normalization scales are rank one, so they intentionally remain
    outside the decoupled weight-decay update.
    """

    return jax.tree_util.tree_map(lambda value: bool(value.ndim >= 2), params)


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
    loss, gradients = jax.value_and_grad(
        lambda candidate: cross_entropy(candidate, x, y, config, attention_fn)
    )(params)
    gradients = jax.tree_util.tree_map(lambda grad: grad.astype(jnp.float32), gradients)
    raw_gradients = gradients
    squared_norms = [jnp.sum(jnp.square(grad)) for grad in jax.tree_util.tree_leaves(gradients)]
    grad_norm = jnp.sqrt(sum(squared_norms))
    clip_scale = jnp.minimum(1.0, config.grad_clip / (grad_norm + 1.0e-6))
    gradients = jax.tree_util.tree_map(lambda grad: grad * clip_scale, gradients)

    step = optimizer["step"] + jnp.asarray(1, dtype=jnp.int32)
    lr = learning_rate(step, config)
    m = jax.tree_util.tree_map(
        lambda old, grad: config.beta1 * old + (1.0 - config.beta1) * grad,
        optimizer["m"],
        gradients,
    )
    v = jax.tree_util.tree_map(
        lambda old, grad: config.beta2 * old + (1.0 - config.beta2) * jnp.square(grad),
        optimizer["v"],
        gradients,
    )
    bias_correction1 = 1.0 - config.beta1**step.astype(jnp.float32)
    bias_correction2 = 1.0 - config.beta2**step.astype(jnp.float32)

    def update(
        parameter: jax.Array,
        first: jax.Array,
        second: jax.Array,
        should_decay: bool,
    ) -> jax.Array:
        adam = (first / bias_correction1) / (jnp.sqrt(second / bias_correction2) + 1.0e-8)
        decay = config.weight_decay * parameter if should_decay else 0.0
        return parameter - lr * (adam + decay)

    params = jax.tree_util.tree_map(update, params, m, v, decay_mask)
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


def diagnostic_scopes(tree: Mapping[str, Any]) -> tuple[tuple[str, int | None, tuple[Any, ...]], ...]:
    """Group a parameter-shaped tree into stable logical report scopes."""

    embeddings = tuple(
        jax.tree_util.tree_leaves(
            (tree["token_embedding"], tree["position_embedding"])
        )
    )
    blocks = tuple(
        (
            "block",
            layer,
            tuple(jax.tree_util.tree_leaves(block)),
        )
        for layer, block in enumerate(tree["blocks"])
    )
    final_norm = tuple(
        jax.tree_util.tree_leaves(
            (tree["final_ln_scale"], tree["final_ln_bias"])
        )
    )
    return (
        ("overall", None, tuple(jax.tree_util.tree_leaves(tree))),
        ("embeddings", None, embeddings),
        *blocks,
        ("final_norm", None, final_norm),
    )


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
    variance_sum = sum(
        (jnp.sum(jnp.square(value - mean)) for value in values32), zero
    )
    third_sum = sum(
        (jnp.sum(jnp.power(value - mean, 3)) for value in values32), zero
    )
    fourth_sum = sum(
        (jnp.sum(jnp.power(value - mean, 4)) for value in values32), zero
    )
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
        diagnostic_scopes(tree)
        for tree in (params_after, raw_gradients, updates)
    )
    scope_count = len(family_scopes[0])
    return jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    _diagnostic_stat_vector(family_scopes[family][scope][2])
                    for family in range(len(_DIAGNOSTIC_FAMILIES))
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
            params["token_embedding"],
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    else:
        logits = gpt_logits(params, x, config, attention_fn)[..., : config.semantic_vocab_size]
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(
            log_probabilities, y[..., None], axis=-1
        )[..., 0]
        losses = -selected
    mask = mask.astype(jnp.float32)
    return (
        jnp.sum(losses * mask, dtype=jnp.float32),
        jnp.sum(mask, dtype=jnp.float32),
    )


def should_run_validation_probe(step: int, config: Config) -> bool:
    """Return whether this step gets a non-canonical fixed-prefix probe."""

    return (
        config.val_every > 0
        and step < config.steps
        and step % config.val_every == 0
    )


def should_run_diagnostics(step: int, config: Config) -> bool:
    """Capture the first/final updates plus the configured sparse cadence."""

    return config.diagnostics_every > 0 and (
        step == 1
        or step == config.steps
        or step % config.diagnostics_every == 0
    )


def evaluate_validation_prefix(
    params: Any,
    dataset: TokenDataset,
    compiled_eval: Any,
    data_sharding: NamedSharding,
    config: Config,
    batches: int,
    *,
    mesh: Mesh | None = None,
    process_index: int = 0,
    process_count: int = 1,
) -> tuple[float, float]:
    """Synchronously evaluate batches ``0..batches-1`` of the fixed prefix."""

    if batches <= 0:
        raise ValueError("validation batch count must be positive")
    started = time.perf_counter()
    loss_sum = 0.0
    scored_tokens = 0
    if process_count > 1 and mesh is None:
        raise ValueError("a global mesh is required for multi-process evaluation")
    local_batch = local_batch_size(config.batch_size, process_count)
    mask_host = np.ones((local_batch, config.seq_len), dtype=np.float32)
    if mesh is None:
        mask = jax.device_put(mask_host, data_sharding)
    else:
        mask = put_host_local_array(
            mask_host, mesh, P("data", None), data_sharding, process_count
        )
    for eval_index in range(batches):
        eval_x_host, eval_y_host = dataset.validation_batch(
            eval_index,
            config.batch_size,
            config.seq_len,
            config.semantic_vocab_size,
        )
        eval_x_host = rank_local_slice(eval_x_host, process_index, process_count)
        eval_y_host = rank_local_slice(eval_y_host, process_index, process_count)
        if mesh is None:
            eval_x = jax.device_put(eval_x_host, data_sharding)
            eval_y = jax.device_put(eval_y_host, data_sharding)
        else:
            eval_x = put_host_local_array(
                eval_x_host, mesh, P("data", None), data_sharding, process_count
            )
            eval_y = put_host_local_array(
                eval_y_host, mesh, P("data", None), data_sharding, process_count
            )
        batch_loss_sum, batch_scored = local_device_get(
            compiled_eval(params, eval_x, eval_y, mask)
        )
        loss_sum += float(batch_loss_sum)
        scored_tokens += int(batch_scored)
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    expected_tokens = batches * config.batch_size * config.seq_len
    if scored_tokens != expected_tokens:
        raise RuntimeError(
            f"validation executable scored {scored_tokens:,} tokens; expected "
            f"{expected_tokens:,}"
        )
    return (
        finite_metric("validation_loss", loss_sum / scored_tokens),
        finite_metric("validation_seconds", elapsed, positive=True),
    )


def downstream_batches(
    domain: DownstreamDomain, config: Config
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Pack documents into fixed eval shapes without cross-document targets."""

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for document in domain.documents:
        target = document.score_offset
        remaining = document.scored_tokens
        while remaining:
            count = min(config.seq_len, remaining)
            x = np.zeros((config.seq_len,), dtype=np.int32)
            y = np.zeros((config.seq_len,), dtype=np.int32)
            mask = np.zeros((config.seq_len,), dtype=np.float32)
            x[:count] = domain.tokens[target - 1 : target - 1 + count]
            y[:count] = domain.tokens[target : target + count]
            mask[:count] = 1.0
            rows.append((x, y, mask))
            target += count
            remaining -= count

    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for start in range(0, len(rows), config.batch_size):
        batch_rows = rows[start : start + config.batch_size]
        x = np.zeros((config.batch_size, config.seq_len), dtype=np.int32)
        y = np.zeros((config.batch_size, config.seq_len), dtype=np.int32)
        mask = np.zeros((config.batch_size, config.seq_len), dtype=np.float32)
        for row, (row_x, row_y, row_mask) in enumerate(batch_rows):
            x[row], y[row], mask[row] = row_x, row_y, row_mask
        batches.append((x, y, mask))
    packed_tokens = int(sum(float(mask.sum()) for _, _, mask in batches))
    if packed_tokens != domain.scored_tokens:
        raise AssertionError(
            f"packed {packed_tokens:,} targets for {domain.name}; expected "
            f"{domain.scored_tokens:,}"
        )
    return tuple(batches)


def evaluate_downstream_domain(
    params: Any,
    domain: DownstreamDomain,
    compiled_eval: Any,
    data_sharding: NamedSharding,
    config: Config,
    *,
    mesh: Mesh | None = None,
    process_index: int = 0,
    process_count: int = 1,
) -> dict[str, float | int]:
    """Evaluate one domain with exact masking and the shared eval executable."""

    started = time.perf_counter()
    loss_sum = 0.0
    scored_tokens = 0
    if process_count > 1 and mesh is None:
        raise ValueError("a global mesh is required for multi-process evaluation")
    for x_host, y_host, mask_host in downstream_batches(domain, config):
        x_host = rank_local_slice(x_host, process_index, process_count)
        y_host = rank_local_slice(y_host, process_index, process_count)
        mask_host = rank_local_slice(mask_host, process_index, process_count)
        if mesh is None:
            x = jax.device_put(x_host, data_sharding)
            y = jax.device_put(y_host, data_sharding)
            mask = jax.device_put(mask_host, data_sharding)
        else:
            x = put_host_local_array(
                x_host, mesh, P("data", None), data_sharding, process_count
            )
            y = put_host_local_array(
                y_host, mesh, P("data", None), data_sharding, process_count
            )
            mask = put_host_local_array(
                mask_host, mesh, P("data", None), data_sharding, process_count
            )
        batch_loss_sum, batch_scored = local_device_get(
            compiled_eval(params, x, y, mask)
        )
        loss_sum += float(batch_loss_sum)
        scored_tokens += int(batch_scored)
    elapsed = finite_metric(
        f"downstream {domain.name} seconds",
        max(time.perf_counter() - started, 1.0e-12),
        positive=True,
    )
    if scored_tokens != domain.scored_tokens:
        raise RuntimeError(
            f"downstream {domain.name} scored {scored_tokens:,} tokens; expected "
            f"{domain.scored_tokens:,}"
        )
    loss = finite_metric(f"downstream {domain.name} loss", loss_sum / scored_tokens)
    return {
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "scored_tokens": scored_tokens,
        "seconds": elapsed,
    }


def parameter_count(params: Any) -> int:
    return sum(int(value.size) for value in jax.tree_util.tree_leaves(params))


def estimated_flops_per_token(config: Config, params_total: int) -> int:
    """Return the analytic training FLOP estimate used by logs and reports.

    The familiar ``6P`` approximation covers the dense forward and two
    backward matrix products. The tiled output head recomputes vocabulary
    logits in its custom VJP and pads the storage table to a whole tile, so its
    additional work must be recorded for honest equi-FLOP comparisons.
    """

    attention_extent = config.seq_len
    if config.attention_backend != "dense":
        # Both TPU Flash paths right-pad q/k/v to native 128-wide tiles.  The
        # surrounding projections still process only logical tokens, while the
        # quadratic attention products execute across the padded square.
        attention_extent = padded_sequence_length(config.seq_len)
    attention_total_per_sequence = (
        12 * config.layers * config.d_model * attention_extent * attention_extent
    )
    attention_per_logical_token = (
        attention_total_per_sequence + config.seq_len - 1
    ) // config.seq_len
    estimate = 6 * params_total + attention_per_logical_token
    if config.loss_backend == "tiled":
        padded_vocab = (
            (config.vocab_size + config.vocab_tile_size - 1)
            // config.vocab_tile_size
            * config.vocab_tile_size
        )
        dense_head = 6 * config.vocab_size * config.d_model
        tiled_head = 8 * padded_vocab * config.d_model
        estimate += tiled_head - dense_head
    return int(estimate)


def flatten_arrays(tree: Any, prefix: str = "params") -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(value[key], f"{path}/{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")
        else:
            flat[path] = np.asarray(value)

    visit(tree, prefix)
    return flat


def save_checkpoint(
    output_dir: Path,
    params: Any,
    config: Config,
    seed: int,
    attention_runtime: AttentionRuntime,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    host_params = local_device_get(params)
    arrays = flatten_arrays(host_params)
    metadata = {
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
            "dtype": config.dtype_name,
            "attention_backend": config.attention_backend,
            "attention_tuning": attention_runtime_metadata(attention_runtime),
            "loss_backend": config.loss_backend,
            "vocab_tile_size": config.vocab_tile_size,
            "tied_embeddings": True,
        },
    }
    arrays["metadata.json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    destination = output_dir / CHECKPOINT_NAME
    temporary = output_dir / f".{CHECKPOINT_NAME}.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_result(output_dir: Path, result: Mapping[str, Any]) -> None:
    destination = output_dir / "metrics.json"
    temporary = output_dir / ".metrics.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_training_csv(
    output_dir: Path,
    history: np.ndarray,
    config: Config,
    flops_per_token: int | None = None,
) -> None:
    """Atomically persist every optimizer step without timing host transfers."""

    if history.shape != (config.steps, 3):
        raise ValueError(
            f"training history has shape {history.shape}; expected {(config.steps, 3)}"
        )
    if flops_per_token is not None and flops_per_token <= 0:
        raise ValueError("flops_per_token must be positive when provided")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / TRAINING_CSV_NAME
    temporary = output_dir / f".{TRAINING_CSV_NAME}.tmp"
    tokens_per_step = config.batch_size * config.seq_len
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "step",
                "tokens_processed",
                "cumulative_estimated_flops",
                "train_loss",
                "learning_rate",
                "grad_norm",
            )
        )
        for index, (loss, learning_rate_value, grad_norm) in enumerate(history, 1):
            tokens_processed = index * tokens_per_step
            writer.writerow(
                (
                    index,
                    tokens_processed,
                    (
                        tokens_processed * flops_per_token
                        if flops_per_token is not None
                        else ""
                    ),
                    float(loss),
                    float(learning_rate_value),
                    float(grad_norm),
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_diagnostics_csv(
    output_dir: Path,
    points: Sequence[DiagnosticPoint],
    scope_metadata: Sequence[tuple[str, int | None, int]],
    config: Config,
    flops_per_token: int,
) -> None:
    """Atomically persist long-form sparse optimizer diagnostics."""

    if not points:
        raise ValueError("diagnostic history cannot be empty")
    if flops_per_token <= 0:
        raise ValueError("flops_per_token must be positive")
    expected_shape = (
        len(scope_metadata),
        len(_DIAGNOSTIC_FAMILIES),
        len(_DIAGNOSTIC_STATS),
    )
    if points[-1].step != config.steps:
        raise ValueError("diagnostic history must include the final optimizer step")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / DIAGNOSTICS_CSV_NAME
    temporary = output_dir / f".{DIAGNOSTICS_CSV_NAME}.tmp"
    tokens_per_step = config.batch_size * config.seq_len
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
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
        )
        previous_step = 0
        for point in points:
            if point.step <= previous_step or point.step > config.steps:
                raise ValueError("diagnostic steps must be unique and increasing")
            previous_step = point.step
            values = np.asarray(point.values, dtype=np.float32)
            if values.shape != expected_shape:
                raise ValueError(
                    f"diagnostic values have shape {values.shape}; "
                    f"expected {expected_shape}"
                )
            if not np.all(np.isfinite(values)):
                raise FloatingPointError(
                    f"diagnostic values at step {point.step} must be finite"
                )
            tokens_processed = point.step * tokens_per_step
            cumulative_flops = tokens_processed * flops_per_token
            for scope_index, (scope, layer, element_count) in enumerate(scope_metadata):
                for family_index, family in enumerate(_DIAGNOSTIC_FAMILIES):
                    for stat_index, stat in enumerate(_DIAGNOSTIC_STATS):
                        writer.writerow(
                            (
                                point.step,
                                tokens_processed,
                                cumulative_flops,
                                scope,
                                "" if layer is None else layer,
                                family,
                                stat,
                                float(values[scope_index, family_index, stat_index]),
                                element_count,
                            )
                        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_validation_csv(output_dir: Path, rows: Sequence[ValidationRow]) -> None:
    """Persist FineWeb probes/final and optional downstream domain scores."""

    canonical_rows = [row for row in rows if row.canonical]
    if len(canonical_rows) != 1 or canonical_rows[0].kind != "fineweb":
        raise ValueError("validation history must contain one canonical in-distribution row")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / VALIDATION_CSV_NAME
    temporary = output_dir / f".{VALIDATION_CSV_NAME}.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
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
        )
        for row in rows:
            writer.writerow(
                (
                    int(row.step),
                    int(row.tokens_processed),
                    row.kind,
                    row.domain,
                    int(row.validation_tokens),
                    float(row.validation_loss),
                    float(row.perplexity),
                    float(row.validation_seconds),
                    "true" if row.canonical else "false",
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def format_count(value: float) -> str:
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.2f}{suffix}"
    return f"{value:.0f}"


def format_rate(value: float) -> str:
    return format_count(value)


def device_label(devices: Sequence[jax.Device]) -> str:
    kinds = sorted({str(device.device_kind) for device in devices})
    return ", ".join(kinds)


def validate_official_topology(profile: str, devices: Sequence[jax.Device]) -> None:
    """Require four TPU v4 chips per process on a coherent global mesh."""

    if profile != "official":
        return
    local_devices = tuple(jax.local_devices())
    process_count = int(jax.process_count())
    device_count = int(jax.device_count())
    platforms = sorted({str(device.platform) for device in local_devices})
    kinds = sorted({str(device.device_kind) for device in local_devices})
    is_tpu_v4 = all(
        str(device.platform).lower() == "tpu"
        and str(device.device_kind).strip().lower() == "tpu v4"
        for device in local_devices
    )
    if (
        process_count < 1
        or len(local_devices) != 4
        or device_count != 4 * process_count
        or len(devices) != device_count
        or not is_tpu_v4
    ):
        raise RuntimeError(
            "official profile requires exactly 4 local TPU v4 devices per JAX "
            "process and one coherent global device mesh; detected "
            f"process_count={process_count}, device_count={device_count}, "
            f"local_device_count={len(local_devices)}, platforms={platforms}, "
            f"device_kinds={kinds}"
        )


def system_metadata(devices: Sequence[jax.Device]) -> dict[str, Any]:
    platforms = sorted({str(device.platform) for device in devices})

    def optional_int(device: jax.Device, name: str) -> int | None:
        value = getattr(device, name, None)
        return int(value) if value is not None else None

    try:
        libtpu_version = importlib_metadata.version("libtpu")
    except importlib_metadata.PackageNotFoundError:
        libtpu_version = None
    return {
        "python_version": host_platform.python_version(),
        "jax_version": str(jax.__version__),
        "jaxlib_version": str(jaxlib.__version__),
        "libtpu_version": libtpu_version,
        "platform": platforms[0] if len(platforms) == 1 else ",".join(platforms),
        "device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "process_count": int(jax.process_count()),
        "device_kinds": sorted({str(device.device_kind) for device in devices}),
        "device_ids": [optional_int(device, "id") for device in devices],
        "process_indices": [optional_int(device, "process_index") for device in devices],
    }


def inferred_peak_tflops(args: argparse.Namespace, devices: Sequence[jax.Device]) -> float | None:
    if args.peak_tflops is not None:
        return args.peak_tflops
    labels = " ".join(str(device.device_kind).lower() for device in devices)
    # A JAX-visible device on a v4-8 is one full 275-TFLOP/s TPU v4 chip. This
    # only feeds an explicitly labeled theoretical-utilization estimate.
    if "tpu" in labels and "v4" in labels:
        return 275.0 * len(devices)
    return None


def sync_tree(tree: Any) -> None:
    for value in jax.tree_util.tree_leaves(tree):
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()


def initialize_distributed_runtime() -> tuple[int, int]:
    """Initialize JAX's Cloud TPU coordinator before the first device query."""

    distributed = os.environ.get(_DISTRIBUTED_ENV) == "1"
    if distributed:
        # On Cloud TPU VMs JAX discovers the coordinator, process count, and
        # process id from TPU metadata. Every host must enter this call.
        jax.distributed.initialize()
    process_count = int(jax.process_count())
    process_index = int(jax.process_index())
    expected_raw = os.environ.get(_PROCESS_COUNT_ENV)
    if expected_raw is not None:
        try:
            expected = int(expected_raw)
        except ValueError as exc:
            raise ValueError(f"{_PROCESS_COUNT_ENV} must be a positive integer") from exc
        if expected <= 0:
            raise ValueError(f"{_PROCESS_COUNT_ENV} must be a positive integer")
        if process_count != expected:
            raise RuntimeError(
                f"JAX discovered {process_count} processes, but the launcher expected "
                f"{expected}"
            )
    if not 0 <= process_index < process_count:
        raise RuntimeError(
            f"invalid JAX process index {process_index} for {process_count} processes"
        )
    return process_index, process_count


def is_controller_process(process_index: int) -> bool:
    """Keep artifacts on the host that owns the harness, regardless of JAX rank."""

    configured = os.environ.get(_CONTROLLER_HOST_ENV)
    if configured is None:
        return process_index == 0
    local = socket.gethostname().strip().split(".", 1)[0]
    expected = configured.strip().split(".", 1)[0]
    if not expected:
        raise ValueError(f"{_CONTROLLER_HOST_ENV} may not be empty")
    return local == expected


def local_batch_size(global_batch_size: int, process_count: int) -> int:
    if global_batch_size % process_count:
        raise ValueError(
            f"global batch size {global_batch_size} must be divisible by JAX "
            f"process count {process_count}"
        )
    return global_batch_size // process_count


def rank_local_slice(
    value: np.ndarray, process_index: int, process_count: int
) -> np.ndarray:
    """Return this process's contiguous part of a global leading batch axis."""

    size = local_batch_size(int(value.shape[0]), process_count)
    start = process_index * size
    return np.ascontiguousarray(value[start : start + size])


def put_host_local_array(
    value: np.ndarray,
    mesh: Mesh,
    partition_spec: P,
    sharding: NamedSharding,
    process_count: int,
) -> jax.Array:
    """Convert rank-local NumPy data into one globally sharded JAX array."""

    if process_count == 1:
        return jax.device_put(value, sharding)
    return multihost_utils.host_local_array_to_global_array(
        value, mesh, partition_spec
    )


def put_replicated_tree(tree: Any, mesh: Mesh, sharding: NamedSharding, process_count: int) -> Any:
    """Create identical global replicas from the same host value on every rank."""

    if process_count == 1:
        return jax.device_put(tree, sharding)
    return multihost_utils.host_local_array_to_global_array(tree, mesh, P())


def local_device_get(tree: Any) -> Any:
    """Copy replicated values through one addressable shard on multi-host JAX."""

    def fetch(value: Any) -> Any:
        if isinstance(value, jax.Array) and not value.is_fully_addressable:
            return jax.device_get(value.addressable_data(0))
        return jax.device_get(value)

    return jax.tree_util.tree_map(fetch, tree)


def finite_metric(name: str, value: float, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0) or (not positive and value < 0.0):
        qualifier = "finite and positive" if positive else "finite and nonnegative"
        raise FloatingPointError(f"{name} must be {qualifier}, got {value!r}")
    return value


def perplexity_from_loss(loss: float) -> float:
    try:
        perplexity = math.exp(loss)
    except OverflowError as exc:
        raise FloatingPointError(f"loss {loss!r} overflows perplexity") from exc
    return finite_metric("perplexity", perplexity, positive=True)


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

    dataset, vocab_size = load_dataset(args, experiment)
    config = resolve_config(args, platform, vocab_size, experiment)
    capture_window = xprof_step_window(args, config.steps)
    downstream_domains = load_downstream_domains(args, config.semantic_vocab_size)
    diagnostic_mode = args.no_final_validation and args.no_checkpoint
    needs_evaluation = should_compile_evaluation(args, config, downstream_domains)
    if config.batch_size % len(devices):
        raise ValueError(
            f"global batch size {config.batch_size} must be divisible by "
            f"visible device count {len(devices)}"
        )
    local_batch = local_batch_size(config.batch_size, process_count)
    if config.attention_backend != "dense":
        console.phase(
            "Attention tile preflight",
            (
                "AOT-compiling synthetic forward/backward candidates"
                if args.autotune_attention
                else "resolving exact cache, shipped lookup, or shape heuristic"
            ),
        )
    attention_runtime = prepare_attention_runtime(args, config, devices)
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
    flops_per_token = estimated_flops_per_token(config, params_total)
    tokens_processed = config.steps * config.batch_size * config.seq_len

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
            ("train / val tokens", f"{len(dataset.train):,} / {len(dataset.validation):,}"),
            (
                "downstream",
                (
                    f"{len(downstream_domains)} domains / "
                    f"{sum(domain.scored_tokens for domain in downstream_domains):,} scored"
                    if downstream_domains
                    else "not requested"
                ),
            ),
            ("model", f"L{config.layers} D{config.d_model} H{config.heads} MLP×{config.mlp_mult}"),
            ("parameters", format_count(params_total)),
            ("global batch", f"{config.batch_size} × {config.seq_len} tokens"),
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
            ("train tokens", format_count(tokens_processed)),
            ("estimated FLOPs", format_count(flops_per_token * tokens_processed)),
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
    attention_fn = make_mesh_attention(config, mesh, attention_runtime.tiles)
    params = put_replicated_tree(host_params, mesh, replicated, process_count)
    optimizer = put_replicated_tree(host_optimizer, mesh, replicated, process_count)
    del host_params, host_optimizer

    train_rng = np.random.default_rng(
        args.seed + 1 + process_index * 1_000_003
    )
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
            p, o, x, y, config, decay_mask, attention_fn
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
        diagnostic_executable = jax.jit(
            lambda p, o, x, y: diagnostic_train_step(
                p, o, x, y, config, decay_mask, attention_fn
            ),
            in_shardings=(replicated, replicated, data_sharding, data_sharding),
            donate_argnums=(0, 1),
        ).lower(params, optimizer, sample_x, sample_y).compile()
        diagnostic_compile_seconds = (
            time.perf_counter() - diagnostic_compile_started
        )

    # Compile evaluation exactly once when it is requested. Diagnostic XProf
    # runs can skip this executable entirely, keeping their setup focused on the
    # training step being inspected.
    compiled_eval: Any | None = None
    sample_mask: jax.Array | None = None
    eval_compile_seconds = 0.0
    if needs_evaluation:
        sample_mask_host = np.ones(
            (local_batch, config.seq_len), dtype=np.float32
        )
        sample_mask = put_host_local_array(
            sample_mask_host,
            mesh,
            P("data", None),
            data_sharding,
            process_count,
        )
        console.phase("Compiling evaluation", "reused by probes and final validation")
        eval_compile_started = time.perf_counter()
        compiled_eval = jax.jit(
            lambda p, x, y, mask: eval_step(
                p, x, y, mask, config, attention_fn
            ),
            in_shardings=(replicated, data_sharding, data_sharding, data_sharding),
        ).lower(params, sample_x, sample_y, sample_mask).compile()
        eval_compile_seconds = time.perf_counter() - eval_compile_started
    total_compile_seconds = (
        train_compile_seconds
        + diagnostic_compile_seconds
        + eval_compile_seconds
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
        multihost_utils.sync_global_devices("speedrun-training-start")
    train_started = time.perf_counter()
    xprof_dir = (
        args.xprof_dir.expanduser().resolve() if capture_window is not None else None
    )
    trace_active = False
    try:
        for step_index in range(1, config.steps + 1):
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
                    multihost_utils.sync_global_devices(
                        "speedrun-xprof-capture-started"
                    )

            annotation = (
                jax.profiler.StepTraceAnnotation("train", step_num=step_index)
                if trace_active
                else nullcontext()
            )
            with annotation:
                # Keep the host sampling, transfer, dispatch, and any logging
                # synchronization inside the step annotation. This exposes input
                # gaps alongside TPU execution in the same XProf timeline.
                batch_x, batch_y = dataset.batch(
                    "train",
                    train_rng,
                    local_batch,
                    config.seq_len,
                    config.semantic_vocab_size,
                )
                batch_x = put_host_local_array(
                    batch_x, mesh, P("data", None), data_sharding, process_count
                )
                batch_y = put_host_local_array(
                    batch_y, mesh, P("data", None), data_sharding, process_count
                )
                if should_run_diagnostics(step_index, config):
                    if diagnostic_executable is None:  # defensive invariant
                        raise AssertionError("diagnostic executable was not compiled")
                    params, optimizer, last_metrics, diagnostic_values_at_step = (
                        diagnostic_executable(params, optimizer, batch_x, batch_y)
                    )
                    diagnostic_device_points.append(
                        (step_index, diagnostic_values_at_step)
                    )
                else:
                    params, optimizer, last_metrics = executable(
                        params, optimizer, batch_x, batch_y
                    )
                if should_run_validation_probe(step_index, config):
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
                        config,
                        config.val_probe_batches,
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
                    or step_index == config.steps
                    or step_index % config.log_every == 0
                )
                if should_log:
                    host_metrics = local_device_get(last_metrics)
                    elapsed_so_far = max(time.perf_counter() - train_started, 1.0e-12)
                    seen_tokens = step_index * config.batch_size * config.seq_len
                    console.step(
                        step_index,
                        config.steps,
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
                    multihost_utils.sync_global_devices(
                        "speedrun-xprof-capture-finished"
                    )
                if trace_active:
                    jax.profiler.stop_trace()
                    trace_active = False
                    console.phase("XProf capture saved", str(xprof_dir))
                if process_count > 1:
                    multihost_utils.sync_global_devices(
                        "speedrun-xprof-capture-stopped"
                    )
    finally:
        if trace_active:
            # Avoid leaving process-global profiler state active when a sampled
            # batch or training step raises midway through the capture window.
            jax.profiler.stop_trace()

    if last_metrics is None:  # defensive: argparse prevents zero steps
        raise AssertionError("training produced no metrics")
    # Sparse diagnostic reductions are part of benchmark time even if their
    # result branch is otherwise independent of the next optimizer state.
    sync_tree((params, optimizer, last_metrics, diagnostic_device_points))
    if process_count > 1:
        multihost_utils.sync_global_devices("speedrun-training-finished")
    train_seconds = max(time.perf_counter() - train_started, 1.0e-12)
    final_train = local_device_get(last_metrics)
    training_history = np.asarray(
        local_device_get(optimizer["history"]), dtype=np.float32
    )
    diagnostic_points = tuple(
        DiagnosticPoint(step, np.asarray(local_device_get(values), dtype=np.float32))
        for step, values in diagnostic_device_points
    )
    train_loss = finite_metric("train_loss", float(final_train["loss"]))

    if diagnostic_mode:
        output_dir = args.output_dir.expanduser().resolve()
        if is_controller:
            write_training_csv(
                output_dir, training_history, config, flops_per_token=flops_per_token
            )
            if diagnostic_points:
                write_diagnostics_csv(
                    output_dir,
                    diagnostic_points,
                    diagnostic_metadata,
                    config,
                    flops_per_token,
                )
        diagnostic_rate = finite_metric(
            "tokens_per_second", tokens_processed / train_seconds, positive=True
        )
        assert capture_window is not None and xprof_dir is not None
        console.table(
            "profile complete",
            (
                ("training steps", f"{config.steps:,}"),
                ("captured steps", f"{capture_window[0]}..{capture_window[1]}"),
                ("train loss", f"{train_loss:.4f}"),
                ("diagnostic rate", f"{format_rate(diagnostic_rate)} tok/s"),
                ("training curve", output_dir / TRAINING_CSV_NAME),
                (
                    "diagnostics",
                    (
                        output_dir / DIAGNOSTICS_CSV_NAME
                        if diagnostic_points
                        else "disabled"
                    ),
                ),
                ("XProf trace", xprof_dir),
            ),
        )
        if process_count > 1:
            multihost_utils.sync_global_devices("speedrun-profile-artifacts-written")
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
        config,
        config.eval_batches,
        mesh=mesh,
        process_index=process_index,
        process_count=process_count,
    )
    validation_rows.append(
        ValidationRow(
            step=config.steps,
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
                config,
                mesh=mesh,
                process_index=process_index,
                process_count=process_count,
            )
            downstream_results[domain.name] = domain_result
            validation_rows.append(
                ValidationRow(
                    step=config.steps,
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
                step=config.steps,
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
    artifact_names = [TRAINING_CSV_NAME, VALIDATION_CSV_NAME]
    if diagnostic_points:
        artifact_names.append(DIAGNOSTICS_CSV_NAME)
    artifact_names.append(CHECKPOINT_NAME)
    console.phase("Artifacts", " + ".join(artifact_names))
    if is_controller:
        write_training_csv(
            output_dir, training_history, config, flops_per_token=flops_per_token
        )
        if diagnostic_points:
            write_diagnostics_csv(
                output_dir,
                diagnostic_points,
                diagnostic_metadata,
                config,
                flops_per_token,
            )
        write_validation_csv(output_dir, validation_rows)
        save_checkpoint(output_dir, params, config, args.seed, attention_runtime)

    tokens_per_second = finite_metric(
        "tokens_per_second", tokens_processed / train_seconds, positive=True
    )
    total_flops = int(flops_per_token * tokens_processed)
    achieved_tflops = finite_metric(
        "achieved_tflops", total_flops / train_seconds / 1.0e12
    )
    peak_tflops = inferred_peak_tflops(args, devices)
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
        "track": args.track,
        "profile": profile,
        "seed": int(args.seed),
        "checkpoint": CHECKPOINT_NAME,
        "artifacts": {
            "training_curve": TRAINING_CSV_NAME,
            "validation_curve": VALIDATION_CSV_NAME,
            **(
                {"diagnostics": DIAGNOSTICS_CSV_NAME}
                if diagnostic_points
                else {}
            ),
        },
        "system": {
            **system_metadata(devices),
            "controller_process_index": process_index,
        },
        "contract": {
            "model_id": "reference-gpt-v1",
            "dataset_id": dataset_id,
            "tokenizer_id": tokenizer_id,
            "sequence_length": config.seq_len,
            "model": contract_model_metadata(config),
        },
        # Kernel choices are implementation provenance, not part of the fixed
        # sample-efficiency model contract. Keeping this sibling object separate
        # preserves exact compatibility with the harness reference contract.
        "implementation": implementation_metadata(config, attention_runtime),
        "evaluations": evaluations,
        "metrics": {
            "train_seconds": finite_metric("train_seconds", train_seconds, positive=True),
            "tokens_processed": int(tokens_processed),
            "training_token_budget": int(tokens_processed),
            "training_steps": int(config.steps),
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
        multihost_utils.sync_global_devices("speedrun-final-artifacts-written")
    return result if is_controller else None


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        # A concise colored-ish error is useful interactively; a traceback can be
        # requested naturally via Python's exception chaining during development.
        print(f"\nerror: {error}", file=sys.stderr)
        if os.environ.get("SPEEDRUN_DEBUG") == "1":
            raise
        return 1
    if result is not None:
        print(
            RESULT_PREFIX
            + json.dumps(
                result, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
