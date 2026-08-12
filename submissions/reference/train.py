#!/usr/bin/env python3
"""A compact, dependency-light GPT trainer for the GPT TPU speedrun.

Everything involved in training lives in this file.  The default model is sized
for a TPU v4-8 and uses pure JAX: model state is replicated while the global
batch is sharded over every visible device.  ``--smoke`` selects a tiny CPU-
friendly configuration and the built-in byte corpus means the script never
requires a download.

Prepared data can be supplied as a directory of llm.c-style FineWeb shards,
individual NumPy/token/text files, or repeatable explicit shard paths. The
final stdout line is a machine-readable competition result and is
intentionally never colorized.
"""

from __future__ import annotations

import argparse
import csv
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform as host_platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


RESULT_PREFIX = "SPEEDRUN_RESULT="
CHECKPOINT_NAME = "checkpoint.npz"
TRAINING_CSV_NAME = "training.csv"
SCHEMA_VERSION = 1
_VALID_TRACKS = ("open", "sample_efficiency")
_VALID_PROFILES = ("smoke", "dev", "official")


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
    log_every: int
    vocab_size: int
    compute_dtype: Any
    dtype_name: str


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

    def __init__(self, mode: str) -> None:
        auto = sys.stderr.isatty() and "NO_COLOR" not in os.environ
        self.enabled = mode == "always" or (mode == "auto" and auto)

    def paint(self, text: object, *styles: str) -> str:
        raw = str(text)
        if not self.enabled or not styles:
            return raw
        prefix = "".join(self.COLORS[s] for s in styles)
        return f"{prefix}{raw}{self.COLORS['reset']}"

    def banner(self) -> None:
        mark = self.paint("◆", "magenta", "bold")
        title = self.paint(" GPT TPU SPEEDRUN ", "white", "bold")
        print(
            f"\n  {mark}{title}{self.paint('reference / jax', 'cyan')}\n",
            file=sys.stderr,
        )

    def table(self, title: str, rows: Sequence[tuple[str, object]]) -> None:
        width = max(52, *(len(str(k)) + len(str(v)) + 7 for k, v in rows))
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
            key_text = f"{key:<20}"
            value_text = str(value)
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

    def success(self, validation_loss: float, elapsed: float) -> None:
        print(
            f"\n  {self.paint('✓', 'green', 'bold')} "
            f"validation loss {self.paint(f'{validation_loss:.4f}', 'green', 'bold')} "
            f"in {self.paint(f'{elapsed:.3f}s', 'white', 'bold')}\n",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a decoder-only GPT with JAX on TPU (or run a tiny CPU smoke test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    run = parser.add_argument_group("run")
    run.add_argument("--output-dir", type=Path, default=Path("runs/reference"))
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--steps", type=positive_int, default=None)
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
    run.add_argument("--eval-batches", type=positive_int, default=None)
    run.add_argument("--log-every", type=positive_int, default=None)
    run.add_argument("--color", choices=("auto", "always", "never"), default="auto")

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
    data.add_argument("--vocab-size", type=positive_int, default=None)
    data.add_argument("--dataset-id", default=None, help="stable dataset identifier for records")
    data.add_argument("--tokenizer-id", default=None, help="stable tokenizer identifier for records")
    data.add_argument(
        "--data-format",
        choices=("auto", "raw", "llmc"),
        default="auto",
        help="raw binaries or llm.c 256-int-header shards",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--batch-size", type=positive_int, default=None, help="global batch size")
    model.add_argument("--seq-len", type=positive_int, default=None)
    model.add_argument("--layers", type=positive_int, default=None)
    model.add_argument("--heads", type=positive_int, default=None)
    model.add_argument("--d-model", type=positive_int, default=None)
    model.add_argument("--mlp-mult", type=positive_int, default=4)
    model.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--learning-rate", type=float, default=3.0e-4)
    optim.add_argument("--min-lr-ratio", type=float, default=0.1)
    optim.add_argument("--warmup-steps", type=nonnegative_int, default=None)
    optim.add_argument("--weight-decay", type=float, default=0.1)
    optim.add_argument("--beta1", type=float, default=0.9)
    optim.add_argument("--beta2", type=float, default=0.95)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="hardware bf16 peak for the whole mesh; enables an MFU estimate",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke and args.profile not in (None, "smoke"):
        raise ValueError("--smoke cannot be combined with a non-smoke --profile")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be nonnegative")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        raise ValueError("--beta1 and --beta2 must be in [0, 1)")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")
    if args.peak_tflops is not None and args.peak_tflops <= 0.0:
        raise ValueError("--peak-tflops must be positive")


def selected_profile(args: argparse.Namespace) -> str:
    return "smoke" if args.smoke else (args.profile or "dev")


def resolve_config(args: argparse.Namespace, platform: str, vocab_size: int) -> Config:
    # Defaults deliberately distinguish a meaningful v4-8 run from a cheap smoke
    # test. Every field remains directly overridable for competition entries.
    profile = selected_profile(args)
    defaults = {
        # Four keeps this profile valid on both CPU and all four v4-8 devices.
        "smoke": dict(steps=2, batch=4, sequence=32, layers=2, heads=2, width=64, eval=1),
        "dev": dict(steps=100, batch=32, sequence=256, layers=6, heads=6, width=384, eval=8),
        # GPT-2-small model shape. The iteration count is a conservative initial
        # calibration setting, not a claim that this reference reaches 3.28.
        "official": dict(
            steps=19_073,
            batch=32,
            sequence=1024,
            layers=12,
            heads=12,
            width=768,
            # 320 * 32 * 1024 = the manifest's fixed 10,485,760-token prefix.
            eval=320,
        ),
    }[profile]
    steps = args.steps if args.steps is not None else defaults["steps"]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch"]
    seq_len = args.seq_len if args.seq_len is not None else defaults["sequence"]
    layers = args.layers if args.layers is not None else defaults["layers"]
    heads = args.heads if args.heads is not None else defaults["heads"]
    d_model = args.d_model if args.d_model is not None else defaults["width"]
    if profile == "official":
        validation_tokens = 10_485_760
        predictions_per_batch = batch_size * seq_len
        if validation_tokens % predictions_per_batch:
            raise ValueError(
                "official validation requires batch_size * seq_len to divide "
                f"{validation_tokens:,} exactly; got {predictions_per_batch:,}"
            )
        required_eval_batches = validation_tokens // predictions_per_batch
        if args.eval_batches is not None and args.eval_batches != required_eval_batches:
            raise ValueError(
                "official validation must cover exactly 10,485,760 predictions; "
                f"use --eval-batches {required_eval_batches} for the selected shape"
            )
        eval_batches = required_eval_batches
    elif args.eval_batches is not None:
        eval_batches = args.eval_batches
    else:
        eval_batches = defaults["eval"]
    # Keep the UI lively without forcing a host synchronization on every short
    # calibration step. Step 1 and the final step are always printed separately.
    log_every = args.log_every if args.log_every is not None else max(10, steps // 20)
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else min(steps, {"smoke": 1, "dev": 10, "official": 715}[profile])
    )
    warmup_steps = min(warmup_steps, steps)

    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "bfloat16" if platform == "tpu" and profile != "smoke" else "float32"
    compute_dtype = jnp.bfloat16 if dtype_name == "bfloat16" else jnp.float32

    if d_model % heads:
        raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")

    return Config(
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        layers=layers,
        heads=heads,
        d_model=d_model,
        mlp_mult=args.mlp_mult,
        learning_rate=args.learning_rate,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        eval_batches=eval_batches,
        log_every=log_every,
        vocab_size=vocab_size,
        compute_dtype=compute_dtype,
        dtype_name=dtype_name,
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


def load_dataset(args: argparse.Namespace) -> tuple[TokenDataset, int]:
    if args.data_path is None and not args.train_data and not args.val_data:
        dataset = built_in_dataset(args.seed)
        vocab_size = args.vocab_size or 256
        return dataset, vocab_size

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

    vocab_size = args.vocab_size or (256 if selected_profile(args) == "smoke" else 50_304)
    source = f"{len(train_shards)} train + {len(validation_shards)} val shard(s)"
    return (
        TokenDataset(ShardedTokens(train_shards), ShardedTokens(validation_shards), source),
        vocab_size,
    )


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


def gpt_logits(params: Mapping[str, Any], tokens: jax.Array, config: Config) -> jax.Array:
    dtype = config.compute_dtype
    batch, length = tokens.shape
    del batch
    x = params["token_embedding"][tokens].astype(dtype)
    x = x + params["position_embedding"][:length].astype(dtype)
    head_dim = config.d_model // config.heads
    causal = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))[None, None, :, :]

    for block in params["blocks"]:
        residual = x
        x_norm = layer_norm(x, block["ln1_scale"], block["ln1_bias"], dtype)
        qkv = linear(x_norm, block["qkv_w"], block["qkv_b"], dtype)
        query, key, value = jnp.split(qkv, 3, axis=-1)
        query = query.reshape(tokens.shape[0], length, config.heads, head_dim)
        key = key.reshape(tokens.shape[0], length, config.heads, head_dim)
        value = value.reshape(tokens.shape[0], length, config.heads, head_dim)
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

    x = layer_norm(x, params["final_ln_scale"], params["final_ln_bias"], dtype)
    # Weight tying avoids a second vocabulary-sized parameter matrix.
    return jnp.einsum(
        "btd,vd->btv", x, params["token_embedding"].astype(dtype)
    ).astype(jnp.float32)


def cross_entropy(params: Mapping[str, Any], x: jax.Array, y: jax.Array, config: Config) -> jax.Array:
    logits = gpt_logits(params, x, config)
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


def train_step(
    params: Any,
    optimizer: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    decay_mask: Any | None = None,
) -> tuple[Any, dict[str, Any], dict[str, jax.Array]]:
    if decay_mask is None:
        decay_mask = weight_decay_mask(params)
    loss, gradients = jax.value_and_grad(cross_entropy)(params, x, y, config)
    gradients = jax.tree_util.tree_map(lambda grad: grad.astype(jnp.float32), gradients)
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
    return params, {"step": step, "m": m, "v": v, "history": history}, {
        "loss": loss,
        "grad_norm": grad_norm,
        "learning_rate": lr,
    }


def eval_step(params: Any, x: jax.Array, y: jax.Array, config: Config) -> jax.Array:
    return cross_entropy(params, x, y, config)


def parameter_count(params: Any) -> int:
    return sum(int(value.size) for value in jax.tree_util.tree_leaves(params))


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


def save_checkpoint(output_dir: Path, params: Any, config: Config, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    host_params = jax.device_get(params)
    arrays = flatten_arrays(host_params)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "model": {
            "vocab_size": config.vocab_size,
            "seq_len": config.seq_len,
            "layers": config.layers,
            "heads": config.heads,
            "d_model": config.d_model,
            "mlp_mult": config.mlp_mult,
            "dtype": config.dtype_name,
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


def write_training_csv(output_dir: Path, history: np.ndarray, config: Config) -> None:
    """Atomically persist every optimizer step without timing host transfers."""

    if history.shape != (config.steps, 3):
        raise ValueError(
            f"training history has shape {history.shape}; expected {(config.steps, 3)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / TRAINING_CSV_NAME
    temporary = output_dir / f".{TRAINING_CSV_NAME}.tmp"
    tokens_per_step = config.batch_size * config.seq_len
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("step", "tokens_processed", "train_loss", "learning_rate", "grad_norm"))
        for index, (loss, learning_rate_value, grad_norm) in enumerate(history, 1):
            writer.writerow(
                (
                    index,
                    index * tokens_per_step,
                    float(loss),
                    float(learning_rate_value),
                    float(grad_norm),
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
    """Require the competition's exact single-host TPU v4-8 topology."""

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
        process_count != 1
        or len(local_devices) != 4
        or device_count != 4
        or len(devices) != 4
        or not is_tpu_v4
    ):
        raise RuntimeError(
            "official profile requires one JAX process with exactly 4 local TPU v4 "
            "devices (one TPU v4-8); detected "
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


def finite_metric(name: str, value: float, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0) or (not positive and value < 0.0):
        qualifier = "finite and positive" if positive else "finite and nonnegative"
        raise FloatingPointError(f"{name} must be {qualifier}, got {value!r}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    console = Console(args.color)
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

    dataset, vocab_size = load_dataset(args)
    config = resolve_config(args, platform, vocab_size)
    if config.batch_size % len(devices):
        raise ValueError(
            f"global batch size {config.batch_size} must be divisible by "
            f"visible device count {len(devices)}"
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
    params_total = parameter_count(host_params)
    flops_per_token = int(
        6 * params_total + 12 * config.layers * config.d_model * config.seq_len
    )
    tokens_processed = config.steps * config.batch_size * config.seq_len

    console.table(
        "run configuration",
        (
            ("devices", f"{len(devices)} × {device_label(devices)}"),
            ("mesh", f"data={len(devices)} (replicated model)"),
            ("dataset", dataset.source),
            ("train / val tokens", f"{len(dataset.train):,} / {len(dataset.validation):,}"),
            ("model", f"L{config.layers} D{config.d_model} H{config.heads} MLP×{config.mlp_mult}"),
            ("parameters", format_count(params_total)),
            ("global batch", f"{config.batch_size} × {config.seq_len} tokens"),
            ("compute", config.dtype_name),
            ("train tokens", format_count(tokens_processed)),
            ("estimated FLOPs", format_count(flops_per_token * tokens_processed)),
        ),
    )

    mesh = Mesh(np.asarray(devices, dtype=object), ("data",))
    replicated = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("data", None))
    params = jax.device_put(host_params, replicated)
    optimizer = jax.device_put(host_optimizer, replicated)
    del host_params, host_optimizer

    train_rng = np.random.default_rng(args.seed + 1)
    # Compilation may not inspect real data. Shapes and dtypes are sufficient.
    sample_x = np.zeros((config.batch_size, config.seq_len), dtype=np.int32)
    sample_y = np.zeros((config.batch_size, config.seq_len), dtype=np.int32)
    sample_x = jax.device_put(sample_x, data_sharding)
    sample_y = jax.device_put(sample_y, data_sharding)

    compiled_step = jax.jit(
        lambda p, o, x, y: train_step(p, o, x, y, config, decay_mask),
        in_shardings=(replicated, replicated, data_sharding, data_sharding),
        donate_argnums=(0, 1),
    )
    console.phase("Compiling train step", "compilation is outside train_seconds")
    compile_started = time.perf_counter()
    executable = compiled_step.lower(params, optimizer, sample_x, sample_y).compile()
    train_compile_seconds = time.perf_counter() - compile_started
    sync_tree((params, optimizer))
    console.phase("Training", f"compiled in {train_compile_seconds:.2f}s")

    last_metrics: Mapping[str, jax.Array] | None = None
    train_started = time.perf_counter()
    for step_index in range(1, config.steps + 1):
        batch_x, batch_y = dataset.batch(
            "train", train_rng, config.batch_size, config.seq_len, config.vocab_size
        )
        batch_x = jax.device_put(batch_x, data_sharding)
        batch_y = jax.device_put(batch_y, data_sharding)
        params, optimizer, last_metrics = executable(params, optimizer, batch_x, batch_y)
        should_log = (
            step_index == 1
            or step_index == config.steps
            or step_index % config.log_every == 0
        )
        if should_log:
            host_metrics = jax.device_get(last_metrics)
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

    if last_metrics is None:  # defensive: argparse prevents zero steps
        raise AssertionError("training produced no metrics")
    sync_tree((params, optimizer, last_metrics))
    train_seconds = max(time.perf_counter() - train_started, 1.0e-12)
    final_train = jax.device_get(last_metrics)
    training_history = np.asarray(jax.device_get(optimizer["history"]), dtype=np.float32)
    train_loss = finite_metric("train_loss", float(final_train["loss"]))

    console.phase("Compiling evaluation", "compilation is outside train_seconds")
    eval_x, eval_y = dataset.validation_batch(
        0, config.batch_size, config.seq_len, config.vocab_size
    )
    eval_x = jax.device_put(eval_x, data_sharding)
    eval_y = jax.device_put(eval_y, data_sharding)
    eval_compile_started = time.perf_counter()
    compiled_eval = jax.jit(
        lambda p, x, y: eval_step(p, x, y, config),
        in_shardings=(replicated, data_sharding, data_sharding),
    ).lower(params, eval_x, eval_y).compile()
    eval_compile_seconds = time.perf_counter() - eval_compile_started
    total_compile_seconds = train_compile_seconds + eval_compile_seconds
    console.phase(
        "Validation",
        f"{config.eval_batches} deterministic batches; compiled in {eval_compile_seconds:.2f}s",
    )
    validation_losses: list[float] = [float(jax.device_get(compiled_eval(params, eval_x, eval_y)))]
    for eval_index in range(1, config.eval_batches):
        eval_x_host, eval_y_host = dataset.validation_batch(
            eval_index, config.batch_size, config.seq_len, config.vocab_size
        )
        eval_x = jax.device_put(eval_x_host, data_sharding)
        eval_y = jax.device_put(eval_y_host, data_sharding)
        validation_losses.append(float(jax.device_get(compiled_eval(params, eval_x, eval_y))))
    validation_loss = finite_metric("validation_loss", float(np.mean(validation_losses)))

    output_dir = args.output_dir.expanduser().resolve()
    console.phase("Artifacts", f"{TRAINING_CSV_NAME} + {CHECKPOINT_NAME}")
    write_training_csv(output_dir, training_history, config)
    save_checkpoint(output_dir, params, config, args.seed)

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
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "track": args.track,
        "profile": profile,
        "seed": int(args.seed),
        "checkpoint": CHECKPOINT_NAME,
        "artifacts": {"training_curve": TRAINING_CSV_NAME},
        "system": system_metadata(devices),
        "contract": {
            "model_id": "reference-gpt-v1",
            "dataset_id": dataset_id,
            "tokenizer_id": tokenizer_id,
            "sequence_length": config.seq_len,
            "model": {
                "layers": config.layers,
                "heads": config.heads,
                "d_model": config.d_model,
                "mlp_mult": config.mlp_mult,
                "vocab_size": config.vocab_size,
                "tied_embeddings": True,
            },
        },
        "metrics": {
            "train_seconds": finite_metric("train_seconds", train_seconds, positive=True),
            "tokens_processed": int(tokens_processed),
            "validation_loss": validation_loss,
            "validation_tokens": int(
                config.eval_batches * config.batch_size * config.seq_len
            ),
            "train_loss": train_loss,
            "parameters": int(params_total),
            "flops_per_token": int(flops_per_token),
            "estimated_total_flops": total_flops,
            "tokens_per_second": tokens_per_second,
            "achieved_tflops": achieved_tflops,
            "mfu_estimate": finite_metric("mfu_estimate", mfu),
            "train_compile_seconds": finite_metric(
                "train_compile_seconds", train_compile_seconds
            ),
            "eval_compile_seconds": finite_metric(
                "eval_compile_seconds", eval_compile_seconds
            ),
            "total_compile_seconds": finite_metric(
                "total_compile_seconds", total_compile_seconds
            ),
        },
    }
    write_result(output_dir, result)
    console.success(validation_loss, train_seconds)
    return result


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
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
