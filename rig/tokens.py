"""Runtime token access: shards, batches, and downstream eval documents.

Separate from :mod:`rig.data`, which *prepares* a corpus -- downloads it,
verifies manifests, and installs it into a cache. This module is what a running
trainer touches: memory-mapped shards, the deterministic batch stream, and the
packed documents a downstream evaluation scores.

Nothing here knows what model is training. Functions take plain values rather
than a recipe's config or argparse namespace, so every recipe loads data the
same way and a fork cannot quietly diverge on how it samples.

The sampling contract matters more than it looks. ``ShuffledEpochBatchStream``
walks non-overlapping windows under a keyed permutation, so every JAX process
derives the same global order from the seed alone and consumes only its
rank-local slice. Hosts therefore never duplicate examples inside a global
batch, and no index array is ever materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import re

import numpy as np


_UINT64_MASK = (1 << 64) - 1
_DOMAIN_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


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


def _mix_uint64(value: int) -> int:
    """Return a stable SplitMix64 avalanche without relying on Python hashes."""

    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _permute_bounded(value: int, size: int, key: int) -> int:
    """Pseudo-randomly permute ``range(size)`` with O(1) auxiliary memory.

    A six-round balanced Feistel network permutes the next even-bit power-of-two
    domain. Cycle walking restricts that permutation to the requested interval.
    The domain is less than four times ``size``, so the expected work is bounded
    and no multi-billion-element index array is ever materialized.
    """

    if not 0 <= value < size:
        raise ValueError("permutation input must lie inside its domain")
    if size == 1:
        return 0
    bits = max(2, (size - 1).bit_length())
    if bits % 2:
        bits += 1
    half_bits = bits // 2
    half_mask = (1 << half_bits) - 1

    def feistel(candidate: int) -> int:
        left = candidate >> half_bits
        right = candidate & half_mask
        for round_index in range(6):
            round_key = _mix_uint64(key + round_index * 0xD1B54A32D192ED03)
            mixed = _mix_uint64(right ^ round_key) & half_mask
            left, right = right, left ^ mixed
        return (left << half_bits) | right

    candidate = value
    while True:
        candidate = feistel(candidate)
        if candidate < size:
            return candidate


class ShuffledEpochBatchStream:
    """Deterministic, distributed, no-replacement windows within each epoch.

    Shards are reordered once per epoch and all non-overlapping target windows
    across that layout are traversed through a keyed permutation. Every process
    constructs the same global order and takes only its contiguous rank-local
    part of each global batch. This keeps hosts disjoint without communication,
    while auxiliary memory remains O(number_of_shards + local_batch_size).
    """

    def __init__(
        self,
        tokens: ShardedTokens,
        *,
        global_batch_size: int,
        seq_len: int,
        vocab_size: int,
        seed: int,
        process_index: int = 0,
        process_count: int = 1,
    ) -> None:
        if global_batch_size <= 0 or seq_len <= 0 or vocab_size <= 0:
            raise ValueError(
                "stream batch, sequence, and vocabulary sizes must be positive"
            )
        if process_count <= 0 or not 0 <= process_index < process_count:
            raise ValueError("invalid process index/count for shuffled stream")
        if global_batch_size % process_count:
            raise ValueError("global batch size must be divisible by process count")
        self.tokens = tokens
        self.global_batch_size = int(global_batch_size)
        self.local_batch_size = global_batch_size // process_count
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed) & _UINT64_MASK
        self.process_index = int(process_index)
        self.window_counts = np.asarray(
            [(len(shard) - 1) // seq_len for shard in tokens.shards], dtype=np.int64
        )
        self.windows_per_epoch = int(self.window_counts.sum())
        if self.windows_per_epoch <= 0:
            raise ValueError(
                f"no training shard contains a full {seq_len:,}-target window"
            )
        self.usable_tokens_per_epoch = self.windows_per_epoch * seq_len
        self._global_cursor = 0
        self._planned_epoch = -1
        self._ordered_shards = np.empty((0,), dtype=np.int64)
        self._ordered_cumulative = np.empty((0,), dtype=np.int64)
        self._epoch_key = 0

    def _prepare_epoch(self, epoch: int) -> None:
        seed_words = (
            self.seed & 0xFFFFFFFF,
            self.seed >> 32,
            epoch & 0xFFFFFFFF,
            epoch >> 32,
        )
        rng = np.random.default_rng(np.random.SeedSequence(seed_words))
        self._ordered_shards = rng.permutation(len(self.tokens.shards)).astype(
            np.int64, copy=False
        )
        ordered_counts = self.window_counts[self._ordered_shards]
        self._ordered_cumulative = np.cumsum(ordered_counts, dtype=np.int64)
        self._epoch_key = _mix_uint64(self.seed ^ _mix_uint64(epoch))
        self._planned_epoch = epoch

    def _window(self, global_ordinal: int) -> np.ndarray:
        epoch, epoch_ordinal = divmod(global_ordinal, self.windows_per_epoch)
        if epoch != self._planned_epoch:
            self._prepare_epoch(epoch)
        permuted = _permute_bounded(
            int(epoch_ordinal), self.windows_per_epoch, self._epoch_key
        )
        ordered_index = int(
            np.searchsorted(self._ordered_cumulative, permuted, side="right")
        )
        previous = (
            int(self._ordered_cumulative[ordered_index - 1]) if ordered_index else 0
        )
        shard_index = int(self._ordered_shards[ordered_index])
        local_window = permuted - previous
        start = local_window * self.seq_len
        return self.tokens.shards[shard_index][start : start + self.seq_len + 1]

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        rank_start = self._global_cursor + self.process_index * self.local_batch_size
        windows = np.empty(
            (self.local_batch_size, self.seq_len + 1), dtype=np.int32
        )
        for row in range(self.local_batch_size):
            windows[row] = self._window(rank_start + row)
        self._global_cursor += self.global_batch_size
        observed_min = int(windows.min())
        observed_max = int(windows.max())
        if observed_min < 0 or observed_max >= self.vocab_size:
            raise ValueError(
                f"streamed token ids [{observed_min}, {observed_max}] do not fit "
                f"vocab_size={self.vocab_size}"
            )
        return (
            np.ascontiguousarray(windows[:, :-1]),
            np.ascontiguousarray(windows[:, 1:]),
        )


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
    *,
    data_path: Path | None,
    train_data: Sequence[Path],
    val_data: Sequence[Path],
    data_dtype: str,
    data_format: str,
    val_fraction: float,
    seed: int,
) -> TokenDataset:
    """Open the training and validation shards named by a run's arguments.

    Takes plain values, not a namespace: which vocabulary a recipe declares is
    a schema question and stays with the recipe, so this returns only the
    dataset.
    """

    if data_path is None and not train_data and not val_data:
        return built_in_dataset(seed)

    train_paths = [path.expanduser().resolve() for path in train_data]
    validation_paths = [path.expanduser().resolve() for path in val_data]
    if data_path is not None:
        data_path = data_path.expanduser().resolve()
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
        location = data_path if data_path is not None else "explicit arguments"
        raise FileNotFoundError(f"no training shards found from {location}")

    train_shards = [
        load_token_file(path, data_dtype, data_format) for path in train_paths
    ]
    if validation_paths:
        validation_shards = [
            load_token_file(path, data_dtype, data_format)
            for path in validation_paths
        ]
    else:
        train_shards, validation_shards = split_shards(train_shards, val_fraction)

    source = f"{len(train_shards)} train + {len(validation_shards)} val shard(s)"
    return TokenDataset(
        ShardedTokens(train_shards), ShardedTokens(validation_shards), source
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
    *,
    manifest: Path | None,
    root: Path | None,
    documents: Sequence[Path],
    vocab_size: int,
) -> tuple[DownstreamDomain, ...]:
    """Load canonical manifest shards or repeatable standalone documents."""

    if manifest is None and not documents:
        return ()
    if manifest is not None:
        manifest_path = manifest.expanduser().resolve()
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
            root.expanduser().resolve()
            if root is not None
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
    for specification in documents:
        if not isinstance(specification, str) or "=" not in specification:
            raise ValueError("--downstream-data must use DOMAIN=PATH")
        name, raw_path = specification.split("=", 1)
        if not _DOMAIN_NAME.fullmatch(name) or not raw_path:
            raise ValueError(f"invalid --downstream-data value: {specification!r}")
        grouped.setdefault(name, []).append(
            load_token_file(Path(raw_path), "uint16", "auto")
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


def downstream_batches(
    domain: DownstreamDomain, *, seq_len: int, batch_size: int
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Pack documents into fixed eval shapes without cross-document targets."""

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for document in domain.documents:
        target = document.score_offset
        remaining = document.scored_tokens
        while remaining:
            count = min(seq_len, remaining)
            x = np.zeros((seq_len,), dtype=np.int32)
            y = np.zeros((seq_len,), dtype=np.int32)
            mask = np.zeros((seq_len,), dtype=np.float32)
            x[:count] = domain.tokens[target - 1 : target - 1 + count]
            y[:count] = domain.tokens[target : target + count]
            mask[:count] = 1.0
            rows.append((x, y, mask))
            target += count
            remaining -= count

    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        x = np.zeros((batch_size, seq_len), dtype=np.int32)
        y = np.zeros((batch_size, seq_len), dtype=np.int32)
        mask = np.zeros((batch_size, seq_len), dtype=np.float32)
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
