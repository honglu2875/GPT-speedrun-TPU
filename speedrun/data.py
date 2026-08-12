"""Deterministic data preparation for the GPT TPU speedrun.

This module deliberately uses only the Python standard library.  Importing it
never performs I/O beyond defining constants; downloads happen only through an
explicit :func:`prepare_dataset` call.

Shards use the llm.c GPT-2 layout: a 1,024-byte header containing 256
little-endian int32 values followed by little-endian uint16 token IDs.  Header
slots 0, 1, and 2 are the magic number, format version, and token count.
"""

from __future__ import annotations

import hashlib
import json
import os
from array import array
from dataclasses import dataclass
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen


MAGIC = 20_240_520
FORMAT_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
TOKEN_BYTES = 2
_HEADER = struct.Struct("<256i")
_CHUNK_BYTES = 8 * 1024 * 1024
_MANIFEST_DIR = Path(__file__).resolve().parent.parent / "data" / "manifests"
_PROFILE_ALIASES = {
    "classic": "fineweb10b-gpt2",
    "fineweb": "fineweb10b-gpt2",
    "smoke": "smoke",
}

Progress = Callable[[str, int, int], None]


class DataError(ValueError):
    """Raised when a manifest or token shard violates the data contract."""


@dataclass(frozen=True)
class ShardHeader:
    magic: int
    version: int
    token_count: int


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    token_count: int
    byte_count: int
    sha256: str | None


@dataclass(frozen=True)
class PreparedDataset:
    """Paths and stable identity of one verified dataset selection."""

    name: str
    root: Path
    manifest_path: Path
    manifest_sha256: str
    train_files: tuple[Path, ...]
    validation_files: tuple[Path, ...]
    train_tokens: int
    validation_tokens: int
    validation_prefix_tokens: int = 0


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def read_header(path: str | os.PathLike[str]) -> ShardHeader:
    """Read and validate the fixed portion of an llm.c GPT-2 shard header."""

    shard = Path(path)
    try:
        with shard.open("rb") as handle:
            raw = handle.read(HEADER_BYTES)
    except OSError as exc:
        raise DataError(f"cannot read shard {shard}: {exc}") from exc
    if len(raw) != HEADER_BYTES:
        raise DataError(
            f"{shard}: truncated header ({len(raw)} bytes; expected {HEADER_BYTES})"
        )
    values = _HEADER.unpack(raw)
    header = ShardHeader(values[0], values[1], values[2])
    if header.magic != MAGIC:
        raise DataError(f"{shard}: bad magic {header.magic}; expected {MAGIC}")
    if header.version != FORMAT_VERSION:
        raise DataError(
            f"{shard}: unsupported version {header.version}; expected {FORMAT_VERSION}"
        )
    if header.token_count < 0:
        raise DataError(f"{shard}: negative token count {header.token_count}")
    return header


def validate_shard(
    path: str | os.PathLike[str],
    *,
    expected_tokens: int | None = None,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    verify_hash: bool = True,
) -> ShardInfo:
    """Validate header fields, exact length, and optionally SHA-256."""

    shard = Path(path)
    header = read_header(shard)
    actual_bytes = shard.stat().st_size
    encoded_bytes = HEADER_BYTES + header.token_count * TOKEN_BYTES
    if actual_bytes != encoded_bytes:
        raise DataError(
            f"{shard}: length is {actual_bytes:,} bytes, but header declares "
            f"{header.token_count:,} tokens ({encoded_bytes:,} bytes total)"
        )
    if expected_tokens is not None and header.token_count != expected_tokens:
        raise DataError(
            f"{shard}: contains {header.token_count:,} tokens; expected "
            f"{expected_tokens:,}"
        )
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise DataError(
            f"{shard}: length is {actual_bytes:,} bytes; expected "
            f"{expected_bytes:,}"
        )
    actual_sha256 = None
    if expected_sha256 is not None and verify_hash:
        actual_sha256 = sha256_file(shard)
        if actual_sha256.lower() != expected_sha256.lower():
            raise DataError(
                f"{shard}: SHA-256 is {actual_sha256}; expected {expected_sha256}"
            )
    return ShardInfo(shard, header.token_count, actual_bytes, actual_sha256)


def manifest_path(name: str) -> Path:
    canonical = _PROFILE_ALIASES.get(name, name)
    if not canonical or Path(canonical).name != canonical:
        raise DataError(f"invalid built-in manifest name: {name!r}")
    path = _MANIFEST_DIR / f"{canonical}.json"
    if not path.is_file():
        choices = ", ".join(p.stem for p in sorted(_MANIFEST_DIR.glob("*.json")))
        raise DataError(f"unknown data manifest {name!r}; available: {choices}")
    return path


def load_manifest(
    manifest: str | os.PathLike[str] | Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Load a built-in manifest name, JSON path, or in-memory manifest."""

    if isinstance(manifest, Mapping):
        payload = dict(manifest)
        source = Path("<in-memory-manifest>")
    else:
        candidate = Path(manifest)
        source = candidate if candidate.is_file() else manifest_path(str(manifest))
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"cannot load manifest {source}: {exc}") from exc
    _validate_manifest(payload, source)
    return payload, source


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def selected_files(
    manifest: Mapping[str, Any], train_shards: int | None = None
) -> tuple[dict[str, Any], ...]:
    """Select all validation files and the first N ordered training files."""

    training = sorted(
        (dict(item) for item in manifest["files"] if item["split"] == "train"),
        key=lambda item: item["path"],
    )
    validation = sorted(
        (dict(item) for item in manifest["files"] if item["split"] == "validation"),
        key=lambda item: item["path"],
    )
    if train_shards is None:
        train_shards = int(manifest.get("default_train_shards", len(training)))
    if train_shards < 1 or train_shards > len(training):
        raise DataError(
            f"train_shards must be between 1 and {len(training)}; got {train_shards}"
        )
    return tuple(validation + training[:train_shards])


def ensure_data_root(path: str | os.PathLike[str], required_bytes: int = 0) -> Path:
    """Create and preflight the exact cache root chosen by the caller.

    Symlinks (for example ``shm -> /dev/shm``) are supported.  Nothing outside
    this root is inspected or removed.
    """

    root = Path(path).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise DataError(f"data path is not a directory: {root}")
        with tempfile.NamedTemporaryFile(prefix=".speedrun-write-check-", dir=root):
            pass
        free = shutil.disk_usage(root).free
    except DataError:
        raise
    except OSError as exc:
        raise DataError(f"data path is not writable: {root} ({exc})") from exc
    if free < required_bytes:
        raise DataError(
            f"not enough free space at {root}: need {required_bytes:,} bytes, "
            f"have {free:,}"
        )
    return root


def verify_dataset(
    manifest: str | os.PathLike[str] | Mapping[str, Any],
    root: str | os.PathLike[str],
    *,
    train_shards: int | None = None,
    verify_hash: bool = True,
) -> PreparedDataset:
    """Verify an already-prepared dataset without network or mutation."""

    payload, source = load_manifest(manifest)
    data_root = Path(root).expanduser().resolve(strict=True)
    if not data_root.is_dir():
        raise DataError(f"data path is not a directory: {data_root}")
    entries = selected_files(payload, train_shards)
    verified: list[tuple[Mapping[str, Any], ShardInfo]] = []
    for entry in entries:
        target = _safe_target(data_root, entry["path"])
        if not target.is_file():
            raise DataError(f"missing dataset shard: {target}")
        info = _validate_entry(target, entry, verify_hash=verify_hash)
        verified.append((entry, info))
    return _prepared_result(payload, source, data_root, verified)


def prepare_dataset(
    manifest: str | os.PathLike[str] | Mapping[str, Any],
    root: str | os.PathLike[str],
    *,
    train_shards: int | None = None,
    force: bool = False,
    offline: bool = False,
    progress: Progress | None = None,
    timeout: float = 60.0,
) -> PreparedDataset:
    """Explicitly generate/download selected shards, then verify them.

    Downloads are resumable and are written to ``<name>.part``.  A complete
    part is validated before an atomic rename, so training never observes a
    partially written target.
    """

    payload, source = load_manifest(manifest)
    entries = selected_files(payload, train_shards)
    data_root = ensure_data_root(root)
    missing_bytes = _required_download_bytes(data_root, entries, force=force)
    ensure_data_root(data_root, missing_bytes)

    verified: list[tuple[Mapping[str, Any], ShardInfo]] = []
    for entry in entries:
        target = _safe_target(data_root, entry["path"])
        if target.is_file() and not force:
            try:
                info = _validate_entry(target, entry, verify_hash=True)
            except DataError as exc:
                raise DataError(
                    f"existing shard is invalid: {target}; pass force=True to replace it"
                ) from exc
            verified.append((entry, info))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry.get("generator"):
            info = _generate_entry(target, entry, progress=progress)
        else:
            url = entry.get("url")
            if not url:
                raise DataError(
                    f"{target} is missing and manifest provides no URL; place a matching "
                    "user-supplied shard there, then run verification"
                )
            if offline:
                raise DataError(
                    f"offline preparation cannot download missing shard {target}; "
                    "copy it into the data root or run without offline=True"
                )
            info = _download_entry(target, entry, timeout=timeout, progress=progress)
        verified.append((entry, info))

    # Every entry above underwent one definitive header/length/hash validation.
    # Re-reading all shards here would add another full pass over atomically
    # installed files without strengthening their integrity guarantee.
    return _prepared_result(payload, source, data_root, verified)


def prepare_smoke(
    root: str | os.PathLike[str], *, force: bool = False
) -> PreparedDataset:
    """Generate the tiny deterministic, CPU-friendly smoke dataset."""

    return prepare_dataset("smoke", root, force=force)


def prepare(
    path: str | os.PathLike[str],
    profile: str | os.PathLike[str] | Mapping[str, Any] = "classic",
    *,
    train_shards: int | None = None,
    offline: bool = False,
    check_only: bool = False,
    force: bool = False,
    progress: Progress | None = None,
    timeout: float = 60.0,
) -> PreparedDataset:
    """Wizard/CLI-friendly entry point with no prompting of its own.

    ``check_only`` is strictly read-only.  ``offline`` permits deterministic
    generation and existing files but forbids all HTTP requests.
    """

    if check_only:
        return verify_dataset(profile, path, train_shards=train_shards)
    return prepare_dataset(
        profile,
        path,
        train_shards=train_shards,
        force=force,
        offline=offline,
        progress=progress,
        timeout=timeout,
    )


def _validate_manifest(payload: Mapping[str, Any], source: Path) -> None:
    if payload.get("schema_version") != 1:
        raise DataError(f"{source}: unsupported or missing schema_version")
    if not isinstance(payload.get("name"), str) or not payload["name"]:
        raise DataError(f"{source}: missing manifest name")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise DataError(f"{source}: files must be a non-empty list")
    seen: set[str] = set()
    splits: set[str] = set()
    validation_tokens = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise DataError(f"{source}: every file entry must be an object")
        relative = entry.get("path")
        relative_path = Path(relative) if isinstance(relative, str) else None
        if (
            not isinstance(relative, str)
            or not relative
            or relative_path == Path(".")
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise DataError(f"{source}: unsafe shard path {relative!r}")
        if relative in seen:
            raise DataError(f"{source}: duplicate shard path {relative!r}")
        seen.add(relative)
        split = entry.get("split")
        if split not in {"train", "validation"}:
            raise DataError(f"{source}: invalid split for {relative!r}: {split!r}")
        splits.add(split)
        tokens = entry.get("tokens")
        byte_count = entry.get("bytes")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise DataError(f"{source}: invalid token count for {relative!r}")
        if byte_count != HEADER_BYTES + tokens * TOKEN_BYTES:
            raise DataError(f"{source}: inconsistent byte count for {relative!r}")
        if split == "validation":
            validation_tokens += tokens
        expected_hash = entry.get("sha256")
        if expected_hash is not None and (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in expected_hash)
        ):
            raise DataError(f"{source}: invalid SHA-256 for {relative!r}")
        if not entry.get("url") and not entry.get("generator"):
            # Deliberately supported for user-supplied, offline shards.
            continue
    if splits != {"train", "validation"}:
        raise DataError(f"{source}: manifest needs train and validation files")
    validation_prefix = payload.get("validation_prefix_tokens", validation_tokens)
    if (
        isinstance(validation_prefix, bool)
        or not isinstance(validation_prefix, int)
        or validation_prefix <= 0
        or validation_prefix > validation_tokens
    ):
        raise DataError(
            f"{source}: validation_prefix_tokens must be between 1 and "
            f"{validation_tokens:,}"
        )


def _safe_target(root: Path, relative: str) -> Path:
    root = root.resolve(strict=True)
    target = (root / relative).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise DataError(f"shard path escapes data root: {relative!r}") from exc
    return target


def _validate_entry(path: Path, entry: Mapping[str, Any], verify_hash: bool) -> ShardInfo:
    return validate_shard(
        path,
        expected_tokens=int(entry["tokens"]),
        expected_bytes=int(entry["bytes"]),
        expected_sha256=entry.get("sha256"),
        verify_hash=verify_hash,
    )


def _prepared_result(
    payload: Mapping[str, Any],
    source: Path,
    root: Path,
    verified: Sequence[tuple[Mapping[str, Any], ShardInfo]],
) -> PreparedDataset:
    """Build a dataset result from entries already validated exactly once."""

    train_paths: list[Path] = []
    validation_paths: list[Path] = []
    train_tokens = validation_tokens = 0
    for entry, info in verified:
        if entry["split"] == "train":
            train_paths.append(info.path)
            train_tokens += info.token_count
        else:
            validation_paths.append(info.path)
            validation_tokens += info.token_count
    return PreparedDataset(
        name=str(payload["name"]),
        root=root,
        manifest_path=source.resolve() if source.exists() else source,
        manifest_sha256=manifest_digest(payload),
        train_files=tuple(train_paths),
        validation_files=tuple(validation_paths),
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        validation_prefix_tokens=int(
            payload.get("validation_prefix_tokens", validation_tokens)
        ),
    )


def _part_size(path: Path) -> int | None:
    """Return a partial file's size while refusing links and special files."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DataError(f"cannot inspect partial file {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise DataError(f"refusing symlink partial file: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise DataError(f"partial path is not a regular file: {path}")
    return metadata.st_size


def _open_part(path: Path, *, append: bool) -> Any:
    """Safely open a checked partial file without following a final symlink."""

    _part_size(path)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= os.O_APPEND if append else os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o666)
    except OSError as exc:
        raise DataError(f"cannot open partial file {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise DataError(f"partial path is not a regular file: {path}")
        return os.fdopen(descriptor, "ab" if append else "wb")
    except Exception:
        os.close(descriptor)
        raise


def _required_download_bytes(
    root: Path, entries: Sequence[Mapping[str, Any]], *, force: bool
) -> int:
    required = 0
    for entry in entries:
        target = _safe_target(root, str(entry["path"]))
        if target.is_file() and not force:
            # The main preparation loop performs the one definitive hash pass.
            # An invalid existing target is reported there rather than replaced.
            continue
        part = target.with_name(target.name + ".part")
        part_bytes = _part_size(part)
        partial_bytes = min(part_bytes or 0, int(entry["bytes"]))
        required += int(entry["bytes"]) - partial_bytes
    return required


def _write_header(handle: Any, token_count: int) -> None:
    values = [0] * HEADER_INTS
    values[0] = MAGIC
    values[1] = FORMAT_VERSION
    values[2] = token_count
    handle.write(_HEADER.pack(*values))


def _smoke_token_stream(count: int, seed: int) -> Iterable[int]:
    """Stable xorshift32 + repeating motifs; independent of Python RNG details."""

    state = seed & 0xFFFFFFFF or 0x6D2B79F5
    for index in range(count):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        position = index % 127
        if position == 0:
            yield 255  # deterministic document delimiter
        elif position % 17 == 0:
            yield (state >> 24) & 31
        else:
            yield (position * 29 + (index // 127) * 7) % 251


def _generate_entry(
    target: Path, entry: Mapping[str, Any], progress: Progress | None
) -> ShardInfo:
    spec = entry["generator"]
    if spec.get("algorithm") != "xorshift32-motif-v1":
        raise DataError(f"unsupported smoke generator: {spec.get('algorithm')!r}")
    part = target.with_name(target.name + ".part")
    token_count = int(entry["tokens"])
    tokens = array("H", _smoke_token_stream(token_count, int(spec["seed"])))
    if sys.byteorder != "little":
        tokens.byteswap()
    try:
        with _open_part(part, append=False) as handle:
            _write_header(handle, token_count)
            handle.write(tokens.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        info = _validate_entry(part, entry, verify_hash=True)
        os.replace(part, target)
        _fsync_directory(target.parent)
    except DataError:
        raise
    except OSError as exc:
        raise DataError(f"cannot generate {target}: {exc}") from exc
    if progress:
        progress(str(entry["path"]), int(entry["bytes"]), int(entry["bytes"]))
    return ShardInfo(target, info.token_count, info.byte_count, info.sha256)


def _download_entry(
    target: Path,
    entry: Mapping[str, Any],
    *,
    timeout: float,
    progress: Progress | None,
) -> ShardInfo:
    part = target.with_name(target.name + ".part")
    expected = int(entry["bytes"])
    start = _part_size(part) or 0
    if start == expected:
        try:
            info = _validate_entry(part, entry, verify_hash=True)
            os.replace(part, target)
            _fsync_directory(target.parent)
            return ShardInfo(target, info.token_count, info.byte_count, info.sha256)
        except DataError:
            start = 0
    elif start > expected:
        start = 0

    headers = {"User-Agent": "gpt-speedrun-tpu-data/1"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = Request(str(entry["url"]), headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as exc:
        raise DataError(f"download failed for {entry['path']}: HTTP {exc.code}") from exc
    except OSError as exc:
        raise DataError(f"download failed for {entry['path']}: {exc}") from exc

    with response:
        status = getattr(response, "status", response.getcode())
        if start and status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise DataError(
                    f"server returned an unexpected Content-Range for {entry['path']}: "
                    f"{content_range!r}"
                )
            mode = "ab"
        elif status == 200:
            # The server ignored Range.  Restart safely instead of appending.
            start = 0
            mode = "wb"
        else:
            raise DataError(f"download failed for {entry['path']}: HTTP {status}")
        try:
            with _open_part(part, append=mode == "ab") as handle:
                completed = start
                if progress:
                    progress(str(entry["path"]), completed, expected)
                while True:
                    try:
                        block = response.read(_CHUNK_BYTES)
                    except Exception as exc:
                        raise DataError(
                            f"download interrupted for {entry['path']}: {exc}"
                        ) from exc
                    if not block:
                        break
                    completed += handle.write(block)
                    if completed > expected:
                        raise DataError(
                            f"download for {entry['path']} exceeded expected size {expected:,}"
                        )
                    if progress:
                        progress(str(entry["path"]), completed, expected)
                handle.flush()
                os.fsync(handle.fileno())
        except DataError:
            raise
        except OSError as exc:
            raise DataError(f"cannot write download {part}: {exc}") from exc

    info = _validate_entry(part, entry, verify_hash=True)
    try:
        os.replace(part, target)
        _fsync_directory(target.parent)
    except OSError as exc:
        raise DataError(f"cannot install downloaded shard {target}: {exc}") from exc
    return ShardInfo(target, info.token_count, info.byte_count, info.sha256)


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DataError",
    "FORMAT_VERSION",
    "HEADER_BYTES",
    "MAGIC",
    "PreparedDataset",
    "ShardHeader",
    "ShardInfo",
    "ensure_data_root",
    "load_manifest",
    "manifest_digest",
    "manifest_path",
    "prepare_dataset",
    "prepare",
    "prepare_smoke",
    "read_header",
    "selected_files",
    "sha256_file",
    "validate_shard",
    "verify_dataset",
]
