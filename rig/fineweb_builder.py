"""Bounded-memory, resumable FineWeb-to-llm.c shard preparation.

The production entry point lives in ``scripts/prepare_fineweb.py`` so the
large, preparation-only PyArrow and tiktoken dependencies do not become
runtime dependencies of the speedrun harness.  This module intentionally
imports neither package at module import time; its small core is unit-testable
with synthetic document sources and tokenizers.

The source contract is the immutable, globally shuffled
``HuggingFaceFW/fineweb_100BT-shuffled`` revision below.  Output variants are
nested prefixes of one token stream.  Completed shards are kept once in a
private pool and hard-linked into the user-facing variant directories.
"""

from __future__ import annotations

from array import array
import base64
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import zlib

from speedrun.data import (
    FORMAT_VERSION,
    HEADER_BYTES,
    HEADER_INTS,
    MAGIC,
    TOKEN_BYTES,
    sha256_file,
    validate_shard,
)


SOURCE_REPOSITORY = "HuggingFaceFW/fineweb_100BT-shuffled"
SOURCE_REVISION = "ee8552966e3d6a5fee2f317f2ae0b342be03d998"
SOURCE_FILE_COUNT = 100
SOURCE_DOCUMENT_COUNT = 160_677_091
UPSTREAM_GLOBAL_SHUFFLE_SEED = 42
TOKENIZER_NAME = "gpt2"
TIKTOKEN_VERSION = "0.11.0"
PYARROW_VERSION = "19.0.1"
EOT_TOKEN = 50_256
VOCAB_SIZE = 50_257
DEFAULT_SHARD_TOKENS = 100_000_000
DEFAULT_VALIDATION_TOKENS = 100_000_000
DEFAULT_HERO_TOKENS = 75_000_000_000
DEFAULT_RESERVE_BYTES = 16 * 1024**3
DEFAULT_SOURCE_DATE_CUTOFF = "2024-04-01"
CHECKPOINT_SCHEMA_VERSION = 1
SOURCE_INVENTORY_SCHEMA_VERSION = 1
BUILDER_VERSION = 1
_HEADER = struct.Struct("<256i")
_IO_CHUNK_BYTES = 8 * 1024 * 1024


class FineWebBuildError(RuntimeError):
    """Raised when a build cannot continue without risking wrong output."""


@dataclass(frozen=True)
class SourceFile:
    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class SourceInventory:
    repository: str
    revision: str
    files: tuple[SourceFile, ...]
    document_count: int = SOURCE_DOCUMENT_COUNT
    global_shuffle_seed: int = UPSTREAM_GLOBAL_SHUFFLE_SEED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_INVENTORY_SCHEMA_VERSION,
            "repository": self.repository,
            "revision": self.revision,
            "split": "train",
            "documents": self.document_count,
            "upstream_shuffle": {
                "scope": "global documents",
                "seed": self.global_shuffle_seed,
                "provenance": "dataset card claim",
            },
            "files": [
                {
                    "path": item.path,
                    "bytes": item.byte_count,
                    "sha256": item.sha256,
                }
                for item in self.files
            ],
        }

    @property
    def digest(self) -> str:
        return canonical_json_sha256(self.as_dict())


@dataclass(frozen=True)
class ExclusionPolicy:
    """Fail-closed temporal filter plus defensive Fresh10 exclusions."""

    source_date_before: str
    normalized_urls: frozenset[str]
    exact_text_sha256: frozenset[str]
    raw_source_sha256: frozenset[str]

    def validate(self) -> None:
        try:
            cutoff = date.fromisoformat(self.source_date_before)
        except ValueError as exc:
            raise FineWebBuildError("source_date_before must be an ISO date") from exc
        if cutoff != date(2024, 4, 1):
            raise FineWebBuildError(
                "the reproducible FineWeb build requires source dates before 2024-04-01"
            )
        if len(self.normalized_urls) != 40:
            raise FineWebBuildError(
                f"Fresh10 exclusion policy must contain 40 URLs; got {len(self.normalized_urls)}"
            )
        if any(not _is_sha256(value) for value in self.exact_text_sha256):
            raise FineWebBuildError(
                "Fresh10 text exclusion contains an invalid SHA-256"
            )
        if any(not _is_sha256(value) for value in self.raw_source_sha256):
            raise FineWebBuildError("Fresh10 raw exclusion contains an invalid SHA-256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source_date": {
                "operator": "<",
                "cutoff": self.source_date_before,
                "missing_or_invalid": "exclude",
            },
            "fresh10": {
                "normalized_urls": sorted(self.normalized_urls),
                "exact_canonical_text_sha256": sorted(self.exact_text_sha256),
                "raw_source_sha256": sorted(self.raw_source_sha256),
                "notes": (
                    "URL and canonical-text hashes are defensive. Raw artifact "
                    "hashes are also compared opportunistically to UTF-8 row bytes, "
                    "but are not normalized-content hashes. The pre-2024-04-01 "
                    "date cutoff is the primary temporal isolation boundary."
                ),
            },
        }

    @property
    def digest(self) -> str:
        return canonical_json_sha256(self.as_dict())

    def reason(self, text: str, url: str, source_date: str) -> str | None:
        parsed_date = _parse_source_date(source_date)
        if parsed_date is None:
            return "missing_or_invalid_date"
        if parsed_date >= date.fromisoformat(self.source_date_before):
            return "date_cutoff"
        if normalize_url(url) in self.normalized_urls:
            return "fresh10_url"
        text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_digest in self.exact_text_sha256:
            return "fresh10_text_sha256"
        if text_digest in self.raw_source_sha256:
            return "fresh10_raw_sha256"
        return None


@dataclass(frozen=True)
class Variant:
    directory: str
    total_tokens: int


@dataclass(frozen=True)
class BuildConfig:
    root: Path
    hero_tokens: int = DEFAULT_HERO_TOKENS
    shard_tokens: int = DEFAULT_SHARD_TOKENS
    validation_tokens: int = DEFAULT_VALIDATION_TOKENS
    reserve_bytes: int = DEFAULT_RESERVE_BYTES
    max_document_bytes: int = 16 * 1024**2
    batch_rows: int = 256
    tokenizer_threads: int = 16
    variant_override: tuple[Variant, ...] = ()

    def variants(self) -> tuple[Variant, ...]:
        if self.variant_override:
            return self.variant_override
        return (
            Variant("2B", 2_000_000_000),
            Variant("4B", 4_000_000_000),
            Variant("8B", 8_000_000_000),
            Variant("hero", self.hero_tokens),
        )

    def validate(self) -> None:
        if self.shard_tokens <= 0:
            raise FineWebBuildError("shard_tokens must be positive")
        if self.validation_tokens != self.shard_tokens:
            raise FineWebBuildError(
                "validation_tokens must equal one shard; production uses 100M"
            )
        if self.shard_tokens > 2_147_483_647:
            raise FineWebBuildError("llm.c v1 stores token count in signed int32")
        if not self.variant_override and self.hero_tokens < 8_000_000_000:
            raise FineWebBuildError("hero_tokens must be at least 8B")
        variants = self.variants()
        if not variants:
            raise FineWebBuildError("at least one output variant is required")
        if len({variant.directory for variant in variants}) != len(variants):
            raise FineWebBuildError("variant directory names must be unique")
        if tuple(sorted(variant.total_tokens for variant in variants)) != tuple(
            variant.total_tokens for variant in variants
        ):
            raise FineWebBuildError(
                "variants must be ordered by increasing token target"
            )
        for variant in variants:
            if (
                not variant.directory
                or Path(variant.directory).name != variant.directory
                or variant.directory.startswith(".")
            ):
                raise FineWebBuildError(
                    f"unsafe variant directory: {variant.directory!r}"
                )
            if variant.total_tokens % self.shard_tokens:
                raise FineWebBuildError(
                    f"{variant.directory} token target must be divisible by shard_tokens"
                )
            if variant.total_tokens <= self.validation_tokens:
                raise FineWebBuildError(
                    f"{variant.directory} needs at least one training shard"
                )
        if self.max_document_bytes <= 0:
            raise FineWebBuildError("max_document_bytes must be positive")
        if self.batch_rows <= 0 or self.tokenizer_threads <= 0:
            raise FineWebBuildError("batch_rows and tokenizer_threads must be positive")
        if self.reserve_bytes < 0:
            raise FineWebBuildError("reserve_bytes cannot be negative")


@dataclass(frozen=True)
class DocumentBatch:
    texts: tuple[str, ...]
    next_cursors: tuple[Mapping[str, int], ...]
    document_ids: tuple[str, ...]
    urls: tuple[str, ...] = ()
    source_dates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.urls:
            object.__setattr__(
                self, "urls", ("https://example.invalid/",) * len(self.texts)
            )
        if not self.source_dates:
            object.__setattr__(self, "source_dates", ("2020-01-01",) * len(self.texts))
        if not (
            len(self.texts)
            == len(self.next_cursors)
            == len(self.document_ids)
            == len(self.urls)
            == len(self.source_dates)
        ):
            raise FineWebBuildError("document batch columns have different lengths")


class DocumentSource(Protocol):
    def initial_cursor(self) -> Mapping[str, int]: ...

    def iter_batches(
        self, cursor: Mapping[str, int], batch_rows: int
    ) -> Iterable[DocumentBatch]: ...


class TokenEncoder(Protocol):
    name: str
    implementation: str
    version: str

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]: ...


@dataclass(frozen=True)
class ShardRecord:
    path: str
    split: str
    tokens: int
    byte_count: int
    sha256: str
    documents_processed: int = 0
    source_utf8_bytes: int = 0
    exclusion_counts: Mapping[str, int] | None = None
    validation_boundary_discarded_tokens: int = 0
    validation_boundary_document_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "split": self.split,
            "tokens": self.tokens,
            "bytes": self.byte_count,
            "sha256": self.sha256,
        }

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            **self.as_dict(),
            "prefix_provenance": {
                "documents_processed": self.documents_processed,
                "source_utf8_bytes": self.source_utf8_bytes,
                "excluded_documents": dict(
                    sorted((self.exclusion_counts or {}).items())
                ),
                "validation_boundary": {
                    "document_disjoint": True,
                    "discarded_tokens": self.validation_boundary_discarded_tokens,
                    "document_id_sha256": self.validation_boundary_document_sha256,
                },
            },
        }


@dataclass
class BuildState:
    core_digest: str
    cursor: dict[str, int]
    pending_tokens: array
    shards: list[ShardRecord]
    documents_processed: int = 0
    source_utf8_bytes: int = 0
    last_document_id: str = ""
    exclusion_counts: dict[str, int] | None = None
    validation_boundary_discarded_tokens: int = 0
    validation_boundary_document_sha256: str = ""


@dataclass(frozen=True)
class BuildResult:
    root: Path
    completed_shards: int
    completed_tokens: int
    stopped_at: str
    complete_variants: tuple[str, ...]


@dataclass(frozen=True)
class ProbeResult:
    examined_documents: int
    accepted_documents: int
    examined_gpt2_tokens: int
    accepted_gpt2_tokens: int
    examined_utf8_bytes: int
    accepted_utf8_bytes: int
    exclusion_documents: Mapping[str, int]
    exclusion_gpt2_tokens: Mapping[str, int]
    next_cursor: Mapping[str, int]

    @property
    def estimated_token_retention(self) -> float:
        if not self.examined_gpt2_tokens:
            return 0.0
        return self.accepted_gpt2_tokens / self.examined_gpt2_tokens


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalize_url(value: str) -> str:
    """Canonicalize a URL for defensive exact-source exclusion."""

    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        parsed = urlsplit(value.strip())
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        if not scheme or not hostname:
            return value.strip()
        port = parsed.port
        if port is not None and not (
            (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        ):
            hostname = f"{hostname}:{port}"
        path = parsed.path or "/"
        query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
        return urlunsplit((scheme, hostname, path, query, ""))
    except ValueError:
        return value.strip()


def load_fresh10_exclusion_policy(
    manifest_path: str | os.PathLike[str],
) -> ExclusionPolicy:
    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FineWebBuildError(
            f"cannot load Fresh10 exclusions from {path}: {exc}"
        ) from exc
    domains = payload.get("domains") if isinstance(payload, dict) else None
    if not isinstance(domains, list):
        raise FineWebBuildError("Fresh10 exclusion manifest has no domains")
    urls: set[str] = set()
    text_hashes: set[str] = set()
    raw_hashes: set[str] = set()
    document_count = 0
    for domain in domains:
        documents = domain.get("documents") if isinstance(domain, dict) else None
        if not isinstance(documents, list):
            raise FineWebBuildError("Fresh10 exclusion domain has no documents")
        for document in documents:
            if not isinstance(document, dict):
                raise FineWebBuildError("Fresh10 exclusion document is malformed")
            normalized = normalize_url(str(document.get("source_url", "")))
            text_digest = str(document.get("text_sha256", "")).lower()
            raw_digest = str(document.get("raw_sha256", "")).lower()
            if (
                not normalized
                or not _is_sha256(text_digest)
                or not _is_sha256(raw_digest)
            ):
                raise FineWebBuildError(
                    "Fresh10 exclusion document lacks URL/hash provenance"
                )
            urls.add(normalized)
            text_hashes.add(text_digest)
            raw_hashes.add(raw_digest)
            document_count += 1
    if document_count != 40:
        raise FineWebBuildError(
            f"Fresh10 exclusion manifest must contain 40 documents; got {document_count}"
        )
    policy = ExclusionPolicy(
        DEFAULT_SOURCE_DATE_CUTOFF,
        frozenset(urls),
        frozenset(text_hashes),
        frozenset(raw_hashes),
    )
    policy.validate()
    return policy


def configure_cache_root(root: str | os.PathLike[str]) -> dict[str, Path]:
    """Force all preparation caches and temporary files under ``root``.

    This function must run before importing tiktoken, PyArrow, datasets, or
    huggingface_hub.  Values are overwritten rather than inherited so a
    machine-level cache setting cannot silently spill onto the boot disk.
    """

    selected = _select_root(Path(root).expanduser())
    cache = _ensure_private_directory(selected / ".cache", root=selected)
    paths = {
        "hf_home": _ensure_private_directory(cache / "huggingface", root=selected),
        "hf_datasets": _ensure_private_directory(cache / "datasets", root=selected),
        "hf_hub": _ensure_private_directory(cache / "hub", root=selected),
        "tiktoken": _ensure_private_directory(cache / "tiktoken", root=selected),
        "tmp": _ensure_private_directory(cache / "tmp", root=selected),
        "source": _ensure_private_directory(cache / "source", root=selected),
    }
    environment = {
        "HF_HOME": paths["hf_home"],
        "HF_DATASETS_CACHE": paths["hf_datasets"],
        "HUGGINGFACE_HUB_CACHE": paths["hf_hub"],
        "HF_HUB_CACHE": paths["hf_hub"],
        "TIKTOKEN_CACHE_DIR": paths["tiktoken"],
        "XDG_CACHE_HOME": cache,
        "TMPDIR": paths["tmp"],
        "TEMP": paths["tmp"],
        "TMP": paths["tmp"],
    }
    for name, path in environment.items():
        os.environ[name] = str(path)
    tempfile.tempdir = None
    return paths


def ensure_build_work_directory(root: str | os.PathLike[str]) -> Path:
    """Return the symlink-safe private metadata directory under a selected root."""

    selected = _select_root(Path(root).expanduser())
    return _ensure_private_directory(selected / ".fineweb-build", root=selected)


def fetch_source_inventory(timeout: float = 60.0) -> SourceInventory:
    """Fetch and validate immutable LFS metadata for all 100 Parquet files."""

    repository = quote(SOURCE_REPOSITORY, safe="/")
    url = (
        f"https://huggingface.co/api/datasets/{repository}/tree/"
        f"{SOURCE_REVISION}/data?recursive=true&expand=true&limit=100"
    )
    request = Request(url, headers={"User-Agent": "gpt-tpu-speedrun-fineweb/1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise FineWebBuildError(
            f"cannot fetch pinned FineWeb inventory: {type(exc).__name__}"
        ) from exc
    return source_inventory_from_tree(payload)


def source_inventory_from_tree(payload: Any) -> SourceInventory:
    if not isinstance(payload, list):
        raise FineWebBuildError("FineWeb tree API returned a non-list payload")
    files: list[SourceFile] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        path = item.get("path")
        lfs = item.get("lfs")
        if not isinstance(path, str) or not isinstance(lfs, dict):
            continue
        if not path.startswith("data/train-") or not path.endswith(".parquet"):
            continue
        byte_count = lfs.get("size")
        digest = lfs.get("oid")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not _is_sha256(digest)
        ):
            raise FineWebBuildError(f"invalid LFS metadata for {path}")
        files.append(SourceFile(path, byte_count, str(digest).lower()))
    files.sort(key=lambda item: item.path)
    expected_paths = [
        f"data/train-{index:05d}-of-00100.parquet" for index in range(100)
    ]
    if [item.path for item in files] != expected_paths:
        raise FineWebBuildError(
            "pinned FineWeb inventory is not the expected contiguous 100-file split"
        )
    return SourceInventory(SOURCE_REPOSITORY, SOURCE_REVISION, tuple(files))


def load_or_fetch_inventory(
    path: Path, *, fetcher: Callable[[], SourceInventory] = fetch_source_inventory
) -> SourceInventory:
    if path.is_file():
        _validate_regular_file(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FineWebBuildError(
                f"cannot read source inventory {path}: {exc}"
            ) from exc
        return source_inventory_from_dict(payload)
    inventory = fetcher()
    _atomic_write_json(path, inventory.as_dict())
    return inventory


def source_inventory_from_dict(payload: Any) -> SourceInventory:
    if not isinstance(payload, dict):
        raise FineWebBuildError("source inventory must be a JSON object")
    if (
        payload.get("schema_version") != SOURCE_INVENTORY_SCHEMA_VERSION
        or payload.get("repository") != SOURCE_REPOSITORY
        or payload.get("revision") != SOURCE_REVISION
    ):
        raise FineWebBuildError(
            "source inventory repository/revision does not match pin"
        )
    upstream = payload.get("upstream_shuffle")
    if (
        not isinstance(upstream, dict)
        or upstream.get("seed") != UPSTREAM_GLOBAL_SHUFFLE_SEED
    ):
        raise FineWebBuildError("source inventory has wrong global shuffle provenance")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise FineWebBuildError("source inventory files must be a list")
    tree = [
        {
            "type": "file",
            "path": item.get("path") if isinstance(item, dict) else None,
            "lfs": {
                "size": item.get("bytes") if isinstance(item, dict) else None,
                "oid": item.get("sha256") if isinstance(item, dict) else None,
            },
        }
        for item in raw_files
    ]
    inventory = source_inventory_from_tree(tree)
    if payload.get("documents") != SOURCE_DOCUMENT_COUNT:
        raise FineWebBuildError("source inventory document count does not match pin")
    return inventory


class SourceFileCache:
    """A one-file, resumable source cache rooted under the selected SHM path."""

    def __init__(
        self,
        root: Path,
        inventory: SourceInventory,
        *,
        timeout: float = 60.0,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.root = _ensure_private_directory(root, root=root.parent.parent)
        self.inventory = inventory
        self.timeout = timeout
        self.progress = progress
        self.active_path = self.root / "active.json"
        self.final_path = self.root / "current.parquet"
        self.part_path = self.root / "current.parquet.part"

    def get(self, index: int) -> Path:
        try:
            source = self.inventory.files[index]
        except IndexError as exc:
            raise FineWebBuildError(
                f"source file index is out of range: {index}"
            ) from exc
        self._activate(source)
        if self.final_path.is_file():
            try:
                _validate_regular_file(self.final_path)
                if self.final_path.stat().st_size != source.byte_count:
                    raise FineWebBuildError("cached source file has wrong length")
                if sha256_file(self.final_path) != source.sha256:
                    raise FineWebBuildError("cached source file has wrong SHA-256")
                return self.final_path
            except FineWebBuildError:
                _safe_unlink(self.final_path)
        self._download(source)
        return self.final_path

    def release(self, index: int) -> None:
        """Delete the fully consumed compressed Parquet file, never output shards."""

        source = self.inventory.files[index]
        active = self._read_active()
        if active and active.get("path") == source.path:
            _safe_unlink(self.final_path)
            _safe_unlink(self.part_path)
            _safe_unlink(self.active_path)
            _fsync_directory(self.root)

    def release_active(self) -> None:
        """Remove whichever single source file is active, including a partial."""

        self._read_active()
        _safe_unlink(self.final_path)
        _safe_unlink(self.part_path)
        _safe_unlink(self.active_path)
        _fsync_directory(self.root)

    def _activate(self, source: SourceFile) -> None:
        desired = {
            "path": source.path,
            "bytes": source.byte_count,
            "sha256": source.sha256,
        }
        active = self._read_active()
        if active == desired:
            return
        _safe_unlink(self.final_path)
        _safe_unlink(self.part_path)
        _atomic_write_json(self.active_path, desired)

    def _read_active(self) -> dict[str, Any] | None:
        if not self.active_path.exists():
            return None
        _validate_regular_file(self.active_path)
        try:
            value = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FineWebBuildError(f"invalid active source marker: {exc}") from exc
        if not isinstance(value, dict):
            raise FineWebBuildError("active source marker must be a JSON object")
        return value

    def _download(self, source: SourceFile) -> None:
        start = _regular_file_size(self.part_path) or 0
        if start == source.byte_count:
            if sha256_file(self.part_path) == source.sha256:
                os.replace(self.part_path, self.final_path)
                _fsync_directory(self.root)
                return
            _safe_unlink(self.part_path)
            start = 0
        if start > source.byte_count:
            _safe_unlink(self.part_path)
            start = 0
        repository = quote(self.inventory.repository, safe="/")
        remote_path = quote(source.path, safe="/")
        url = (
            f"https://huggingface.co/datasets/{repository}/resolve/"
            f"{self.inventory.revision}/{remote_path}?download=true"
        )
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": "gpt-tpu-speedrun-fineweb/1",
        }
        if start:
            headers["Range"] = f"bytes={start}-"
        request = Request(url, headers=headers)
        try:
            response = urlopen(request, timeout=self.timeout)
        except HTTPError as exc:
            raise FineWebBuildError(
                f"source download failed for {source.path}: HTTP {exc.code}"
            ) from exc
        except OSError as exc:
            raise FineWebBuildError(
                f"source download failed for {source.path}: {exc}"
            ) from exc
        with response:
            status = getattr(response, "status", response.getcode())
            if start and status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-"):
                    raise FineWebBuildError(
                        f"unexpected Content-Range for {source.path}: {content_range!r}"
                    )
                append = True
            elif status == 200:
                start = 0
                append = False
            else:
                raise FineWebBuildError(
                    f"source download failed for {source.path}: HTTP {status}"
                )
            completed = start
            try:
                with _open_regular(self.part_path, append=append) as handle:
                    if self.progress:
                        self.progress(source.path, completed, source.byte_count)
                    while True:
                        block = response.read(_IO_CHUNK_BYTES)
                        if not block:
                            break
                        completed += handle.write(block)
                        if completed > source.byte_count:
                            raise FineWebBuildError(
                                f"download exceeded expected size for {source.path}"
                            )
                        if self.progress:
                            self.progress(source.path, completed, source.byte_count)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FineWebBuildError:
                raise
            except Exception as exc:
                raise FineWebBuildError(
                    f"cannot cache source file {source.path}: {type(exc).__name__}"
                ) from exc
        if completed != source.byte_count:
            raise FineWebBuildError(
                f"short download for {source.path}: {completed:,}/{source.byte_count:,} bytes"
            )
        if sha256_file(self.part_path) != source.sha256:
            raise FineWebBuildError(f"SHA-256 mismatch for downloaded {source.path}")
        os.replace(self.part_path, self.final_path)
        _fsync_directory(self.root)


class ParquetDocumentSource:
    """Iterate one local Parquet file at a time from an exact row cursor."""

    def __init__(self, inventory: SourceInventory, cache: SourceFileCache) -> None:
        self.inventory = inventory
        self.cache = cache

    def initial_cursor(self) -> Mapping[str, int]:
        return {"file": 0, "row_group": 0, "row": 0}

    def iter_batches(
        self, cursor: Mapping[str, int], batch_rows: int
    ) -> Iterable[DocumentBatch]:
        try:
            import importlib.metadata
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - exercised by production script
            raise FineWebBuildError(
                "PyArrow is required; run scripts/prepare_fineweb.py with uv"
            ) from exc
        del pa
        actual_pyarrow = importlib.metadata.version("pyarrow")
        if actual_pyarrow != PYARROW_VERSION:
            raise FineWebBuildError(
                f"PyArrow {PYARROW_VERSION} is required for reproducibility; "
                f"found {actual_pyarrow}"
            )
        file_start, row_group_start, row_start = _validate_source_cursor(
            cursor, len(self.inventory.files)
        )
        for file_index in range(file_start, len(self.inventory.files)):
            local = self.cache.get(file_index)
            parquet = pq.ParquetFile(local, memory_map=False, pre_buffer=False)
            first_group = row_group_start if file_index == file_start else 0
            for row_group in range(first_group, parquet.metadata.num_row_groups):
                group_rows = parquet.metadata.row_group(row_group).num_rows
                first_row = (
                    row_start
                    if (file_index == file_start and row_group == first_group)
                    else 0
                )
                if first_row < 0 or first_row > group_rows:
                    raise FineWebBuildError(
                        f"checkpoint row {first_row} is invalid for source row group"
                    )
                offset = 0
                batches = parquet.iter_batches(
                    batch_size=batch_rows,
                    row_groups=[row_group],
                    columns=["text", "id", "url", "date"],
                    use_threads=True,
                )
                for record_batch in batches:
                    batch_end = offset + record_batch.num_rows
                    if batch_end <= first_row:
                        offset = batch_end
                        continue
                    crop = max(0, first_row - offset)
                    if crop:
                        record_batch = record_batch.slice(crop)
                        offset += crop
                    texts = record_batch.column(0).to_pylist()
                    identifiers = record_batch.column(1).to_pylist()
                    urls = record_batch.column(2).to_pylist()
                    source_dates = record_batch.column(3).to_pylist()
                    next_cursors: list[Mapping[str, int]] = []
                    for relative_index in range(len(texts)):
                        next_row = offset + relative_index + 1
                        if next_row < group_rows:
                            next_cursor = {
                                "file": file_index,
                                "row_group": row_group,
                                "row": next_row,
                            }
                        elif row_group + 1 < parquet.metadata.num_row_groups:
                            next_cursor = {
                                "file": file_index,
                                "row_group": row_group + 1,
                                "row": 0,
                            }
                        else:
                            next_cursor = {
                                "file": file_index + 1,
                                "row_group": 0,
                                "row": 0,
                            }
                        next_cursors.append(next_cursor)
                    if any(not isinstance(text, str) for text in texts):
                        raise FineWebBuildError(
                            "FineWeb source contains a non-string text"
                        )
                    if any(not isinstance(url, str) for url in urls):
                        raise FineWebBuildError(
                            "FineWeb source contains a non-string URL"
                        )
                    yield DocumentBatch(
                        tuple(texts),
                        tuple(next_cursors),
                        tuple(str(value) for value in identifiers),
                        tuple(urls),
                        tuple(
                            str(value) if value is not None else ""
                            for value in source_dates
                        ),
                    )
                    offset += record_batch.num_rows
                if offset != group_rows:
                    raise FineWebBuildError(
                        f"PyArrow yielded {offset:,}/{group_rows:,} rows from a row group"
                    )
            self.cache.release(file_index)
            row_group_start = 0
            row_start = 0


class TiktokenGPT2Encoder:
    name = TOKENIZER_NAME
    implementation = "tiktoken"
    version = TIKTOKEN_VERSION

    def __init__(self, threads: int) -> None:
        try:
            import importlib.metadata
            import tiktoken
        except ImportError as exc:  # pragma: no cover - exercised by production script
            raise FineWebBuildError(
                "tiktoken is required; run scripts/prepare_fineweb.py with uv"
            ) from exc
        actual = importlib.metadata.version("tiktoken")
        if actual != self.version:
            raise FineWebBuildError(
                f"tiktoken {self.version} is required for reproducibility; found {actual}"
            )
        self.encoding = tiktoken.get_encoding(TOKENIZER_NAME)
        if self.encoding.eot_token != EOT_TOKEN or self.encoding.n_vocab != VOCAB_SIZE:
            raise FineWebBuildError(
                "loaded tokenizer is not the 50,257-token GPT-2 encoding"
            )
        self.threads = threads

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        return self.encoding.encode_ordinary_batch(
            list(texts), num_threads=self.threads
        )


def probe_fineweb(
    source: DocumentSource,
    encoder: TokenEncoder,
    exclusions: ExclusionPolicy,
    *,
    examined_token_target: int = 1_000_000,
    batch_rows: int = 256,
    max_document_bytes: int = 16 * 1024**2,
) -> ProbeResult:
    """Tokenize a globally shuffled source sample and estimate cutoff retention."""

    exclusions.validate()
    if examined_token_target <= 0:
        raise FineWebBuildError("probe token target must be positive")
    cursor = dict(source.initial_cursor())
    examined_documents = accepted_documents = 0
    examined_tokens = accepted_tokens = 0
    examined_bytes = accepted_bytes = 0
    excluded_documents: dict[str, int] = {}
    excluded_tokens: dict[str, int] = {}
    for batch in source.iter_batches(cursor, batch_rows):
        text_bytes = [len(text.encode("utf-8")) for text in batch.texts]
        if any(count > max_document_bytes for count in text_bytes):
            largest = max(text_bytes)
            raise FineWebBuildError(
                f"probe document exceeds max_document_bytes: {largest:,} > "
                f"{max_document_bytes:,}"
            )
        encoded = list(encoder.encode_batch(batch.texts))
        if len(encoded) != len(batch.texts):
            raise FineWebBuildError("tokenizer returned the wrong probe batch length")
        for index, token_ids in enumerate(encoded):
            invalid = next((token for token in token_ids if token >= VOCAB_SIZE), None)
            if invalid is not None:
                raise FineWebBuildError(
                    f"tokenizer emitted invalid GPT-2 token {invalid}"
                )
            row_tokens = len(token_ids) + 1
            row_bytes = text_bytes[index]
            reason = exclusions.reason(
                batch.texts[index], batch.urls[index], batch.source_dates[index]
            )
            examined_documents += 1
            examined_tokens += row_tokens
            examined_bytes += row_bytes
            cursor = dict(batch.next_cursors[index])
            if reason is None:
                accepted_documents += 1
                accepted_tokens += row_tokens
                accepted_bytes += row_bytes
            else:
                excluded_documents[reason] = excluded_documents.get(reason, 0) + 1
                excluded_tokens[reason] = excluded_tokens.get(reason, 0) + row_tokens
            if examined_tokens >= examined_token_target:
                return ProbeResult(
                    examined_documents,
                    accepted_documents,
                    examined_tokens,
                    accepted_tokens,
                    examined_bytes,
                    accepted_bytes,
                    dict(sorted(excluded_documents.items())),
                    dict(sorted(excluded_tokens.items())),
                    cursor,
                )
    raise FineWebBuildError("FineWeb source ended before the probe token target")


def build_fineweb(
    config: BuildConfig,
    inventory: SourceInventory,
    exclusions: ExclusionPolicy,
    source: DocumentSource,
    encoder: TokenEncoder,
    *,
    through: str = "hero",
    stop_after_shards: int | None = None,
    preflight: bool = True,
    progress: Callable[[str], None] | None = None,
) -> BuildResult:
    """Build nested prefix variants, resuming at the latest atomic checkpoint."""

    config.validate()
    exclusions.validate()
    root = _select_root(config.root.expanduser())
    layout = _BuildLayout.create(root)
    variants = config.variants()
    variant_by_name = {variant.directory: variant for variant in variants}
    if through not in variant_by_name:
        raise FineWebBuildError(
            f"through must be one of {', '.join(variant_by_name)}; got {through!r}"
        )
    target = variant_by_name[through]
    core = _core_contract(config, inventory, exclusions, encoder)
    core_digest = canonical_json_sha256(core)
    state = _load_state(layout.checkpoint, core_digest, source.initial_cursor())
    _validate_completed_pool(layout, state, config.shard_tokens)
    if len(state.shards) > target.total_tokens // config.shard_tokens:
        raise FineWebBuildError(
            f"checkpoint already contains more data than requested through {through}"
        )
    if preflight:
        _preflight_capacity(layout, state, config, inventory, target.total_tokens)
    _write_plan_files(layout, config, inventory, exclusions, encoder, core, variants)
    _sync_variant_links(
        layout,
        variants,
        state.shards,
        config,
        inventory,
        exclusions,
        encoder,
        core,
    )

    if stop_after_shards is not None and stop_after_shards < 0:
        raise FineWebBuildError("stop_after_shards cannot be negative")
    started_with = len(state.shards)
    target_shards = target.total_tokens // config.shard_tokens
    batch_iterator = iter(source.iter_batches(state.cursor, config.batch_rows))
    pending = state.pending_tokens
    current_batch: DocumentBatch | None = None
    current_encoded: Sequence[Sequence[int] | None] = ()
    batch_index = 0

    while len(state.shards) < target_shards:
        if (
            stop_after_shards is not None
            and len(state.shards) - started_with >= stop_after_shards
        ):
            break
        ordinal = len(state.shards)
        split = "validation" if ordinal == 0 else "train"
        filename = _shard_filename(ordinal)
        writer = _AtomicShardWriter(layout.pool / filename, config.shard_tokens)
        while not writer.full:
            if pending:
                consumed = writer.write(pending)
                pending = array("H", pending[consumed:])
                continue
            if current_batch is None or batch_index >= len(current_batch.texts):
                try:
                    current_batch = next(batch_iterator)
                except StopIteration as exc:
                    writer.close()
                    raise FineWebBuildError(
                        "FineWeb source ended before the requested token target"
                    ) from exc
                reasons = [
                    exclusions.reason(text, url, source_date)
                    for text, url, source_date in zip(
                        current_batch.texts,
                        current_batch.urls,
                        current_batch.source_dates,
                    )
                ]
                accepted_texts = [
                    text
                    for text, reason in zip(current_batch.texts, reasons)
                    if reason is None
                ]
                for text in accepted_texts:
                    encoded_bytes = len(text.encode("utf-8"))
                    if encoded_bytes > config.max_document_bytes:
                        writer.close()
                        raise FineWebBuildError(
                            f"retained source document exceeds max_document_bytes: "
                            f"{encoded_bytes:,} > {config.max_document_bytes:,}"
                        )
                try:
                    encoded_values = list(encoder.encode_batch(accepted_texts))
                except BaseException:
                    writer.close()
                    raise
                if len(encoded_values) != len(accepted_texts):
                    writer.close()
                    raise FineWebBuildError("tokenizer returned the wrong batch length")
                accepted_encoded = iter(encoded_values)
                aligned: list[Sequence[int] | None] = []
                for reason in reasons:
                    aligned.append(
                        None if reason is not None else next(accepted_encoded)
                    )
                current_encoded = tuple(aligned)
                if len(current_encoded) != len(current_batch.texts):
                    writer.close()
                    raise FineWebBuildError("tokenizer returned the wrong batch length")
                batch_index = 0
            assert current_batch is not None
            text = current_batch.texts[batch_index]
            encoded = current_encoded[batch_index]
            state.cursor = dict(current_batch.next_cursors[batch_index])
            state.last_document_id = current_batch.document_ids[batch_index]
            if encoded is None:
                reason = exclusions.reason(
                    text,
                    current_batch.urls[batch_index],
                    current_batch.source_dates[batch_index],
                )
                assert reason is not None
                counts = state.exclusion_counts or {}
                counts[reason] = counts.get(reason, 0) + 1
                state.exclusion_counts = counts
                batch_index += 1
                continue
            document_tokens = array("H", [EOT_TOKEN])
            try:
                document_tokens.extend(encoded)
            except OverflowError as exc:
                writer.close()
                raise FineWebBuildError(
                    "tokenizer emitted a token outside uint16"
                ) from exc
            invalid = next(
                (token for token in document_tokens if token >= VOCAB_SIZE), None
            )
            if invalid is not None:
                writer.close()
                raise FineWebBuildError(
                    f"tokenizer emitted invalid GPT-2 token {invalid}"
                )
            state.documents_processed += 1
            state.source_utf8_bytes += len(text.encode("utf-8"))
            state.last_document_id = current_batch.document_ids[batch_index]
            batch_index += 1
            pending = document_tokens

        digest = writer.finish()
        if ordinal == 0 and pending:
            state.validation_boundary_discarded_tokens = len(pending)
            state.validation_boundary_document_sha256 = hashlib.sha256(
                state.last_document_id.encode("utf-8")
            ).hexdigest()
            pending = array("H")
        record = ShardRecord(
            filename,
            split,
            config.shard_tokens,
            HEADER_BYTES + TOKEN_BYTES * config.shard_tokens,
            digest,
            state.documents_processed,
            state.source_utf8_bytes,
            dict(state.exclusion_counts or {}),
            state.validation_boundary_discarded_tokens,
            state.validation_boundary_document_sha256,
        )
        new_state = BuildState(
            core_digest=state.core_digest,
            cursor=dict(state.cursor),
            pending_tokens=pending,
            shards=[*state.shards, record],
            documents_processed=state.documents_processed,
            source_utf8_bytes=state.source_utf8_bytes,
            last_document_id=state.last_document_id,
            exclusion_counts=dict(state.exclusion_counts or {}),
            validation_boundary_discarded_tokens=(
                state.validation_boundary_discarded_tokens
            ),
            validation_boundary_document_sha256=(
                state.validation_boundary_document_sha256
            ),
        )
        _commit_shard_and_checkpoint(layout, writer, record, new_state)
        state = new_state
        _sync_variant_links(
            layout,
            variants,
            state.shards,
            config,
            inventory,
            exclusions,
            encoder,
            core,
        )
        if progress:
            progress(
                f"completed {filename}: {len(state.shards) * config.shard_tokens:,}/"
                f"{target.total_tokens:,} tokens"
            )

    completed_variants = tuple(
        variant.directory
        for variant in variants
        if len(state.shards) >= variant.total_tokens // config.shard_tokens
    )
    return BuildResult(
        root=root,
        completed_shards=len(state.shards),
        completed_tokens=len(state.shards) * config.shard_tokens,
        stopped_at=through,
        complete_variants=completed_variants,
    )


@dataclass(frozen=True)
class _BuildLayout:
    root: Path
    work: Path
    pool: Path
    checkpoint: Path
    checkpoint_next: Path
    inventory: Path

    @classmethod
    def create(cls, root: Path) -> _BuildLayout:
        work = _ensure_private_directory(root / ".fineweb-build", root=root)
        pool = _ensure_private_directory(work / "shards", root=root)
        return cls(
            root=root,
            work=work,
            pool=pool,
            checkpoint=work / "checkpoint.json",
            checkpoint_next=work / "checkpoint.next.json",
            inventory=work / "source.json",
        )


class _AtomicShardWriter:
    def __init__(self, target: Path, token_count: int) -> None:
        self.target = target
        self.part = target.with_name(target.name + ".part")
        self.token_count = token_count
        self.written = 0
        self.handle = _open_regular(self.part, append=False)
        header = [0] * HEADER_INTS
        header[0] = MAGIC
        header[1] = FORMAT_VERSION
        header[2] = token_count
        raw_header = _HEADER.pack(*header)
        self.handle.write(raw_header)
        self.digest = hashlib.sha256(raw_header)
        self.finished = False

    @property
    def full(self) -> bool:
        return self.written == self.token_count

    def write(self, tokens: Sequence[int]) -> int:
        if self.finished:
            raise FineWebBuildError("cannot write a finished shard")
        count = min(len(tokens), self.token_count - self.written)
        if not count:
            return 0
        chunk = array("H", tokens[:count])
        if sys.byteorder != "little":
            chunk.byteswap()
        raw = chunk.tobytes()
        try:
            self.handle.write(raw)
        except BaseException:
            self.close()
            raise
        self.digest.update(raw)
        self.written += count
        return count

    def finish(self) -> str:
        if self.finished:
            return self.digest.hexdigest()
        if not self.full:
            self.close()
            raise FineWebBuildError(
                f"cannot finish short shard: {self.written:,}/{self.token_count:,} tokens"
            )
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
        except BaseException:
            self.close()
            raise
        self.finished = True
        expected = HEADER_BYTES + TOKEN_BYTES * self.token_count
        if self.part.stat().st_size != expected:
            raise FineWebBuildError("completed shard part has the wrong byte length")
        return self.digest.hexdigest()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _core_contract(
    config: BuildConfig,
    inventory: SourceInventory,
    exclusions: ExclusionPolicy,
    encoder: TokenEncoder,
) -> dict[str, Any]:
    module_path = Path(__file__).resolve()
    script_path = module_path.parent.parent / "scripts" / "prepare_fineweb.py"
    return {
        "builder_version": BUILDER_VERSION,
        "builder_module_sha256": sha256_file(module_path),
        "entrypoint_sha256": sha256_file(script_path),
        "pyarrow_version": PYARROW_VERSION,
        "source_inventory_sha256": inventory.digest,
        "source_order": "upstream-global-shuffle-order",
        "exclusion_policy_sha256": exclusions.digest,
        "exclusion_policy": exclusions.as_dict(),
        "tokenizer": {
            "name": encoder.name,
            "implementation": encoder.implementation,
            "version": encoder.version,
            "document_prefix_token": EOT_TOKEN,
            "vocab_size": VOCAB_SIZE,
        },
        "format": {
            "name": "llm.c-gpt2-v1",
            "header_bytes": HEADER_BYTES,
            "magic": MAGIC,
            "version": FORMAT_VERSION,
            "token_dtype": "little-endian uint16",
        },
        "shard_tokens": config.shard_tokens,
        "validation_tokens": config.validation_tokens,
        "max_document_bytes": config.max_document_bytes,
        "split_policy": (
            "first validation_tokens form validation; discard the remainder of its "
            "boundary document; subsequent whole-document stream forms train"
        ),
    }


def _write_plan_files(
    layout: _BuildLayout,
    config: BuildConfig,
    inventory: SourceInventory,
    exclusions: ExclusionPolicy,
    encoder: TokenEncoder,
    core: Mapping[str, Any],
    variants: Sequence[Variant],
) -> None:
    if layout.inventory.is_file():
        existing = source_inventory_from_dict(
            json.loads(layout.inventory.read_text(encoding="utf-8"))
        )
        if existing.digest != inventory.digest:
            raise FineWebBuildError(
                "saved source inventory differs from requested inventory"
            )
    else:
        _atomic_write_json(layout.inventory, inventory.as_dict())
    _atomic_write_json(layout.work / "exclusions.json", exclusions.as_dict())
    plan = {
        "schema_version": 1,
        "core": dict(core),
        "core_sha256": canonical_json_sha256(core),
        "variants": [
            {
                "directory": variant.directory,
                "total_tokens": variant.total_tokens,
                "validation_tokens": config.validation_tokens,
                "training_tokens": variant.total_tokens - config.validation_tokens,
                "shards": variant.total_tokens // config.shard_tokens,
            }
            for variant in variants
        ],
        "resource_policy": {
            "root": str(layout.root),
            "cache_root": str(layout.root / ".cache"),
            "one_source_parquet_at_a_time": True,
            "source_file_deleted_after_consumption": True,
            "reserve_bytes": config.reserve_bytes,
            "batch_rows": config.batch_rows,
            "tokenizer_threads": config.tokenizer_threads,
        },
    }
    _atomic_write_json(layout.work / "plan.json", plan)
    for variant in variants:
        destination = _ensure_private_directory(
            layout.root / variant.directory, root=layout.root
        )
        variant_plan = {
            "schema_version": 1,
            "status": "manifest.json appears only after this prefix is complete",
            "directory": variant.directory,
            "total_tokens": variant.total_tokens,
            "validation_tokens": config.validation_tokens,
            "training_tokens": variant.total_tokens - config.validation_tokens,
            "source_inventory_sha256": inventory.digest,
            "exclusion_policy_sha256": exclusions.digest,
            "core_sha256": canonical_json_sha256(core),
        }
        _atomic_write_json(destination / "BUILD_PLAN.json", variant_plan)


def _load_state(
    path: Path, core_digest: str, initial_cursor: Mapping[str, int]
) -> BuildState:
    if not path.is_file():
        return BuildState(core_digest, dict(initial_cursor), array("H"), [])
    _validate_regular_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FineWebBuildError(f"cannot load build checkpoint: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise FineWebBuildError("unsupported or malformed build checkpoint")
    if payload.get("core_sha256") != core_digest:
        raise FineWebBuildError(
            "build settings/source/tokenizer differ from the existing checkpoint"
        )
    cursor = payload.get("source_cursor")
    if not isinstance(cursor, dict) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in cursor.values()
    ):
        raise FineWebBuildError("checkpoint has an invalid source cursor")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list):
        raise FineWebBuildError("checkpoint shard ledger must be a list")
    shards: list[ShardRecord] = []
    for ordinal, item in enumerate(raw_shards):
        if not isinstance(item, dict):
            raise FineWebBuildError("checkpoint contains a malformed shard record")
        provenance = item.get("prefix_provenance", {})
        if not isinstance(provenance, dict):
            raise FineWebBuildError("checkpoint shard provenance is malformed")
        excluded = provenance.get("excluded_documents", {})
        _validate_counter_mapping(excluded, "checkpoint prefix exclusions")
        boundary = provenance.get("validation_boundary", {})
        if not isinstance(boundary, dict):
            raise FineWebBuildError("checkpoint validation boundary is malformed")
        discarded_tokens = _nonnegative_int(
            boundary.get("discarded_tokens", 0),
            "checkpoint boundary discarded tokens",
        )
        boundary_document = str(boundary.get("document_id_sha256", ""))
        if boundary_document and not _is_sha256(boundary_document):
            raise FineWebBuildError("checkpoint boundary document hash is invalid")
        record = ShardRecord(
            path=str(item.get("path")),
            split=str(item.get("split")),
            tokens=_positive_int(item.get("tokens"), "checkpoint shard tokens"),
            byte_count=_positive_int(item.get("bytes"), "checkpoint shard bytes"),
            sha256=str(item.get("sha256")),
            documents_processed=_nonnegative_int(
                provenance.get("documents_processed", 0),
                "checkpoint prefix documents",
            ),
            source_utf8_bytes=_nonnegative_int(
                provenance.get("source_utf8_bytes", 0),
                "checkpoint prefix source bytes",
            ),
            exclusion_counts={str(key): int(value) for key, value in excluded.items()},
            validation_boundary_discarded_tokens=discarded_tokens,
            validation_boundary_document_sha256=boundary_document,
        )
        if (
            record.path != _shard_filename(ordinal)
            or record.split != ("validation" if ordinal == 0 else "train")
            or not _is_sha256(record.sha256)
        ):
            raise FineWebBuildError("checkpoint shard ledger is not canonical")
        shards.append(record)
    pending = _decode_pending(payload.get("pending"))
    exclusion_counts = payload.get("exclusion_counts", {})
    _validate_counter_mapping(exclusion_counts, "checkpoint exclusions")
    state_boundary_hash = str(payload.get("validation_boundary_document_sha256", ""))
    if state_boundary_hash and not _is_sha256(state_boundary_hash):
        raise FineWebBuildError("checkpoint boundary document hash is invalid")
    return BuildState(
        core_digest=core_digest,
        cursor={str(key): int(value) for key, value in cursor.items()},
        pending_tokens=pending,
        shards=shards,
        documents_processed=_nonnegative_int(
            payload.get("documents_processed"), "documents_processed"
        ),
        source_utf8_bytes=_nonnegative_int(
            payload.get("source_utf8_bytes"), "source_utf8_bytes"
        ),
        last_document_id=str(payload.get("last_document_id", "")),
        exclusion_counts={
            str(key): int(value) for key, value in exclusion_counts.items()
        },
        validation_boundary_discarded_tokens=_nonnegative_int(
            payload.get("validation_boundary_discarded_tokens", 0),
            "checkpoint boundary discarded tokens",
        ),
        validation_boundary_document_sha256=state_boundary_hash,
    )


def _state_payload(state: BuildState) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "core_sha256": state.core_digest,
        "source_cursor": state.cursor,
        "pending": _encode_pending(state.pending_tokens),
        "shards": [record.checkpoint_dict() for record in state.shards],
        "documents_processed": state.documents_processed,
        "source_utf8_bytes": state.source_utf8_bytes,
        "last_document_id": state.last_document_id,
        "exclusion_counts": dict(sorted((state.exclusion_counts or {}).items())),
        "validation_boundary_discarded_tokens": (
            state.validation_boundary_discarded_tokens
        ),
        "validation_boundary_document_sha256": (
            state.validation_boundary_document_sha256
        ),
    }


def _encode_pending(tokens: array) -> dict[str, Any]:
    copy = array("H", tokens)
    if sys.byteorder != "little":
        copy.byteswap()
    raw = copy.tobytes()
    return {
        "encoding": "base64-zlib-le-uint16-v1",
        "tokens": len(tokens),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(zlib.compress(raw, level=6)).decode("ascii"),
    }


def _decode_pending(payload: Any) -> array:
    if (
        not isinstance(payload, dict)
        or payload.get("encoding") != "base64-zlib-le-uint16-v1"
    ):
        raise FineWebBuildError("checkpoint pending-token encoding is invalid")
    count = _nonnegative_int(payload.get("tokens"), "pending tokens")
    digest = payload.get("sha256")
    encoded = payload.get("data")
    if not _is_sha256(digest) or not isinstance(encoded, str):
        raise FineWebBuildError("checkpoint pending-token metadata is invalid")
    try:
        raw = zlib.decompress(base64.b64decode(encoded, validate=True))
    except (ValueError, zlib.error) as exc:
        raise FineWebBuildError("checkpoint pending tokens are corrupt") from exc
    if len(raw) != TOKEN_BYTES * count or hashlib.sha256(raw).hexdigest() != digest:
        raise FineWebBuildError("checkpoint pending-token length/hash mismatch")
    tokens = array("H")
    tokens.frombytes(raw)
    if sys.byteorder != "little":
        tokens.byteswap()
    if any(token >= VOCAB_SIZE for token in tokens):
        raise FineWebBuildError("checkpoint contains an invalid GPT-2 token")
    return tokens


def _commit_shard_and_checkpoint(
    layout: _BuildLayout,
    writer: _AtomicShardWriter,
    record: ShardRecord,
    state: BuildState,
) -> None:
    _write_json_file(layout.checkpoint_next, _state_payload(state))
    target = writer.target
    if target.exists():
        _validate_regular_file(target)
        validate_shard(
            target,
            expected_tokens=record.tokens,
            expected_bytes=record.byte_count,
            expected_sha256=record.sha256,
        )
        _safe_unlink(writer.part)
    else:
        os.replace(writer.part, target)
        _fsync_directory(target.parent)
    os.replace(layout.checkpoint_next, layout.checkpoint)
    _fsync_directory(layout.checkpoint.parent)


def _validate_completed_pool(
    layout: _BuildLayout, state: BuildState, shard_tokens: int
) -> None:
    for ordinal, record in enumerate(state.shards):
        target = layout.pool / _shard_filename(ordinal)
        validate_shard(
            target,
            expected_tokens=shard_tokens,
            expected_bytes=record.byte_count,
            expected_sha256=record.sha256,
        )
    next_final = layout.pool / _shard_filename(len(state.shards))
    later_final = layout.pool / _shard_filename(len(state.shards) + 1)
    if later_final.exists():
        raise FineWebBuildError(
            "shard pool is more than one shard ahead of its checkpoint"
        )
    if next_final.is_symlink():
        raise FineWebBuildError(f"refusing symlink in shard pool: {next_final}")


def _sync_variant_links(
    layout: _BuildLayout,
    variants: Sequence[Variant],
    shards: Sequence[ShardRecord],
    config: BuildConfig,
    inventory: SourceInventory,
    exclusions: ExclusionPolicy,
    encoder: TokenEncoder,
    core: Mapping[str, Any],
) -> None:
    for variant in variants:
        required = variant.total_tokens // config.shard_tokens
        destination = _ensure_private_directory(
            layout.root / variant.directory, root=layout.root
        )
        for record in shards[:required]:
            source = layout.pool / record.path
            target = destination / record.path
            _ensure_hard_link(source, target, record)
        if len(shards) >= required:
            _atomic_write_json(destination / "source.json", inventory.as_dict())
            _atomic_write_json(destination / "exclusions.json", exclusions.as_dict())
            manifest = _variant_manifest(
                variant,
                shards[:required],
                config,
                inventory,
                exclusions,
                encoder,
                shards[required - 1].exclusion_counts or {},
                core,
            )
            _atomic_write_json(destination / "manifest.json", manifest)


def _variant_manifest(
    variant: Variant,
    shards: Sequence[ShardRecord],
    config: BuildConfig,
    inventory: SourceInventory,
    exclusions: ExclusionPolicy,
    encoder: TokenEncoder,
    exclusion_counts: Mapping[str, int],
    core: Mapping[str, Any],
) -> dict[str, Any]:
    train_count = len(shards) - 1
    label = f"{variant.total_tokens // 1_000_000_000}b"
    return {
        "schema_version": 1,
        "name": f"fineweb-{label}-gpt2",
        "description": (
            f"Nested {variant.total_tokens:,}-token prefix of the pinned globally "
            "shuffled FineWeb 100BT stream."
        ),
        "license": {
            "dataset": "ODC-By-1.0",
            "url": "https://opendatacommons.org/licenses/by/1-0/",
            "code": "Apache-2.0",
            "notes": (
                "Prepared tokens remain subject to FineWeb/Common Crawl source terms; "
                "the repository code license does not replace corpus terms."
            ),
        },
        "format": {
            "name": "llm.c-gpt2-v1",
            "header_bytes": HEADER_BYTES,
            "header_dtype": "little-endian int32",
            "magic": MAGIC,
            "version": FORMAT_VERSION,
            "token_dtype": "little-endian uint16",
        },
        "source": {
            "dataset": inventory.repository,
            "revision": inventory.revision,
            "global_shuffle_seed": inventory.global_shuffle_seed,
            "global_shuffle_provenance": "upstream dataset card claim",
            "inventory_path": "source.json",
            "inventory_sha256": inventory.digest,
            "selection": f"first {variant.total_tokens:,} prepared tokens",
            "source_dataset_card": (
                "https://huggingface.co/datasets/HuggingFaceFW/fineweb_100BT-shuffled"
            ),
            "source_date_before": exclusions.source_date_before,
            "exclusion_policy_path": "exclusions.json",
            "exclusion_policy_sha256": exclusions.digest,
            "excluded_documents_at_prefix_end": dict(sorted(exclusion_counts.items())),
        },
        "tokenizer": {
            "name": encoder.name,
            "implementation": encoder.implementation,
            "implementation_version": encoder.version,
            "document_prefix_token": EOT_TOKEN,
            "vocab_size": VOCAB_SIZE,
        },
        "preparation": {
            "builder": "gpt-tpu-speedrun fineweb builder",
            "builder_version": BUILDER_VERSION,
            "builder_module_sha256": core["builder_module_sha256"],
            "entrypoint_sha256": core["entrypoint_sha256"],
            "pyarrow_version": core["pyarrow_version"],
            "core_sha256": canonical_json_sha256(core),
            "shard_tokens": config.shard_tokens,
            "split_policy": (
                f"first {config.validation_tokens:,} tokens validation; discard the "
                "rest of its boundary document; remaining documents train"
            ),
            "validation_train_document_disjoint": True,
            "validation_boundary_discarded_tokens": (
                shards[0].validation_boundary_discarded_tokens
            ),
            "validation_boundary_document_id_sha256": (
                shards[0].validation_boundary_document_sha256
            ),
            "nested_prefix": True,
        },
        "default_train_shards": train_count,
        "validation_prefix_tokens": config.validation_tokens,
        "files": [record.as_dict() for record in shards],
    }


def _preflight_capacity(
    layout: _BuildLayout,
    state: BuildState,
    config: BuildConfig,
    inventory: SourceInventory,
    target_tokens: int,
) -> None:
    target_shards = target_tokens // config.shard_tokens
    remaining_shards = max(0, target_shards - len(state.shards))
    shard_bytes = HEADER_BYTES + TOKEN_BYTES * config.shard_tokens
    source_bytes = max(item.byte_count for item in inventory.files)
    active_part_bytes = shard_bytes
    required = (
        remaining_shards * shard_bytes
        + source_bytes
        + active_part_bytes
        + config.reserve_bytes
    )
    free = shutil.disk_usage(layout.root).free
    if free < required:
        raise FineWebBuildError(
            f"not enough free space under {layout.root}: need {required:,} bytes "
            f"({remaining_shards} output shards + largest source parquet + one "
            f"output part + {config.reserve_bytes:,} reserve), have {free:,}. "
            "Build through 8B first or lower --hero-tokens."
        )


def _ensure_hard_link(source: Path, target: Path, record: ShardRecord) -> None:
    if target.exists():
        _validate_regular_file(target)
        source_stat = source.stat()
        target_stat = target.stat()
        if (source_stat.st_dev, source_stat.st_ino) == (
            target_stat.st_dev,
            target_stat.st_ino,
        ):
            return
        validate_shard(
            target,
            expected_tokens=record.tokens,
            expected_bytes=record.byte_count,
            expected_sha256=record.sha256,
        )
        return
    temporary = target.with_name(target.name + ".link.part")
    _safe_unlink(temporary)
    try:
        os.link(source, temporary, follow_symlinks=False)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except OSError as exc:
        _safe_unlink(temporary)
        raise FineWebBuildError(
            f"cannot hard-link shared shard into {target.parent}: {exc}"
        ) from exc


def _shard_filename(ordinal: int) -> str:
    if ordinal < 0:
        raise FineWebBuildError("negative shard ordinal")
    if ordinal == 0:
        return "fineweb_val_000000.bin"
    return f"fineweb_train_{ordinal:06d}.bin"


def _validate_source_cursor(
    cursor: Mapping[str, int], file_count: int
) -> tuple[int, int, int]:
    expected = {"file", "row_group", "row"}
    if set(cursor) != expected:
        raise FineWebBuildError("source cursor must contain file, row_group, and row")
    values = tuple(cursor[key] for key in ("file", "row_group", "row"))
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise FineWebBuildError("source cursor values must be nonnegative integers")
    if values[0] > file_count:
        raise FineWebBuildError("source cursor file is beyond the inventory")
    if values[0] == file_count and values[1:] != (0, 0):
        raise FineWebBuildError("terminal source cursor must be normalized")
    return values


def _select_root(path: Path) -> Path:
    """Create and resolve the explicitly selected root; that root may be a symlink."""

    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise FineWebBuildError(
            f"cannot create/access build directory {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise FineWebBuildError(f"build path is not a directory: {resolved}")
    return resolved


def _ensure_private_directory(path: Path, *, root: Path) -> Path:
    """Create a child directory while rejecting every symlink below ``root``."""

    selected = root.resolve(strict=True)
    unresolved = Path(os.path.abspath(path))
    try:
        relative = unresolved.relative_to(selected)
    except ValueError as exc:
        raise FineWebBuildError(
            f"build directory escapes selected root {selected}: {path}"
        ) from exc
    current = selected
    try:
        for part in relative.parts:
            current = current / part
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                current.mkdir(mode=0o700)
                metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise FineWebBuildError(
                    f"refusing symlink below selected build root: {current}"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise FineWebBuildError(f"build path is not a directory: {current}")
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(selected)
    except FineWebBuildError:
        raise
    except (OSError, ValueError) as exc:
        raise FineWebBuildError(
            f"cannot create/access build directory {path}: {exc}"
        ) from exc
    return resolved


def _validate_regular_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FineWebBuildError(f"cannot inspect file {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise FineWebBuildError(f"refusing non-regular file: {path}")


def _regular_file_size(path: Path) -> int | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise FineWebBuildError(f"cannot inspect file {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise FineWebBuildError(f"refusing non-regular file: {path}")
    return metadata.st_size


def _open_regular(path: Path, *, append: bool) -> Any:
    _regular_file_size(path)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= os.O_APPEND if append else os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise FineWebBuildError(f"cannot open {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FineWebBuildError(f"refusing non-regular file: {path}")
        return os.fdopen(descriptor, "ab" if append else "wb")
    except Exception:
        os.close(descriptor)
        raise


def _safe_unlink(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise FineWebBuildError(f"cannot inspect temporary file {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise FineWebBuildError(
            f"refusing to replace non-regular temporary file: {path}"
        )
    try:
        path.unlink()
    except OSError as exc:
        raise FineWebBuildError(f"cannot remove temporary file {path}: {exc}") from exc


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    _write_json_file(temporary, payload)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(payload)
    try:
        with _open_regular(path, append=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FineWebBuildError(f"cannot write metadata {path}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FineWebBuildError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FineWebBuildError(f"{field} must be a nonnegative integer")
    return value


def _validate_counter_mapping(value: Any, field: str) -> None:
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for key, count in value.items()
    ):
        raise FineWebBuildError(f"{field} must map strings to nonnegative integers")


def _parse_source_date(value: str) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )
