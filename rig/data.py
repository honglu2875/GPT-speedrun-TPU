"""Deterministic data preparation for the GPT TPU rig.

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
from datetime import date
from pathlib import Path
import re
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
_HF_DATASET_REPOSITORY = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
_HF_REPOSITORY_MAX_LENGTH = 96

# Stable order used by manifests, evaluators, CSV output, and result records.
FRESH10_DOMAINS = (
    "science",
    "medicine",
    "software",
    "history",
    "fiction",
    "government",
    "legal",
    "economics",
    "climate",
    "education",
)
FRESH10_DOCUMENTS_PER_DOMAIN = 4
FRESH10_SCORED_TOKENS_PER_DOCUMENT = 2_048
FRESH10_CONTEXT_TOKENS_PER_DOCUMENT = 1
FRESH10_SCORED_TOKENS_PER_DOMAIN = (
    FRESH10_DOCUMENTS_PER_DOMAIN * FRESH10_SCORED_TOKENS_PER_DOCUMENT
)
FRESH10_TOKENS_PER_DOMAIN = FRESH10_SCORED_TOKENS_PER_DOMAIN + (
    FRESH10_DOCUMENTS_PER_DOMAIN * FRESH10_CONTEXT_TOKENS_PER_DOCUMENT
)

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


@dataclass(frozen=True)
class Fresh10Document:
    """One independently scored document span within a Fresh10 shard."""

    document_id: str
    title: str
    authors: tuple[str, ...]
    publisher: str
    source_url: str
    published_date: str
    retrieved_date: str
    license_name: str
    license_url: str
    extraction_notes: str
    raw_sha256: str
    text_path: str
    text_bytes: int
    text_sha256: str
    token_offset: int
    token_count: int
    score_offset: int
    scored_tokens: int


@dataclass(frozen=True)
class Fresh10Domain:
    """A verified domain shard plus exact document-boundary metadata."""

    name: str
    path: Path
    token_count: int
    scored_tokens: int
    sha256: str
    documents: tuple[Fresh10Document, ...]


@dataclass(frozen=True)
class PreparedFresh10:
    """Verified Fresh10 downstream-perplexity corpus."""

    name: str
    root: Path
    manifest_path: Path
    manifest_sha256: str
    domains: tuple[Fresh10Domain, ...]

    @property
    def scored_tokens(self) -> int:
        return sum(domain.scored_tokens for domain in self.domains)


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


def load_fresh10_manifest(
    manifest: str | os.PathLike[str] | Mapping[str, Any] = "fresh10",
) -> tuple[dict[str, Any], Path]:
    """Load and validate a Fresh10 manifest without touching shard data."""

    if isinstance(manifest, Mapping):
        payload = dict(manifest)
        source = Path("<in-memory-fresh10-manifest>")
    else:
        candidate = Path(manifest)
        source = candidate if candidate.is_file() else manifest_path(str(manifest))
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"cannot load Fresh10 manifest {source}: {exc}") from exc
    _validate_fresh10_manifest(payload, source)
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
        with tempfile.NamedTemporaryFile(prefix=".rig-write-check-", dir=root):
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


def verify_fresh10(
    root: str | os.PathLike[str],
    manifest: str | os.PathLike[str] | Mapping[str, Any] = "fresh10",
    *,
    verify_hash: bool = True,
) -> PreparedFresh10:
    """Verify all ten Fresh10 domain shards without network or mutation."""

    payload, source = load_fresh10_manifest(manifest)
    data_root = Path(root).expanduser().resolve(strict=True)
    if not data_root.is_dir():
        raise DataError(f"data path is not a directory: {data_root}")
    verified: list[tuple[Mapping[str, Any], ShardInfo]] = []
    for domain in payload["domains"]:
        target = _safe_target(data_root, str(domain["path"]))
        if not target.is_file():
            raise DataError(f"missing Fresh10 domain shard: {target}")
        info = _validate_entry(target, domain, verify_hash=verify_hash)
        _validate_fresh10_boundaries(target, domain)
        verified.append((domain, info))
    return _prepared_fresh10_result(payload, source, data_root, verified)


def prepare_fresh10(
    root: str | os.PathLike[str],
    manifest: str | os.PathLike[str] | Mapping[str, Any] = "fresh10",
    *,
    force: bool = False,
    offline: bool = False,
    progress: Progress | None = None,
    timeout: float = 60.0,
) -> PreparedFresh10:
    """Download and verify the immutable Fresh10 domain shards.

    Each domain is installed atomically from an adjacent ``.part`` file.  The
    source manifest is validated before the cache is mutated, and every final
    file is checked by header, exact length, and SHA-256.
    """

    payload, source = load_fresh10_manifest(manifest)
    entries = tuple(payload["domains"])
    data_root = ensure_data_root(root)
    missing_bytes = _required_download_bytes(data_root, entries, force=force)
    ensure_data_root(data_root, missing_bytes)

    verified: list[tuple[Mapping[str, Any], ShardInfo]] = []
    for domain in entries:
        target = _safe_target(data_root, str(domain["path"]))
        if target.is_file() and not force:
            try:
                info = _validate_entry(target, domain, verify_hash=True)
                _validate_fresh10_boundaries(target, domain)
            except DataError as exc:
                raise DataError(
                    f"existing Fresh10 shard is invalid: {target}; pass force=True "
                    "to replace it"
                ) from exc
            verified.append((domain, info))
            continue
        url = domain["url"]
        if offline:
            raise DataError(
                f"offline preparation cannot download missing Fresh10 shard {target}; "
                "copy it into the data root or run without offline=True"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        info = _download_entry(target, domain, timeout=timeout, progress=progress)
        _validate_fresh10_boundaries(target, domain)
        verified.append((domain, info))
    return _prepared_fresh10_result(payload, source, data_root, verified)


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
            or "\\" in relative
            or relative_path.as_posix() != relative
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


def _validate_fresh10_manifest(payload: Mapping[str, Any], source: Path) -> None:
    if payload.get("schema_version") != 1 or payload.get("kind") != "fresh10":
        raise DataError(f"{source}: unsupported or missing Fresh10 schema/kind")
    if not isinstance(payload.get("name"), str) or not payload["name"]:
        raise DataError(f"{source}: missing Fresh10 manifest name")

    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise DataError(f"{source}: missing Fresh10 tokenizer metadata")
    if (
        tokenizer.get("name") != "gpt2"
        or tokenizer.get("vocab_size") != 50_257
        or tokenizer.get("eot_token") != 50_256
        or tokenizer.get("document_prefix_token") != 50_256
    ):
        raise DataError(
            f"{source}: Fresh10 requires GPT-2 tokens, vocab_size 50257, and "
            "EOT/document_prefix_token 50256"
        )
    format_info = payload.get("format")
    if not isinstance(format_info, dict) or format_info != {
        "name": "llm.c-gpt2-v1",
        "header_bytes": HEADER_BYTES,
        "magic": MAGIC,
        "version": FORMAT_VERSION,
        "token_dtype": "little-endian uint16",
    }:
        raise DataError(f"{source}: Fresh10 requires the llm.c GPT-2 v1 shard format")

    not_before_raw = payload.get("publication_not_before")
    not_before = _manifest_date(not_before_raw, source, "publication_not_before")
    prepared = payload.get("prepared_source")
    if not isinstance(prepared, dict):
        raise DataError(f"{source}: missing prepared_source metadata")
    repository = prepared.get("repository")
    revision = prepared.get("revision")
    if (
        not isinstance(repository, str)
        or len(repository) > _HF_REPOSITORY_MAX_LENGTH
        or _HF_DATASET_REPOSITORY.fullmatch(repository) is None
        or not isinstance(revision, str)
        or len(revision) not in {40, 64}
        or any(character not in "0123456789abcdefABCDEF" for character in revision)
    ):
        raise DataError(f"{source}: invalid prepared repository or immutable revision")

    domains = payload.get("domains")
    if not isinstance(domains, list) or len(domains) != len(FRESH10_DOMAINS):
        raise DataError(
            f"{source}: Fresh10 needs exactly {len(FRESH10_DOMAINS)} domains"
        )
    actual_names = tuple(
        domain.get("name") if isinstance(domain, dict) else None for domain in domains
    )
    if actual_names != FRESH10_DOMAINS:
        raise DataError(
            f"{source}: Fresh10 domain order must be {', '.join(FRESH10_DOMAINS)}"
        )

    seen_paths: set[str] = set()
    seen_documents: set[str] = set()
    expected_url_prefix = (
        f"https://huggingface.co/datasets/{repository}/resolve/{revision}/"
    )
    for domain in domains:
        name = str(domain["name"])
        relative = domain.get("path")
        _validate_relative_shard_path(relative, source)
        assert isinstance(relative, str)  # narrowed by validation helper
        if relative in seen_paths:
            raise DataError(f"{source}: duplicate Fresh10 shard path {relative!r}")
        seen_paths.add(relative)
        if Path(relative).suffix.lower() != ".bin":
            raise DataError(f"{source}: Fresh10 shard must use a .bin path: {relative!r}")

        tokens = domain.get("tokens")
        scored_tokens = domain.get("scored_tokens")
        if tokens != FRESH10_TOKENS_PER_DOMAIN:
            raise DataError(
                f"{source}: {name} must contain exactly "
                f"{FRESH10_TOKENS_PER_DOMAIN:,} stored tokens"
            )
        if scored_tokens != FRESH10_SCORED_TOKENS_PER_DOMAIN:
            raise DataError(
                f"{source}: {name} must declare exactly "
                f"{FRESH10_SCORED_TOKENS_PER_DOMAIN:,} scored tokens"
            )
        if domain.get("bytes") != HEADER_BYTES + tokens * TOKEN_BYTES:
            raise DataError(f"{source}: inconsistent byte count for {relative!r}")
        _manifest_hash(domain.get("sha256"), source, f"{relative} SHA-256")
        url = domain.get("url")
        if not isinstance(url, str) or url != expected_url_prefix + relative:
            raise DataError(
                f"{source}: {relative!r} URL must pin prepared revision {revision}"
            )

        documents = domain.get("documents")
        if not isinstance(documents, list) or len(documents) != FRESH10_DOCUMENTS_PER_DOMAIN:
            raise DataError(
                f"{source}: {name} needs exactly "
                f"{FRESH10_DOCUMENTS_PER_DOMAIN} documents"
            )
        cursor = 0
        scored_total = 0
        for document in documents:
            if not isinstance(document, dict):
                raise DataError(f"{source}: every {name} document must be an object")
            document_id = document.get("id")
            if (
                not isinstance(document_id, str)
                or not document_id
                or document_id in seen_documents
            ):
                raise DataError(f"{source}: invalid or duplicate document id {document_id!r}")
            seen_documents.add(document_id)
            for field in ("title", "publisher"):
                if not isinstance(document.get(field), str) or not document[field].strip():
                    raise DataError(f"{source}: {document_id} has invalid {field}")
            authors = document.get("authors")
            if (
                not isinstance(authors, list)
                or not authors
                or any(
                    not isinstance(author, str) or not author.strip()
                    for author in authors
                )
                or len(set(authors)) != len(authors)
            ):
                raise DataError(
                    f"{source}: {document_id} needs unique, nonempty authors"
                )
            source_url = document.get("source_url")
            if not isinstance(source_url, str) or not source_url.startswith("https://"):
                raise DataError(f"{source}: {document_id} needs an HTTPS source_url")
            published = _manifest_date(
                document.get("published_date"), source, f"{document_id} published_date"
            )
            retrieved = _manifest_date(
                document.get("retrieved_date"), source, f"{document_id} retrieved_date"
            )
            if published < not_before or published > retrieved:
                raise DataError(
                    f"{source}: {document_id} publication date is outside the declared "
                    "freshness/retrieval interval"
                )
            license_info = document.get("license")
            if not isinstance(license_info, dict):
                raise DataError(f"{source}: {document_id} needs license metadata")
            if not isinstance(license_info.get("name"), str) or not license_info["name"]:
                raise DataError(f"{source}: {document_id} has invalid license name")
            license_url = license_info.get("url")
            if not isinstance(license_url, str) or not license_url.startswith("https://"):
                raise DataError(f"{source}: {document_id} needs an HTTPS license URL")
            if (
                not isinstance(document.get("extraction_notes"), str)
                or not document["extraction_notes"].strip()
            ):
                raise DataError(f"{source}: {document_id} needs extraction notes")
            _manifest_hash(
                document.get("raw_sha256"), source, f"{document_id} raw_sha256"
            )
            text_path = document.get("text_path")
            _validate_relative_shard_path(text_path, source)
            assert isinstance(text_path, str)
            if Path(text_path).suffix.lower() != ".txt":
                raise DataError(
                    f"{source}: {document_id} canonical text must use a .txt path"
                )
            if text_path in seen_paths:
                raise DataError(f"{source}: duplicate Fresh10 path {text_path!r}")
            seen_paths.add(text_path)
            if (
                isinstance(document.get("text_bytes"), bool)
                or not isinstance(document.get("text_bytes"), int)
                or document["text_bytes"] <= 0
            ):
                raise DataError(f"{source}: {document_id} has invalid text_bytes")
            _manifest_hash(
                document.get("text_sha256"), source, f"{document_id} text_sha256"
            )

            if (
                document.get("token_offset") != cursor
                or document.get("token_count")
                != FRESH10_SCORED_TOKENS_PER_DOCUMENT
                + FRESH10_CONTEXT_TOKENS_PER_DOCUMENT
                or document.get("score_offset")
                != cursor + FRESH10_CONTEXT_TOKENS_PER_DOCUMENT
                or document.get("scored_tokens")
                != FRESH10_SCORED_TOKENS_PER_DOCUMENT
            ):
                raise DataError(f"{source}: invalid fixed token span for {document_id}")
            cursor += int(document["token_count"])
            scored_total += int(document["scored_tokens"])
        if cursor != tokens or scored_total != scored_tokens:
            raise DataError(f"{source}: document spans do not cover {name} exactly")


def _validate_relative_shard_path(relative: Any, source: Path) -> None:
    relative_path = Path(relative) if isinstance(relative, str) else None
    if (
        not isinstance(relative, str)
        or not relative
        or relative_path == Path(".")
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or any(part in {"", "."} for part in relative_path.parts)
        or "\\" in relative
        or relative_path.as_posix() != relative
    ):
        raise DataError(f"{source}: unsafe shard path {relative!r}")


def _manifest_hash(value: Any, source: Path, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise DataError(f"{source}: invalid {field}")


def _manifest_date(value: Any, source: Path, field: str) -> date:
    if not isinstance(value, str):
        raise DataError(f"{source}: invalid {field}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DataError(f"{source}: invalid {field}") from exc


def _safe_target(root: Path, relative: str) -> Path:
    # Resolve the caller-selected root first so an intentional cache-root link
    # (for example ``shm/ -> /dev/shm``) remains supported.  Links below that
    # boundary are never part of the cache contract: following one could alias
    # two manifest entries or let ``force`` replace an unrelated file.
    root = root.resolve(strict=True)
    unresolved = root / relative
    current = root
    for part in Path(relative).parts:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise DataError(f"cannot inspect shard path {current}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise DataError(f"refusing symlink below data root: {current}")
    target = unresolved.resolve(strict=False)
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


def _validate_fresh10_boundaries(
    path: Path, domain: Mapping[str, Any]
) -> None:
    """Prove every token is GPT-2-valid and each document starts at EOT."""

    try:
        with path.open("rb") as handle:
            handle.seek(HEADER_BYTES)
            tokens = array("H")
            tokens.frombytes(handle.read())
            if sys.byteorder != "little":
                tokens.byteswap()
            invalid = next((token for token in tokens if token > 50_256), None)
            if invalid is not None:
                raise DataError(
                    f"{path}: token id {invalid} exceeds the GPT-2 vocabulary maximum 50256"
                )
            for document in domain["documents"]:
                token_offset = int(document["token_offset"])
                if token_offset >= len(tokens):
                    raise DataError(
                        f"{path}: truncated document boundary for {document['id']}"
                    )
                token = tokens[token_offset]
                if token != 50_256:
                    raise DataError(
                        f"{path}: document {document['id']} starts with token {token}; "
                        "expected GPT-2 EOT token 50256"
                    )
    except DataError:
        raise
    except OSError as exc:
        raise DataError(f"cannot inspect Fresh10 boundaries in {path}: {exc}") from exc


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


def _prepared_fresh10_result(
    payload: Mapping[str, Any],
    source: Path,
    root: Path,
    verified: Sequence[tuple[Mapping[str, Any], ShardInfo]],
) -> PreparedFresh10:
    domains: list[Fresh10Domain] = []
    for domain, info in verified:
        documents = tuple(
            Fresh10Document(
                document_id=str(document["id"]),
                title=str(document["title"]),
                authors=tuple(str(author) for author in document["authors"]),
                publisher=str(document["publisher"]),
                source_url=str(document["source_url"]),
                published_date=str(document["published_date"]),
                retrieved_date=str(document["retrieved_date"]),
                license_name=str(document["license"]["name"]),
                license_url=str(document["license"]["url"]),
                extraction_notes=str(document["extraction_notes"]),
                raw_sha256=str(document["raw_sha256"]).lower(),
                text_path=str(document["text_path"]),
                text_bytes=int(document["text_bytes"]),
                text_sha256=str(document["text_sha256"]).lower(),
                token_offset=int(document["token_offset"]),
                token_count=int(document["token_count"]),
                score_offset=int(document["score_offset"]),
                scored_tokens=int(document["scored_tokens"]),
            )
            for document in domain["documents"]
        )
        domains.append(
            Fresh10Domain(
                name=str(domain["name"]),
                path=info.path,
                token_count=info.token_count,
                scored_tokens=int(domain["scored_tokens"]),
                sha256=str(domain["sha256"]).lower(),
                documents=documents,
            )
        )
    return PreparedFresh10(
        name=str(payload["name"]),
        root=root,
        manifest_path=source.resolve() if source.exists() else source,
        manifest_sha256=manifest_digest(payload),
        domains=tuple(domains),
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

    headers = {"User-Agent": "gpt-rig-tpu-data/1"}
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
    "FRESH10_CONTEXT_TOKENS_PER_DOCUMENT",
    "FRESH10_DOCUMENTS_PER_DOMAIN",
    "FRESH10_DOMAINS",
    "FRESH10_SCORED_TOKENS_PER_DOCUMENT",
    "FRESH10_SCORED_TOKENS_PER_DOMAIN",
    "FRESH10_TOKENS_PER_DOMAIN",
    "FORMAT_VERSION",
    "HEADER_BYTES",
    "MAGIC",
    "PreparedDataset",
    "PreparedFresh10",
    "Fresh10Document",
    "Fresh10Domain",
    "ShardHeader",
    "ShardInfo",
    "ensure_data_root",
    "load_manifest",
    "load_fresh10_manifest",
    "manifest_digest",
    "manifest_path",
    "prepare_dataset",
    "prepare_fresh10",
    "prepare",
    "prepare_smoke",
    "read_header",
    "selected_files",
    "sha256_file",
    "validate_shard",
    "verify_dataset",
    "verify_fresh10",
]
