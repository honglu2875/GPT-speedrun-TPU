# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "huggingface-hub==0.34.4",
#   "numpy==2.5.2",
#   "pyyaml==6.0.3",
# ]
# ///
"""Build, verify, and publish the v4 IsoFLOP evidence as a closed archive.

The verifier is deliberately contained in this file.  A byte-for-byte copy is
placed in every archive as ``verify.py``; it loads the experiment implementation
from the deterministic Git source archive rather than trusting an installed
checkout.  Publication is one-way: no credential is read until a complete local
archive has passed both byte and semantic verification.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from types import ModuleType
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


LAUNCH_COMMIT = "317ca1dae79cc75acf2ae48583b0392e8bb95114"
SOURCE_ARCHIVE_BYTES = 2_549_760
SOURCE_ARCHIVE_SHA256 = "dc3d58bfbb0075dc4fb14060fd5336eb539e1ea6b508b2fef3665024c924b017"
SUITE_REPOSITORY_PATH = "sweeps/current_budget_isoflop_v4/suite.yaml"
DEFAULT_REPOSITORY = "quintic/gpt-tpu-speedrun-scaling-evidence"
DEFAULT_PREFIX = "current_budget_isoflop_v4"
DEFAULT_RECEIPT = Path("data/manifests/scaling/current_budget_isoflop_v4.json")
MANIFEST_NAME = "archive-manifest.json"
SOURCE_PREFIX = "source/"
SCHEMA_VERSION = 1
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40,64}\Z")
HF_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_TOKEN_BYTES = 4 * 1024
READ_CHUNK = 1024 * 1024
HTTP_ATTEMPTS = 5
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024
MAX_INVENTORY_FILES = 4_096
ROLE_MAX_BYTES = {
    "documentation": 2 * 1024 * 1024,
    "verifier": 4 * 1024 * 1024,
    "source_archive": 4 * 1024 * 1024,
    "final_fit_json": 64 * 1024 * 1024,
    "final_fit_markdown": 8 * 1024 * 1024,
    "slice_fit": 32 * 1024 * 1024,
    "learning_rate_selection": 64 * 1024 * 1024,
    "run_manifest": 16 * 1024 * 1024,
    "config": 2 * 1024 * 1024,
    "trainer": 8 * 1024 * 1024,
    "trainer_support": 8 * 1024 * 1024,
    "training_curve": 256 * 1024 * 1024,
    "validation_curve": 64 * 1024 * 1024,
    "diagnostics": 512 * 1024 * 1024,
    "metrics": 32 * 1024 * 1024,
    "result": 32 * 1024 * 1024,
    "stability_admission": 32 * 1024 * 1024,
}
PUBLISHED_4B_MANIFEST_PATH = "data/manifests/fineweb-scaled-gpt2/4B.json"
PUBLISHED_4B_MANIFEST_BYTES = 14_846
PUBLISHED_4B_MANIFEST_SHA256 = (
    "99ac90a5e1bf5371053e46a3f378a2d507002d80a854a6b1372f39747edc2c14"
)
PUBLISHED_4B_MANIFEST_CANONICAL_SHA256 = (
    "92b21722236814a046a621cb4a801928dcf0318ca52693e11a3f32b1c6998dc1"
)

RUN_FILES: tuple[tuple[str, str], ...] = (
    ("run-manifest.json", "run_manifest"),
    ("work/config.yaml", "config"),
    ("work/train.py", "trainer"),
    ("work/speedrun/__init__.py", "trainer_support"),
    ("work/speedrun/kernels/__init__.py", "trainer_support"),
    ("work/speedrun/kernels/autotune.py", "trainer_support"),
    ("work/speedrun/kernels/linear_cross_entropy.py", "trainer_support"),
    ("work/speedrun/kernels/pallas_linear_cross_entropy.py", "trainer_support"),
    ("work/speedrun/kernels/tpu_flash_attention.py", "trainer_support"),
    ("artifacts/training.csv", "training_curve"),
    ("artifacts/validation.csv", "validation_curve"),
    ("artifacts/diagnostics.csv", "diagnostics"),
    ("artifacts/metrics.json", "metrics"),
    ("artifacts/result.json", "result"),
    ("artifacts/stability-admission.json", "stability_admission"),
)
FIT_PATHS = (
    "runs/fit.json",
    "runs/fit.md",
    "runs/fits/c025.json",
    "runs/fits/c050.json",
    "runs/fits/c100.json",
)
MANIFEST_TOP_KEYS = {
    "schema_version",
    "kind",
    "archive_id",
    "identity",
    "publication_target",
    "source_archive",
    "study",
    "inventory",
}
SEMANTIC_FLAGS = (
    "all_runs_recomputed",
    "prospective_run_set_complete",
    "required_control_present",
    "published_4b_manifest_bound",
    "metrics_equal_results",
    "admissions_recomputed",
    "learning_rate_selections_recomputed",
    "slice_fits_recomputed",
    "final_fit_recomputed",
    "scaling_law_present",
)


class EvidenceError(ValueError):
    """The evidence archive is incomplete, mutable, or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise EvidenceError(f"{label} must be a string-keyed object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise EvidenceError(f"{label} has the wrong schema ({'; '.join(details)})")


def _positive_integer(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be an immutable lowercase commit OID")
    return value


def normalized_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvidenceError(f"{label} must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise EvidenceError(f"{label} must be a normalized relative POSIX path")
    if value != path.as_posix():
        raise EvidenceError(f"{label} is not canonical POSIX spelling")
    return value


def validate_repo_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 96
        or value.count("/") != 1
        or any(
            not component
            or len(component) > 96
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", component) is None
            or component[-1] in {"-", "."}
            or "--" in component
            or ".." in component
            or component.endswith(".git")
            for component in value.split("/")
        )
    ):
        raise EvidenceError(f"invalid Hugging Face repository id: {value!r}")
    return value


def validate_prefix(value: str) -> str:
    normalized_path(value, "publication prefix")
    if any(SAFE_COMPONENT.fullmatch(part) is None for part in value.split("/")):
        raise EvidenceError("publication prefix has an unsafe component")
    return value.rstrip("/")


def lexical_absolute(path: Path) -> Path:
    """Make a path absolute without resolving or following any component."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def open_directory_chain(path: Path, *, create: bool) -> int:
    """Open one absolute directory path component-by-component without links."""

    absolute = lexical_absolute(path)
    if not absolute.is_absolute():  # pragma: no cover - lexical_absolute guarantees it
        raise EvidenceError("internal path normalization did not produce an absolute path")
    flags = _directory_open_flags()
    try:
        descriptor = os.open(os.sep, flags)
    except OSError as exc:  # pragma: no cover - a usable POSIX root is fundamental
        raise EvidenceError(f"cannot open filesystem root safely: {exc}") from exc
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    # A concurrent creator is acceptable only if the next
                    # O_NOFOLLOW open proves that it created a real directory.
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise EvidenceError(f"path is not a directory: {absolute}")
        return descriptor
    except (EvidenceError, OSError) as exc:
        os.close(descriptor)
        if isinstance(exc, EvidenceError):
            raise
        raise EvidenceError(
            f"directory path contains a missing, linked, or non-directory component: {absolute}"
        ) from exc


@contextmanager
def open_parent_directory(path: Path, *, create: bool) -> Iterator[tuple[Path, int, str]]:
    """Yield a pinned parent descriptor and safe leaf for one output/input path."""

    absolute = lexical_absolute(path)
    leaf = absolute.name
    if not leaf or leaf in {".", ".."} or "/" in leaf or "\\" in leaf:
        raise EvidenceError(f"unsafe final path component: {path}")
    descriptor = open_directory_chain(absolute.parent, create=create)
    try:
        yield absolute, descriptor, leaf
    finally:
        os.close(descriptor)


def _directory_entry_signature(
    descriptor: int, name: str
) -> tuple[int, ...] | None:
    try:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise EvidenceError(f"cannot safely inspect directory entry {name!r}: {exc}") from exc
    return _stat_signature(metadata)


def read_token_file(
    path: Path,
    *,
    expected_signature: tuple[int, ...] | None = None,
    expected_parent_identity: tuple[int, int] | None = None,
) -> str:
    """Read one bounded mode-0600 token through a pinned no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    with open_parent_directory(path, create=False) as (_absolute, parent, leaf):
        parent_metadata = os.fstat(parent)
        parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
        if (
            expected_parent_identity is not None
            and parent_identity != expected_parent_identity
        ):
            raise EvidenceError("Hugging Face token parent changed after preflight")
        try:
            descriptor = os.open(leaf, flags, dir_fd=parent)
        except OSError as exc:
            raise EvidenceError(
                "Hugging Face token path must be a regular file with no linked component"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if (
                expected_signature is not None
                and _stat_signature(before) != expected_signature
            ):
                raise EvidenceError("Hugging Face token changed after preflight")
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise EvidenceError(
                    "Hugging Face token path must be one regular, non-hard-linked file"
                )
            if before.st_uid != os.getuid():
                raise EvidenceError("Hugging Face token file must be owned by the current user")
            if stat.S_IMODE(before.st_mode) != 0o600:
                raise EvidenceError("Hugging Face token file must have exact mode 0600")
            if before.st_size > MAX_TOKEN_BYTES:
                raise EvidenceError(
                    f"Hugging Face token file exceeds {MAX_TOKEN_BYTES} bytes"
                )
            remaining = before.st_size
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    raise EvidenceError("Hugging Face token file was truncated while reading")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise EvidenceError("Hugging Face token file grew while reading")
            after = os.fstat(descriptor)
            current = _directory_entry_signature(parent, leaf)
            if (
                _stat_signature(before) != _stat_signature(after)
                or current != _stat_signature(after)
            ):
                raise EvidenceError("Hugging Face token path changed while reading")
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
    try:
        raw = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"cannot decode Hugging Face token file: {exc}") from exc
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].startswith("HF_TOKEN="):
        raise EvidenceError("token file must contain exactly one HF_TOKEN=... assignment")
    token = lines[0].removeprefix("HF_TOKEN=").strip()
    if TOKEN_PATTERN.fullmatch(token) is None:
        raise EvidenceError("HF_TOKEN value has an unexpected format")
    return token


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def scan_regular_tree(
    root: Path, *, directories: dict[str, tuple[int, ...]] | None = None
) -> dict[str, tuple[int, ...]]:
    """Return a recursive file inventory using lstat and no pathname globs.

    Callers enforcing a closed tree pass ``directories={}`` and compare both
    returned files and populated directories to their exact allowlists.  This
    makes even an unexpected empty directory observable.
    """

    try:
        root_meta = root.lstat()
    except OSError as exc:
        raise EvidenceError(f"cannot inspect directory {root}: {exc}") from exc
    if not stat.S_ISDIR(root_meta.st_mode):
        raise EvidenceError(f"expected a regular directory, not a link: {root}")
    if directories is not None:
        directories["."] = _stat_signature(root_meta)
    found: dict[str, tuple[int, ...]] = {}
    pending: list[tuple[Path, PurePosixPath | None]] = [(root, None)]
    while pending:
        directory, relative_parent = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise EvidenceError(f"cannot enumerate {directory}: {exc}") from exc
        for entry in entries:
            if entry.name in {".", ".."} or "/" in entry.name or "\\" in entry.name:
                raise EvidenceError(f"unsafe directory entry in {directory}: {entry.name!r}")
            relative = (
                PurePosixPath(entry.name)
                if relative_parent is None
                else relative_parent / entry.name
            )
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise EvidenceError(f"cannot inspect {entry.path}: {exc}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise EvidenceError(f"links are forbidden in evidence trees: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                if directories is not None:
                    directories[relative.as_posix()] = _stat_signature(metadata)
                pending.append((Path(entry.path), relative))
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise EvidenceError(f"hard-linked files are forbidden: {relative}")
                found[relative.as_posix()] = _stat_signature(metadata)
            else:
                raise EvidenceError(f"non-regular archive object is forbidden: {relative}")
    return dict(sorted(found.items()))


def expected_parent_directories(paths: Iterable[str]) -> set[str]:
    expected: set[str] = {"."}
    for raw in paths:
        path = PurePosixPath(raw)
        for parent in path.parents:
            if parent != PurePosixPath("."):
                expected.add(parent.as_posix())
    return expected


def read_regular_once(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    """Open without following links and reject mutation during the single read."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot safely open regular file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(f"expected a regular file: {path}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise EvidenceError(f"file exceeds the {maximum_bytes}-byte limit: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK, remaining))
            if not chunk:
                raise EvidenceError(f"file was truncated while reading: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise EvidenceError(f"file grew while reading: {path}")
        after = os.fstat(descriptor)
        if _stat_signature(before) != _stat_signature(after):
            raise EvidenceError(f"file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def copy_regular_once(source: Path, destination: Path) -> tuple[int, str]:
    """Copy and hash one source stream, rejecting links and concurrent writes."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot safely open source file {source}: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(f"source is not a regular file: {source}")
        if destination.exists() or destination.is_symlink():
            raise EvidenceError(f"refusing to overwrite staged path: {destination}")
        with destination.open("xb") as output:
            while True:
                chunk = os.read(descriptor, READ_CHUNK)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                total += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(descriptor)
        if _stat_signature(before) != _stat_signature(after) or total != before.st_size:
            destination.unlink(missing_ok=True)
            raise EvidenceError(f"source changed while being copied: {source}")
    finally:
        os.close(descriptor)
    return total, digest.hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise EvidenceError(f"refusing to overwrite archive file: {path}")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid {label} JSON: {exc}") from exc
    return _mapping(value, label)


def read_json_regular(path: Path, label: str) -> dict[str, Any]:
    return read_json_bytes(
        read_regular_once(path, maximum_bytes=MAX_MANIFEST_BYTES), label
    )


def inventory_digest(files: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(files)))


def validate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact schema and return a plain, non-normalized copy."""

    manifest = dict(payload)
    _exact_keys(manifest, MANIFEST_TOP_KEYS, "archive manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError("unsupported archive manifest schema_version")
    if manifest["kind"] != "gpt_tpu_speedrun_scaling_evidence":
        raise EvidenceError("archive manifest kind is invalid")
    archive_id = manifest["archive_id"]
    if not isinstance(archive_id, str) or SAFE_COMPONENT.fullmatch(archive_id) is None:
        raise EvidenceError("archive_id is not filesystem safe")

    identity = _mapping(manifest["identity"], "identity")
    _exact_keys(
        identity,
        {
            "suite_id",
            "suite_sha256",
            "template_sha256",
            "execution_fingerprint",
            "trainer_sha256",
            "seed",
            "launch_commit",
        },
        "identity",
    )
    if identity["suite_id"] != "current_budget_isoflop_v4":
        raise EvidenceError("identity.suite_id is not the v4 study")
    for key in (
        "suite_sha256",
        "template_sha256",
        "execution_fingerprint",
        "trainer_sha256",
    ):
        _sha256(identity[key], f"identity.{key}")
    _positive_integer(identity["seed"], "identity.seed", allow_zero=True)
    if _commit(identity["launch_commit"], "identity.launch_commit") != LAUNCH_COMMIT:
        raise EvidenceError(f"identity.launch_commit must equal {LAUNCH_COMMIT}")

    publication = _mapping(manifest["publication_target"], "publication_target")
    _exact_keys(publication, {"repository", "directory"}, "publication_target")
    validate_repo_id(publication["repository"])
    directory = normalized_path(publication["directory"], "publication_target.directory")
    if PurePosixPath(directory).name != archive_id:
        raise EvidenceError("publication directory must end with archive_id")

    source = _mapping(manifest["source_archive"], "source_archive")
    _exact_keys(source, {"path", "commit", "prefix", "bytes", "sha256"}, "source_archive")
    source_path = normalized_path(source["path"], "source_archive.path")
    expected_source_name = f"provenance/source-{identity['launch_commit']}.tar"
    if source_path != expected_source_name:
        raise EvidenceError("source archive path does not bind the launch commit")
    if _commit(source["commit"], "source_archive.commit") != identity["launch_commit"]:
        raise EvidenceError("source archive commit differs from identity")
    if source["prefix"] != SOURCE_PREFIX:
        raise EvidenceError("source archive prefix must be source/")
    _positive_integer(source["bytes"], "source_archive.bytes")
    _sha256(source["sha256"], "source_archive.sha256")

    study = _mapping(manifest["study"], "study")
    _exact_keys(
        study,
        {
            "runs",
            "classifications",
            "learning_rate_selections",
            "fit_paths",
            "can_estimate_scaling_exponent",
        },
        "study",
    )
    runs = study["runs"]
    if not isinstance(runs, list) or not runs or runs != sorted(set(runs)):
        raise EvidenceError("study.runs must be a nonempty sorted unique list")
    if any(not isinstance(item, str) or SAFE_COMPONENT.fullmatch(item) is None for item in runs):
        raise EvidenceError("study.runs contains an unsafe identifier")
    classifications = _mapping(study["classifications"], "study.classifications")
    _exact_keys(classifications, {"stable", "suspect", "rejected"}, "study.classifications")
    classified: list[str] = []
    for name in ("stable", "suspect", "rejected"):
        values = classifications[name]
        if (
            not isinstance(values, list)
            or any(
                not isinstance(item, str) or SAFE_COMPONENT.fullmatch(item) is None
                for item in values
            )
            or values != sorted(set(values))
        ):
            raise EvidenceError(f"study.classifications.{name} must be sorted and unique")
        classified.extend(values)
    if sorted(classified) != runs or len(classified) != len(set(classified)):
        raise EvidenceError("study classifications must partition the run list")
    selections = study["learning_rate_selections"]
    if (
        not isinstance(selections, list)
        or not selections
        or selections != sorted(set(selections))
        or any(
            not isinstance(item, str)
            or not item.startswith("runs/learning-rate-selections/")
            or not item.endswith(".json")
            or normalized_path(item, "learning-rate selection") != item
            for item in selections
        )
    ):
        raise EvidenceError("learning-rate selections must be sorted canonical JSON paths")
    fit_paths = _mapping(study["fit_paths"], "study.fit_paths")
    _exact_keys(fit_paths, {"final_json", "final_markdown", "slices"}, "study.fit_paths")
    if (
        fit_paths["final_json"] != "runs/fit.json"
        or fit_paths["final_markdown"] != "runs/fit.md"
        or fit_paths["slices"]
        != ["runs/fits/c025.json", "runs/fits/c050.json", "runs/fits/c100.json"]
    ):
        raise EvidenceError("study.fit_paths differs from the exact v4 fit set")
    if study["can_estimate_scaling_exponent"] is not True:
        raise EvidenceError("publication requires a fully bracketed scaling law")

    inventory = _mapping(manifest["inventory"], "inventory")
    _exact_keys(inventory, {"file_count", "total_bytes", "files"}, "inventory")
    files = inventory["files"]
    if (
        not isinstance(files, list)
        or not files
        or len(files) > MAX_INVENTORY_FILES
    ):
        raise EvidenceError("inventory.files must be a nonempty list")
    if inventory["file_count"] != len(files):
        raise EvidenceError("inventory.file_count differs from files")
    paths: list[str] = []
    total_bytes = 0
    for index, raw in enumerate(files):
        item = _mapping(raw, f"inventory.files[{index}]")
        _exact_keys(item, {"path", "bytes", "sha256", "role"}, f"inventory.files[{index}]")
        path = normalized_path(item["path"], f"inventory.files[{index}].path")
        if path == MANIFEST_NAME:
            raise EvidenceError("archive manifest must not inventory itself")
        paths.append(path)
        total_bytes += _positive_integer(
            item["bytes"], f"inventory.files[{index}].bytes", allow_zero=True
        )
        _sha256(item["sha256"], f"inventory.files[{index}].sha256")
        role = item["role"]
        if not isinstance(role, str) or role not in ROLE_MAX_BYTES:
            raise EvidenceError(f"inventory.files[{index}].role is unknown")
        if item["bytes"] > ROLE_MAX_BYTES[role]:
            raise EvidenceError(
                f"inventory.files[{index}] exceeds the {role} byte bound"
            )
    if paths != sorted(set(paths)):
        raise EvidenceError("inventory files must be sorted by unique path")
    if inventory["total_bytes"] != total_bytes:
        raise EvidenceError("inventory.total_bytes differs from file records")
    if total_bytes > MAX_ARCHIVE_BYTES:
        raise EvidenceError("inventory exceeds the bounded archive byte budget")

    expected_nonrun = {
        "README.md",
        "verify.py",
        source_path,
        *FIT_PATHS,
        *selections,
    }
    expected_run_paths = {
        f"runs/{run}/{relative}" for run in runs for relative, _role in RUN_FILES
    }
    expected = expected_nonrun | expected_run_paths
    if set(paths) != expected:
        missing = sorted(expected - set(paths))
        extra = sorted(set(paths) - expected)
        raise EvidenceError(
            "inventory is not the exact closed v4 archive: "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    role_by_path = {item["path"]: item["role"] for item in files}
    expected_roles = {
        "README.md": "documentation",
        "verify.py": "verifier",
        source_path: "source_archive",
        "runs/fit.json": "final_fit_json",
        "runs/fit.md": "final_fit_markdown",
        "runs/fits/c025.json": "slice_fit",
        "runs/fits/c050.json": "slice_fit",
        "runs/fits/c100.json": "slice_fit",
        **{path: "learning_rate_selection" for path in selections},
        **{
            f"runs/{run}/{relative}": role
            for run in runs
            for relative, role in RUN_FILES
        },
    }
    if role_by_path != expected_roles:
        raise EvidenceError("inventory roles differ from the exact path-role contract")
    source_entry = next(item for item in files if item["path"] == source_path)
    if (
        source_entry["bytes"] != source["bytes"]
        or source_entry["sha256"] != source["sha256"]
        or source_entry["role"] != "source_archive"
    ):
        raise EvidenceError("source_archive metadata differs from its inventory entry")
    expected_archive_id = (
        f"{identity['suite_id']}-{inventory_digest(files)[:16]}-"
        f"{identity['launch_commit'][:12]}"
    )
    if archive_id != expected_archive_id:
        raise EvidenceError("archive_id does not bind the complete file inventory")
    return manifest


def _inventory_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["path"]): dict(item)
        for item in _mapping(manifest["inventory"], "inventory")["files"]
    }


def snapshot_bundle(
    bundle: Path,
    destination: Path,
    *,
    pinned_files: Mapping[str, tuple[int, ...]] | None = None,
    pinned_directories: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Read each bundle object once into a private verified snapshot."""

    # Publication passes a symlink-free resolved path plus its complete pinned
    # inventory. Standalone verification resolves its user-selected root once.
    if pinned_files is None and pinned_directories is None:
        bundle = bundle.expanduser().resolve(strict=True)
    elif pinned_files is None or pinned_directories is None:
        raise EvidenceError("pinned bundle files/directories must be supplied together")
    else:
        bundle = lexical_absolute(bundle)
    if destination.exists() or destination.is_symlink():
        raise EvidenceError(f"snapshot destination already exists: {destination}")
    destination.mkdir(parents=True)
    source_directories: dict[str, tuple[int, ...]] = {}
    source_inventory = scan_regular_tree(bundle, directories=source_directories)
    if pinned_files is not None and (
        source_inventory != pinned_files or source_directories != pinned_directories
    ):
        raise EvidenceError("pinned bundle changed between preflight and snapshot")
    if MANIFEST_NAME not in source_inventory:
        raise EvidenceError(f"bundle lacks {MANIFEST_NAME}")
    manifest_bytes = read_regular_once(
        bundle / MANIFEST_NAME, maximum_bytes=MAX_MANIFEST_BYTES
    )
    manifest = validate_manifest(read_json_bytes(manifest_bytes, "archive manifest"))
    expected_files = _inventory_index(manifest)
    expected_paths = {MANIFEST_NAME, *expected_files}
    if set(source_inventory) != expected_paths:
        missing = sorted(expected_paths - set(source_inventory))
        extra = sorted(set(source_inventory) - expected_paths)
        raise EvidenceError(
            f"bundle tree is not closed: missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    expected_directories = expected_parent_directories(expected_paths)
    if set(source_directories) != expected_directories:
        raise EvidenceError(
            "bundle directory tree is not closed: "
            f"missing={sorted(expected_directories - set(source_directories))[:5]!r}, "
            f"extra={sorted(set(source_directories) - expected_directories)[:5]!r}"
        )
    write_new(destination / MANIFEST_NAME, manifest_bytes)
    for relative in sorted(expected_files):
        expected = expected_files[relative]
        size, digest = copy_regular_once(bundle / relative, destination / relative)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise EvidenceError(f"bundle object differs from inventory: {relative}")
    after_directories: dict[str, tuple[int, ...]] = {}
    after_inventory = scan_regular_tree(bundle, directories=after_directories)
    if source_inventory != after_inventory or source_directories != after_directories:
        raise EvidenceError("bundle source tree changed while it was being snapshotted")
    return manifest


def _safe_extract_source(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive, mode="r:") as handle:
            members = handle.getmembers()
            if not members:
                raise EvidenceError("source archive is empty")
            for member in members:
                name = normalized_path(member.name.rstrip("/"), "source tar member")
                if name != "source" and not name.startswith(SOURCE_PREFIX):
                    raise EvidenceError("source tar member escapes source/ prefix")
                if not (member.isdir() or member.isfile()):
                    raise EvidenceError(f"source tar contains a link or special file: {name}")
            handle.extractall(destination, filter="data")
    except (OSError, tarfile.TarError) as exc:
        raise EvidenceError(f"invalid deterministic source archive: {exc}") from exc
    root = destination / "source"
    if not root.is_dir() or root.is_symlink():
        raise EvidenceError("source archive did not extract one source/ root")
    return root


@contextmanager
def archived_scaling_module(bundle: Path, manifest: Mapping[str, Any]) -> Iterator[tuple[ModuleType, Path]]:
    """Import speedrun.scaling solely from the hash-pinned launch source tar."""

    source_info = _mapping(manifest["source_archive"], "source_archive")
    with tempfile.TemporaryDirectory(prefix="scaling-evidence-source-") as directory:
        extracted = _safe_extract_source(
            bundle / str(source_info["path"]), Path(directory) / "unpacked"
        )
        old_path = list(sys.path)
        old_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "speedrun" or name.startswith("speedrun.")
        }
        for name in old_modules:
            del sys.modules[name]
        sys.path.insert(0, str(extracted))
        try:
            module = importlib.import_module("speedrun.scaling")
            module_path = Path(module.__file__).resolve()
            try:
                module_path.relative_to(extracted.resolve())
            except ValueError as exc:
                raise EvidenceError("speedrun.scaling was not loaded from the source archive") from exc
            yield module, extracted
        finally:
            for name in list(sys.modules):
                if name == "speedrun" or name.startswith("speedrun."):
                    del sys.modules[name]
            sys.modules.update(old_modules)
            sys.path[:] = old_path


def semantic_verify(bundle: Path, manifest: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute every raw admission and all derived study outputs."""

    manifest = validate_manifest(manifest)
    identity = _mapping(manifest["identity"], "identity")
    study = _mapping(manifest["study"], "study")
    runs_root = bundle / "runs"
    source_info = _mapping(manifest["source_archive"], "source_archive")
    source_path = bundle / str(source_info["path"])
    source_size, source_digest = hash_regular_file(source_path)
    if (
        source_size != SOURCE_ARCHIVE_BYTES
        or source_digest != SOURCE_ARCHIVE_SHA256
        or source_info["bytes"] != SOURCE_ARCHIVE_BYTES
        or source_info["sha256"] != SOURCE_ARCHIVE_SHA256
    ):
        raise EvidenceError(
            "source archive is not the exact deterministic Git archive of the launch commit"
        )
    with archived_scaling_module(bundle, manifest) as (scaling, source_root):
        try:
            suite = scaling.load_suite(source_root / SUITE_REPOSITORY_PATH)
        except Exception as exc:
            raise EvidenceError(f"cannot load the archived v4 suite: {exc}") from exc
        expected_identity = {
            "suite_id": suite["suite_id"],
            "suite_sha256": suite["suite_sha256"],
            "template_sha256": suite["template_sha256"],
            "execution_fingerprint": suite["execution_fingerprint"],
            "trainer_sha256": suite["trainer_source_sha256"],
            "seed": suite["seed"],
            "launch_commit": identity["launch_commit"],
        }
        if dict(identity) != expected_identity:
            raise EvidenceError("archive identity differs from the archived suite")
        points = {str(point["id"]): point for point in suite["all_variants"]}
        run_names = list(study["runs"])
        if any(name not in points for name in run_names):
            raise EvidenceError("archive contains a run not declared by the v4 suite")

        measurements: dict[str, dict[str, Any]] = {}
        for name in run_names:
            metrics_bytes = read_regular_once(
                runs_root / name / "artifacts" / "metrics.json"
            )
            result_bytes = read_regular_once(
                runs_root / name / "artifacts" / "result.json"
            )
            if metrics_bytes != result_bytes:
                raise EvidenceError(f"{name}: metrics.json and result.json are not byte-identical")
            try:
                measurements[name] = scaling._read_run(suite, points[name], runs_root)
            except Exception as exc:
                raise EvidenceError(f"{name}: raw run/admission recomputation failed: {exc}") from exc

        validate_published_4b_binding(
            source_root=source_root,
            suite=suite,
            measurements=measurements,
        )

        classifications = {"stable": [], "suspect": [], "rejected": []}
        for name in run_names:
            admission = measurements[name].get("stability_admission")
            if not isinstance(admission, Mapping):
                raise EvidenceError(f"{name}: v4 stability admission is missing")
            classification = admission.get("classification")
            if classification not in classifications:
                raise EvidenceError(f"{name}: invalid stability classification")
            classifications[str(classification)].append(name)
        for values in classifications.values():
            values.sort()
        if classifications != study["classifications"]:
            raise EvidenceError("manifest classification partition differs from recomputation")

        selection_groups = prospective_selection_groups(
            suite,
            run_names=run_names,
            measurements=measurements,
        )
        expected_selection_paths = {
            f"runs/learning-rate-selections/{slice_id}/{shape_id}.json"
            for slice_id, shape_id in selection_groups
        }
        if set(study["learning_rate_selections"]) != expected_selection_paths:
            raise EvidenceError(
                "learning-rate selection inventory does not exactly cover every "
                "required or begun slice/shape calibration"
            )

        accounted_runs = {
            str(point["id"])
            for point in suite["controls"]
        }
        for relative in sorted(expected_selection_paths):
            parts = PurePosixPath(relative).parts
            if len(parts) != 4:
                raise EvidenceError(f"invalid v4 learning-rate selection path: {relative}")
            slice_id = parts[2]
            filename = parts[3]
            if not filename.endswith(".json"):
                raise EvidenceError(f"invalid v4 learning-rate selection path: {relative}")
            shape_id = filename[:-5]
            existing = read_json_regular(bundle / relative, relative)
            try:
                recomputed = scaling.select_learning_rate(
                    suite,
                    shape_id=shape_id,
                    slice_id=slice_id,
                    runs_path=runs_root,
                )
            except Exception as exc:
                raise EvidenceError(f"{slice_id}/{shape_id}: LR recomputation failed: {exc}") from exc
            if recomputed != existing:
                raise EvidenceError(f"{slice_id}/{shape_id}: LR selection JSON differs")
            candidates = recomputed.get("candidates")
            if not isinstance(candidates, list):
                raise EvidenceError(f"{slice_id}/{shape_id}: LR candidates are malformed")
            for candidate in candidates:
                if not isinstance(candidate, Mapping) or not isinstance(
                    candidate.get("id"), str
                ):
                    raise EvidenceError(
                        f"{slice_id}/{shape_id}: LR candidate identity is malformed"
                    )
                accounted_runs.add(str(candidate["id"]))
        if accounted_runs != set(run_names):
            raise EvidenceError(
                "archived runs are not exactly the required control plus every "
                "candidate in a recomputed LR selection"
            )

        try:
            final_fit = scaling.fit_results(suite, runs_root)
        except Exception as exc:
            raise EvidenceError(f"final fit recomputation failed: {exc}") from exc
        existing_fit = read_json_regular(runs_root / "fit.json", "runs/fit.json")
        if final_fit != existing_fit:
            raise EvidenceError("runs/fit.json differs from full recomputation")
        if (
            final_fit.get("can_estimate_scaling_exponent") is not True
            or not isinstance(final_fit.get("scaling_law"), Mapping)
        ):
            raise EvidenceError("publication requires a fully bracketed final scaling law")
        control_ids = [str(point["id"]) for point in suite["controls"]]
        if control_ids != ["c100_n124_control"]:
            raise EvidenceError("archived launch suite has an unexpected control contract")
        fitted_controls = final_fit.get("controls")
        if (
            not isinstance(fitted_controls, list)
            or [item.get("id") for item in fitted_controls] != control_ids
        ):
            raise EvidenceError("final fit does not contain the mandatory c100_n124_control")
        slices = final_fit.get("slices")
        if not isinstance(slices, list) or [item.get("slice") for item in slices] != [
            "c025",
            "c050",
            "c100",
        ]:
            raise EvidenceError("final fit has the wrong ordered slice set")
        for fitted in slices:
            slice_id = str(fitted["slice"])
            existing_slice = read_json_regular(
                runs_root / "fits" / f"{slice_id}.json",
                f"runs/fits/{slice_id}.json",
            )
            if fitted != existing_slice:
                raise EvidenceError(f"{slice_id}: stored slice fit differs from recomputation")
        with tempfile.TemporaryDirectory(prefix="scaling-evidence-fit-") as directory:
            generated_json, generated_markdown = scaling.write_fit(
                final_fit, Path(directory) / "fit.json"
            )
            if read_regular_once(generated_json) != read_regular_once(runs_root / "fit.json"):
                raise EvidenceError("final fit JSON bytes are not deterministic")
            if read_regular_once(generated_markdown) != read_regular_once(runs_root / "fit.md"):
                raise EvidenceError("final fit Markdown differs from recomputation")

    return {flag: True for flag in SEMANTIC_FLAGS}


def validate_published_4b_binding(
    *,
    source_root: Path,
    suite: Mapping[str, Any],
    measurements: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind every run to the exact public, checked-in 4B shard inventory."""

    path = source_root / PUBLISHED_4B_MANIFEST_PATH
    payload = read_regular_once(path, maximum_bytes=PUBLISHED_4B_MANIFEST_BYTES)
    if (
        len(payload) != PUBLISHED_4B_MANIFEST_BYTES
        or sha256_bytes(payload) != PUBLISHED_4B_MANIFEST_SHA256
    ):
        raise EvidenceError("archived published 4B manifest differs from its exact pin")
    public = read_json_bytes(payload, "published 4B manifest")
    canonical_public = json.dumps(
        public, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if sha256_bytes(canonical_public) != PUBLISHED_4B_MANIFEST_CANONICAL_SHA256:
        raise EvidenceError("archived published 4B canonical manifest hash differs")
    files = public.get("files")
    source = public.get("source")
    tokenizer = public.get("tokenizer")
    preparation = public.get("preparation")
    if (
        public.get("schema_version") != 1
        or public.get("name") != suite["dataset"]["id"]
        or not isinstance(files, list)
        or len(files) != 40
        or not isinstance(source, Mapping)
        or not isinstance(tokenizer, Mapping)
        or not isinstance(preparation, Mapping)
        or source.get("dataset") != suite["dataset"]["source_repository"]
        or source.get("revision") != suite["dataset"]["source_revision"]
        or source.get("inventory_sha256")
        != suite["dataset"]["source_inventory_sha256"]
        or source.get("exclusion_policy_sha256")
        != suite["dataset"]["exclusion_policy_sha256"]
        or tokenizer.get("implementation_version")
        != suite["dataset"]["tokenizer_version"]
        or preparation.get("core_sha256")
        != suite["dataset"]["preparation_core_sha256"]
    ):
        raise EvidenceError("published 4B identity differs from the v4 suite")
    prepared_revision = source.get("prepared_revision")
    prepared_repository = source.get("prepared_repository")
    if (
        not isinstance(prepared_revision, str)
        or HF_COMMIT_PATTERN.fullmatch(prepared_revision) is None
        or prepared_repository != "quintic/fineweb-scaled-gpt2"
    ):
        raise EvidenceError("published 4B prepared repository/revision is not immutable")
    expected_shards: list[dict[str, Any]] = []
    for index, raw in enumerate(files):
        if not isinstance(raw, Mapping):
            raise EvidenceError(f"published 4B files[{index}] is malformed")
        expected_keys = {"path", "split", "tokens", "bytes", "sha256", "url"}
        if set(raw) != expected_keys:
            raise EvidenceError(f"published 4B files[{index}] has unexpected fields")
        path_value = raw["path"]
        expected_url = (
            "https://huggingface.co/datasets/"
            f"{prepared_repository}/resolve/{prepared_revision}/4B/{path_value}"
        )
        if (
            not isinstance(path_value, str)
            or raw["url"] != expected_url
            or raw["tokens"] != 100_000_000
            or raw["bytes"] != 200_001_024
            or not isinstance(raw["sha256"], str)
            or SHA256_PATTERN.fullmatch(raw["sha256"]) is None
        ):
            raise EvidenceError(f"published 4B files[{index}] differs from its contract")
        expected_shards.append(
            {key: raw[key] for key in ("path", "split", "tokens", "bytes", "sha256")}
        )
    for run_name, measurement in measurements.items():
        provenance = measurement.get("_dataset_provenance")
        if not isinstance(provenance, Mapping):
            raise EvidenceError(f"{run_name}: dataset provenance is missing")
        production = provenance.get("production")
        if (
            provenance.get("name") != public["name"]
            or provenance.get("manifest_raw_sha256")
            != PUBLISHED_4B_MANIFEST_SHA256
            or provenance.get("manifest_canonical_sha256")
            != PUBLISHED_4B_MANIFEST_CANONICAL_SHA256
            or not isinstance(production, Mapping)
            or production.get("source_inventory_sha256")
            != source["inventory_sha256"]
            or production.get("exclusion_policy_sha256")
            != source["exclusion_policy_sha256"]
            or production.get("preparation_core_sha256")
            != preparation["core_sha256"]
            or production.get("builder_module_sha256")
            != preparation["builder_module_sha256"]
            or production.get("entrypoint_sha256")
            != preparation["entrypoint_sha256"]
            or production.get("source_date_before")
            != source["source_date_before"]
            or production.get("validation_train_document_disjoint") is not True
            or production.get("validation_boundary_discarded_tokens")
            != preparation["validation_boundary_discarded_tokens"]
            or production.get("validation_boundary_document_id_sha256")
            != preparation["validation_boundary_document_id_sha256"]
            or provenance.get("shards") != expected_shards
        ):
            raise EvidenceError(
                f"{run_name}: dataset identity/shards differ from the exact "
                "published 4B manifest"
            )


def prospective_selection_groups(
    suite: Mapping[str, Any],
    *,
    run_names: Sequence[str],
    measurements: Mapping[str, Mapping[str, Any]],
) -> set[tuple[str, str]]:
    """Prove the archived trials are a completed prospective v4 state machine."""

    points = {str(point["id"]): point for point in suite["all_variants"]}
    completed = set(run_names)
    if not completed.issubset(points):
        raise EvidenceError(
            "completed run set contains identities outside the archived v4 suite"
        )
    control_ids = {str(point["id"]) for point in suite["controls"]}
    if control_ids != {"c100_n124_control"}:
        raise EvidenceError("v4 publication requires exactly c100_n124_control")
    if not control_ids.issubset(completed):
        raise EvidenceError("mandatory c100_n124_control evidence is missing")

    base_groups = {
        (str(compute_slice["id"]), str(shape["shape_id"]))
        for compute_slice in suite["compute_slices"]
        for shape in suite["fit_shapes"]
    }
    extension_shape_ids = {
        str(shape["shape_id"]) for shape in suite["optional_extension_shapes"]
    }
    begun_extension_groups = {
        (str(points[name]["slice"]), str(points[name]["shape_id"]))
        for name in completed - control_ids
        if str(points[name]["shape_id"]) in extension_shape_ids
    }
    groups = base_groups | begun_extension_groups

    calibration_points = list(suite["calibrations"]) + list(
        suite["extension_calibrations"]
    )
    adaptive_points = list(suite["adaptive_calibrations"])
    expected_run_ids = set(control_ids)

    def classification(point: Mapping[str, Any]) -> str:
        point_id = str(point["id"])
        measurement = measurements.get(point_id)
        if measurement is None:
            raise EvidenceError(f"{point_id}: prospectively required LR trial is missing")
        admission = measurement.get("stability_admission")
        if not isinstance(admission, Mapping) or admission.get("classification") not in {
            "stable",
            "suspect",
            "rejected",
        }:
            raise EvidenceError(f"{point_id}: stability admission is malformed")
        return str(admission["classification"])

    def selection_state(
        launched: Sequence[Mapping[str, Any]], group_label: str
    ) -> str:
        ordered = sorted(launched, key=lambda point: float(point["learning_rate"]))
        stable = [point for point in ordered if classification(point) == "stable"]
        if not stable:
            raise EvidenceError(f"{group_label}: no stable LR candidate")
        selected = min(
            stable,
            key=lambda point: (
                float(measurements[str(point["id"])]["validation_loss"]),
                float(point["learning_rate"]),
            ),
        )
        index = ordered.index(selected)
        if index == 0 or classification(ordered[index - 1]) != "stable":
            return "lower"
        if index == len(ordered) - 1:
            return "upper"
        return "valid"

    for slice_id, shape_id in sorted(groups):
        group_label = f"{slice_id}/{shape_id}"
        initial = sorted(
            (
                point
                for point in calibration_points
                if str(point["slice"]) == slice_id
                and str(point["shape_id"]) == shape_id
            ),
            key=lambda point: float(point["learning_rate"]),
        )
        if len(initial) != len(suite["learning_rate_candidates"]):
            raise EvidenceError(f"{slice_id}/{shape_id}: initial LR definition is incomplete")
        lower = [
            next(
                point
                for point in adaptive_points
                if str(point["slice"]) == slice_id
                and str(point["shape_id"]) == shape_id
                and float(point["learning_rate"]) == float(candidate["value"])
            )
            for candidate in suite["learning_rate_search"]["lower"]
        ]
        upper = [
            next(
                point
                for point in adaptive_points
                if str(point["slice"]) == slice_id
                and str(point["shape_id"]) == shape_id
                and float(point["learning_rate"]) == float(candidate["value"])
            )
            for candidate in suite["learning_rate_search"]["upper"]
        ]

        launched: list[Mapping[str, Any]] = []
        for point in initial:
            if str(point["id"]) not in completed:
                raise EvidenceError(
                    f"{group_label}: stable initial LR prefix has a missing suffix"
                )
            launched.append(point)
            expected_run_ids.add(str(point["id"]))
            if classification(point) != "stable":
                break
        # The actual staged controller performs a two-stable-point lower
        # recovery before attempting selection when lr200 itself is ineligible.
        if classification(initial[0]) != "stable":
            if len(launched) != 1:
                raise EvidenceError(f"{group_label}: ran beyond the lr200 frontier")
            for point in lower:
                if str(point["id"]) not in completed:
                    raise EvidenceError(
                        f"{group_label}: mandatory lower-LR recovery is incomplete"
                    )
                if classification(point) != "stable":
                    raise EvidenceError(
                        f"{group_label}: lower-LR recovery crossed an ineligible frontier"
                    )
                launched.append(point)
                expected_run_ids.add(str(point["id"]))

        lower_index = sum(point in launched for point in lower)
        upper_index = sum(point in launched for point in upper)
        while True:
            state = selection_state(launched, group_label)
            if state == "valid":
                break
            table = lower if state == "lower" else upper
            index = lower_index if state == "lower" else upper_index
            if index >= len(table):
                raise EvidenceError(
                    f"{group_label}: bounded {state} LR search exhausted without a bracket"
                )
            point = table[index]
            if str(point["id"]) not in completed:
                raise EvidenceError(
                    f"{group_label}: prospectively necessary {state} LR trial is missing"
                )
            if state == "lower" and classification(point) != "stable":
                raise EvidenceError(
                    f"{group_label}: lower LR expansion is ineligible"
                )
            launched.append(point)
            expected_run_ids.add(str(point["id"]))
            if state == "lower":
                lower_index += 1
            else:
                upper_index += 1

    if completed != expected_run_ids:
        omitted = sorted(expected_run_ids - completed)
        unnecessary = sorted(completed - expected_run_ids)
        raise EvidenceError(
            "completed run set differs from the exact minimal prospective state: "
            f"missing={omitted[:8]!r}, unnecessary={unnecessary[:8]!r}"
        )
    return groups


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Verify a local/archive download through a private read-once snapshot."""

    with tempfile.TemporaryDirectory(prefix="scaling-evidence-snapshot-") as directory:
        snapshot = Path(directory) / "archive"
        manifest = snapshot_bundle(bundle, snapshot)
        flags = verify_frozen_snapshot(snapshot, manifest)
        return {
            "archive_id": manifest["archive_id"],
            "manifest_sha256": sha256_bytes(read_regular_once(snapshot / MANIFEST_NAME)),
            "inventory_sha256": inventory_digest(manifest["inventory"]["files"]),
            "file_count": manifest["inventory"]["file_count"] + 1,
            "semantic_verification": flags,
        }


def verify_frozen_snapshot(
    snapshot: Path,
    manifest: Mapping[str, Any],
    *,
    expected_manifest: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    """Verify the one retained upload snapshot without making another copy."""

    manifest = validate_manifest(manifest)
    raw_manifest = read_regular_once(
        snapshot / MANIFEST_NAME, maximum_bytes=MAX_MANIFEST_BYTES
    )
    if raw_manifest != canonical_json_bytes(manifest):
        raise EvidenceError("snapshot archive manifest is not exact canonical JSON")
    if expected_manifest is not None:
        expected = validate_manifest(expected_manifest)
        if manifest != expected or raw_manifest != canonical_json_bytes(expected):
            raise EvidenceError("retained upload snapshot manifest identity differs")
    before_directories: dict[str, tuple[int, ...]] = {}
    before_files = scan_regular_tree(snapshot, directories=before_directories)
    flags = semantic_verify(snapshot, manifest)
    after_directories: dict[str, tuple[int, ...]] = {}
    after_files = scan_regular_tree(snapshot, directories=after_directories)
    if before_files != after_files or before_directories != after_directories:
        raise EvidenceError("semantic recomputation changed the closed archive tree")
    return flags


def set_snapshot_permissions(snapshot: Path, *, sealed: bool) -> None:
    """Seal the retained private snapshot read-only, or reopen it for cleanup."""

    directories: dict[str, tuple[int, ...]] = {}
    files = scan_regular_tree(snapshot, directories=directories)
    file_mode = 0o400 if sealed else 0o600
    directory_mode = 0o500 if sealed else 0o700
    for relative in files:
        os.chmod(snapshot / relative, file_mode, follow_symlinks=False)
    ordered_directories = sorted(
        (path for path in directories if path != "."),
        key=lambda value: len(PurePosixPath(value).parts),
        reverse=sealed,
    )
    for relative in ordered_directories:
        os.chmod(snapshot / relative, directory_mode, follow_symlinks=False)
    os.chmod(snapshot, directory_mode, follow_symlinks=False)


def deterministic_git_archive(repository: Path, commit: str, destination: Path) -> tuple[int, str]:
    """Write the exact deterministic ``git archive`` byte stream for a commit."""

    # build_archive has already resolved this input root exactly once.  Do not
    # re-resolve a user-controlled alias between the build preflight and Git.
    repository = lexical_absolute(repository)
    descriptor = open_directory_chain(repository, create=False)
    os.close(descriptor)
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.decode("ascii").strip()
    if resolved != commit:
        raise EvidenceError(f"launch commit did not resolve exactly: {resolved}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise EvidenceError(f"refusing to overwrite source archive: {destination}")
    try:
        with destination.open("xb") as output:
            completed = subprocess.run(
                [
                    "git",
                    "archive",
                    "--format=tar",
                    f"--prefix={SOURCE_PREFIX}",
                    commit,
                ],
                cwd=repository,
                check=False,
                stdout=output,
                stderr=subprocess.PIPE,
            )
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        raise EvidenceError(f"cannot create deterministic source archive: {exc}") from exc
    if completed.returncode:
        destination.unlink(missing_ok=True)
        raise EvidenceError(
            "git archive failed: " + completed.stderr.decode("utf-8", errors="replace")
        )
    size, digest = hash_regular_file(destination)
    if size != SOURCE_ARCHIVE_BYTES or digest != SOURCE_ARCHIVE_SHA256:
        raise EvidenceError(
            "git archive bytes differ from the pinned deterministic launch archive"
        )
    with tempfile.TemporaryDirectory(prefix="scaling-source-repeat-") as directory:
        repeated = Path(directory) / "source.tar"
        with repeated.open("xb") as output:
            check = subprocess.run(
                [
                    "git",
                    "archive",
                    "--format=tar",
                    f"--prefix={SOURCE_PREFIX}",
                    commit,
                ],
                cwd=repository,
                check=False,
                stdout=output,
                stderr=subprocess.PIPE,
            )
        if check.returncode:
            raise EvidenceError("repeat git archive failed")
        repeat_size, repeat_digest = hash_regular_file(repeated)
        if (size, digest) != (repeat_size, repeat_digest):
            raise EvidenceError("git archive was not byte-deterministic across two emissions")
    return size, digest


def hash_regular_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(f"not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if _stat_signature(before) != _stat_signature(after):
            raise EvidenceError(f"file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return total, digest.hexdigest()


def _load_suite_from_source_archive(source_archive: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="scaling-suite-source-") as directory:
        extracted = _safe_extract_source(source_archive, Path(directory) / "unpacked")
        old_path = list(sys.path)
        old_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "speedrun" or name.startswith("speedrun.")
        }
        for name in old_modules:
            del sys.modules[name]
        sys.path.insert(0, str(extracted))
        try:
            scaling = importlib.import_module("speedrun.scaling")
            suite = scaling.load_suite(extracted / SUITE_REPOSITORY_PATH)
            public = {
                "all_variants": [dict(item) for item in suite["all_variants"]],
                "suite_id": suite["suite_id"],
                "suite_sha256": suite["suite_sha256"],
                "template_sha256": suite["template_sha256"],
                "execution_fingerprint": suite["execution_fingerprint"],
                "trainer_source_sha256": suite["trainer_source_sha256"],
                "seed": suite["seed"],
            }
            return public
        finally:
            for name in list(sys.modules):
                if name == "speedrun" or name.startswith("speedrun."):
                    del sys.modules[name]
            sys.modules.update(old_modules)
            sys.path[:] = old_path


def discover_runs_source(runs: Path, declared_points: set[str]) -> tuple[list[str], list[str], dict[str, tuple[int, ...]]]:
    """Validate the source run tree against the exact run/derived allowlist."""

    # The caller supplies the one canonical root chosen at workflow preflight.
    # A later symlink substitution must be rejected by scan_regular_tree rather
    # than silently followed to a different evidence tree.
    runs = lexical_absolute(runs)
    directories: dict[str, tuple[int, ...]] = {}
    inventory = scan_regular_tree(runs, directories=directories)
    required_derived = {
        "fit.json",
        "fit.md",
        "fits/c025.json",
        "fits/c050.json",
        "fits/c100.json",
    }
    run_names = sorted(
        {
            PurePosixPath(path).parts[0]
            for path in inventory
            if PurePosixPath(path).parts[0] in declared_points
        }
    )
    if not run_names:
        raise EvidenceError("run source contains no completed declared v4 runs")
    selections = sorted(
        path
        for path in inventory
        if path.startswith("learning-rate-selections/")
    )
    if not selections:
        raise EvidenceError("run source has no learning-rate selections")
    for path in selections:
        parts = PurePosixPath(path).parts
        if (
            len(parts) != 3
            or parts[0] != "learning-rate-selections"
            or parts[1] not in {"c025", "c050", "c100"}
            or not parts[2].endswith(".json")
            or SAFE_COMPONENT.fullmatch(parts[2][:-5]) is None
        ):
            raise EvidenceError(f"unexpected learning-rate selection path: {path}")
    expected = set(required_derived) | set(selections)
    for name in run_names:
        expected.update(f"{name}/{relative}" for relative, _role in RUN_FILES)
    if set(inventory) != expected:
        missing = sorted(expected - set(inventory))
        extra = sorted(set(inventory) - expected)
        raise EvidenceError(
            "run source is not the strict 15-file/run closed tree: "
            f"missing={missing[:8]!r}, extra={extra[:8]!r}"
        )
    expected_directories = expected_parent_directories(expected)
    if set(directories) != expected_directories:
        raise EvidenceError(
            "run source directory tree is not closed: "
            f"missing={sorted(expected_directories - set(directories))[:8]!r}, "
            f"extra={sorted(set(directories) - expected_directories)[:8]!r}"
        )
    total_bytes = 0
    for relative, metadata in inventory.items():
        archive_path = f"runs/{relative}"
        role = _role_for_archive_path(archive_path, run_names, selections)
        size = metadata[6]
        if size > ROLE_MAX_BYTES[role]:
            raise EvidenceError(
                f"run source {relative} exceeds the {role} byte bound"
            )
        total_bytes += size
    if total_bytes > MAX_ARCHIVE_BYTES:
        raise EvidenceError("run source exceeds the bounded archive byte budget")
    return run_names, selections, inventory


def _classification_from_admission(path: Path, run_name: str) -> str:
    payload = read_json_regular(path, f"{run_name}.stability-admission")
    classification = payload.get("classification")
    if classification not in {"stable", "suspect", "rejected"}:
        raise EvidenceError(f"{run_name}: invalid stability classification")
    return str(classification)


def archive_readme(identity: Mapping[str, Any]) -> bytes:
    return (
        "# GPT TPU speedrun scaling evidence\n\n"
        f"Suite: `{identity['suite_id']}`  \n"
        f"Launch source: `{identity['launch_commit']}`\n\n"
        "This is the closed raw-evidence bundle for the one-seed, local v4 "
        "IsoFLOP study. It is not a universal Chinchilla law. Every learning-rate "
        "trial is retained, including quarantined trials.\n\n"
        "Verify all hashes and recompute admissions, LR choices, slice fits, and "
        "the final fit with:\n\n"
        "```bash\n"
        "python verify.py verify --bundle .\n"
        "```\n\n"
        "`archive-manifest.json` is excluded from its own inventory; the external "
        "publication receipt pins its SHA-256 and immutable repository revision.\n"
    ).encode("utf-8")


def _role_for_archive_path(path: str, run_names: Sequence[str], selections: Sequence[str]) -> str:
    fixed = {
        "README.md": "documentation",
        "verify.py": "verifier",
        f"provenance/source-{LAUNCH_COMMIT}.tar": "source_archive",
        "runs/fit.json": "final_fit_json",
        "runs/fit.md": "final_fit_markdown",
        "runs/fits/c025.json": "slice_fit",
        "runs/fits/c050.json": "slice_fit",
        "runs/fits/c100.json": "slice_fit",
        **{f"runs/{path}": "learning_rate_selection" for path in selections},
    }
    if path in fixed:
        return fixed[path]
    for run in run_names:
        prefix = f"runs/{run}/"
        if path.startswith(prefix):
            relative = path.removeprefix(prefix)
            roles = dict(RUN_FILES)
            if relative in roles:
                return roles[relative]
    raise EvidenceError(f"no archive role for {path}")


def create_owned_temporary_directory(parent: int, label: str) -> tuple[str, tuple[int, ...]]:
    """Create one unpredictable mode-0700 directory below a pinned parent."""

    for _attempt in range(128):
        name = f".{label}.tmp-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            continue
        signature = _directory_entry_signature(parent, name)
        if signature is None or not stat.S_ISDIR(signature[2]):  # pragma: no cover
            raise EvidenceError("new temporary archive directory is not a directory")
        if signature[4] != os.getuid():  # pragma: no cover - mkdir owns its result
            raise EvidenceError("new temporary archive directory has the wrong owner")
        return name, signature
    raise EvidenceError("could not allocate a unique temporary archive directory")


def cleanup_owned_temporary_directory(
    parent: int, name: str, signature: tuple[int, ...]
) -> None:
    """Remove only the unchanged temporary root allocated by this invocation."""

    current = _directory_entry_signature(parent, name)
    if current is None:
        return
    if (
        current[:2] != signature[:2]
        or not stat.S_ISDIR(current[2])
        or current[4] != signature[4]
    ):
        raise EvidenceError(
            "refusing to clean a temporary archive directory whose identity changed"
        )
    try:
        shutil.rmtree(name, dir_fd=parent)
    except OSError as exc:
        raise EvidenceError(f"cannot clean owned temporary archive directory: {exc}") from exc


def rename_directory_noreplace(
    source_parent: int,
    source: str,
    destination_parent: int,
    destination: str,
) -> None:
    """Atomically install a directory without replacing any existing object."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - Linux release environment
        raise EvidenceError("atomic renameat2(RENAME_NOREPLACE) is unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent,
        os.fsencode(source),
        destination_parent,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise EvidenceError(
            "archive output appeared concurrently; refusing to replace it"
        )
    raise EvidenceError(
        "atomic no-replace archive installation failed: "
        f"{os.strerror(error) if error else 'unknown renameat2 error'}"
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either absolute path contains the other."""

    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def preflight_build_output(
    *, repository_root: Path, runs: Path, output: Path
) -> Path:
    """Reject output/source overlap without creating any filesystem object."""

    output_path = lexical_absolute(output)
    try:
        # strict=False follows only already-existing aliases and appends a
        # missing suffix.  This is read-only and catches a parent symlink into a
        # source tree before open_directory_chain could create below it.
        canonical_projection = output_path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError("cannot safely project archive output path") from exc
    for label, source in (("repository", repository_root), ("runs", runs)):
        for spelling in (output_path, canonical_projection):
            if _paths_overlap(spelling, source):
                raise EvidenceError(
                    f"archive output must be disjoint from the {label} source tree"
                )
    return output_path


def build_archive(
    *,
    repository_root: Path,
    runs: Path,
    output: Path,
    repo_id: str,
    prefix: str,
    launch_commit: str = LAUNCH_COMMIT,
    _pinned_runs: PinnedEvidenceRoot | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build atomically from a stable source tree, then fully reverify it."""

    if launch_commit != LAUNCH_COMMIT:
        raise EvidenceError(f"v4 publication pins launch commit {LAUNCH_COMMIT}")
    repository_root = repository_root.expanduser().resolve(strict=True)
    repository_descriptor = open_directory_chain(repository_root, create=False)
    os.close(repository_descriptor)
    if _pinned_runs is None:
        runs = runs.expanduser().resolve(strict=True)
        pinned_runs = pin_evidence_root(runs, resolve_once=False)
    else:
        runs = lexical_absolute(runs)
        if runs != _pinned_runs.path:
            raise EvidenceError("pinned runs root differs from the build input")
        current_runs = pin_evidence_root(runs, resolve_once=False)
        if (
            current_runs.files != _pinned_runs.files
            or current_runs.directories != _pinned_runs.directories
        ):
            raise EvidenceError("run source changed after workflow preflight")
        pinned_runs = _pinned_runs
    output = preflight_build_output(
        repository_root=repository_root,
        runs=runs,
        output=output,
    )
    with open_parent_directory(output, create=True) as (output, output_parent, output_leaf):
        if _directory_entry_signature(output_parent, output_leaf) is not None:
            raise EvidenceError(
                f"archive output already exists; use `publish --bundle` to resume: {output}"
            )
        temporary_name, temporary_signature = create_owned_temporary_directory(
            output_parent, output_leaf
        )
        temporary = Path(f"/proc/self/fd/{output_parent}/{temporary_name}")
        completed = False
        try:
            source_relative = f"provenance/source-{launch_commit}.tar"
            source_size, source_digest = deterministic_git_archive(
                repository_root, launch_commit, temporary / source_relative
            )
            suite = _load_suite_from_source_archive(temporary / source_relative)
            if suite["suite_id"] != "current_budget_isoflop_v4":
                raise EvidenceError("launch source does not contain the v4 suite")
            declared = {str(point["id"]) for point in suite["all_variants"]}
            run_names, source_selections, before_inventory = discover_runs_source(
                runs, declared
            )
            before_directories: dict[str, tuple[int, ...]] = {}
            checked_before_inventory = scan_regular_tree(
                runs, directories=before_directories
            )
            if (
                checked_before_inventory != before_inventory
                or (
                    checked_before_inventory != pinned_runs.files
                    or before_directories != pinned_runs.directories
                )
            ):
                raise EvidenceError(
                    "run source changed after workflow preflight"
                )

            copied: list[dict[str, Any]] = []
            for relative in sorted(before_inventory):
                size, digest = copy_regular_once(
                    runs / relative, temporary / "runs" / relative
                )
                archive_path = f"runs/{relative}"
                copied.append(
                    {
                        "path": archive_path,
                        "bytes": size,
                        "sha256": digest,
                        "role": _role_for_archive_path(
                            archive_path, run_names, source_selections
                        ),
                    }
                )
            after_directories: dict[str, tuple[int, ...]] = {}
            after_inventory = scan_regular_tree(
                runs, directories=after_directories
            )
            if (
                before_inventory != after_inventory
                or before_directories != after_directories
                or (
                    after_inventory != pinned_runs.files
                    or after_directories != pinned_runs.directories
                )
            ):
                raise EvidenceError(
                    "run source tree changed while the archive was copied"
                )

            classifications = {"stable": [], "suspect": [], "rejected": []}
            for run in run_names:
                classification = _classification_from_admission(
                    temporary
                    / "runs"
                    / run
                    / "artifacts"
                    / "stability-admission.json",
                    run,
                )
                classifications[classification].append(run)
            for values in classifications.values():
                values.sort()

            identity = {
                "suite_id": suite["suite_id"],
                "suite_sha256": suite["suite_sha256"],
                "template_sha256": suite["template_sha256"],
                "execution_fingerprint": suite["execution_fingerprint"],
                "trainer_sha256": suite["trainer_source_sha256"],
                "seed": suite["seed"],
                "launch_commit": launch_commit,
            }
            write_new(temporary / "README.md", archive_readme(identity))
            verifier_source = Path(__file__).resolve(strict=True)
            copy_regular_once(verifier_source, temporary / "verify.py")

            for path, role in (
                ("README.md", "documentation"),
                ("verify.py", "verifier"),
                (source_relative, "source_archive"),
            ):
                size, digest = hash_regular_file(temporary / path)
                copied.append(
                    {"path": path, "bytes": size, "sha256": digest, "role": role}
                )
            copied.sort(key=lambda item: str(item["path"]))
            archive_id = (
                f"{suite['suite_id']}-{inventory_digest(copied)[:16]}-"
                f"{launch_commit[:12]}"
            )

            staged_fit = read_json_regular(
                temporary / "runs" / "fit.json", "runs/fit.json"
            )
            if staged_fit.get("can_estimate_scaling_exponent") is not True:
                raise EvidenceError("stored fit cannot estimate a scaling exponent")
            selection_paths = [f"runs/{path}" for path in source_selections]
            study = {
                "runs": run_names,
                "classifications": classifications,
                "learning_rate_selections": selection_paths,
                "fit_paths": {
                    "final_json": "runs/fit.json",
                    "final_markdown": "runs/fit.md",
                    "slices": [
                        "runs/fits/c025.json",
                        "runs/fits/c050.json",
                        "runs/fits/c100.json",
                    ],
                },
                "can_estimate_scaling_exponent": True,
            }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "kind": "gpt_tpu_speedrun_scaling_evidence",
                "archive_id": archive_id,
                "identity": identity,
                "publication_target": {
                    "repository": validate_repo_id(repo_id),
                    "directory": f"{validate_prefix(prefix)}/{archive_id}",
                },
                "source_archive": {
                    "path": source_relative,
                    "commit": launch_commit,
                    "prefix": SOURCE_PREFIX,
                    "bytes": source_size,
                    "sha256": source_digest,
                },
                "study": study,
                "inventory": {
                    "file_count": len(copied),
                    "total_bytes": sum(int(item["bytes"]) for item in copied),
                    "files": copied,
                },
            }
            validate_manifest(manifest)
            write_new(temporary / MANIFEST_NAME, canonical_json_bytes(manifest))
            verify_bundle(temporary)
            final_directories: dict[str, tuple[int, ...]] = {}
            final_inventory = scan_regular_tree(
                runs, directories=final_directories
            )
            if (
                final_inventory != pinned_runs.files
                or final_directories != pinned_runs.directories
            ):
                raise EvidenceError(
                    "run source changed before atomic archive installation"
                )
            if _directory_entry_signature(output_parent, output_leaf) is not None:
                raise EvidenceError(
                    "archive output appeared while the bundle was being built"
                )
            rename_directory_noreplace(
                output_parent,
                temporary_name,
                output_parent,
                output_leaf,
            )
            installed_signature = _directory_entry_signature(
                output_parent, output_leaf
            )
            if (
                installed_signature is None
                or installed_signature[:2] != temporary_signature[:2]
                or not stat.S_ISDIR(installed_signature[2])
                or installed_signature[4] != temporary_signature[4]
            ):
                raise EvidenceError(
                    "atomically installed archive is not the owned temporary root"
                )
            os.fsync(output_parent)
            completed = True
            return output, manifest
        finally:
            if not completed:
                cleanup_owned_temporary_directory(
                    output_parent, temporary_name, temporary_signature
                )


def _remote_item_kind(item: Any) -> str:
    value = getattr(item, "type", None)
    if isinstance(value, str):
        return value
    return item.__class__.__name__.lower()


def hf_commit_oid(value: Any, label: str) -> str:
    if not isinstance(value, str) or HF_COMMIT_PATTERN.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be an exact lowercase 40-hex commit OID")
    return value


def remote_closed_tree(api: Any, *, repo_id: str, revision: str, directory: str) -> dict[str, int | None]:
    """List an immutable HF subtree anonymously and require files only at leaves."""

    prefix = directory.rstrip("/") + "/"
    try:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=directory,
            recursive=True,
            expand=True,
            revision=revision,
            repo_type="dataset",
            token=False,
        )
        found: dict[str, int | None] = {}
        for item in entries:
            path = getattr(item, "path", None)
            if not isinstance(path, str) or not path.startswith(prefix):
                if path == directory and "folder" in _remote_item_kind(item):
                    continue
                raise EvidenceError("anonymous remote tree returned an out-of-prefix object")
            relative = normalized_path(path.removeprefix(prefix), "remote tree path")
            kind = _remote_item_kind(item)
            if "folder" in kind or "directory" in kind or kind in {"tree", "dir"}:
                continue
            if "file" not in kind and kind not in {"blob"}:
                raise EvidenceError(f"anonymous remote tree has unsupported object: {path}")
            size = getattr(item, "size", None)
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise EvidenceError(f"anonymous remote file has invalid size: {path}")
            if relative in found:
                raise EvidenceError(f"anonymous remote tree duplicates {relative}")
            found[relative] = size
        return dict(sorted(found.items()))
    except EvidenceError:
        raise
    except Exception as exc:
        raise EvidenceError(f"cannot list anonymous immutable remote tree: {exc}") from exc


def _download_url(repo_id: str, revision: str, path: str) -> str:
    quoted_repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    quoted_path = "/".join(quote(part, safe="") for part in path.split("/"))
    return f"https://huggingface.co/datasets/{quoted_repo}/resolve/{revision}/{quoted_path}"


def anonymous_download(
    url: str, destination: Path, *, expected_bytes: int
) -> tuple[int, str]:
    """Stream one public object without an authorization header."""

    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or expected_bytes > MAX_ARCHIVE_BYTES
    ):
        raise EvidenceError("anonymous object has an invalid byte bound")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise EvidenceError(f"refusing to overwrite anonymous download: {destination}")
    last_error: BaseException | None = None
    for attempt in range(HTTP_ATTEMPTS):
        digest = hashlib.sha256()
        total = 0
        temporary = destination.with_name(
            f".{destination.name}.part-{secrets.token_hex(16)}"
        )
        temporary_signature: tuple[int, ...] | None = None
        installed = False

        def clean_owned(path: Path) -> None:
            if temporary_signature is None:
                return
            try:
                current = _stat_signature(path.lstat())
            except FileNotFoundError:
                return
            if (
                current[:2] == temporary_signature[:2]
                and stat.S_ISREG(current[2])
                and current[4] == temporary_signature[4]
            ):
                path.unlink()

        request = Request(url, headers={"User-Agent": "gpt-tpu-scaling-evidence/1"})
        if request.has_header("Authorization"):
            raise EvidenceError("anonymous verification request unexpectedly has authorization")
        try:
            with urlopen(request, timeout=300) as response, temporary.open("xb") as output:
                temporary_signature = _stat_signature(os.fstat(output.fileno()))
                while True:
                    chunk = response.read(READ_CHUNK)
                    if not chunk:
                        break
                    if total + len(chunk) > expected_bytes:
                        raise EvidenceError(
                            "anonymous object exceeds its manifest byte bound"
                        )
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if total != expected_bytes:
                raise EvidenceError(
                    "anonymous object length differs from its manifest byte count"
                )
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise EvidenceError(
                    f"anonymous download target appeared concurrently: {destination}"
                ) from exc
            installed = True
            temporary.unlink()
            temporary_signature = None
            return total, digest.hexdigest()
        except EvidenceError:
            if installed:
                clean_owned(destination)
            clean_owned(temporary)
            raise
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if installed:
                clean_owned(destination)
            clean_owned(temporary)
            if attempt + 1 < HTTP_ATTEMPTS:
                time.sleep(min(2**attempt, 8))
    raise EvidenceError(f"anonymous download failed for {url}: {last_error}")


def anonymous_revalidate(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    directory: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Download the exact public immutable tree, hash every byte, and recompute."""

    manifest = validate_manifest(manifest)
    revision = hf_commit_oid(revision, "anonymous verification revision")
    try:
        info = api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=False,
        )
    except Exception as exc:
        raise EvidenceError(
            "cannot resolve the anonymous immutable revision"
        ) from exc
    if getattr(info, "sha", None) != revision:
        raise EvidenceError(
            "anonymous immutable revision resolved to a different commit"
        )
    expected = {MANIFEST_NAME: None, **{path: item for path, item in _inventory_index(manifest).items()}}
    remote = remote_closed_tree(
        api, repo_id=repo_id, revision=revision, directory=directory
    )
    if set(remote) != set(expected):
        raise EvidenceError(
            "anonymous immutable tree is not closed: "
            f"missing={sorted(set(expected) - set(remote))[:5]!r}, "
            f"extra={sorted(set(remote) - set(expected))[:5]!r}"
        )
    manifest_bytes = canonical_json_bytes(manifest)
    expected_details = {
        MANIFEST_NAME: {"bytes": len(manifest_bytes), "sha256": sha256_bytes(manifest_bytes)},
        **_inventory_index(manifest),
    }
    remote_total = 0
    for path, remote_size in remote.items():
        if remote_size != expected_details[path]["bytes"]:
            raise EvidenceError(f"anonymous remote size differs for {path}")
        remote_total += remote_size
    if remote_total > MAX_ARCHIVE_BYTES + MAX_MANIFEST_BYTES:
        raise EvidenceError("anonymous remote tree exceeds its total byte bound")
    with tempfile.TemporaryDirectory(prefix="scaling-evidence-anonymous-") as directory_name:
        download_root = Path(directory_name) / "archive"
        download_root.mkdir()
        for relative in sorted(expected_details):
            remote_path = f"{directory}/{relative}"
            size, digest = anonymous_download(
                _download_url(repo_id, revision, remote_path),
                download_root / relative,
                expected_bytes=int(expected_details[relative]["bytes"]),
            )
            details = expected_details[relative]
            if size != details["bytes"] or digest != details["sha256"]:
                raise EvidenceError(f"anonymous remote SHA-256 differs for {relative}")
        verification = verify_bundle(download_root)
    return verification


def publication_receipt(
    *,
    manifest: Mapping[str, Any],
    revision: str,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = validate_manifest(manifest)
    manifest_bytes = canonical_json_bytes(manifest)
    flags = _mapping(verification.get("semantic_verification"), "semantic verification")
    if flags != {flag: True for flag in SEMANTIC_FLAGS}:
        raise EvidenceError("anonymous semantic verification is incomplete")
    inventory = _mapping(manifest["inventory"], "inventory")
    return {
        "schema_version": 1,
        "kind": "gpt_tpu_speedrun_scaling_evidence_receipt",
        "archive_id": manifest["archive_id"],
        "identity": manifest["identity"],
        "publication": {
            "repository": manifest["publication_target"]["repository"],
            "revision": hf_commit_oid(revision, "publication revision"),
            "directory": manifest["publication_target"]["directory"],
            "url": (
                "https://huggingface.co/datasets/"
                f"{manifest['publication_target']['repository']}/tree/{revision}/"
                f"{manifest['publication_target']['directory']}"
            ),
        },
        "archive_manifest": {
            "path": f"{manifest['publication_target']['directory']}/{MANIFEST_NAME}",
            "bytes": len(manifest_bytes),
            "sha256": sha256_bytes(manifest_bytes),
        },
        "inventory": {
            "file_count": inventory["file_count"],
            "total_bytes": inventory["total_bytes"],
            "canonical_sha256": inventory_digest(inventory["files"]),
            "files": inventory["files"],
        },
        "anonymous_verification": {
            "immutable_revision": True,
            "archive_manifest": True,
            "closed_tree_inventory": True,
            "all_file_sha256": True,
            "semantic_revalidation": True,
            **flags,
        },
    }


def _read_directory_entry_once(
    parent: int,
    leaf: str,
    signature: tuple[int, ...],
    *,
    maximum_bytes: int,
) -> bytes:
    """Read one already-lstat'd leaf and prove its identity stayed fixed."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(leaf, flags, dir_fd=parent)
    except OSError as exc:
        raise EvidenceError(f"cannot safely open existing receipt {leaf!r}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            _stat_signature(before) != signature
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_size > maximum_bytes
        ):
            raise EvidenceError("existing receipt is not one bounded current-user file")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK, remaining))
            if not chunk:
                raise EvidenceError("existing receipt was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise EvidenceError("existing receipt grew while reading")
        after = os.fstat(descriptor)
        if (
            _stat_signature(after) != signature
            or _directory_entry_signature(parent, leaf) != signature
        ):
            raise EvidenceError("existing receipt changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _existing_receipt_is_identical(parent: int, leaf: str, payload: bytes) -> bool:
    signature = _directory_entry_signature(parent, leaf)
    if signature is None:
        return False
    if (
        not stat.S_ISREG(signature[2])
        or signature[3] != 1
        or signature[4] != os.getuid()
    ):
        raise EvidenceError("receipt output exists but is not one current-user file")
    existing = _read_directory_entry_once(
        parent, leaf, signature, maximum_bytes=MAX_MANIFEST_BYTES
    )
    if existing != payload:
        raise EvidenceError(
            "existing receipt differs; refusing to overwrite immutable evidence"
        )
    return True


def write_atomic(
    path: Path,
    payload: bytes,
    *,
    expected_parent_identity: tuple[int, int] | None = None,
) -> None:
    """Install one immutable receipt, or accept an identical existing receipt."""

    if len(payload) > MAX_MANIFEST_BYTES:
        raise EvidenceError("receipt payload exceeds the bounded receipt size")
    with open_parent_directory(path, create=True) as (_absolute, parent, leaf):
        parent_metadata = os.fstat(parent)
        parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
        if (
            expected_parent_identity is not None
            and parent_identity != expected_parent_identity
        ):
            raise EvidenceError("receipt parent changed after publication preflight")
        if _existing_receipt_is_identical(parent, leaf, payload):
            return
        temporary = f".{leaf}.tmp-{secrets.token_hex(16)}"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor: int | None = None
        temporary_signature: tuple[int, ...] | None = None

        def clean_owned_temporary() -> None:
            if temporary_signature is None:
                return
            current = _directory_entry_signature(parent, temporary)
            if current is None:
                return
            if (
                current[:2] != temporary_signature[:2]
                or not stat.S_ISREG(current[2])
                or current[4] != temporary_signature[4]
            ):
                raise EvidenceError(
                    "refusing to clean a receipt temporary whose identity changed"
                )
            os.unlink(temporary, dir_fd=parent)

        try:
            descriptor = os.open(temporary, flags, 0o600, dir_fd=parent)
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:  # pragma: no cover - POSIX writes either progress or fail
                    raise EvidenceError("receipt temporary file made no write progress")
                written += count
            os.fsync(descriptor)
            temporary_signature = _stat_signature(os.fstat(descriptor))
            os.close(descriptor)
            descriptor = None
            if _directory_entry_signature(parent, temporary) != temporary_signature:
                raise EvidenceError("receipt temporary changed before installation")
            try:
                # link(2) has atomic no-replace semantics for the destination.
                os.link(
                    temporary,
                    leaf,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                if not _existing_receipt_is_identical(parent, leaf, payload):
                    raise EvidenceError("receipt appeared without an identical payload")
            else:
                installed = _directory_entry_signature(parent, leaf)
                if (
                    installed is None
                    or installed[:2] != temporary_signature[:2]
                    or not stat.S_ISREG(installed[2])
                    or installed[4] != os.getuid()
                ):
                    raise EvidenceError("installed receipt identity differs from temporary")
            clean_owned_temporary()
            temporary_signature = None
            os.fsync(parent)
        except (EvidenceError, OSError) as exc:
            if descriptor is not None:
                os.close(descriptor)
            try:
                clean_owned_temporary()
            except (EvidenceError, OSError) as cleanup_exc:
                exc.add_note(f"temporary receipt cleanup also failed: {cleanup_exc}")
            if isinstance(exc, EvidenceError):
                raise
            raise EvidenceError(f"cannot atomically install receipt output: {exc}") from exc


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class PinnedEvidenceRoot:
    path: Path
    files: Mapping[str, tuple[int, ...]]
    directories: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class PublicationPathState:
    bundle: PinnedEvidenceRoot
    evidence_roots: tuple[PinnedEvidenceRoot, ...]
    token_file: Path
    token_signature: tuple[int, ...]
    token_parent_identity: tuple[int, int]
    receipt_output: Path
    receipt_parent_identity: tuple[int, int]


def pin_evidence_root(path: Path, *, resolve_once: bool = True) -> PinnedEvidenceRoot:
    """Capture one closed source-tree identity for later exact revalidation."""

    root = (
        path.expanduser().resolve(strict=True)
        if resolve_once
        else lexical_absolute(path)
    )
    descriptor = open_directory_chain(root, create=False)
    os.close(descriptor)
    directories: dict[str, tuple[int, ...]] = {}
    files = scan_regular_tree(root, directories=directories)
    return PinnedEvidenceRoot(
        path=root,
        files=dict(files),
        directories=dict(directories),
    )


def _require_disjoint_roots(roots: Sequence[PinnedEvidenceRoot]) -> None:
    """Reject containment or inode aliasing among bundle/evidence trees."""

    for index, left in enumerate(roots):
        left_inodes = {
            (signature[0], signature[1])
            for signature in (*left.files.values(), *left.directories.values())
        }
        for right in roots[index + 1 :]:
            if _paths_overlap(left.path, right.path):
                raise EvidenceError("bundle/evidence roots must be disjoint")
            right_inodes = {
                (signature[0], signature[1])
                for signature in (*right.files.values(), *right.directories.values())
            }
            if left_inodes & right_inodes:
                raise EvidenceError("bundle/evidence roots must not contain inode aliases")


def _inspect_publication_aliases(
    *,
    roots: Sequence[PinnedEvidenceRoot],
    token_path: Path,
    receipt_path: Path,
) -> tuple[tuple[int, ...], tuple[int, int], tuple[int, int]]:
    """Recheck leaf safety and cross-tree inode aliases using pinned roots."""

    root_file_inodes = {
        (signature[0], signature[1])
        for root in roots
        for signature in root.files.values()
    }
    root_directory_inodes = {
        (signature[0], signature[1])
        for root in roots
        for signature in root.directories.values()
    }
    for root in roots:
        if _path_is_within(token_path, root.path):
            raise EvidenceError("token path must be disjoint from bundle/evidence roots")
        if _path_is_within(receipt_path, root.path):
            raise EvidenceError("receipt path must be disjoint from bundle/evidence roots")
    if token_path == receipt_path:
        raise EvidenceError("receipt path must not alias the token path")

    identities: dict[str, tuple[int, ...] | None] = {}
    parent_identities: dict[str, tuple[int, int]] = {}
    for label, path in (("token", token_path), ("receipt", receipt_path)):
        with open_parent_directory(path, create=(label == "receipt")) as (
            _absolute,
            parent,
            leaf,
        ):
            parent_metadata = os.fstat(parent)
            parent_identities[label] = (
                parent_metadata.st_dev,
                parent_metadata.st_ino,
            )
            identities[label] = _directory_entry_signature(parent, leaf)
    token_identity = identities["token"]
    if token_identity is None or not stat.S_ISREG(token_identity[2]):
        raise EvidenceError("token path must name an existing regular file")
    if (
        token_identity[3] != 1
        or token_identity[4] != os.getuid()
        or stat.S_IMODE(token_identity[2]) != 0o600
        or token_identity[6] > MAX_TOKEN_BYTES
    ):
        raise EvidenceError(
            "token preflight requires one current-user regular file with exact "
            f"mode 0600 and at most {MAX_TOKEN_BYTES} bytes"
        )
    receipt_identity = identities["receipt"]
    if receipt_identity is not None and (
        not stat.S_ISREG(receipt_identity[2])
        or receipt_identity[3] != 1
        or receipt_identity[4] != os.getuid()
    ):
        raise EvidenceError("existing receipt must be one current-user regular file")
    if (token_identity[0], token_identity[1]) in root_file_inodes:
        raise EvidenceError("token file aliases a bundle/evidence object")
    if receipt_identity is not None and (
        receipt_identity[0],
        receipt_identity[1],
    ) in root_file_inodes:
        raise EvidenceError("receipt file aliases a bundle/evidence object")
    if receipt_identity is not None and (
        receipt_identity[0],
        receipt_identity[1],
    ) == (token_identity[0], token_identity[1]):
        raise EvidenceError("receipt file aliases the token file")
    if parent_identities["receipt"] in root_directory_inodes:
        raise EvidenceError("receipt parent aliases a bundle/evidence directory")
    return (
        token_identity,
        parent_identities["token"],
        parent_identities["receipt"],
    )


def validate_publication_paths(
    *,
    bundle: Path,
    token_file: Path,
    receipt_output: Path,
    evidence_roots: Sequence[Path] = (),
    pinned_evidence_roots: Sequence[PinnedEvidenceRoot] = (),
) -> PublicationPathState:
    """Resolve inputs once and pin every tree/token identity before credentials."""

    try:
        bundle_path = bundle.expanduser().resolve(strict=True)
        resolved_evidence_roots = [
            path.expanduser().resolve(strict=True) for path in evidence_roots
        ]
    except OSError as exc:
        raise EvidenceError(
            "bundle/evidence input root cannot be resolved once to a directory"
        ) from exc
    token_path = lexical_absolute(token_file)
    receipt_path = lexical_absolute(receipt_output)
    pinned_roots: list[PinnedEvidenceRoot] = [
        pin_evidence_root(bundle_path, resolve_once=False),
        *(pin_evidence_root(path, resolve_once=False) for path in resolved_evidence_roots),
    ]
    for expected in pinned_evidence_roots:
        fixed_path = lexical_absolute(expected.path)
        current = pin_evidence_root(fixed_path, resolve_once=False)
        if (
            current.files != expected.files
            or current.directories != expected.directories
        ):
            raise EvidenceError(
                f"pinned evidence root changed after workflow preflight: {fixed_path}"
            )
        pinned_roots.append(expected)
    _require_disjoint_roots(pinned_roots)
    (
        token_signature,
        token_parent_identity,
        receipt_parent_identity,
    ) = _inspect_publication_aliases(
        roots=pinned_roots,
        token_path=token_path,
        receipt_path=receipt_path,
    )
    return PublicationPathState(
        bundle=pinned_roots[0],
        evidence_roots=tuple(pinned_roots[1:]),
        token_file=token_path,
        token_signature=token_signature,
        token_parent_identity=token_parent_identity,
        receipt_output=receipt_path,
        receipt_parent_identity=receipt_parent_identity,
    )


def revalidate_publication_paths(state: PublicationPathState) -> None:
    """Require pinned source trees/token to remain exact; recheck receipt aliases."""

    roots = (state.bundle, *state.evidence_roots)
    for root in roots:
        directories: dict[str, tuple[int, ...]] = {}
        files = scan_regular_tree(root.path, directories=directories)
        if files != root.files or directories != root.directories:
            raise EvidenceError(
                f"pinned publication input changed after preflight: {root.path}"
            )
    (
        token_signature,
        token_parent_identity,
        receipt_parent_identity,
    ) = _inspect_publication_aliases(
        roots=roots,
        token_path=state.token_file,
        receipt_path=state.receipt_output,
    )
    if (
        token_signature != state.token_signature
        or token_parent_identity != state.token_parent_identity
    ):
        raise EvidenceError("token identity changed after publication preflight")
    if receipt_parent_identity != state.receipt_parent_identity:
        raise EvidenceError("receipt parent identity changed after publication preflight")


def exact_snapshot_inventory(
    snapshot: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    expected_details = {
        MANIFEST_NAME: {
            "bytes": len(canonical_json_bytes(manifest)),
            "sha256": sha256_bytes(canonical_json_bytes(manifest)),
        },
        **_inventory_index(manifest),
    }
    directories: dict[str, tuple[int, ...]] = {}
    files = scan_regular_tree(snapshot, directories=directories)
    if set(files) != set(expected_details):
        raise EvidenceError("retained upload snapshot is not the exact manifest allowlist")
    total = 0
    for relative, metadata in files.items():
        if metadata[6] != expected_details[relative]["bytes"]:
            raise EvidenceError(f"retained snapshot size differs for {relative}")
        total += metadata[6]
    if total > MAX_ARCHIVE_BYTES + MAX_MANIFEST_BYTES:
        raise EvidenceError("retained upload snapshot exceeds its total byte bound")
    if set(directories) != expected_parent_directories(expected_details):
        raise EvidenceError("retained upload snapshot directory tree is not closed")
    return files, directories


def open_verified_upload_object(
    *,
    snapshot: Path,
    relative: str,
    expected: Mapping[str, Any],
    sealed_signature: tuple[int, ...],
) -> tuple[Any, tuple[int, ...]]:
    """Open and hash one sealed upload object without following its leaf."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(snapshot / relative, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open retained upload object: {relative}") from exc
    handle = os.fdopen(descriptor, "rb", closefd=True)
    try:
        before = os.fstat(descriptor)
        before_signature = _stat_signature(before)
        if (
            before_signature != sealed_signature
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected["bytes"]
        ):
            raise EvidenceError(
                f"retained upload object changed before commit: {relative}"
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, READ_CHUNK)
            if not chunk:
                break
            if total + len(chunk) > expected["bytes"]:
                raise EvidenceError(
                    f"retained upload object exceeds its byte bound: {relative}"
                )
            digest.update(chunk)
            total += len(chunk)
        after_hash = os.fstat(descriptor)
        if (
            _stat_signature(after_hash) != before_signature
            or total != expected["bytes"]
            or digest.hexdigest() != expected["sha256"]
        ):
            raise EvidenceError(
                f"retained upload object differs from its manifest: {relative}"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return handle, before_signature
    except BaseException:
        handle.close()
        raise


def publish_archive(
    *,
    bundle: Path,
    token_file: Path,
    receipt_output: Path,
    expected_manifest: Mapping[str, Any] | None = None,
    evidence_roots: Sequence[Path] = (),
    pinned_evidence_roots: Sequence[PinnedEvidenceRoot] = (),
) -> dict[str, Any]:
    """Verify one retained snapshot before token access and upload exactly it."""

    path_state = validate_publication_paths(
        bundle=bundle,
        token_file=token_file,
        receipt_output=receipt_output,
        evidence_roots=evidence_roots,
        pinned_evidence_roots=pinned_evidence_roots,
    )
    with tempfile.TemporaryDirectory(prefix="scaling-evidence-upload-") as name:
        upload_snapshot = Path(name) / "archive"
        manifest = snapshot_bundle(
            path_state.bundle.path,
            upload_snapshot,
            pinned_files=path_state.bundle.files,
            pinned_directories=path_state.bundle.directories,
        )
        verify_frozen_snapshot(
            upload_snapshot,
            manifest,
            expected_manifest=expected_manifest,
        )
        set_snapshot_permissions(upload_snapshot, sealed=True)
        sealed_files, sealed_directories = exact_snapshot_inventory(
            upload_snapshot, manifest
        )
        # Credential access is deliberately after the sole retained upload
        # snapshot has passed byte, identity, source-stability, and semantic gates.
        try:
            revalidate_publication_paths(path_state)
            token = read_token_file(
                path_state.token_file,
                expected_signature=path_state.token_signature,
                expected_parent_identity=path_state.token_parent_identity,
            )
            try:
                from huggingface_hub import CommitOperationAdd, HfApi
            except ImportError as exc:
                raise EvidenceError("huggingface-hub is required for publication") from exc
            repo_id = str(manifest["publication_target"]["repository"])
            directory = str(manifest["publication_target"]["directory"])
            api = HfApi(token=token)
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=False,
                    exist_ok=True,
                    token=token,
                )
                parent_info = api.repo_info(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision="main",
                    token=token,
                )
                parent_commit = hf_commit_oid(
                    getattr(parent_info, "sha", None), "upload parent commit"
                )
                remote_files = api.list_repo_files(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=parent_commit,
                    token=token,
                )
                if not isinstance(remote_files, list) or any(
                    not isinstance(path, str) for path in remote_files
                ):
                    raise EvidenceError("authenticated parent file inventory is malformed")
                collision = any(
                    path == directory or path.startswith(directory + "/")
                    for path in remote_files
                )
                if collision:
                    revision = parent_commit
                else:
                    expected_paths = [MANIFEST_NAME, *sorted(_inventory_index(manifest))]
                    manifest_bytes = canonical_json_bytes(manifest)
                    expected_details = {
                        MANIFEST_NAME: {
                            "bytes": len(manifest_bytes),
                            "sha256": sha256_bytes(manifest_bytes),
                        },
                        **_inventory_index(manifest),
                    }
                    with ExitStack() as handles:
                        operations = []
                        opened: list[tuple[str, Any, tuple[int, ...]]] = []
                        for relative in expected_paths:
                            handle, before_signature = open_verified_upload_object(
                                snapshot=upload_snapshot,
                                relative=relative,
                                expected=expected_details[relative],
                                sealed_signature=sealed_files[relative],
                            )
                            handles.callback(handle.close)
                            opened.append((relative, handle, before_signature))
                            operations.append(
                                CommitOperationAdd(
                                    path_in_repo=f"{directory}/{relative}",
                                    path_or_fileobj=handle,
                                )
                            )
                        result = api.create_commit(
                            repo_id=repo_id,
                            repo_type="dataset",
                            operations=operations,
                            parent_commit=parent_commit,
                            revision="main",
                            commit_message=(
                                "Publish immutable scaling evidence "
                                f"{manifest['archive_id']}"
                            ),
                            token=token,
                        )
                        for relative, handle, before_signature in opened:
                            try:
                                after_signature = _stat_signature(
                                    os.fstat(handle.fileno())
                                )
                            except (OSError, ValueError) as exc:
                                raise EvidenceError(
                                    "upload client closed or invalidated retained "
                                    f"object: {relative}"
                                ) from exc
                            if after_signature != before_signature:
                                raise EvidenceError(
                                    "retained upload object changed during commit: "
                                    f"{relative}"
                                )
                    revision = hf_commit_oid(
                        getattr(result, "oid", None), "uploaded immutable revision"
                    )
            except Exception as exc:
                raise EvidenceError(
                    "authenticated Hugging Face repository creation/upload failed"
                ) from exc
        finally:
            current_files, current_directories = exact_snapshot_inventory(
                upload_snapshot, manifest
            )
            if (
                current_files != sealed_files
                or current_directories != sealed_directories
            ):
                set_snapshot_permissions(upload_snapshot, sealed=False)
                raise EvidenceError("retained upload snapshot changed during upload")
            # TemporaryDirectory can now remove only the snapshot we created.
            set_snapshot_permissions(upload_snapshot, sealed=False)
    anonymous_api = HfApi(token=False)
    verification = anonymous_revalidate(
        anonymous_api,
        repo_id=repo_id,
        revision=revision,
        directory=directory,
        manifest=manifest,
    )
    receipt = publication_receipt(
        manifest=manifest, revision=revision, verification=verification
    )
    revalidate_publication_paths(path_state)
    write_atomic(
        path_state.receipt_output,
        canonical_json_bytes(receipt),
        expected_parent_identity=path_state.receipt_parent_identity,
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, verify, and publish closed v4 scaling evidence"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    build.add_argument("--runs", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--repo-id", default=DEFAULT_REPOSITORY)
    build.add_argument("--prefix", default=DEFAULT_PREFIX)
    build.add_argument("--launch-commit", default=LAUNCH_COMMIT)
    build.add_argument("--token-file", type=Path)
    build.add_argument("--receipt-output", type=Path, default=DEFAULT_RECEIPT)
    build.add_argument(
        "--dry-run",
        action="store_true",
        help="build and fully verify locally; never read a token or use the network",
    )
    publish = commands.add_parser(
        "publish",
        help="resume publication from one already-built bundle after full revalidation",
    )
    publish.add_argument("--bundle", type=Path, required=True)
    publish.add_argument("--token-file", type=Path, required=True)
    publish.add_argument("--receipt-output", type=Path, default=DEFAULT_RECEIPT)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "verify":
            print(json.dumps(verify_bundle(args.bundle), indent=2, sort_keys=True))
            return 0
        if args.command == "publish":
            receipt = publish_archive(
                bundle=args.bundle,
                token_file=args.token_file,
                receipt_output=args.receipt_output,
            )
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0
        # A combined build+publish resolves and inventories the runs root once
        # before building.  The same canonical identity is required by both the
        # builder and the pre-credential publication gate; retargeting the
        # original CLI symlink can never select a second tree.
        workflow_runs = (
            None if args.dry_run else pin_evidence_root(args.runs)
        )
        build_runs = args.runs if workflow_runs is None else workflow_runs.path
        bundle, manifest = build_archive(
            repository_root=args.repository_root,
            runs=build_runs,
            output=args.output,
            repo_id=args.repo_id,
            prefix=args.prefix,
            launch_commit=args.launch_commit,
            _pinned_runs=workflow_runs,
        )
        plan = {
            "archive": str(bundle),
            "archive_id": manifest["archive_id"],
            "repository": manifest["publication_target"]["repository"],
            "directory": manifest["publication_target"]["directory"],
            "files": manifest["inventory"]["file_count"] + 1,
            "bytes_excluding_manifest": manifest["inventory"]["total_bytes"],
            "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
            "dry_run": bool(args.dry_run),
        }
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if args.token_file is None:
            raise EvidenceError("--token-file is required unless --dry-run is used")
        receipt = publish_archive(
            bundle=bundle,
            token_file=args.token_file,
            receipt_output=args.receipt_output,
            expected_manifest=manifest,
            pinned_evidence_roots=(workflow_runs,),
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (EvidenceError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
