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
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
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
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
READ_CHUNK = 1024 * 1024
HTTP_ATTEMPTS = 5

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
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", value
    ):
        raise EvidenceError(f"invalid Hugging Face repository id: {value!r}")
    return value


def validate_prefix(value: str) -> str:
    normalized_path(value, "publication prefix")
    if any(SAFE_COMPONENT.fullmatch(part) is None for part in value.split("/")):
        raise EvidenceError("publication prefix has an unsafe component")
    return value.rstrip("/")


def read_token_file(path: Path) -> str:
    """Read exactly one mode-0600 HF_TOKEN assignment without shell parsing."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"cannot inspect Hugging Face token file {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise EvidenceError("Hugging Face token path must be a regular file, not a link")
    if metadata.st_uid != os.getuid():
        raise EvidenceError("Hugging Face token file must be owned by the current user")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise EvidenceError("Hugging Face token file must have mode 0600 or stricter")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"cannot read Hugging Face token file: {exc}") from exc
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
    if not isinstance(files, list) or not files:
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
        if not isinstance(item["role"], str) or not item["role"]:
            raise EvidenceError(f"inventory.files[{index}].role must be nonempty")
    if paths != sorted(set(paths)):
        raise EvidenceError("inventory files must be sorted by unique path")
    if inventory["total_bytes"] != total_bytes:
        raise EvidenceError("inventory.total_bytes differs from file records")

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


def snapshot_bundle(bundle: Path, destination: Path) -> dict[str, Any]:
    """Read each bundle object once into a private verified snapshot."""

    bundle = bundle.expanduser().resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise EvidenceError(f"snapshot destination already exists: {destination}")
    destination.mkdir(parents=True)
    source_directories: dict[str, tuple[int, ...]] = {}
    source_inventory = scan_regular_tree(bundle, directories=source_directories)
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

        for relative in study["learning_rate_selections"]:
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


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Verify a local/archive download through a private read-once snapshot."""

    with tempfile.TemporaryDirectory(prefix="scaling-evidence-snapshot-") as directory:
        snapshot = Path(directory) / "archive"
        manifest = snapshot_bundle(bundle, snapshot)
        before_directories: dict[str, tuple[int, ...]] = {}
        before_files = scan_regular_tree(snapshot, directories=before_directories)
        flags = semantic_verify(snapshot, manifest)
        after_directories: dict[str, tuple[int, ...]] = {}
        after_files = scan_regular_tree(snapshot, directories=after_directories)
        if before_files != after_files or before_directories != after_directories:
            raise EvidenceError("semantic recomputation changed the closed archive tree")
        return {
            "archive_id": manifest["archive_id"],
            "manifest_sha256": sha256_bytes(read_regular_once(snapshot / MANIFEST_NAME)),
            "inventory_sha256": inventory_digest(manifest["inventory"]["files"]),
            "file_count": manifest["inventory"]["file_count"] + 1,
            "semantic_verification": flags,
        }


def deterministic_git_archive(repository: Path, commit: str, destination: Path) -> tuple[int, str]:
    """Write the exact deterministic ``git archive`` byte stream for a commit."""

    repository = repository.expanduser().resolve(strict=True)
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

    runs = runs.expanduser().resolve(strict=True)
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


def build_archive(
    *,
    repository_root: Path,
    runs: Path,
    output: Path,
    repo_id: str,
    prefix: str,
    launch_commit: str = LAUNCH_COMMIT,
) -> tuple[Path, dict[str, Any]]:
    """Build atomically from a stable source tree, then fully reverify it."""

    if launch_commit != LAUNCH_COMMIT:
        raise EvidenceError(f"v4 publication pins launch commit {LAUNCH_COMMIT}")
    repository_root = repository_root.expanduser().resolve(strict=True)
    runs = runs.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise EvidenceError(f"archive output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        source_relative = f"provenance/source-{launch_commit}.tar"
        source_size, source_digest = deterministic_git_archive(
            repository_root, launch_commit, temporary / source_relative
        )
        suite = _load_suite_from_source_archive(temporary / source_relative)
        if suite["suite_id"] != "current_budget_isoflop_v4":
            raise EvidenceError("launch source does not contain the v4 suite")
        declared = {str(point["id"]) for point in suite["all_variants"]}
        run_names, source_selections, before_inventory = discover_runs_source(runs, declared)
        before_directories: dict[str, tuple[int, ...]] = {}
        checked_before_inventory = scan_regular_tree(runs, directories=before_directories)
        if checked_before_inventory != before_inventory:
            raise EvidenceError("run source changed during initial inventory validation")

        copied: list[dict[str, Any]] = []
        for relative in sorted(before_inventory):
            size, digest = copy_regular_once(runs / relative, temporary / "runs" / relative)
            archive_path = f"runs/{relative}"
            copied.append(
                {
                    "path": archive_path,
                    "bytes": size,
                    "sha256": digest,
                    "role": _role_for_archive_path(archive_path, run_names, source_selections),
                }
            )
        after_directories: dict[str, tuple[int, ...]] = {}
        after_inventory = scan_regular_tree(runs, directories=after_directories)
        if before_inventory != after_inventory or before_directories != after_directories:
            raise EvidenceError("run source tree changed while the archive was copied")

        classifications = {"stable": [], "suspect": [], "rejected": []}
        for run in run_names:
            classification = _classification_from_admission(
                temporary / "runs" / run / "artifacts" / "stability-admission.json",
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
            copied.append({"path": path, "bytes": size, "sha256": digest, "role": role})
        copied.sort(key=lambda item: str(item["path"]))
        archive_id = (
            f"{suite['suite_id']}-{inventory_digest(copied)[:16]}-"
            f"{launch_commit[:12]}"
        )

        staged_fit = read_json_regular(temporary / "runs" / "fit.json", "runs/fit.json")
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
        temporary.replace(output)
        return output, manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _remote_item_kind(item: Any) -> str:
    value = getattr(item, "type", None)
    if isinstance(value, str):
        return value
    return item.__class__.__name__.lower()


def remote_closed_tree(api: Any, *, repo_id: str, revision: str, directory: str) -> dict[str, int | None]:
    """List an immutable HF subtree anonymously and require files only at leaves."""

    prefix = directory.rstrip("/") + "/"
    try:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=directory,
            recursive=True,
            expand=False,
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
            if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
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


def anonymous_download(url: str, destination: Path) -> tuple[int, str]:
    """Stream one public object without an authorization header."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise EvidenceError(f"refusing to overwrite anonymous download: {destination}")
    last_error: BaseException | None = None
    for attempt in range(HTTP_ATTEMPTS):
        digest = hashlib.sha256()
        total = 0
        temporary = destination.with_name(destination.name + ".part")
        temporary.unlink(missing_ok=True)
        request = Request(url, headers={"User-Agent": "gpt-tpu-scaling-evidence/1"})
        if request.has_header("Authorization"):
            raise EvidenceError("anonymous verification request unexpectedly has authorization")
        try:
            with urlopen(request, timeout=300) as response, temporary.open("xb") as output:
                while True:
                    chunk = response.read(READ_CHUNK)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
            return total, digest.hexdigest()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
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
    for path, remote_size in remote.items():
        if remote_size is not None and remote_size != expected_details[path]["bytes"]:
            raise EvidenceError(f"anonymous remote size differs for {path}")
    with tempfile.TemporaryDirectory(prefix="scaling-evidence-anonymous-") as directory_name:
        download_root = Path(directory_name) / "archive"
        download_root.mkdir()
        for relative in sorted(expected_details):
            remote_path = f"{directory}/{relative}"
            size, digest = anonymous_download(
                _download_url(repo_id, revision, remote_path), download_root / relative
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
            "revision": _commit(revision, "publication revision"),
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


def write_atomic(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise EvidenceError(f"refusing symlink output: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise EvidenceError(f"temporary output already exists: {temporary}")
    write_new(temporary, payload)
    temporary.replace(path)


def publish_archive(
    *,
    bundle: Path,
    manifest: Mapping[str, Any],
    token_file: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Upload a separately verified snapshot, then verify it anonymously."""

    # Token access is intentionally after complete local verification.
    verify_bundle(bundle)
    token = read_token_file(token_file)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise EvidenceError("huggingface-hub is required for publication") from exc
    repo_id = str(manifest["publication_target"]["repository"])
    directory = str(manifest["publication_target"]["directory"])
    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="scaling-evidence-upload-") as name:
        upload_snapshot = Path(name) / "archive"
        snap_manifest = snapshot_bundle(bundle, upload_snapshot)
        semantic_verify(upload_snapshot, snap_manifest)
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True,
                token=token,
            )
            result = api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(upload_snapshot),
                path_in_repo=directory,
                commit_message=f"Publish immutable scaling evidence {manifest['archive_id']}",
                token=token,
            )
        except Exception as exc:
            raise EvidenceError(
                "authenticated Hugging Face repository creation/upload failed"
            ) from exc
    revision = getattr(result, "oid", None)
    if not isinstance(revision, str) or COMMIT_PATTERN.fullmatch(revision) is None:
        try:
            info = api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        except Exception as exc:
            raise EvidenceError(
                "authenticated Hugging Face revision lookup failed"
            ) from exc
        revision = getattr(info, "sha", None)
    revision = _commit(revision, "uploaded immutable revision")
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
    write_atomic(receipt_output, canonical_json_bytes(receipt))
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
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "verify":
            print(json.dumps(verify_bundle(args.bundle), indent=2, sort_keys=True))
            return 0
        bundle, manifest = build_archive(
            repository_root=args.repository_root,
            runs=args.runs,
            output=args.output,
            repo_id=args.repo_id,
            prefix=args.prefix,
            launch_commit=args.launch_commit,
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
            manifest=manifest,
            token_file=args.token_file,
            receipt_output=args.receipt_output,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (EvidenceError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
