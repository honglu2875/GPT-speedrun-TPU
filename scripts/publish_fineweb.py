# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "hf-xet==1.1.7",
#   "huggingface-hub==0.34.4",
# ]
# ///
"""Publish completed scaled FineWeb variants with immutable download URLs."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote, urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rig.data import DataError, load_manifest, validate_shard  # noqa: E402
from rig.data import FORMAT_VERSION, HEADER_BYTES, MAGIC  # noqa: E402
from rig.data_routing import (  # noqa: E402
    SCALED_BUILDER_SHA256,
    SCALED_CORE_SHA256,
    SCALED_ENTRYPOINT_SHA256,
    SCALED_EXCLUSION_POLICY_SHA256,
    SCALED_SOURCE_INVENTORY_SHA256,
    scaled_variant_named,
    validate_scaled_manifest_contract,
)
import rig.frozen  # noqa: F401  (registers the frozen builder's legacy import name)
from rig.fineweb_builder import (  # noqa: E402
    FineWebBuildError,
    canonical_json_bytes,
    canonical_json_sha256,
    configure_cache_root,
)


DEFAULT_REPOSITORY = "quintic/fineweb-scaled-gpt2"
VARIANT_ORDER = ("2B", "4B", "8B", "hero")
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}\Z")
METADATA_FILENAMES = ("BUILD_PLAN.json", "exclusions.json", "source.json")
PROVENANCE_FILES = (
    (
        REPOSITORY_ROOT / "rig" / "fineweb_builder.py",
        "provenance/fineweb_builder.py",
        SCALED_BUILDER_SHA256,
    ),
    (
        REPOSITORY_ROOT / "scripts" / "prepare_fineweb.py",
        "provenance/prepare_fineweb.py",
        SCALED_ENTRYPOINT_SHA256,
    ),
)
HTTP_ATTEMPTS = 5
MAX_JSON_BYTES = 16 * 1024 * 1024
PUBLICATION_LEDGER_SCHEMA_VERSION = 1
PLAN_IDENTITY_FIELDS = (
    "repository",
    "public",
    "manifest_output",
    "source_inventory_sha256",
    "exclusion_policy_sha256",
    "core_sha256",
)
PLAN_FIELDS = (*PLAN_IDENTITY_FIELDS, "variants", "shards")
ANONYMOUS_VERIFICATION_FLAGS = (
    "manifest",
    "closed_tree_inventory",
    "all_shard_lfs_sizes_and_sha256",
    "selected_shard_headers",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Upload completed FineWeb variants to one Hugging Face dataset repo"
    )
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--repo-id", default=DEFAULT_REPOSITORY)
    result.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_ORDER,
        default=["2B", "4B", "8B"],
        help="Completed variants to upload in increasing order",
    )
    result.add_argument(
        "--token-file",
        type=Path,
        default=REPOSITORY_ROOT / ".env.hf",
        help="Mode-0600 file containing exactly HF_TOKEN=...",
    )
    result.add_argument(
        "--card",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "fineweb-scaled-card.md",
    )
    result.add_argument(
        "--private",
        action="store_true",
        help="Create a private repo (default is the requested public dataset)",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all data and print the publication plan without network writes",
    )
    result.add_argument(
        "--manifest-output",
        type=Path,
        help=(
            "Directory for verified URL-bearing manifests (default: "
            "<root>/.fineweb-build/staged-manifests)"
        ),
    )
    return result


def read_token_file(path: Path) -> str:
    """Read one token without shell parsing, interpolation, or logging."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FineWebBuildError(
            f"cannot inspect Hugging Face token file {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise FineWebBuildError(
            "Hugging Face token path must be a regular file, not a link"
        )
    if metadata.st_uid != os.getuid():
        raise FineWebBuildError(
            "Hugging Face token file must be owned by the current user"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise FineWebBuildError(
            "Hugging Face token file must not be group/world accessible"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise FineWebBuildError(f"cannot read Hugging Face token file: {exc}") from exc
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].startswith("HF_TOKEN="):
        raise FineWebBuildError(
            "token file must contain exactly one HF_TOKEN=... assignment"
        )
    token = lines[0].removeprefix("HF_TOKEN=").strip()
    if not TOKEN_PATTERN.fullmatch(token):
        raise FineWebBuildError("HF_TOKEN value has an unexpected format")
    return token


def validate_repo_id(value: str) -> str:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", value
    ):
        raise FineWebBuildError(f"invalid Hugging Face repository id: {value!r}")
    return value


def validate_variants(root: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    required = publication_validation_chain(names)
    validate_provenance_sources()
    complete: dict[str, dict[str, Any]] = {}
    previous: list[str] = []
    validated_inodes: dict[tuple[int, int], tuple[int, int, str]] = {}
    for name in required:
        directory = safe_child_directory(root, name)
        manifest_path = directory / "manifest.json"
        if not is_regular_nonsymlink(manifest_path):
            raise FineWebBuildError(
                f"{name} is incomplete: {manifest_path} does not exist"
            )
        try:
            manifest, source_path = load_manifest(manifest_path)
            variant = scaled_variant_named(name)
            validate_scaled_manifest_contract(
                manifest, variant, source_path, require_publication=False
            )
        except (DataError, KeyError, TypeError, ValueError) as exc:
            raise FineWebBuildError(
                f"{name} manifest violates the production publication contract: {exc}"
            ) from exc
        expected_names = {
            "manifest.json",
            *METADATA_FILENAMES,
            *(str(entry["path"]) for entry in manifest["files"]),
        }
        validate_closed_directory(directory, expected_names)
        for entry in manifest["files"]:
            path = directory / entry["path"]
            if not is_regular_nonsymlink(path):
                raise FineWebBuildError(f"refusing non-regular shard: {path}")
            metadata = path.lstat()
            signature = (
                int(entry["tokens"]),
                int(entry["bytes"]),
                str(entry["sha256"]),
            )
            inode = (metadata.st_dev, metadata.st_ino)
            prior_signature = validated_inodes.get(inode)
            if prior_signature is not None and prior_signature != signature:
                raise FineWebBuildError(
                    f"hard-linked shard has conflicting manifest metadata: {path}"
                )
            if prior_signature is None:
                try:
                    validate_shard(
                        path,
                        expected_tokens=signature[0],
                        expected_bytes=signature[1],
                        expected_sha256=signature[2],
                    )
                except (DataError, OSError) as exc:
                    raise FineWebBuildError(
                        f"invalid publication shard {path}: {exc}"
                    ) from exc
                validated_inodes[inode] = signature
        hashes = [str(entry["sha256"]) for entry in manifest["files"]]
        if previous and hashes[: len(previous)] != previous:
            raise FineWebBuildError(
                f"{name} is not a nested prefix of the prior variant"
            )
        previous = hashes
        source = read_json_regular(directory / "source.json")
        exclusion_policy = read_json_regular(directory / "exclusions.json")
        plan = read_json_regular(directory / "BUILD_PLAN.json")
        source_info = manifest.get("source", {})
        if not isinstance(source_info, dict):
            raise FineWebBuildError(f"{name} manifest source metadata is malformed")
        if canonical_json_sha256(source) != SCALED_SOURCE_INVENTORY_SHA256:
            raise FineWebBuildError(
                f"{name} source.json does not match the pinned production inventory"
            )
        if canonical_json_sha256(exclusion_policy) != SCALED_EXCLUSION_POLICY_SHA256:
            raise FineWebBuildError(
                f"{name} exclusions.json does not match the pinned production policy"
            )
        preparation = manifest.get("preparation", {})
        if not isinstance(preparation, dict) or preparation.get(
            "core_sha256"
        ) != SCALED_CORE_SHA256:
            raise FineWebBuildError(f"{name} manifest has the wrong production core")
        expected_plan = {
            "schema_version": 1,
            "status": "manifest.json appears only after this prefix is complete",
            "directory": name,
            "total_tokens": variant.total_tokens,
            "validation_tokens": 100_000_000,
            "training_tokens": variant.train_capacity,
            "source_inventory_sha256": SCALED_SOURCE_INVENTORY_SHA256,
            "exclusion_policy_sha256": SCALED_EXCLUSION_POLICY_SHA256,
            "core_sha256": SCALED_CORE_SHA256,
        }
        if plan != expected_plan:
            mismatches = sorted(
                key
                for key in set(plan) | set(expected_plan)
                if plan.get(key) != expected_plan.get(key)
            )
            raise FineWebBuildError(
                f"{name} BUILD_PLAN differs in: {', '.join(mismatches)}"
            )
        complete[name] = manifest
    return {name: complete[name] for name in names}


def publication_validation_chain(names: Sequence[str]) -> tuple[str, ...]:
    """Return all local predecessors needed to authenticate the requested prefix."""

    if not names:
        raise FineWebBuildError("at least one publication variant is required")
    try:
        order = [VARIANT_ORDER.index(name) for name in names]
    except ValueError as exc:
        raise FineWebBuildError("unknown FineWeb publication variant") from exc
    if order != sorted(set(order)):
        raise FineWebBuildError("variants must be unique and ordered 2B, 4B, 8B, hero")
    return VARIANT_ORDER[: max(order) + 1]


def expected_publication_shards(variant: str) -> int:
    """Return the fixed validation-plus-training shard count for one variant."""

    selected = scaled_variant_named(variant)
    return selected.train_shards + 1


def required_preserved_predecessors(
    requested_variants: Sequence[str],
) -> tuple[str, ...]:
    """Return nested predecessors omitted from this upload request."""

    if not requested_variants:
        return ()
    requested = set(requested_variants)
    furthest = max(VARIANT_ORDER.index(name) for name in requested_variants)
    return tuple(
        name for name in VARIANT_ORDER[:furthest] if name not in requested
    )


def require_preserved_predecessor_chain(
    *,
    requested_variants: Sequence[str],
    prior_variants: Sequence[str],
    receipts: Mapping[str, Any],
) -> None:
    """Require a complete trusted ledger for predecessors not being uploaded."""

    required = required_preserved_predecessors(requested_variants)
    if not required:
        return
    prior = set(prior_variants)
    missing_plan = [name for name in required if name not in prior]
    missing_receipts = [name for name in required if name not in receipts]
    if missing_plan or missing_receipts:
        details: list[str] = []
        if missing_plan:
            details.append(f"missing from prior plan: {', '.join(missing_plan)}")
        if missing_receipts:
            details.append(
                f"missing verified receipts: {', '.join(missing_receipts)}"
            )
        raise FineWebBuildError(
            "publication request omits nested predecessors and requires their "
            "complete verified prior plan and receipts; " + "; ".join(details)
        )


def validate_publication_plan(
    plan: Mapping[str, Any], *, label: str
) -> tuple[str, ...]:
    """Validate one exact schema-v1 publication plan without normalizing it."""

    if set(plan) != set(PLAN_FIELDS):
        raise FineWebBuildError(f"{label} has an unexpected publication plan schema")
    repository = plan.get("repository")
    public = plan.get("public")
    manifest_output = plan.get("manifest_output")
    if not isinstance(repository, str) or not repository:
        raise FineWebBuildError(f"{label} repository identity is malformed")
    if not isinstance(public, bool):
        raise FineWebBuildError(f"{label} public/private identity is malformed")
    if (
        not isinstance(manifest_output, str)
        or not manifest_output
        or not Path(manifest_output).is_absolute()
    ):
        raise FineWebBuildError(f"{label} manifest output identity is malformed")
    for field in (
        "source_inventory_sha256",
        "exclusion_policy_sha256",
        "core_sha256",
    ):
        value = plan.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise FineWebBuildError(f"{label} {field} identity is malformed")

    variants = plan.get("variants")
    shards = plan.get("shards")
    if not isinstance(variants, list) or not variants:
        raise FineWebBuildError(f"{label} variants must be a non-empty list")
    if any(not isinstance(name, str) for name in variants):
        raise FineWebBuildError(f"{label} contains a malformed variant name")
    canonical = tuple(name for name in VARIANT_ORDER if name in variants)
    if tuple(variants) != canonical or len(canonical) != len(variants):
        raise FineWebBuildError(
            f"{label} variants are not unique and in canonical publication order"
        )
    if not isinstance(shards, dict) or set(shards) != set(canonical):
        raise FineWebBuildError(f"{label} shard counts do not match its variants")
    for name in canonical:
        count = shards.get(name)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count != expected_publication_shards(name)
        ):
            raise FineWebBuildError(
                f"{label} has the wrong fixed shard count for {name}"
            )
    return canonical


def validate_preserved_receipt(
    *,
    variant: str,
    receipt: Any,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate one prior row before it can enter a cumulative ledger."""

    if variant not in VARIANT_ORDER or not isinstance(receipt, dict):
        raise FineWebBuildError("publication ledger contains a malformed receipt")
    public = plan["public"]
    expected_keys = {
        "shard_revision",
        "manifest_revision",
        "manifest_sha256",
        "repository",
        "staged_manifest",
    }
    if public:
        expected_keys.add("anonymous_verification")
    if set(receipt) != expected_keys:
        raise FineWebBuildError(
            f"publication ledger receipt for {variant} has an unexpected schema"
        )
    if receipt.get("repository") != plan["repository"]:
        raise FineWebBuildError(
            f"publication ledger receipt repository differs for {variant}"
        )
    for field in ("shard_revision", "manifest_revision"):
        value = receipt.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise FineWebBuildError(
                f"publication ledger receipt has an invalid {field} for {variant}"
            )
    manifest_sha256 = receipt.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", manifest_sha256
    ):
        raise FineWebBuildError(
            f"publication ledger receipt has an invalid manifest hash for {variant}"
        )
    expected_staged = str(Path(str(plan["manifest_output"])) / f"{variant}.json")
    if receipt.get("staged_manifest") != expected_staged:
        raise FineWebBuildError(
            f"publication ledger staged manifest differs for {variant}"
        )
    staged_path = Path(expected_staged)
    try:
        staging_metadata = staged_path.parent.lstat()
    except OSError as exc:
        raise FineWebBuildError(
            f"cannot inspect publication ledger staging directory: {exc}"
        ) from exc
    if stat.S_ISLNK(staging_metadata.st_mode) or not stat.S_ISDIR(
        staging_metadata.st_mode
    ):
        raise FineWebBuildError(
            "publication ledger staging directory must be a real directory"
        )
    staged = read_json_regular(staged_path)
    if canonical_json_sha256(staged) != manifest_sha256:
        raise FineWebBuildError(
            f"publication ledger staged manifest hash differs for {variant}"
        )
    try:
        # Validate the same parsed object whose canonical hash was authenticated,
        # avoiding a second file read between the hash and contract checks.
        staged, _in_memory_source = load_manifest(staged)
        validate_scaled_manifest_contract(
            staged,
            scaled_variant_named(variant),
            staged_path,
            # Repository identity is a publisher setting.  Validate every
            # repository-independent production invariant here, then bind the
            # staged source and every URL to the plan/receipt immediately below.
            require_publication=False,
        )
    except DataError as exc:
        raise FineWebBuildError(
            "publication ledger staged manifest violates the complete production "
            f"contract for {variant}: {exc}"
        ) from exc
    source = staged.get("source")
    if not isinstance(source, dict) or (
        source.get("prepared_repository") != plan["repository"]
        or source.get("prepared_revision") != receipt["shard_revision"]
    ):
        raise FineWebBuildError(
            f"publication ledger staged manifest identity differs for {variant}"
        )
    entries = staged.get("files")
    if not isinstance(entries, list) or not entries:
        raise FineWebBuildError(
            f"publication ledger staged manifest has no files for {variant}"
        )
    repository = quote(str(plan["repository"]), safe="/")
    revision = quote(str(receipt["shard_revision"]), safe="")
    for entry in entries:
        entry_path = entry.get("path") if isinstance(entry, dict) else None
        if (
            not isinstance(entry_path, str)
            or not entry_path
            or Path(entry_path).name != entry_path
        ):
            raise FineWebBuildError(
                f"publication ledger staged manifest is malformed for {variant}"
            )
        remote_path = quote(f"{variant}/{entry_path}", safe="/")
        expected_url = (
            f"https://huggingface.co/datasets/{repository}/resolve/"
            f"{revision}/{remote_path}"
        )
        if entry.get("url") != expected_url:
            raise FineWebBuildError(
                f"publication ledger staged manifest URL differs for {variant}"
            )
    if public:
        verification = receipt.get("anonymous_verification")
        if not isinstance(verification, dict) or set(verification) != set(
            ANONYMOUS_VERIFICATION_FLAGS
        ):
            raise FineWebBuildError(
                f"publication ledger anonymous verification is malformed for {variant}"
            )
        if any(
            verification.get(flag) is not True
            for flag in ANONYMOUS_VERIFICATION_FLAGS
        ):
            raise FineWebBuildError(
                "publication ledger lacks successful anonymous verification for "
                f"{variant}"
            )
    return json.loads(json.dumps(receipt))


def load_and_merge_publication_ledger(
    path: Path, requested_plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a fail-closed ledger and merge the new request in canonical order."""

    requested_variants = validate_publication_plan(
        requested_plan, label="requested publication"
    )
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        prior: dict[str, Any] | None = None
    except OSError as exc:
        raise FineWebBuildError(f"cannot inspect publication ledger: {exc}") from exc
    else:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise FineWebBuildError(
                "publication ledger must be a regular non-symlink file"
            )
        prior = read_json_regular(path)

    if prior is None:
        require_preserved_predecessor_chain(
            requested_variants=requested_variants,
            prior_variants=(),
            receipts={},
        )
        merged_plan = dict(requested_plan)
        merged_plan["variants"] = list(requested_variants)
        merged_plan["shards"] = {
            name: expected_publication_shards(name) for name in requested_variants
        }
        return merged_plan, {}

    expected_ledger_keys = {"schema_version", *PLAN_FIELDS, "receipts"}
    if set(prior) != expected_ledger_keys:
        raise FineWebBuildError("publication ledger has an unexpected schema")
    schema_version = prior.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or schema_version != PUBLICATION_LEDGER_SCHEMA_VERSION
    ):
        raise FineWebBuildError("publication ledger has an unsupported schema version")
    prior_plan = {field: prior[field] for field in PLAN_FIELDS}
    prior_variants = validate_publication_plan(
        prior_plan, label="publication ledger"
    )
    for field in PLAN_IDENTITY_FIELDS:
        if prior_plan[field] != requested_plan[field]:
            raise FineWebBuildError(
                f"publication ledger identity differs in {field}"
            )
    raw_receipts = prior.get("receipts")
    if not isinstance(raw_receipts, dict) or not set(raw_receipts).issubset(
        set(prior_variants)
    ):
        raise FineWebBuildError(
            "publication ledger receipts do not match its planned variants"
        )
    receipts = {
        name: validate_preserved_receipt(
            variant=name, receipt=raw_receipts[name], plan=requested_plan
        )
        for name in VARIANT_ORDER
        if name in raw_receipts
    }
    require_preserved_predecessor_chain(
        requested_variants=requested_variants,
        prior_variants=prior_variants,
        receipts=receipts,
    )
    merged_variants = tuple(
        name
        for name in VARIANT_ORDER
        if name in prior_variants or name in requested_variants
    )
    merged_plan = dict(requested_plan)
    merged_plan["variants"] = list(merged_variants)
    merged_plan["shards"] = {
        name: expected_publication_shards(name) for name in merged_variants
    }
    return merged_plan, receipts


def validate_closed_directory(directory: Path, expected_names: set[str]) -> None:
    """Reject every unplanned entry before handing a folder to the Hub client."""

    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise FineWebBuildError(f"cannot enumerate publication folder {directory}: {exc}") from exc
    actual_names = {entry.name for entry in entries}
    missing = sorted(expected_names - actual_names)
    extras = sorted(actual_names - expected_names)
    if missing or extras or len(entries) != len(expected_names):
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extras:
            details.append(f"unexpected {', '.join(extras)}")
        raise FineWebBuildError(
            f"{directory} is not the exact closed upload inventory: " + "; ".join(details)
        )
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError as exc:
            raise FineWebBuildError(f"cannot inspect publication entry {entry}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise FineWebBuildError(
                f"publication inventory entries must be regular non-symlink files: {entry}"
            )


def validate_provenance_sources() -> None:
    """Ensure uploaded preparation sources are the exact frozen artifacts."""

    for path, _remote_path, expected_sha256 in PROVENANCE_FILES:
        if not is_regular_nonsymlink(path):
            raise FineWebBuildError(f"frozen provenance source is not regular: {path}")
        if sha256_path(path) != expected_sha256:
            raise FineWebBuildError(
                f"frozen provenance source hash changed: {path.relative_to(REPOSITORY_ROOT)}"
            )


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise FineWebBuildError(f"cannot hash publication file {path}: {exc}") from exc
    return digest.hexdigest()


def safe_child_directory(root: Path, name: str) -> Path:
    selected = root.resolve(strict=True)
    candidate = selected / name
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise FineWebBuildError(
            f"cannot inspect variant directory {candidate}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise FineWebBuildError(f"variant path must be a real directory: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(selected)
    except ValueError as exc:
        raise FineWebBuildError(
            f"variant path escapes publication root: {candidate}"
        ) from exc
    return resolved


def read_json_regular(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise FineWebBuildError(f"metadata path is not a regular file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except FineWebBuildError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FineWebBuildError(
            f"cannot read publication metadata {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise FineWebBuildError(f"publication metadata must be an object: {path}")
    return value


def is_regular_nonsymlink(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def publication_manifest(
    manifest: Mapping[str, Any], repo_id: str, variant: str, shard_revision: str
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", shard_revision):
        raise FineWebBuildError("Hub returned a non-immutable shard revision")
    payload = json.loads(json.dumps(manifest))
    repository = quote(repo_id, safe="/")
    revision = quote(shard_revision, safe="")
    for entry in payload["files"]:
        remote_path = quote(f"{variant}/{entry['path']}", safe="/")
        entry["url"] = (
            f"https://huggingface.co/datasets/{repository}/resolve/"
            f"{revision}/{remote_path}"
        )
    payload["source"]["prepared_repository"] = repo_id
    payload["source"]["prepared_revision"] = shard_revision
    payload["source"]["integrity"] = (
        "Per-file SHA-256 plus immutable prepared_revision URLs; repeated nested "
        "prefix content is expected to be deduplicated by Hub Xet storage."
    )
    return payload


def publish(
    *,
    api: Any,
    root: Path,
    repo_id: str,
    manifests: Mapping[str, Mapping[str, Any]],
    card: Path,
    token: str,
    private: bool,
    manifest_output: Path | None = None,
    receipt_callback: Callable[[Mapping[str, Any]], None] | None = None,
    initial_receipts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add dataset card and preparation provenance",
    )
    for local_path, remote_path, _expected_sha256 in PROVENANCE_FILES:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Add exact frozen preparation source",
        )
    receipts: dict[str, Any] = {
        name: json.loads(json.dumps(initial_receipts[name]))
        for name in VARIANT_ORDER
        if initial_receipts is not None and name in initial_receipts
    }
    for variant, manifest in manifests.items():
        allowed = [str(entry["path"]) for entry in manifest["files"]]
        allowed.extend(METADATA_FILENAMES)
        shard_commit = api.upload_folder(
            folder_path=str(root / variant),
            path_in_repo=variant,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            allow_patterns=allowed,
            commit_message=f"Upload {variant} immutable token shards",
        )
        shard_revision = immutable_commit_revision(shard_commit, "shard upload")
        remote_manifest = publication_manifest(
            manifest, repo_id, variant, shard_revision
        )
        manifest_commit = api.upload_file(
            path_or_fileobj=BytesIO(canonical_json_bytes(remote_manifest)),
            path_in_repo=f"{variant}/manifest.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Pin {variant} download manifest",
        )
        manifest_revision = immutable_commit_revision(
            manifest_commit, "manifest upload"
        )
        receipts[variant] = {
            "shard_revision": shard_revision,
            "manifest_revision": manifest_revision,
            "manifest_sha256": canonical_json_sha256(remote_manifest),
            "repository": repo_id,
        }
        expected_remote = remote_inventory(root / variant, remote_manifest)
        if not private:
            anonymous_verify_variant(
                repo_id=repo_id,
                variant=variant,
                manifest=remote_manifest,
                manifest_revision=manifest_revision,
                shard_revision=shard_revision,
                expected_remote=expected_remote,
            )
            receipts[variant]["anonymous_verification"] = {
                flag: True for flag in ANONYMOUS_VERIFICATION_FLAGS
            }
        if manifest_output is not None:
            staged = manifest_output / f"{variant}.json"
            write_json_atomic(staged, remote_manifest)
            receipts[variant]["staged_manifest"] = str(staged)
        if receipt_callback:
            receipt_callback(
                {name: receipts[name] for name in VARIANT_ORDER if name in receipts}
            )
    return {name: receipts[name] for name in VARIANT_ORDER if name in receipts}


def immutable_commit_revision(commit: Any, operation: str) -> str:
    """Extract the exact commit returned by one Hub mutation; never race HEAD."""

    revision = str(getattr(commit, "oid", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise FineWebBuildError(
            f"Hub {operation} did not return an immutable 40-character commit"
        )
    return revision


def remote_inventory(
    directory: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Describe every expected file in the final remote variant subtree."""

    expected: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        expected.append(
            {
                "path": str(entry["path"]),
                "bytes": int(entry["bytes"]),
                "sha256": str(entry["sha256"]),
                "lfs_required": True,
            }
        )
    for name in METADATA_FILENAMES:
        path = directory / name
        expected.append(
            {
                "path": name,
                "bytes": path.lstat().st_size,
                "sha256": sha256_path(path),
                "lfs_required": False,
            }
        )
    manifest_bytes = canonical_json_bytes(manifest)
    expected.append(
        {
            "path": "manifest.json",
            "bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "lfs_required": False,
        }
    )
    return expected


def anonymous_verify_variant(
    *,
    repo_id: str,
    variant: str,
    manifest: Mapping[str, Any],
    manifest_revision: str,
    shard_revision: str,
    expected_remote: Sequence[Mapping[str, Any]] | None = None,
    timeout: float = 120.0,
) -> None:
    """Verify the complete public tree and selected headers without credentials."""

    repository = quote(repo_id, safe="/")
    manifest_url = (
        f"https://huggingface.co/datasets/{repository}/resolve/"
        f"{quote(manifest_revision, safe='')}/{quote(variant, safe='')}/manifest.json"
    )
    try:
        raw_manifest, _headers, _status = request_bytes(
            Request(
                manifest_url,
                headers={"User-Agent": "gpt-tpu-rig-publish/1"},
            ),
            timeout=timeout,
            maximum_bytes=MAX_JSON_BYTES,
            label=f"anonymous manifest verification for {variant}",
        )
        remote_manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FineWebBuildError(
            f"anonymous manifest verification failed for {variant}: malformed JSON"
        ) from exc
    if canonical_json_sha256(remote_manifest) != canonical_json_sha256(manifest):
        raise FineWebBuildError(f"anonymous manifest bytes differ for {variant}")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise FineWebBuildError(f"published {variant} manifest has no files")
    selected_indices = sorted({0, 1 if len(entries) > 1 else 0, len(entries) - 1})
    selected = [entries[index] for index in selected_indices]
    tree = fetch_tree_pages(repository, variant, manifest_revision, timeout=timeout)
    if expected_remote is None:
        expected_remote = [
            {
                **entry,
                "lfs_required": True,
            }
            for entry in entries
        ]
    validate_remote_tree_entries(tree, variant, expected_remote)

    for entry in selected:
        shard_url = str(entry["url"])
        header, headers, status = request_bytes(
            Request(
                shard_url,
                headers={
                    "Accept-Encoding": "identity",
                    "Range": f"bytes=0-{HEADER_BYTES - 1}",
                    "User-Agent": "gpt-tpu-rig-publish/1",
                },
            ),
            timeout=timeout,
            maximum_bytes=HEADER_BYTES,
            label=f"anonymous header verification for {variant}/{entry['path']}",
        )
        expected_range = f"bytes 0-{HEADER_BYTES - 1}/{int(entry['bytes'])}"
        if (
            status != 206
            or headers.get("content-range") != expected_range
            or headers.get("content-length") != str(HEADER_BYTES)
        ):
            raise FineWebBuildError(
                f"anonymous server did not honor the exact Range request for "
                f"{variant}/{entry['path']}"
            )
        if len(header) != HEADER_BYTES:
            raise FineWebBuildError(
                f"anonymous header is truncated for {variant}/{entry['path']}"
            )
        values = struct.unpack("<256i", header)
        if (values[0], values[1], values[2]) != (
            MAGIC,
            FORMAT_VERSION,
            int(entry["tokens"]),
        ):
            raise FineWebBuildError(
                f"anonymous header mismatch for {variant}/{entry['path']}"
            )


def fetch_tree_pages(
    repository: str, variant: str, revision: str, *, timeout: float
) -> list[dict[str, Any]]:
    url: str | None = (
        f"https://huggingface.co/api/datasets/{repository}/tree/"
        f"{quote(revision, safe='')}/{quote(variant, safe='')}"
        "?recursive=true&expand=true&limit=100"
    )
    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    while url is not None:
        if url in seen_urls:
            raise FineWebBuildError(f"anonymous Hub tree pagination loops for {variant}")
        seen_urls.add(url)
        try:
            raw, headers, _status = request_bytes(
                Request(url, headers={"User-Agent": "gpt-tpu-rig-publish/1"}),
                timeout=timeout,
                maximum_bytes=MAX_JSON_BYTES,
                label=f"anonymous tree verification for {variant}",
            )
            payload = json.loads(raw.decode("utf-8"))
            link = headers.get("link", "")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FineWebBuildError(
                f"anonymous tree verification failed for {variant}: malformed JSON"
            ) from exc
        if not isinstance(payload, list):
            raise FineWebBuildError(f"anonymous Hub tree for {variant} is not a list")
        if any(not isinstance(item, dict) for item in payload):
            raise FineWebBuildError(
                f"anonymous Hub tree for {variant} contains a malformed entry"
            )
        items.extend(payload)
        url = next_link(link)
        if url is not None:
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
                raise FineWebBuildError(
                    f"anonymous Hub tree returned an unsafe next link for {variant}"
                )
    return items


def request_bytes(
    request: Request,
    *,
    timeout: float,
    maximum_bytes: int,
    label: str,
    attempts: int = HTTP_ATTEMPTS,
) -> tuple[bytes, dict[str, str], int | None]:
    """Read one bounded anonymous response with transient-safe retries."""

    if attempts <= 0:
        raise FineWebBuildError("HTTP attempt count must be positive")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(maximum_bytes + 1)
                if len(payload) > maximum_bytes:
                    raise FineWebBuildError(f"{label} exceeded its response size limit")
                headers = {
                    str(key).lower(): str(value).strip()
                    for key, value in response.headers.items()
                }
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                return payload, headers, status
        except FineWebBuildError:
            raise
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
        except Exception as exc:
            # HTTP client implementations may expose transport failures through
            # their own exception types. Do not include their text, which can
            # contain request details.
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise FineWebBuildError(
        f"{label} failed after {attempts} attempts: {type(last_error).__name__}"
    ) from last_error


def next_link(header: str) -> str | None:
    for item in header.split(","):
        segments = [segment.strip() for segment in item.split(";")]
        if len(segments) >= 2 and 'rel="next"' in segments[1:]:
            candidate = segments[0]
            if candidate.startswith("<") and candidate.endswith(">"):
                return candidate[1:-1]
    return None


def validate_remote_tree_entries(
    tree: Any, variant: str, expected: Sequence[Mapping[str, Any]]
) -> None:
    if not isinstance(tree, list):
        raise FineWebBuildError(f"anonymous Hub tree for {variant} is not a list")
    by_name: dict[str, Mapping[str, Any]] = {}
    prefix = f"{variant}/"
    for item in tree:
        if not isinstance(item, dict) or item.get("type") != "file":
            raise FineWebBuildError(
                f"anonymous Hub tree contains a non-file entry below {variant}"
            )
        remote_path = str(item.get("path", ""))
        if not remote_path.startswith(prefix):
            raise FineWebBuildError(
                f"anonymous Hub tree contains an out-of-scope path below {variant}"
            )
        relative = remote_path[len(prefix) :]
        if not relative or "/" in relative or relative in by_name:
            raise FineWebBuildError(
                f"anonymous Hub tree contains an invalid path below {variant}"
            )
        by_name[relative] = item
    expected_names = {str(entry["path"]) for entry in expected}
    if set(by_name) != expected_names or len(expected_names) != len(expected):
        missing = sorted(expected_names - set(by_name))
        extras = sorted(set(by_name) - expected_names)
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if extras:
            detail.append(f"unexpected {', '.join(extras)}")
        raise FineWebBuildError(
            f"anonymous Hub tree is not the exact {variant} inventory: "
            + "; ".join(detail)
        )
    for entry in expected:
        item = by_name.get(str(entry["path"]))
        if not isinstance(item, dict):
            raise FineWebBuildError(
                f"anonymous Hub tree is missing {variant}/{entry['path']}"
            )
        if item.get("size") != int(entry["bytes"]):
            raise FineWebBuildError(
                f"anonymous Hub size mismatch for {variant}/{entry['path']}"
            )
        lfs = item.get("lfs")
        if entry.get("lfs_required") and not isinstance(lfs, dict):
            raise FineWebBuildError(
                f"anonymous Hub tree lacks LFS metadata for {variant}/{entry['path']}"
            )
        if isinstance(lfs, dict):
            oid = str(lfs.get("oid", "")).lower().removeprefix("sha256:")
            if (
                oid != str(entry["sha256"]).lower()
                or lfs.get("size") != int(entry["bytes"])
            ):
                raise FineWebBuildError(
                    f"anonymous Hub SHA-256 metadata mismatch for "
                    f"{variant}/{entry['path']}"
                )


def write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        metadata = temporary.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise FineWebBuildError(f"cannot inspect receipt temporary: {exc}") from exc
    else:
        if not stat.S_ISREG(metadata.st_mode):
            raise FineWebBuildError(
                f"refusing non-regular receipt temporary: {temporary}"
            )
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FineWebBuildError(f"receipt temporary is not regular: {temporary}")
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically install a staged JSON file without following links."""

    try:
        parent_metadata = path.parent.lstat()
    except OSError as exc:
        raise FineWebBuildError(f"cannot inspect JSON output directory: {exc}") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise FineWebBuildError(
            f"JSON output parent must be a real directory: {path.parent}"
        )
    try:
        target_metadata = path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise FineWebBuildError(f"cannot inspect JSON output {path}: {exc}") from exc
    else:
        if stat.S_ISLNK(target_metadata.st_mode) or not stat.S_ISREG(
            target_metadata.st_mode
        ):
            raise FineWebBuildError(f"refusing non-regular JSON output: {path}")
    write_receipt(path, payload)


def ensure_staging_directory(path: Path) -> Path:
    """Create one real staging directory beneath an existing real parent."""

    expanded = path.expanduser().absolute()
    try:
        parent_metadata = expanded.parent.lstat()
    except OSError as exc:
        raise FineWebBuildError(
            f"cannot inspect staging parent {expanded.parent}: {exc}"
        ) from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise FineWebBuildError(
            f"manifest staging parent must be a real directory: {expanded.parent}"
        )
    try:
        metadata = expanded.lstat()
    except FileNotFoundError:
        try:
            expanded.mkdir(mode=0o700)
        except OSError as exc:
            raise FineWebBuildError(
                f"cannot create manifest staging directory {expanded}: {exc}"
            ) from exc
    except OSError as exc:
        raise FineWebBuildError(
            f"cannot inspect manifest staging directory {expanded}: {exc}"
        ) from exc
    else:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FineWebBuildError(
                f"manifest staging path must be a real directory: {expanded}"
            )
    return expanded.resolve(strict=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        root = Path(args.root).expanduser().resolve(strict=True)
        configure_cache_root(root)
        repo_id = validate_repo_id(args.repo_id)
        manifests = validate_variants(root, args.variants)
        card_input = Path(args.card).expanduser().absolute()
        if not is_regular_nonsymlink(card_input):
            raise FineWebBuildError(
                f"dataset card is not a regular non-symlink file: {card_input}"
            )
        card = card_input.resolve(strict=True)
        staging_display = (
            Path(args.manifest_output).expanduser().absolute()
            if args.manifest_output is not None
            else root / ".fineweb-build" / "staged-manifests"
        )
        requested_plan = {
            "repository": repo_id,
            "public": not args.private,
            "variants": list(manifests),
            "shards": {
                name: len(manifest["files"]) for name, manifest in manifests.items()
            },
            "manifest_output": str(staging_display),
            "source_inventory_sha256": SCALED_SOURCE_INVENTORY_SHA256,
            "exclusion_policy_sha256": SCALED_EXCLUSION_POLICY_SHA256,
            "core_sha256": SCALED_CORE_SHA256,
        }
        receipt_path = safe_child_directory(root, ".fineweb-build") / "publication.json"
        plan, initial_receipts = load_and_merge_publication_ledger(
            receipt_path, requested_plan
        )
        if args.dry_run:
            print(
                json.dumps(
                    {**plan, "receipts": initial_receipts, "dry_run": True},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        token = read_token_file(args.token_file)
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise FineWebBuildError(
                "huggingface_hub is required; run this file with `uv run --script`"
            ) from exc
        manifest_output = ensure_staging_directory(staging_display)

        def save_receipts(receipts: Mapping[str, Any]) -> None:
            write_receipt(
                receipt_path,
                {"schema_version": 1, **plan, "receipts": receipts},
            )

        receipts = publish(
            api=HfApi(),
            root=root,
            repo_id=repo_id,
            manifests=manifests,
            card=card,
            token=token,
            private=args.private,
            manifest_output=manifest_output,
            receipt_callback=save_receipts,
            initial_receipts=initial_receipts,
        )
        save_receipts(receipts)
        print(json.dumps({**plan, "receipts": receipts}, indent=2, sort_keys=True))
        return 0
    except FineWebBuildError as exc:
        print(f"fineweb publication failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Never include exception text: HTTP client errors can contain request
        # details, and the token must not be rendered even on unexpected paths.
        print(
            f"fineweb publication failed unexpectedly: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print(
            "\npublication interrupted; rerun safely; Hub/Xet may deduplicate "
            "repeated content",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
