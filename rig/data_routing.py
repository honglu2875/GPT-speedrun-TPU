"""Training-budget routing for explicit dataset preparation.

This module chooses the corpus that preparation installs and non-smoke runs use.
Published
scaled manifests are deliberately not synthesized here: until an immutable,
URL-bearing manifest is checked in, a scaled route fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from .data import DataError, FORMAT_VERSION, HEADER_BYTES, MAGIC, load_manifest


CLASSIC_TRAIN_CAPACITY = 900_000_000
SCALED_VALIDATION_TOKENS = 100_000_000
SCALED_CACHE_SUBDIRECTORY = Path("fineweb-scaled")
SCALED_MANIFEST_SUBDIRECTORY = "fineweb-scaled-gpt2"
SCALED_PREPARED_REPOSITORY = "quintic/fineweb-scaled-gpt2"
SCALED_SOURCE_REPOSITORY = "HuggingFaceFW/fineweb_100BT-shuffled"
SCALED_SOURCE_REVISION = "ee8552966e3d6a5fee2f317f2ae0b342be03d998"
SCALED_SOURCE_INVENTORY_SHA256 = (
    "02ddc6361cc2f8a3d23b0d8b823c7eb7e2b1663ad3d0eff63e83b373456fc12b"
)
SCALED_EXCLUSION_POLICY_SHA256 = (
    "ab25cabd0781b1046b7ad7b281b4147ff6e27d36977f4e842b8c92573399ad77"
)
SCALED_CORE_SHA256 = (
    "4bbdcb76da837276f6f337b805d37a74e3272b476e01fd198f416097abe19241"
)
SCALED_BUILDER_SHA256 = (
    "26c61bc921af290e6beb28596feb2c50cac5b15a56a2f3adf921682317f6f109"
)
SCALED_ENTRYPOINT_SHA256 = (
    "3a676241de10c3ac7cf36ed19ccbd1c0e419bb90de960d4e14be51a1f225bd5c"
)


@dataclass(frozen=True)
class ScaledVariant:
    name: str
    total_tokens: int

    @property
    def train_capacity(self) -> int:
        return self.total_tokens - SCALED_VALIDATION_TOKENS

    @property
    def train_shards(self) -> int:
        return self.train_capacity // SCALED_VALIDATION_TOKENS

    @property
    def dataset_name(self) -> str:
        return f"fineweb-{self.total_tokens // 1_000_000_000}b-gpt2"


SCALED_VARIANTS = (
    ScaledVariant("2B", 2_000_000_000),
    ScaledVariant("4B", 4_000_000_000),
    ScaledVariant("8B", 8_000_000_000),
    ScaledVariant("hero", 75_000_000_000),
)
MAX_SCALED_TRAIN_CAPACITY = SCALED_VARIANTS[-1].train_capacity


@dataclass(frozen=True)
class PreparationRoute:
    """One fully resolved preparation choice, before touching its cache."""

    profile: str
    label: str
    manifest: str | Path
    train_shards: int
    train_capacity: int | None
    cache_subdirectory: Path
    variant: ScaledVariant | None = None

    @property
    def is_scaled(self) -> bool:
        return self.variant is not None

    def data_root(self, base: str | os.PathLike[str]) -> Path:
        """Return the selected root while rejecting links below ``base``."""

        unresolved_base = Path(base).expanduser()
        resolved_base = unresolved_base.resolve(strict=False)
        if not self.cache_subdirectory.parts:
            return resolved_base
        current = resolved_base
        for part in self.cache_subdirectory.parts:
            current /= part
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                break
            except OSError as exc:
                raise DataError(
                    f"cannot inspect scaled data path {current}: {exc}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise DataError(f"refusing symlink below data root: {current}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise DataError(
                    f"scaled data path component is not a directory: {current}"
                )
        target = (resolved_base / self.cache_subdirectory).resolve(strict=False)
        try:
            target.relative_to(resolved_base)
        except ValueError as exc:  # defensive against unusual filesystem behavior
            raise DataError(f"scaled data path escapes data root: {target}") from exc
        return target

    def summary(self, requested_tokens: int) -> str:
        if not self.is_scaled:
            if self.profile == "official":
                return (
                    f"{requested_tokens:,} training tokens select classic FineWeb "
                    f"({CLASSIC_TRAIN_CAPACITY:,}-token nominal train capacity)"
                )
            return (
                f"the explicit {self.profile} profile selects {self.label}; the "
                f"{requested_tokens:,}-token corpus budget applies when preparing official data"
            )
        assert self.variant is not None
        return (
            f"{requested_tokens:,} training tokens select scaled FineWeb "
            f"{self.variant.name} "
            f"({self.train_capacity:,}-token nominal train capacity)"
        )


def scaled_variant_for_tokens(training_tokens: int) -> ScaledVariant | None:
    """Return the smallest corpus beyond classic that fits ``training_tokens``."""

    _validate_budget(training_tokens)
    if training_tokens <= CLASSIC_TRAIN_CAPACITY:
        return None
    for variant in SCALED_VARIANTS:
        if training_tokens <= variant.train_capacity:
            return variant
    raise DataError(
        f"training-token budget {training_tokens:,} exceeds the largest prepared "
        f"corpus capacity ({MAX_SCALED_TRAIN_CAPACITY:,} hero training tokens)"
    )


def scaled_variant_named(name: str) -> ScaledVariant:
    """Return one exact production variant by its publication folder name."""

    for variant in SCALED_VARIANTS:
        if variant.name == name:
            return variant
    choices = ", ".join(variant.name for variant in SCALED_VARIANTS)
    raise DataError(f"unknown scaled FineWeb variant {name!r}; expected {choices}")


def preparation_route(
    profile: str,
    training_tokens: int,
    *,
    manifest_root: Path | None = None,
) -> PreparationRoute:
    """Map personal preparation settings to one classic or scaled dataset."""

    _validate_budget(training_tokens)
    if profile == "smoke":
        return PreparationRoute(profile, "smoke", "smoke", 1, None, Path())
    if profile == "dev":
        return PreparationRoute(
            profile,
            "classic FineWeb development shard",
            "fineweb10b-gpt2",
            1,
            100_000_000,
            Path(),
        )
    if profile != "official":
        raise DataError(f"unknown data profile: {profile}")
    variant = scaled_variant_for_tokens(training_tokens)
    if variant is None:
        return PreparationRoute(
            profile,
            "classic FineWeb",
            "fineweb10b-gpt2",
            9,
            CLASSIC_TRAIN_CAPACITY,
            Path(),
        )
    root = (
        manifest_root or Path(__file__).resolve().parent.parent / "data" / "manifests"
    )
    return PreparationRoute(
        profile,
        f"scaled FineWeb {variant.name}",
        root / SCALED_MANIFEST_SUBDIRECTORY / f"{variant.name}.json",
        variant.train_shards,
        variant.train_capacity,
        SCALED_CACHE_SUBDIRECTORY / variant.name,
        variant,
    )


def resolve_preparation_manifest(route: PreparationRoute) -> str | Path:
    """Resolve and validate a real manifest for ``route``.

    Scaled network preparation trusts only the repository's checked-in
    publication manifest.  A builder-local manifest intentionally has no URLs
    and is not a bootstrap mechanism for clean or peer machines.
    """

    if not route.is_scaled:
        return route.manifest
    published = Path(route.manifest)
    if _regular_nonsymlink(published):
        payload, source = load_manifest(published)
        assert route.variant is not None
        validate_scaled_manifest_contract(
            payload, route.variant, source, require_publication=True
        )
        return source
    assert route.variant is not None
    raise DataError(
        f"scaled FineWeb {route.variant.name} is selected, but its trusted immutable "
        f"publication manifest is not checked in at {published}. Publish the completed "
        "variant and commit its URL-bearing manifest before using `rig prepare`; "
        "a builder-local manifest or BUILD_PLAN.json is not a network download contract"
    )


def validate_scaled_manifest_contract(
    payload: Mapping[str, Any],
    variant: ScaledVariant,
    source: Path,
    *,
    require_publication: bool,
) -> None:
    """Validate the exact production manifest independent of builder settings.

    The production folder targets are fixed here, including hero at exactly
    75B total / 74.9B train.  The frozen builder hash is an additional lineage
    check, not a substitute for the exact shard count, names, order, sizes, and
    split contract.
    """

    if payload.get("name") != variant.dataset_name:
        raise DataError(
            f"{source}: expected dataset name {variant.dataset_name!r} for "
            f"the {variant.name} route"
        )
    if payload.get("default_train_shards") != variant.train_shards:
        raise DataError(
            f"{source}: {variant.name} must provide exactly {variant.train_shards} "
            "default training shards"
        )
    if payload.get("validation_prefix_tokens") != SCALED_VALIDATION_TOKENS:
        raise DataError(
            f"{source}: scaled validation must contain exactly 100,000,000 tokens"
        )

    expected_paths = ["fineweb_val_000000.bin"] + [
        f"fineweb_train_{index:06d}.bin" for index in range(1, variant.train_shards + 1)
    ]
    files = payload.get("files")
    assert isinstance(files, list)  # load_manifest already proved this
    if [entry.get("path") for entry in files] != expected_paths:
        raise DataError(
            f"{source}: {variant.name} shard inventory/order differs from its exact "
            f"{variant.train_shards}-train-plus-1-validation contract"
        )
    for index, entry in enumerate(files):
        expected_split = "validation" if index == 0 else "train"
        if entry.get("split") != expected_split:
            raise DataError(f"{source}: invalid split for {entry.get('path')}")
        if entry.get("tokens") != SCALED_VALIDATION_TOKENS:
            raise DataError(
                f"{source}: every scaled shard must contain 100,000,000 tokens"
            )
        if entry.get("bytes") != HEADER_BYTES + 2 * SCALED_VALIDATION_TOKENS:
            raise DataError(
                f"{source}: every scaled shard must contain exactly "
                f"{HEADER_BYTES + 2 * SCALED_VALIDATION_TOKENS:,} bytes"
            )
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise DataError(
                f"{source}: every scaled shard requires a lowercase SHA-256"
            )

    format_info = payload.get("format")
    if not isinstance(format_info, Mapping) or (
        format_info.get("name") != "llm.c-gpt2-v1"
        or format_info.get("header_bytes") != HEADER_BYTES
        or format_info.get("header_dtype") != "little-endian int32"
        or format_info.get("magic") != MAGIC
        or format_info.get("version") != FORMAT_VERSION
        or format_info.get("token_dtype") != "little-endian uint16"
    ):
        raise DataError(f"{source}: scaled shard format differs from llm.c GPT-2 v1")
    license_info = payload.get("license")
    if not isinstance(license_info, Mapping) or (
        license_info.get("dataset") != "ODC-By-1.0"
        or license_info.get("url")
        != "https://opendatacommons.org/licenses/by/1-0/"
        or license_info.get("code") != "Apache-2.0"
    ):
        raise DataError(f"{source}: scaled corpus/license attribution is incomplete")
    source_info = payload.get("source")
    if not isinstance(source_info, Mapping) or (
        source_info.get("dataset") != SCALED_SOURCE_REPOSITORY
        or source_info.get("revision") != SCALED_SOURCE_REVISION
        or source_info.get("global_shuffle_seed") != 42
        or source_info.get("inventory_sha256")
        != SCALED_SOURCE_INVENTORY_SHA256
        or source_info.get("exclusion_policy_sha256")
        != SCALED_EXCLUSION_POLICY_SHA256
        or source_info.get("source_date_before") != "2024-04-01"
        or source_info.get("selection")
        != f"first {variant.total_tokens:,} prepared tokens"
    ):
        raise DataError(
            f"{source}: scaled FineWeb source/cutoff identity is not pinned"
        )
    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or (
        tokenizer.get("name") != "gpt2"
        or tokenizer.get("implementation") != "tiktoken"
        or tokenizer.get("implementation_version") != "0.11.0"
        or tokenizer.get("document_prefix_token") != 50_256
        or tokenizer.get("vocab_size") != 50_257
    ):
        raise DataError(
            f"{source}: scaled tokenizer identity differs from GPT-2/tiktoken"
        )
    preparation = payload.get("preparation")
    boundary_tokens = (
        preparation.get("validation_boundary_discarded_tokens")
        if isinstance(preparation, Mapping)
        else None
    )
    boundary_document = (
        preparation.get("validation_boundary_document_id_sha256")
        if isinstance(preparation, Mapping)
        else None
    )
    if not isinstance(preparation, Mapping) or (
        preparation.get("builder_version") != 1
        or preparation.get("core_sha256") != SCALED_CORE_SHA256
        or preparation.get("builder_module_sha256") != SCALED_BUILDER_SHA256
        or preparation.get("entrypoint_sha256") != SCALED_ENTRYPOINT_SHA256
        or preparation.get("pyarrow_version") != "19.0.1"
        or preparation.get("shard_tokens") != SCALED_VALIDATION_TOKENS
        or preparation.get("validation_train_document_disjoint") is not True
        or preparation.get("nested_prefix") is not True
        or isinstance(boundary_tokens, bool)
        or not isinstance(boundary_tokens, int)
        or boundary_tokens < 0
        or not isinstance(boundary_document, str)
        or not re.fullmatch(r"[0-9a-f]{64}", boundary_document)
    ):
        raise DataError(f"{source}: scaled split/preparation contract is incomplete")
    if require_publication:
        _validate_publication_urls(payload, variant, source)


def _validate_publication_urls(
    payload: Mapping[str, Any], variant: ScaledVariant, source: Path
) -> None:
    source_info = payload["source"]
    assert isinstance(source_info, Mapping)
    files = payload["files"]
    assert isinstance(files, list)
    revision = source_info.get("prepared_revision")
    if (
        source_info.get("prepared_repository") != SCALED_PREPARED_REPOSITORY
        or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise DataError(f"{source}: prepared repository/revision is not immutable")
    prefix = (
        f"https://huggingface.co/datasets/{SCALED_PREPARED_REPOSITORY}/resolve/"
        f"{revision}/{variant.name}/"
    )
    for entry in files:
        if entry.get("url") != prefix + str(entry["path"]):
            raise DataError(
                f"{source}: {entry['path']} does not use its immutable prepared revision URL"
            )


def _validate_budget(training_tokens: int) -> None:
    if (
        isinstance(training_tokens, bool)
        or not isinstance(training_tokens, int)
        or training_tokens <= 0
    ):
        raise DataError("training-token budget must be a positive integer")
    if training_tokens > MAX_SCALED_TRAIN_CAPACITY:
        raise DataError(
            f"training-token budget {training_tokens:,} exceeds the largest prepared "
            f"corpus capacity ({MAX_SCALED_TRAIN_CAPACITY:,} hero training tokens)"
        )


def _regular_nonsymlink(path: Path) -> bool:
    try:
        parent_metadata = path.parent.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DataError(
            f"cannot inspect manifest directory {path.parent}: {exc}"
        ) from exc
    if stat.S_ISLNK(parent_metadata.st_mode):
        raise DataError(f"refusing symlink manifest directory: {path.parent}")
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise DataError(f"manifest parent is not a directory: {path.parent}")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DataError(f"cannot inspect manifest {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise DataError(f"refusing symlink manifest: {path}")
    return stat.S_ISREG(metadata.st_mode)


__all__ = [
    "CLASSIC_TRAIN_CAPACITY",
    "MAX_SCALED_TRAIN_CAPACITY",
    "PreparationRoute",
    "SCALED_CACHE_SUBDIRECTORY",
    "SCALED_BUILDER_SHA256",
    "SCALED_CORE_SHA256",
    "SCALED_ENTRYPOINT_SHA256",
    "SCALED_EXCLUSION_POLICY_SHA256",
    "SCALED_MANIFEST_SUBDIRECTORY",
    "SCALED_PREPARED_REPOSITORY",
    "SCALED_SOURCE_INVENTORY_SHA256",
    "SCALED_SOURCE_REPOSITORY",
    "SCALED_SOURCE_REVISION",
    "SCALED_VARIANTS",
    "ScaledVariant",
    "preparation_route",
    "resolve_preparation_manifest",
    "scaled_variant_named",
    "scaled_variant_for_tokens",
    "validate_scaled_manifest_contract",
]
