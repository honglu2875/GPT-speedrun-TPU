from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rig.data import DataError, HEADER_BYTES
from rig.data_routing import (
    CLASSIC_TRAIN_CAPACITY,
    MAX_SCALED_TRAIN_CAPACITY,
    SCALED_BUILDER_SHA256,
    SCALED_CORE_SHA256,
    SCALED_ENTRYPOINT_SHA256,
    SCALED_EXCLUSION_POLICY_SHA256,
    SCALED_SOURCE_INVENTORY_SHA256,
    SCALED_SOURCE_REPOSITORY,
    SCALED_SOURCE_REVISION,
    preparation_route,
    resolve_preparation_manifest,
    scaled_variant_for_tokens,
)


def publication_manifest(variant: str = "2B") -> dict[str, object]:
    totals = {"2B": 2_000_000_000, "4B": 4_000_000_000}
    total = totals[variant]
    train_shards = total // 100_000_000 - 1
    revision = "a" * 40
    names = ["fineweb_val_000000.bin"] + [
        f"fineweb_train_{index:06d}.bin" for index in range(1, train_shards + 1)
    ]
    return {
        "schema_version": 1,
        "name": f"fineweb-{total // 1_000_000_000}b-gpt2",
        "format": {
            "name": "llm.c-gpt2-v1",
            "header_bytes": HEADER_BYTES,
            "header_dtype": "little-endian int32",
            "magic": 20_240_520,
            "version": 1,
            "token_dtype": "little-endian uint16",
        },
        "license": {
            "dataset": "ODC-By-1.0",
            "url": "https://opendatacommons.org/licenses/by/1-0/",
            "code": "Apache-2.0",
        },
        "source": {
            "dataset": SCALED_SOURCE_REPOSITORY,
            "revision": SCALED_SOURCE_REVISION,
            "global_shuffle_seed": 42,
            "source_date_before": "2024-04-01",
            "inventory_sha256": SCALED_SOURCE_INVENTORY_SHA256,
            "exclusion_policy_sha256": SCALED_EXCLUSION_POLICY_SHA256,
            "selection": f"first {total:,} prepared tokens",
            "prepared_repository": "quintic/fineweb-scaled-gpt2",
            "prepared_revision": revision,
        },
        "tokenizer": {
            "name": "gpt2",
            "implementation": "tiktoken",
            "implementation_version": "0.11.0",
            "document_prefix_token": 50_256,
            "vocab_size": 50_257,
        },
        "preparation": {
            "builder_version": 1,
            "core_sha256": SCALED_CORE_SHA256,
            "builder_module_sha256": SCALED_BUILDER_SHA256,
            "entrypoint_sha256": SCALED_ENTRYPOINT_SHA256,
            "pyarrow_version": "19.0.1",
            "shard_tokens": 100_000_000,
            "validation_train_document_disjoint": True,
            "validation_boundary_discarded_tokens": 17,
            "validation_boundary_document_id_sha256": "f" * 64,
            "nested_prefix": True,
        },
        "default_train_shards": train_shards,
        "validation_prefix_tokens": 100_000_000,
        "files": [
            {
                "path": name,
                "split": "validation" if index == 0 else "train",
                "tokens": 100_000_000,
                "bytes": HEADER_BYTES + 200_000_000,
                "sha256": f"{index + 1:064x}",
                "url": (
                    "https://huggingface.co/datasets/"
                    "quintic/fineweb-scaled-gpt2/resolve/"
                    f"{revision}/{variant}/{name}"
                ),
            }
            for index, name in enumerate(names)
        ],
    }


class DataRoutingTests(unittest.TestCase):
    def test_current_default_budget_preserves_classic_route(self) -> None:
        route = preparation_route("official", 624_984_064)
        self.assertFalse(route.is_scaled)
        self.assertEqual(route.manifest, "fineweb10b-gpt2")
        self.assertEqual(route.train_shards, 9)
        self.assertEqual(route.train_capacity, 900_000_000)
        self.assertEqual(
            route.data_root(Path("/tmp/example-data")), Path("/tmp/example-data")
        )

    def test_exact_capacity_boundaries_choose_minimal_corpus(self) -> None:
        cases = (
            (1, None),
            (CLASSIC_TRAIN_CAPACITY, None),
            (CLASSIC_TRAIN_CAPACITY + 1, "2B"),
            (1_900_000_000, "2B"),
            (1_900_000_001, "4B"),
            (3_900_000_000, "4B"),
            (3_900_000_001, "8B"),
            (7_900_000_000, "8B"),
            (7_900_000_001, "hero"),
            (MAX_SCALED_TRAIN_CAPACITY, "hero"),
        )
        for budget, expected in cases:
            with self.subTest(budget=budget):
                selected = scaled_variant_for_tokens(budget)
                self.assertEqual(None if selected is None else selected.name, expected)

        for invalid in (0, -1, True, MAX_SCALED_TRAIN_CAPACITY + 1):
            with self.subTest(invalid=invalid), self.assertRaises(DataError):
                scaled_variant_for_tokens(invalid)  # type: ignore[arg-type]

    def test_smoke_and_dev_do_not_route_but_budget_bounds_are_global(
        self,
    ) -> None:
        self.assertFalse(
            preparation_route("smoke", MAX_SCALED_TRAIN_CAPACITY).is_scaled
        )
        development = preparation_route("dev", MAX_SCALED_TRAIN_CAPACITY)
        self.assertFalse(development.is_scaled)
        self.assertEqual(development.train_shards, 1)
        for profile in ("smoke", "dev", "official"):
            for budget in (0, MAX_SCALED_TRAIN_CAPACITY + 1):
                with (
                    self.subTest(profile=profile, budget=budget),
                    self.assertRaises(DataError),
                ):
                    preparation_route(profile, budget)

    def test_scaled_route_uses_dedicated_nested_root_and_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo-manifests"
            route = preparation_route(
                "official", 1_900_000_000, manifest_root=repository
            )
            self.assertEqual(route.variant.name, "2B")  # type: ignore[union-attr]
            self.assertEqual(route.train_shards, 19)
            self.assertEqual(
                route.manifest, repository / "fineweb-scaled-gpt2" / "2B.json"
            )
            self.assertEqual(
                route.data_root(Path(directory) / "cache"),
                Path(directory) / "cache" / "fineweb-scaled" / "2B",
            )

    def test_selected_base_symlink_is_allowed_but_nested_symlinks_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = root / "physical"
            physical.mkdir()
            selected = root / "shm"
            selected.symlink_to(physical, target_is_directory=True)
            route = preparation_route("official", 1_000_000_000)
            self.assertEqual(
                route.data_root(selected), physical / "fineweb-scaled" / "2B"
            )
            outside = root / "outside"
            outside.mkdir()
            (physical / "fineweb-scaled").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(DataError, "symlink below data root"):
                route.data_root(selected)

    def test_nested_cache_component_must_be_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fineweb-scaled").write_text("not a directory", encoding="utf-8")
            route = preparation_route("official", 1_000_000_000)
            with self.assertRaisesRegex(DataError, "not a directory"):
                route.data_root(root)

    def test_scaled_route_fails_clearly_before_publication_manifest_exists(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            route = preparation_route(
                "official", 1_000_000_000, manifest_root=Path(directory)
            )
            with self.assertRaisesRegex(
                DataError, "trusted immutable publication manifest is not checked in"
            ):
                resolve_preparation_manifest(route)

    def test_checked_in_scaled_manifest_must_have_complete_immutable_urls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_root = Path(directory)
            destination = manifest_root / "fineweb-scaled-gpt2" / "2B.json"
            destination.parent.mkdir()
            payload = publication_manifest()
            destination.write_text(json.dumps(payload), encoding="utf-8")
            route = preparation_route(
                "official", 1_900_000_000, manifest_root=manifest_root
            )
            self.assertEqual(resolve_preparation_manifest(route), destination)

            for entry in payload["files"]:  # type: ignore[index]
                entry.pop("url")  # type: ignore[union-attr]
            payload["source"].pop("prepared_repository")  # type: ignore[index,union-attr]
            payload["source"].pop("prepared_revision")  # type: ignore[index,union-attr]
            destination.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                DataError, "repository/revision is not immutable"
            ):
                resolve_preparation_manifest(route)

    def test_scaled_manifest_rejects_wrong_identity_and_incomplete_inventory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_root = Path(directory)
            destination = manifest_root / "fineweb-scaled-gpt2" / "2B.json"
            destination.parent.mkdir()
            route = preparation_route(
                "official", 1_000_000_000, manifest_root=manifest_root
            )
            cases = (
                ("source", lambda row: row["source"].update(revision="b" * 40)),
                (
                    "preparation",
                    lambda row: row["preparation"].update(
                        builder_module_sha256="b" * 64
                    ),
                ),
                ("inventory", lambda row: row["files"].pop()),
            )
            for label, mutate in cases:
                with self.subTest(label=label):
                    payload = publication_manifest()
                    mutate(payload)  # type: ignore[arg-type]
                    destination.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(DataError):
                        resolve_preparation_manifest(route)

    def test_checked_in_manifest_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            manifest = root / "fineweb-scaled-gpt2" / "2B.json"
            manifest.parent.mkdir()
            manifest.symlink_to(target)
            route = preparation_route("official", 1_000_000_000, manifest_root=root)
            with self.assertRaisesRegex(DataError, "symlink manifest"):
                resolve_preparation_manifest(route)

    def test_checked_in_manifest_directory_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            (target / "2B.json").write_text("{}", encoding="utf-8")
            (root / "fineweb-scaled-gpt2").symlink_to(target, target_is_directory=True)
            route = preparation_route("official", 1_000_000_000, manifest_root=root)
            with self.assertRaisesRegex(DataError, "symlink manifest directory"):
                resolve_preparation_manifest(route)


if __name__ == "__main__":
    unittest.main()
