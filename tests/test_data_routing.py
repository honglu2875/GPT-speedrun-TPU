from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rig.data import DataError, HEADER_BYTES
from rig.data_routing import (
    SCALED_BUILDER_SHA256,
    SCALED_CORE_SHA256,
    SCALED_ENTRYPOINT_SHA256,
    SCALED_EXCLUSION_POLICY_SHA256,
    SCALED_SHARD_TOKENS,
    SCALED_SOURCE_INVENTORY_SHA256,
    SCALED_SOURCE_REPOSITORY,
    SCALED_SOURCE_REVISION,
    dataset_names,
    named_preparation_route,
    resolve_preparation_manifest,
    smoke_preparation_route,
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
        "validation_prefix_tokens": 10_485_760,
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
    def test_classic_route_is_selected_by_name(self) -> None:
        route = named_preparation_route("classic")
        self.assertFalse(route.is_scaled)
        self.assertEqual(route.manifest, "fineweb10b-gpt2")
        self.assertEqual(route.train_shards, 9)
        self.assertEqual(route.train_capacity, 900_000_000)
        self.assertEqual(
            route.data_root(Path("/tmp/example-data")), Path("/tmp/example-data")
        )

    def test_named_routes_are_exact_and_allow_explicit_prefixes(self) -> None:
        self.assertEqual(dataset_names(), ("classic", "2B", "4B", "8B", "hero"))
        capacities = {"2B": 19, "4B": 39, "8B": 79, "hero": 749}
        for name, shards in capacities.items():
            with self.subTest(name=name):
                route = named_preparation_route(name)
                self.assertEqual(route.variant.name, name)  # type: ignore[union-attr]
                self.assertEqual(route.train_shards, shards)
                self.assertEqual(
                    named_preparation_route(name, train_shards=1).train_capacity,
                    100_000_000,
                )
        with self.assertRaises(DataError):
            named_preparation_route("unknown")
        with self.assertRaises(DataError):
            named_preparation_route("2B", train_shards=20)

    def test_checked_in_scaled_variants_share_the_published_validation_split(
        self,
    ) -> None:
        validation_hashes: set[str] = set()
        boundary_records: set[tuple[int, str]] = set()
        for name in ("2B", "4B", "8B", "hero"):
            with self.subTest(name=name):
                route = named_preparation_route(name)
                manifest_path = resolve_preparation_manifest(route)
                payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                validation = payload["files"][0]
                self.assertEqual(validation["path"], "fineweb_val_000000.bin")
                self.assertEqual(validation["split"], "validation")
                self.assertEqual(validation["tokens"], SCALED_SHARD_TOKENS)
                self.assertEqual(payload["validation_prefix_tokens"], 10_485_760)
                self.assertTrue(
                    payload["preparation"]["validation_train_document_disjoint"]
                )
                self.assertNotIn(
                    validation["sha256"],
                    {entry["sha256"] for entry in payload["files"][1:]},
                )
                validation_hashes.add(validation["sha256"])
                boundary_records.add(
                    (
                        payload["preparation"]["validation_boundary_discarded_tokens"],
                        payload["preparation"][
                            "validation_boundary_document_id_sha256"
                        ],
                    )
                )
        self.assertEqual(len(validation_hashes), 1)
        self.assertEqual(len(boundary_records), 1)
        discarded_tokens, _ = next(iter(boundary_records))
        self.assertGreater(discarded_tokens, 0)

    def test_smoke_is_the_only_non_named_route(self) -> None:
        route = smoke_preparation_route()
        self.assertEqual(route.profile, "smoke")
        self.assertEqual(route.manifest, "smoke")
        self.assertFalse(route.is_scaled)

    def test_scaled_route_uses_dedicated_nested_root_and_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo-manifests"
            route = named_preparation_route("2B", manifest_root=repository)
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
            route = named_preparation_route("2B")
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
            route = named_preparation_route("2B")
            with self.assertRaisesRegex(DataError, "not a directory"):
                route.data_root(root)

    def test_scaled_route_fails_clearly_before_publication_manifest_exists(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            route = named_preparation_route("2B", manifest_root=Path(directory))
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
            route = named_preparation_route("2B", manifest_root=manifest_root)
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
            route = named_preparation_route("2B", manifest_root=manifest_root)
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
            route = named_preparation_route("2B", manifest_root=root)
            with self.assertRaisesRegex(DataError, "symlink manifest"):
                resolve_preparation_manifest(route)

    def test_checked_in_manifest_directory_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            (target / "2B.json").write_text("{}", encoding="utf-8")
            (root / "fineweb-scaled-gpt2").symlink_to(target, target_is_directory=True)
            route = named_preparation_route("2B", manifest_root=root)
            with self.assertRaisesRegex(DataError, "symlink manifest directory"):
                resolve_preparation_manifest(route)


if __name__ == "__main__":
    unittest.main()
