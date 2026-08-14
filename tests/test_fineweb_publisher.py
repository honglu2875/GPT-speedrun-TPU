from __future__ import annotations

import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import Request

from scripts.publish_fineweb import (
    ANONYMOUS_VERIFICATION_FLAGS,
    HEADER_BYTES,
    METADATA_FILENAMES,
    PROVENANCE_FILES,
    anonymous_verify_variant,
    expected_publication_shards,
    load_and_merge_publication_ledger,
    main as publisher_main,
    publication_validation_chain,
    publish,
    request_bytes,
    validate_closed_directory,
    validate_remote_tree_entries,
)
import rig.frozen  # noqa: F401  (registers the frozen builder's legacy import name)
from rig.fineweb_builder import (
    FineWebBuildError,
    canonical_json_bytes,
    canonical_json_sha256,
)
from rig.data import DataError, load_manifest as load_data_manifest
from rig.data_routing import (
    SCALED_CORE_SHA256,
    SCALED_EXCLUSION_POLICY_SHA256,
    SCALED_SOURCE_INVENTORY_SHA256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(
        self, payload: bytes, *, status: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self.payload = payload
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int = -1) -> bytes:
        return self.payload if limit < 0 else self.payload[:limit]

    def getcode(self) -> int:
        return self.status


class _FakeHubApi:
    def __init__(self) -> None:
        self.files: list[str] = []
        self.folder_kwargs: list[dict[str, object]] = []

    def create_repo(self, **_kwargs):
        return None

    def upload_file(self, **kwargs):
        self.files.append(str(kwargs["path_in_repo"]))
        if str(kwargs["path_in_repo"]).endswith("/manifest.json"):
            return SimpleNamespace(oid="c" * 40)
        return SimpleNamespace(oid="a" * 40)

    def upload_folder(self, **kwargs):
        self.folder_kwargs.append(kwargs)
        return SimpleNamespace(oid="b" * 40)

    def repo_info(self, **_kwargs):
        return SimpleNamespace(sha="d" * 40)


def _publication_plan(
    staging: Path,
    variants: list[str],
    *,
    repository: str = "quintic/fineweb-scaled-gpt2",
) -> dict[str, object]:
    return {
        "repository": repository,
        "public": True,
        "variants": variants,
        "shards": {
            name: expected_publication_shards(name) for name in variants
        },
        "manifest_output": str(staging),
        "source_inventory_sha256": SCALED_SOURCE_INVENTORY_SHA256,
        "exclusion_policy_sha256": SCALED_EXCLUSION_POLICY_SHA256,
        "core_sha256": SCALED_CORE_SHA256,
    }


def _verified_receipt(
    staging: Path,
    variant: str,
    index: int,
    *,
    repository: str = "quintic/fineweb-scaled-gpt2",
) -> dict[str, object]:
    manifest = json.loads(
        (
            REPOSITORY_ROOT
            / "data"
            / "manifests"
            / "fineweb-scaled-gpt2"
            / f"{variant}.json"
        ).read_text(encoding="utf-8")
    )
    shard_revision = str(manifest["source"]["prepared_revision"])
    manifest_revision = f"{index + 5:x}" * 40
    manifest["source"]["prepared_repository"] = repository
    url_prefix = (
        f"https://huggingface.co/datasets/{repository}/resolve/"
        f"{shard_revision}/{variant}/"
    )
    for entry in manifest["files"]:
        entry["url"] = url_prefix + entry["path"]
    staged = staging / f"{variant}.json"
    staged.write_bytes(canonical_json_bytes(manifest))
    return {
        "shard_revision": shard_revision,
        "manifest_revision": manifest_revision,
        "manifest_sha256": canonical_json_sha256(manifest),
        "repository": repository,
        "staged_manifest": str(staged),
        "anonymous_verification": {
            flag: True for flag in ANONYMOUS_VERIFICATION_FLAGS
        },
    }


class FineWebPublisherTests(unittest.TestCase):
    def test_hero_requires_every_local_predecessor_in_strict_order(self) -> None:
        self.assertEqual(
            publication_validation_chain(["hero"]),
            ("2B", "4B", "8B", "hero"),
        )
        self.assertEqual(
            publication_validation_chain(["4B", "hero"]),
            ("2B", "4B", "8B", "hero"),
        )
        for invalid in (["hero", "8B"], ["2B", "2B"], []):
            with self.subTest(invalid=invalid), self.assertRaises(FineWebBuildError):
                publication_validation_chain(invalid)

    def test_local_upload_inventory_is_closed_and_rejects_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.bin").write_bytes(b"one")
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            expected = {"one.bin", "manifest.json"}
            validate_closed_directory(root, expected)

            (root / "extra.log").write_text("log", encoding="utf-8")
            with self.assertRaisesRegex(FineWebBuildError, "unexpected extra.log"):
                validate_closed_directory(root, expected)
            (root / "extra.log").unlink()
            (root / "one.bin").unlink()
            (root / "one.bin").symlink_to(root / "manifest.json")
            with self.assertRaisesRegex(FineWebBuildError, "non-symlink"):
                validate_closed_directory(root, expected)

    def test_hero_request_preserves_verified_small_variant_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staged"
            staging.mkdir()
            prior_variants = ["2B", "4B", "8B"]
            prior_plan = _publication_plan(staging, prior_variants)
            prior_receipts = {
                name: _verified_receipt(staging, name, index)
                for index, name in enumerate(prior_variants)
            }
            ledger = {
                "schema_version": 1,
                **prior_plan,
                "receipts": prior_receipts,
            }
            ledger_path = root / "publication.json"
            ledger_path.write_bytes(canonical_json_bytes(ledger))

            requested = _publication_plan(staging, ["hero"])
            plan, receipts = load_and_merge_publication_ledger(
                ledger_path, requested
            )

            self.assertEqual(plan["variants"], ["2B", "4B", "8B", "hero"])
            self.assertEqual(
                plan["shards"], {"2B": 20, "4B": 40, "8B": 80, "hero": 750}
            )
            self.assertEqual(receipts, prior_receipts)

    def test_preserved_receipt_supports_the_planned_custom_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staged"
            staging.mkdir()
            repository = "example-user/private-fineweb"
            plan = _publication_plan(
                staging, ["2B"], repository=repository
            )
            receipt = _verified_receipt(
                staging, "2B", 0, repository=repository
            )
            ledger_path = root / "publication.json"
            ledger_path.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        **plan,
                        "receipts": {"2B": receipt},
                    }
                )
            )

            with patch(
                "scripts.publish_fineweb.load_manifest",
                wraps=load_data_manifest,
            ) as validate_manifest:
                merged, receipts = load_and_merge_publication_ledger(
                    ledger_path, plan
                )

            self.assertEqual(merged, plan)
            self.assertEqual(receipts, {"2B": receipt})
            validate_manifest.assert_called_once()
            self.assertIsInstance(validate_manifest.call_args.args[0], dict)

    def test_hero_only_requires_complete_verified_predecessor_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staged"
            staging.mkdir()
            ledger_path = root / "publication.json"
            requested = _publication_plan(staging, ["hero"])

            with self.assertRaisesRegex(
                FineWebBuildError, "complete verified prior plan and receipts"
            ):
                load_and_merge_publication_ledger(ledger_path, requested)

            first_two = ["2B", "4B"]
            first_two_receipts = {
                name: _verified_receipt(staging, name, index)
                for index, name in enumerate(first_two)
            }
            ledger_path.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        **_publication_plan(staging, first_two),
                        "receipts": first_two_receipts,
                    }
                )
            )
            with self.assertRaisesRegex(
                FineWebBuildError,
                "missing from prior plan: 8B; missing verified receipts: 8B",
            ):
                load_and_merge_publication_ledger(ledger_path, requested)

            predecessors = ["2B", "4B", "8B"]
            ledger_path.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        **_publication_plan(staging, predecessors),
                        "receipts": first_two_receipts,
                    }
                )
            )
            with self.assertRaisesRegex(
                FineWebBuildError, "missing verified receipts: 8B"
            ):
                load_and_merge_publication_ledger(ledger_path, requested)

    def test_preserved_manifest_must_satisfy_full_named_production_contract(
        self,
    ) -> None:
        def wrong_variant(payload: dict[str, object]) -> None:
            payload["name"] = "fineweb-4b-gpt2"

        def wrong_count(payload: dict[str, object]) -> None:
            payload["files"].pop()  # type: ignore[union-attr]

        def wrong_order(payload: dict[str, object]) -> None:
            files = payload["files"]
            files[1], files[2] = files[2], files[1]  # type: ignore[index]

        def wrong_shard_name(payload: dict[str, object]) -> None:
            payload["files"][1]["path"] = "fineweb_train_999999.bin"  # type: ignore[index]

        def wrong_bytes(payload: dict[str, object]) -> None:
            payload["files"][1]["bytes"] = 200_001_022  # type: ignore[index]

        def wrong_source(payload: dict[str, object]) -> None:
            payload["source"]["global_shuffle_seed"] = 7  # type: ignore[index]

        def wrong_preparation(payload: dict[str, object]) -> None:
            payload["preparation"]["core_sha256"] = "0" * 64  # type: ignore[index]

        def mutable_url(payload: dict[str, object]) -> None:
            entry = payload["files"][1]  # type: ignore[index]
            entry["url"] = (
                "https://huggingface.co/datasets/quintic/fineweb-scaled-gpt2/"
                f"resolve/main/2B/{entry['path']}"
            )

        mutations = {
            "named variant": (wrong_variant, "complete production contract for 2B"),
            "shard count": (wrong_count, "complete production contract for 2B"),
            "shard order": (wrong_order, "complete production contract for 2B"),
            "shard name": (wrong_shard_name, "complete production contract for 2B"),
            "byte count": (wrong_bytes, "complete production contract for 2B"),
            "source provenance": (wrong_source, "complete production contract for 2B"),
            "preparation provenance": (
                wrong_preparation,
                "complete production contract for 2B",
            ),
            "immutable public URL": (mutable_url, "staged manifest URL differs for 2B"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staged"
            staging.mkdir()
            plan = _publication_plan(staging, ["2B"])
            ledger_path = root / "publication.json"
            for label, (mutate, expected_error) in mutations.items():
                with self.subTest(label=label):
                    receipt = _verified_receipt(staging, "2B", 0)
                    staged = staging / "2B.json"
                    payload = json.loads(staged.read_text(encoding="utf-8"))
                    mutate(payload)
                    staged.write_bytes(canonical_json_bytes(payload))
                    receipt["manifest_sha256"] = canonical_json_sha256(payload)
                    ledger_path.write_bytes(
                        canonical_json_bytes(
                            {
                                "schema_version": 1,
                                **plan,
                                "receipts": {"2B": receipt},
                            }
                        )
                    )
                    with self.assertRaisesRegex(
                        FineWebBuildError,
                        expected_error,
                    ) as caught:
                        load_and_merge_publication_ledger(ledger_path, plan)
                    if label != "immutable public URL":
                        self.assertIsInstance(caught.exception.__cause__, DataError)

    def test_public_ledger_rejects_unverified_or_changed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staged"
            staging.mkdir()
            plan = _publication_plan(staging, ["2B"])
            receipt = _verified_receipt(staging, "2B", 0)
            ledger_path = root / "publication.json"

            unverified = json.loads(json.dumps(receipt))
            unverified["anonymous_verification"]["manifest"] = False
            ledger_path.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        **plan,
                        "receipts": {"2B": unverified},
                    }
                )
            )
            with self.assertRaisesRegex(
                FineWebBuildError, "lacks successful anonymous verification"
            ):
                load_and_merge_publication_ledger(ledger_path, plan)

            ledger_path.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        **plan,
                        "repository": "quintic/different",
                        "receipts": {"2B": receipt},
                    }
                )
            )
            with self.assertRaisesRegex(
                FineWebBuildError, "identity differs in repository"
            ):
                load_and_merge_publication_ledger(ledger_path, plan)

    def test_ledger_fails_closed_on_schema_links_and_staged_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staged"
            staging.mkdir()
            plan = _publication_plan(staging, ["2B"])
            receipt = _verified_receipt(staging, "2B", 0)
            valid = {
                "schema_version": 1,
                **plan,
                "receipts": {"2B": receipt},
            }
            target = root / "target.json"
            target.write_bytes(canonical_json_bytes(valid))
            ledger_path = root / "publication.json"
            ledger_path.symlink_to(target)
            with self.assertRaisesRegex(FineWebBuildError, "non-symlink"):
                load_and_merge_publication_ledger(ledger_path, plan)

            ledger_path.unlink()
            valid["schema_version"] = 2
            ledger_path.write_bytes(canonical_json_bytes(valid))
            with self.assertRaisesRegex(FineWebBuildError, "unsupported schema"):
                load_and_merge_publication_ledger(ledger_path, plan)

            valid["schema_version"] = 1
            ledger_path.write_bytes(canonical_json_bytes(valid))
            (staging / "2B.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FineWebBuildError, "manifest hash differs"):
                load_and_merge_publication_ledger(ledger_path, plan)

    def test_main_rejects_ledger_before_reading_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".fineweb-build").mkdir()
            (root / ".fineweb-build" / "publication.json").write_text(
                "[]\n", encoding="utf-8"
            )
            card = root / "README.md"
            card.write_text("card\n", encoding="utf-8")
            hero_manifest = {"files": [{} for _ in range(750)]}
            with (
                patch("scripts.publish_fineweb.configure_cache_root"),
                patch(
                    "scripts.publish_fineweb.validate_variants",
                    return_value={"hero": hero_manifest},
                ),
                patch("scripts.publish_fineweb.read_token_file") as read_token,
                redirect_stderr(StringIO()),
            ):
                result = publisher_main(
                    [
                        "--root",
                        str(root),
                        "--variants",
                        "hero",
                        "--card",
                        str(card),
                    ]
                )
            self.assertEqual(result, 2)
            read_token.assert_not_called()

    def test_publish_uses_exact_names_uploads_provenance_and_stages_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = root / "2B"
            variant.mkdir()
            shard_name = "fineweb_val_000000.bin"
            (variant / shard_name).write_bytes(b"fixture")
            for name in METADATA_FILENAMES:
                (variant / name).write_text("{}\n", encoding="utf-8")
            card = root / "README.md"
            card.write_text("card\n", encoding="utf-8")
            staged = root / "staged"
            staged.mkdir()
            manifest = {
                "source": {},
                "files": [
                    {
                        "path": shard_name,
                        "split": "validation",
                        "tokens": 100_000_000,
                        "bytes": 200_001_024,
                        "sha256": "a" * 64,
                    }
                ],
            }
            api = _FakeHubApi()
            receipts = publish(
                api=api,
                root=root,
                repo_id="quintic/fineweb-scaled-gpt2",
                manifests={"2B": manifest},
                card=card,
                token="not-logged",
                private=True,
                manifest_output=staged,
            )

            self.assertEqual(
                api.folder_kwargs[0]["allow_patterns"],
                [shard_name, *METADATA_FILENAMES],
            )
            self.assertNotIn("*.bin", api.folder_kwargs[0]["allow_patterns"])
            self.assertTrue(
                {remote for _local, remote, _digest in PROVENANCE_FILES}.issubset(
                    api.files
                )
            )
            staged_payload = json.loads((staged / "2B.json").read_text())
            self.assertEqual(staged_payload["source"]["prepared_revision"], "b" * 40)
            self.assertEqual(receipts["2B"]["manifest_revision"], "c" * 40)

    def test_publish_replaces_only_requested_row_in_initial_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = root / "hero"
            variant.mkdir()
            shard_name = "fineweb_val_000000.bin"
            (variant / shard_name).write_bytes(b"fixture")
            for name in METADATA_FILENAMES:
                (variant / name).write_text("{}\n", encoding="utf-8")
            card = root / "README.md"
            card.write_text("card\n", encoding="utf-8")
            manifest = {
                "source": {},
                "files": [
                    {
                        "path": shard_name,
                        "split": "validation",
                        "tokens": 100_000_000,
                        "bytes": 200_001_024,
                        "sha256": "a" * 64,
                    }
                ],
            }
            initial = {
                name: {"sentinel": name} for name in ("2B", "4B", "8B", "hero")
            }

            receipts = publish(
                api=_FakeHubApi(),
                root=root,
                repo_id="quintic/fineweb-scaled-gpt2",
                manifests={"hero": manifest},
                card=card,
                token="not-logged",
                private=True,
                initial_receipts=initial,
            )

            for name in ("2B", "4B", "8B"):
                self.assertEqual(receipts[name], initial[name])
            self.assertNotIn("sentinel", receipts["hero"])
            self.assertEqual(receipts["hero"]["manifest_revision"], "c" * 40)

    def test_remote_tree_requires_exact_inventory_and_all_lfs_hashes(self) -> None:
        expected = [
            {
                "path": "one.bin",
                "bytes": 12,
                "sha256": "a" * 64,
                "lfs_required": True,
            },
            {
                "path": "manifest.json",
                "bytes": 4,
                "sha256": "b" * 64,
                "lfs_required": False,
            },
        ]
        tree = [
            {
                "type": "file",
                "path": "2B/one.bin",
                "size": 12,
                "lfs": {"oid": "a" * 64, "size": 12},
            },
            {"type": "file", "path": "2B/manifest.json", "size": 4},
        ]
        validate_remote_tree_entries(tree, "2B", expected)
        with self.assertRaisesRegex(FineWebBuildError, "unexpected extra"):
            validate_remote_tree_entries(
                [*tree, {"type": "file", "path": "2B/extra", "size": 1}],
                "2B",
                expected,
            )
        tree[0].pop("lfs")
        with self.assertRaisesRegex(FineWebBuildError, "lacks LFS"):
            validate_remote_tree_entries(tree, "2B", expected)

    def test_anonymous_header_requires_strict_range_response_headers(self) -> None:
        header_values = [0] * 256
        header_values[:3] = [20_240_520, 1, 100_000_000]
        header = struct.pack("<256i", *header_values)
        manifest = {
            "source": {},
            "files": [
                {
                    "path": "fineweb_val_000000.bin",
                    "tokens": 100_000_000,
                    "bytes": 200_001_024,
                    "sha256": "a" * 64,
                    "url": "https://huggingface.co/fixture.bin",
                }
            ],
        }
        tree = [
            {
                "type": "file",
                "path": "2B/fineweb_val_000000.bin",
                "size": 200_001_024,
                "lfs": {"oid": "a" * 64, "size": 200_001_024},
            }
        ]
        with (
            patch(
                "scripts.publish_fineweb.request_bytes",
                side_effect=[
                    (canonical_json_bytes(manifest), {}, 200),
                    (header, {"content-length": str(HEADER_BYTES)}, 206),
                ],
            ),
            patch("scripts.publish_fineweb.fetch_tree_pages", return_value=tree),
            self.assertRaisesRegex(FineWebBuildError, "exact Range"),
        ):
            anonymous_verify_variant(
                repo_id="quintic/fineweb-scaled-gpt2",
                variant="2B",
                manifest=manifest,
                manifest_revision="c" * 40,
                shard_revision="b" * 40,
            )

    def test_anonymous_reads_retry_without_exposing_error_details(self) -> None:
        response = _Response(b"ok", headers={"Content-Length": "2"})
        with (
            patch(
                "scripts.publish_fineweb.urlopen",
                side_effect=[URLError("secret query"), response],
            ) as mocked,
            patch("scripts.publish_fineweb.time.sleep") as sleep,
        ):
            payload, headers, status = request_bytes(
                Request("https://huggingface.co/fixture"),
                timeout=1,
                maximum_bytes=2,
                label="fixture request",
            )
        self.assertEqual(payload, b"ok")
        self.assertEqual(headers["content-length"], "2")
        self.assertEqual(status, 200)
        self.assertEqual(mocked.call_count, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
