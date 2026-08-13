from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import Request

from scripts.publish_fineweb import (
    HEADER_BYTES,
    METADATA_FILENAMES,
    PROVENANCE_FILES,
    anonymous_verify_variant,
    publication_validation_chain,
    publish,
    request_bytes,
    validate_closed_directory,
    validate_remote_tree_entries,
)
from speedrun.fineweb_builder import (
    FineWebBuildError,
    canonical_json_bytes,
)


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
