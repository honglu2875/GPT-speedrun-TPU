from __future__ import annotations

from http.client import IncompleteRead
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from speedrun.data import (
    DataError,
    FORMAT_VERSION,
    HEADER_BYTES,
    MAGIC,
    load_manifest,
    manifest_digest,
    prepare,
    prepare_smoke,
    read_header,
    selected_files,
    sha256_file,
    validate_shard,
    verify_dataset,
)


def write_shard(path: Path, tokens: list[int], *, magic: int = MAGIC, version: int = 1) -> None:
    header = [0] * 256
    header[0] = magic
    header[1] = version
    header[2] = len(tokens)
    path.write_bytes(struct.pack("<256i", *header) + struct.pack(f"<{len(tokens)}H", *tokens))


class FakeResponse:
    def __init__(
        self,
        chunks: list[bytes | Exception],
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.chunks = iter(chunks)
        self.status = status
        self.headers = headers or {}

    def getcode(self) -> int:
        return self.status

    def read(self, _: int) -> bytes:
        try:
            item = next(self.chunks)
        except StopIteration:
            return b""
        if isinstance(item, Exception):
            raise item
        return item

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class ShardTests(unittest.TestCase):
    def test_valid_header_and_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.bin"
            write_shard(path, [0, 1, 50256, 42])
            header = read_header(path)
            self.assertEqual((header.magic, header.version, header.token_count), (MAGIC, 1, 4))
            info = validate_shard(path, expected_tokens=4, expected_bytes=HEADER_BYTES + 8)
            self.assertEqual(info.token_count, 4)

    def test_rejects_truncated_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.bin"
            path.write_bytes(b"short")
            with self.assertRaisesRegex(DataError, "truncated header"):
                read_header(path)

    def test_rejects_bad_magic_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            write_shard(path, [1], magic=7)
            with self.assertRaisesRegex(DataError, "bad magic"):
                validate_shard(path)
            write_shard(path, [1], version=2)
            with self.assertRaisesRegex(DataError, "unsupported version"):
                validate_shard(path)

    def test_rejects_truncated_payload_and_trailing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-length.bin"
            write_shard(path, [1, 2, 3])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(DataError, "header declares"):
                validate_shard(path)
            write_shard(path, [1, 2, 3])
            with path.open("ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(DataError, "header declares"):
                validate_shard(path)

    def test_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hash.bin"
            write_shard(path, [1, 2, 3])
            with self.assertRaisesRegex(DataError, "SHA-256"):
                validate_shard(path, expected_sha256="0" * 64)


class ManifestTests(unittest.TestCase):
    def test_official_manifest_is_pinned(self) -> None:
        manifest, _ = load_manifest("classic")
        self.assertEqual(
            manifest["source"]["prepared_revision"],
            "889765ea1f903759787add96995d81171b632d0c",
        )
        self.assertEqual(manifest["validation_prefix_tokens"], 10_485_760)
        entries = selected_files(manifest)
        self.assertEqual(len(entries), 10)
        for entry in entries:
            self.assertIn(manifest["source"]["prepared_revision"], entry["url"])
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(entry["bytes"], HEADER_BYTES + 2 * entry["tokens"])

    def test_manifest_digest_ignores_mapping_order(self) -> None:
        left = {"z": 1, "a": [2, 3]}
        right = {"a": [2, 3], "z": 1}
        self.assertEqual(manifest_digest(left), manifest_digest(right))

    def test_manifest_rejects_escape_path(self) -> None:
        invalid = {
            "schema_version": 1,
            "name": "bad",
            "files": [
                {"path": "../train.bin", "split": "train", "tokens": 1, "bytes": 1026},
                {"path": "val.bin", "split": "validation", "tokens": 1, "bytes": 1026},
            ],
        }
        with self.assertRaisesRegex(DataError, "unsafe shard path"):
            load_manifest(invalid)

    def test_manifest_rejects_dot_path(self) -> None:
        invalid = {
            "schema_version": 1,
            "name": "bad",
            "files": [
                {"path": ".", "split": "train", "tokens": 1, "bytes": 1026},
                {"path": "val.bin", "split": "validation", "tokens": 1, "bytes": 1026},
            ],
        }
        with self.assertRaisesRegex(DataError, "unsafe shard path"):
            load_manifest(invalid)


class SmokeTests(unittest.TestCase):
    def test_smoke_generation_is_deterministic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = prepare_smoke(first)
            two = prepare_smoke(second)
            self.assertEqual(one.train_tokens, 32768)
            self.assertEqual(one.validation_tokens, 8192)
            paths_one = one.validation_files + one.train_files
            paths_two = two.validation_files + two.train_files
            self.assertEqual([sha256_file(p) for p in paths_one], [sha256_file(p) for p in paths_two])
            self.assertEqual(
                [sha256_file(p) for p in paths_one],
                [
                    "ff2df081e34adda6fba34ea493ca173b746903698bd2da80b977cda50bbacc80",
                    "bcd6d65542490495682b03cf8563dc8e0db37b6ee246e064aeadf13e70160032",
                ],
            )
            again = prepare_smoke(first)
            self.assertEqual(one.manifest_sha256, again.manifest_sha256)

    def test_warm_prepare_hashes_each_existing_shard_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepare_smoke(directory)
            with patch("speedrun.data.sha256_file", wraps=sha256_file) as digest:
                prepare_smoke(directory)
            self.assertEqual(digest.call_count, 2)

    def test_smoke_generation_accepts_symlinked_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            storage = base / "storage"
            storage.mkdir()
            link = base / "shm"
            link.symlink_to(storage, target_is_directory=True)
            result = prepare_smoke(link)
            self.assertEqual(result.root, storage.resolve())
            self.assertTrue((storage / "smoke_train_000001.bin").is_file())

    def test_rejects_symlink_partial_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "cache"
            root.mkdir()
            victim = base / "victim.bin"
            victim.write_bytes(b"sentinel")
            part = root / "smoke_val_000000.bin.part"
            part.symlink_to(victim)

            with self.assertRaisesRegex(DataError, "symlink partial file"):
                prepare_smoke(root)
            self.assertEqual(victim.read_bytes(), b"sentinel")
            self.assertTrue(part.is_symlink())

    def test_rejects_nonregular_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "smoke_val_000000.bin.part").mkdir()
            with self.assertRaisesRegex(DataError, "not a regular file"):
                prepare_smoke(root)

    def test_check_only_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(DataError, "missing dataset shard"):
                prepare(root, "smoke", check_only=True)
            self.assertEqual(list(root.iterdir()), [])

    def test_offline_rejects_missing_network_shard(self) -> None:
        manifest = {
            "schema_version": 1,
            "name": "offline-test",
            "default_train_shards": 1,
            "files": [
                {
                    "path": "val.bin",
                    "split": "validation",
                    "tokens": 1,
                    "bytes": 1026,
                    "sha256": "0" * 64,
                    "url": "https://example.invalid/val.bin",
                },
                {
                    "path": "train.bin",
                    "split": "train",
                    "tokens": 1,
                    "bytes": 1026,
                    "sha256": "0" * 64,
                    "url": "https://example.invalid/train.bin",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DataError, "offline preparation"):
                prepare(directory, manifest, offline=True)

    def test_verify_user_supplied_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_shard(root / "train.bin", [1, 2])
            write_shard(root / "val.bin", [3])
            manifest = {
                "schema_version": 1,
                "name": "local",
                "default_train_shards": 1,
                "files": [
                    {
                        "path": "val.bin",
                        "split": "validation",
                        "tokens": 1,
                        "bytes": 1026,
                        "sha256": sha256_file(root / "val.bin"),
                    },
                    {
                        "path": "train.bin",
                        "split": "train",
                        "tokens": 2,
                        "bytes": 1028,
                        "sha256": sha256_file(root / "train.bin"),
                    },
                ],
            }
            result = verify_dataset(manifest, root)
            self.assertEqual(result.train_tokens, 2)
            self.assertEqual(result.validation_tokens, 1)


class DownloadTests(unittest.TestCase):
    @staticmethod
    def manifest(val_path: Path, train_path: Path) -> dict[str, object]:
        return {
            "schema_version": 1,
            "name": "network-test",
            "default_train_shards": 1,
            "files": [
                {
                    "path": "val.bin",
                    "split": "validation",
                    "tokens": 1,
                    "bytes": val_path.stat().st_size,
                    "sha256": sha256_file(val_path),
                    "url": "https://example.invalid/val.bin",
                },
                {
                    "path": "train.bin",
                    "split": "train",
                    "tokens": 2,
                    "bytes": train_path.stat().st_size,
                    "sha256": sha256_file(train_path),
                    "url": "https://example.invalid/train.bin",
                },
            ],
        }

    def test_resumes_regular_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            fixtures = Path(directory) / "fixtures"
            root.mkdir()
            fixtures.mkdir()
            val_fixture = fixtures / "val.bin"
            train_fixture = fixtures / "train.bin"
            write_shard(val_fixture, [3])
            write_shard(train_fixture, [1, 2])
            manifest = self.manifest(val_fixture, train_fixture)
            (root / "val.bin").write_bytes(val_fixture.read_bytes())
            payload = train_fixture.read_bytes()
            split = 517
            (root / "train.bin.part").write_bytes(payload[:split])

            def open_range(request: object, **_: object) -> FakeResponse:
                self.assertEqual(request.get_header("Range"), f"bytes={split}-")
                return FakeResponse(
                    [payload[split:]],
                    status=206,
                    headers={
                        "Content-Range": f"bytes {split}-{len(payload) - 1}/{len(payload)}"
                    },
                )

            with patch("speedrun.data.urlopen", side_effect=open_range):
                result = prepare(root, manifest)
            self.assertEqual((root / "train.bin").read_bytes(), payload)
            self.assertEqual(result.train_tokens, 2)

    def test_normalizes_midstream_read_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            fixtures = Path(directory) / "fixtures"
            root.mkdir()
            fixtures.mkdir()
            val_fixture = fixtures / "val.bin"
            train_fixture = fixtures / "train.bin"
            write_shard(val_fixture, [3])
            write_shard(train_fixture, [1, 2])
            manifest = self.manifest(val_fixture, train_fixture)
            response = FakeResponse([b"partial", IncompleteRead(b"", 100)])

            with patch("speedrun.data.urlopen", return_value=response):
                with self.assertRaisesRegex(DataError, "download interrupted for val.bin"):
                    prepare(root, manifest)
            self.assertEqual((root / "val.bin.part").read_bytes(), b"partial")


if __name__ == "__main__":
    unittest.main()
