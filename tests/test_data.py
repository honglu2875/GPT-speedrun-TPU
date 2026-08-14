from __future__ import annotations

import copy
from http.client import IncompleteRead
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from rig import data as data_module
from rig.data import (
    DataError,
    FORMAT_VERSION,
    FRESH10_DOMAINS,
    FRESH10_SCORED_TOKENS_PER_DOMAIN,
    FRESH10_TOKENS_PER_DOMAIN,
    HEADER_BYTES,
    MAGIC,
    load_fresh10_manifest,
    load_manifest,
    manifest_digest,
    prepare,
    prepare_fresh10,
    prepare_smoke,
    read_header,
    selected_files,
    sha256_file,
    validate_shard,
    verify_dataset,
    verify_fresh10,
)


def write_shard(path: Path, tokens: list[int], *, magic: int = MAGIC, version: int = 1) -> None:
    header = [0] * 256
    header[0] = magic
    header[1] = version
    header[2] = len(tokens)
    path.write_bytes(struct.pack("<256i", *header) + struct.pack(f"<{len(tokens)}H", *tokens))


def fresh10_tokens() -> list[int]:
    tokens: list[int] = []
    for document_index in range(4):
        tokens.extend([50_256, *([100 + document_index] * 2_048)])
    return tokens


def fresh10_manifest(shard_sha256: str, shard_bytes: int) -> dict[str, object]:
    revision = "a" * 40
    domains: list[dict[str, object]] = []
    for domain in FRESH10_DOMAINS:
        path = f"shards/fresh10_{domain}.bin"
        documents: list[dict[str, object]] = []
        for document_index in range(4):
            token_offset = document_index * 2_049
            document_id = f"{domain}-document-{document_index + 1}"
            documents.append(
                {
                    "id": document_id,
                    "title": f"{domain.title()} document {document_index + 1}",
                    "authors": ["Fixture Author"],
                    "publisher": "Fixture publisher",
                    "source_url": f"https://example.test/{document_id}",
                    "published_date": "2026-01-01",
                    "retrieved_date": "2026-08-12",
                    "license": {
                        "name": "CC0 1.0",
                        "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    },
                    "extraction_notes": "Synthetic test fixture.",
                    "raw_sha256": "b" * 64,
                    "text_path": f"texts/{domain}/{document_id}.txt",
                    "text_bytes": 123,
                    "text_sha256": "c" * 64,
                    "token_offset": token_offset,
                    "token_count": 2_049,
                    "score_offset": token_offset + 1,
                    "scored_tokens": 2_048,
                }
            )
        domains.append(
            {
                "name": domain,
                "path": path,
                "url": (
                    "https://huggingface.co/datasets/quintic/fresh10/resolve/"
                    f"{revision}/{path}"
                ),
                "tokens": FRESH10_TOKENS_PER_DOMAIN,
                "scored_tokens": FRESH10_SCORED_TOKENS_PER_DOMAIN,
                "bytes": shard_bytes,
                "sha256": shard_sha256,
                "documents": documents,
            }
        )
    return {
        "schema_version": 1,
        "kind": "fresh10",
        "name": "fresh10-fixture",
        "publication_not_before": "2025-05-01",
        "prepared_source": {
            "repository": "quintic/fresh10",
            "revision": revision,
        },
        "tokenizer": {
            "name": "gpt2",
            "vocab_size": 50_257,
            "eot_token": 50_256,
            "document_prefix_token": 50_256,
        },
        "format": {
            "name": "llm.c-gpt2-v1",
            "header_bytes": HEADER_BYTES,
            "magic": MAGIC,
            "version": FORMAT_VERSION,
            "token_dtype": "little-endian uint16",
        },
        "domains": domains,
    }


def materialize_fresh10(root: Path) -> dict[str, object]:
    tokens = fresh10_tokens()
    fixture = root / "fixture.bin"
    write_shard(fixture, tokens)
    payload = fixture.read_bytes()
    digest = sha256_file(fixture)
    fixture.unlink()
    manifest = fresh10_manifest(digest, len(payload))
    for domain in manifest["domains"]:  # type: ignore[index]
        target = root / domain["path"]  # type: ignore[index]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return manifest


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

    def test_relative_shard_paths_must_be_canonical_posix(self) -> None:
        for relative in ("a/./b.bin", "a//b.bin", "a\\b.bin", "./a.bin"):
            with self.subTest(relative=relative), self.assertRaisesRegex(
                DataError, "unsafe shard path"
            ):
                data_module._validate_relative_shard_path(relative, Path("manifest.json"))

    def test_fresh10_shard_rejects_out_of_vocabulary_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "domain.bin"
            write_shard(path, [50_256, 1, 50_257])
            domain = {"documents": [{"id": "doc", "token_offset": 0}]}
            with self.assertRaisesRegex(DataError, "GPT-2 vocabulary maximum"):
                data_module._validate_fresh10_boundaries(path, domain)

            write_shard(path, [50_256, 1, 50_256])
            data_module._validate_fresh10_boundaries(path, domain)


class Fresh10Tests(unittest.TestCase):
    def test_valid_manifest_and_shards_preserve_order_spans_and_eot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = materialize_fresh10(root)

            loaded, source = load_fresh10_manifest(manifest)
            self.assertEqual(source, Path("<in-memory-fresh10-manifest>"))
            self.assertEqual(
                tuple(domain["name"] for domain in loaded["domains"]),
                FRESH10_DOMAINS,
            )
            prepared = verify_fresh10(root, manifest)
            self.assertEqual(tuple(domain.name for domain in prepared.domains), FRESH10_DOMAINS)
            self.assertEqual(prepared.scored_tokens, 10 * 8_192)
            self.assertEqual(
                [document.token_offset for document in prepared.domains[0].documents],
                [0, 2_049, 4_098, 6_147],
            )
            self.assertTrue(all(domain.path.is_file() for domain in prepared.domains))

    def test_rejects_wrong_domain_count_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = materialize_fresh10(Path(directory))
            missing = copy.deepcopy(manifest)
            missing["domains"].pop()  # type: ignore[union-attr]
            with self.assertRaisesRegex(DataError, "exactly 10 domains"):
                load_fresh10_manifest(missing)

            reordered = copy.deepcopy(manifest)
            domains = reordered["domains"]  # type: ignore[assignment]
            domains[0], domains[1] = domains[1], domains[0]  # type: ignore[index]
            with self.assertRaisesRegex(DataError, "domain order"):
                load_fresh10_manifest(reordered)

    def test_rejects_unsafe_and_duplicate_canonical_text_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = materialize_fresh10(Path(directory))
            unsafe = copy.deepcopy(manifest)
            unsafe["domains"][0]["documents"][0]["text_path"] = "../escape.txt"  # type: ignore[index]
            with self.assertRaisesRegex(DataError, "unsafe shard path"):
                load_fresh10_manifest(unsafe)

            duplicate = copy.deepcopy(manifest)
            documents = duplicate["domains"][0]["documents"]  # type: ignore[index]
            documents[1]["text_path"] = documents[0]["text_path"]  # type: ignore[index]
            with self.assertRaisesRegex(DataError, "duplicate Fresh10 path"):
                load_fresh10_manifest(duplicate)

    def test_rejects_stale_or_impossible_publication_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = materialize_fresh10(Path(directory))
            for published, retrieved in (
                ("2025-04-30", "2026-08-12"),
                ("2026-08-13", "2026-08-12"),
                ("not-a-date", "2026-08-12"),
            ):
                with self.subTest(published=published):
                    invalid = copy.deepcopy(manifest)
                    document = invalid["domains"][0]["documents"][0]  # type: ignore[index]
                    document["published_date"] = published
                    document["retrieved_date"] = retrieved
                    with self.assertRaisesRegex(DataError, "published_date|freshness"):
                        load_fresh10_manifest(invalid)

    def test_rejects_unpinned_revision_or_download_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = materialize_fresh10(Path(directory))
            moving_revision = copy.deepcopy(manifest)
            moving_revision["prepared_source"]["revision"] = "main"  # type: ignore[index]
            with self.assertRaisesRegex(DataError, "immutable revision"):
                load_fresh10_manifest(moving_revision)

            moving_url = copy.deepcopy(manifest)
            moving_url["domains"][0]["url"] = (  # type: ignore[index]
                "https://huggingface.co/datasets/quintic/fresh10/resolve/main/"
                "shards/fresh10_science.bin"
            )
            with self.assertRaisesRegex(DataError, "URL must pin"):
                load_fresh10_manifest(moving_url)

    def test_rejects_noncanonical_hugging_face_repository_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = materialize_fresh10(Path(directory))
            invalid_repositories = (
                "fresh10",
                "/fresh10",
                "quintic/",
                "quintic/fresh10/extra",
                "quintic/fresh10?download=true",
                "quintic/fresh10#revision",
                "quintic%2Ffresh10",
                "quintic/%2Ffresh10",
                "quint ic/fresh10",
                "q" * 48 + "/" + "f" * 48,
            )
            for repository in invalid_repositories:
                with self.subTest(repository=repository):
                    invalid = copy.deepcopy(manifest)
                    invalid["prepared_source"]["repository"] = repository  # type: ignore[index]
                    with self.assertRaisesRegex(DataError, "invalid prepared repository"):
                        load_fresh10_manifest(invalid)

    def test_rejects_bad_spans_counts_and_hash_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = materialize_fresh10(Path(directory))
            mutations = (
                ("span", lambda value: value["domains"][0]["documents"][1].__setitem__("token_offset", 2_048)),
                ("count", lambda value: value["domains"][0].__setitem__("tokens", 8_195)),
                ("hash", lambda value: value["domains"][0]["documents"][0].__setitem__("text_sha256", "not-a-hash")),
            )
            for label, mutate in mutations:
                with self.subTest(label=label):
                    invalid = copy.deepcopy(manifest)
                    mutate(invalid)
                    with self.assertRaises(DataError):
                        load_fresh10_manifest(invalid)

    def test_requires_explicit_unique_document_authors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = materialize_fresh10(Path(directory))
            for authors in (None, [], [""], ["Fixture Author", "Fixture Author"]):
                with self.subTest(authors=authors):
                    invalid = copy.deepcopy(manifest)
                    document = invalid["domains"][0]["documents"][0]  # type: ignore[index]
                    if authors is None:
                        document.pop("authors")
                    else:
                        document["authors"] = authors
                    with self.assertRaisesRegex(DataError, "authors"):
                        load_fresh10_manifest(invalid)

    def test_rejects_actual_non_eot_document_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = materialize_fresh10(root)
            shard = root / "shards/fresh10_science.bin"
            payload = bytearray(shard.read_bytes())
            second_boundary = HEADER_BYTES + 2 * 2_049
            struct.pack_into("<H", payload, second_boundary, 42)
            shard.write_bytes(payload)

            with self.assertRaisesRegex(DataError, "expected GPT-2 EOT token 50256"):
                verify_fresh10(root, manifest, verify_hash=False)

    def test_offline_missing_shard_is_read_only(self) -> None:
        tokens = fresh10_tokens()
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.bin"
            write_shard(fixture, tokens)
            manifest = fresh10_manifest(sha256_file(fixture), fixture.stat().st_size)
            fixture.unlink()
            with self.assertRaisesRegex(DataError, "offline preparation"):
                prepare_fresh10(directory, manifest, offline=True)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_prepare_downloads_and_verifies_all_ten_shards(self) -> None:
        tokens = fresh10_tokens()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.bin"
            write_shard(fixture, tokens)
            payload = fixture.read_bytes()
            manifest = fresh10_manifest(sha256_file(fixture), len(payload))
            fixture.unlink()

            def open_fixture(*_: object, **__: object) -> FakeResponse:
                return FakeResponse([payload])

            with patch("rig.data.urlopen", side_effect=open_fixture) as opener:
                prepared = prepare_fresh10(root, manifest)
            self.assertEqual(opener.call_count, 10)
            self.assertEqual(tuple(domain.name for domain in prepared.domains), FRESH10_DOMAINS)
            self.assertTrue(all(domain.path.is_file() for domain in prepared.domains))


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
            with patch("rig.data.sha256_file", wraps=sha256_file) as digest:
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

    def test_force_rejects_final_shard_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim.bin"
            victim.write_bytes(b"sentinel")
            shard = root / "smoke_val_000000.bin"
            shard.symlink_to(victim.name)

            with self.assertRaisesRegex(DataError, "symlink below data root"):
                prepare_smoke(root, force=True)
            self.assertEqual(victim.read_bytes(), b"sentinel")
            self.assertTrue(shard.is_symlink())

    def test_rejects_intermediate_symlink_below_resolved_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = root / "storage"
            storage.mkdir()
            (root / "alias").symlink_to(storage, target_is_directory=True)

            with self.assertRaisesRegex(DataError, "symlink below data root"):
                data_module._safe_target(root, "alias/shard.bin")

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

            with patch("rig.data.urlopen", side_effect=open_range):
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

            with patch("rig.data.urlopen", return_value=response):
                with self.assertRaisesRegex(DataError, "download interrupted for val.bin"):
                    prepare(root, manifest)
            self.assertEqual((root / "val.bin.part").read_bytes(), b"partial")


if __name__ == "__main__":
    unittest.main()
