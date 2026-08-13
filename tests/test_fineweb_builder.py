from __future__ import annotations

import json
import os
from http.client import IncompleteRead
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

from scripts.publish_fineweb import (
    next_link,
    publication_manifest,
    read_token_file,
    validate_remote_tree_entries,
)
from speedrun.data import HEADER_BYTES, verify_dataset
from speedrun.fineweb_builder import (
    BuildConfig,
    DocumentBatch,
    ExclusionPolicy,
    FineWebBuildError,
    SourceFileCache,
    SourceFile,
    SourceInventory,
    Variant,
    build_fineweb,
    configure_cache_root,
    ensure_build_work_directory,
    load_fresh10_exclusion_policy,
    normalize_url,
    probe_fineweb,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeSource:
    def __init__(
        self,
        documents: list[tuple[str, str, str, str]],
    ) -> None:
        self.documents = documents
        self.started_at: list[int] = []

    def initial_cursor(self) -> dict[str, int]:
        return {"index": 0}

    def iter_batches(self, cursor: dict[str, int], batch_rows: int):
        start = cursor["index"]
        self.started_at.append(start)
        for batch_start in range(start, len(self.documents), batch_rows):
            rows = self.documents[batch_start : batch_start + batch_rows]
            yield DocumentBatch(
                texts=tuple(row[0] for row in rows),
                next_cursors=tuple(
                    {"index": batch_start + offset + 1} for offset in range(len(rows))
                ),
                document_ids=tuple(row[1] for row in rows),
                urls=tuple(row[2] for row in rows),
                source_dates=tuple(row[3] for row in rows),
            )


class FakeEncoder:
    name = "gpt2"
    implementation = "fixture"
    version = "1"

    def encode_batch(self, texts):
        return [[(ord(character) % 200) + 1 for character in text] for text in texts]


class FailingEncoder(FakeEncoder):
    def encode_batch(self, texts):
        raise RuntimeError("fixture tokenizer failure")


class DifferentEncoder(FakeEncoder):
    version = "2"


class BrokenResponse:
    status = 200
    headers: dict[str, str] = {}

    def getcode(self) -> int:
        return self.status

    def read(self, _: int) -> bytes:
        raise IncompleteRead(b"partial", 100)

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None


def fake_inventory() -> SourceInventory:
    files = tuple(
        SourceFile(
            f"data/train-{index:05d}-of-00100.parquet",
            1,
            f"{index + 1:064x}",
        )
        for index in range(100)
    )
    return SourceInventory(
        "HuggingFaceFW/fineweb_100BT-shuffled",
        "ee8552966e3d6a5fee2f317f2ae0b342be03d998",
        files,
    )


def exclusions() -> ExclusionPolicy:
    return load_fresh10_exclusion_policy(
        REPOSITORY_ROOT / "data" / "manifests" / "fresh10.json"
    )


def documents() -> list[tuple[str, str, str, str]]:
    return [
        (text, f"doc-{index}", f"https://example.test/{index}", "2020-01-01")
        for index, text in enumerate(
            [
                "abcdefghij",
                "klmnop",
                "qrstuvwx",
                "yzabcdef",
                "ghijklmn",
                "opqrstuv",
                "wxyzabcd",
            ]
        )
    ]


def config(root: Path) -> BuildConfig:
    return BuildConfig(
        root=root,
        shard_tokens=7,
        validation_tokens=7,
        reserve_bytes=0,
        batch_rows=3,
        tokenizer_threads=1,
        variant_override=(Variant("tiny", 14), Variant("medium", 28)),
    )


def payload(path: Path) -> list[int]:
    raw = path.read_bytes()[HEADER_BYTES:]
    return list(struct.unpack(f"<{len(raw) // 2}H", raw))


class FineWebBuilderTests(unittest.TestCase):
    def test_exact_shards_resume_inside_document_and_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = FakeSource(documents())
            first = build_fineweb(
                config(root),
                fake_inventory(),
                exclusions(),
                source,
                FakeEncoder(),
                through="medium",
                stop_after_shards=1,
                preflight=False,
            )
            self.assertEqual(first.completed_shards, 1)
            checkpoint = json.loads(
                (root / ".fineweb-build" / "checkpoint.json").read_text()
            )
            self.assertEqual(checkpoint["pending"]["tokens"], 0)
            self.assertGreater(checkpoint["validation_boundary_discarded_tokens"], 0)
            self.assertRegex(
                checkpoint["validation_boundary_document_sha256"], r"^[0-9a-f]{64}$"
            )

            resumed = build_fineweb(
                config(root),
                fake_inventory(),
                exclusions(),
                source,
                FakeEncoder(),
                through="medium",
                preflight=False,
            )
            self.assertEqual(resumed.completed_shards, 4)
            self.assertEqual(
                source.started_at, [0, checkpoint["source_cursor"]["index"]]
            )
            self.assertEqual(
                payload(root / "tiny" / "fineweb_val_000000.bin")[:3],
                [50_256, (ord("a") % 200) + 1, (ord("b") % 200) + 1],
            )
            pool = root / ".fineweb-build" / "shards" / "fineweb_val_000000.bin"
            linked = root / "tiny" / "fineweb_val_000000.bin"
            self.assertEqual(
                (pool.stat().st_dev, pool.stat().st_ino),
                (
                    linked.stat().st_dev,
                    linked.stat().st_ino,
                ),
            )
            prepared = verify_dataset(root / "tiny" / "manifest.json", root / "tiny")
            self.assertEqual(prepared.validation_tokens, 7)
            self.assertEqual(prepared.train_tokens, 7)
            manifest = json.loads((root / "tiny" / "manifest.json").read_text())
            self.assertTrue(
                manifest["preparation"]["validation_train_document_disjoint"]
            )
            self.assertEqual(
                manifest["preparation"]["validation_boundary_discarded_tokens"],
                checkpoint["validation_boundary_discarded_tokens"],
            )

    def test_training_shard_boundary_keeps_document_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_fineweb(
                config(root),
                fake_inventory(),
                exclusions(),
                FakeSource(documents()),
                FakeEncoder(),
                through="medium",
                stop_after_shards=3,
                preflight=False,
            )
            checkpoint = json.loads(
                (root / ".fineweb-build" / "checkpoint.json").read_text()
            )
            self.assertGreater(checkpoint["pending"]["tokens"], 0)

    def test_interrupted_and_uninterrupted_outputs_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            interrupted = base / "interrupted"
            uninterrupted = base / "uninterrupted"
            build_fineweb(
                config(interrupted),
                fake_inventory(),
                exclusions(),
                FakeSource(documents()),
                FakeEncoder(),
                through="medium",
                stop_after_shards=2,
                preflight=False,
            )
            build_fineweb(
                config(interrupted),
                fake_inventory(),
                exclusions(),
                FakeSource(documents()),
                FakeEncoder(),
                through="medium",
                preflight=False,
            )
            build_fineweb(
                config(uninterrupted),
                fake_inventory(),
                exclusions(),
                FakeSource(documents()),
                FakeEncoder(),
                through="medium",
                preflight=False,
            )
            left = sorted((interrupted / "medium").glob("*.bin"))
            right = sorted((uninterrupted / "medium").glob("*.bin"))
            self.assertEqual(
                [path.read_bytes() for path in left],
                [path.read_bytes() for path in right],
            )

    def test_recovers_final_shard_ahead_of_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            expected = base / "expected"
            recovered = base / "recovered"
            build_fineweb(
                config(expected),
                fake_inventory(),
                exclusions(),
                FakeSource(documents()),
                FakeEncoder(),
                through="medium",
                preflight=False,
            )
            build_fineweb(
                config(recovered),
                fake_inventory(),
                exclusions(),
                FakeSource(documents()),
                FakeEncoder(),
                through="medium",
                stop_after_shards=1,
                preflight=False,
            )
            filename = "fineweb_train_000001.bin"
            shutil.copyfile(
                expected / ".fineweb-build" / "shards" / filename,
                recovered / ".fineweb-build" / "shards" / filename,
            )
            result = build_fineweb(
                config(recovered),
                fake_inventory(),
                exclusions(),
                FakeSource(documents()),
                FakeEncoder(),
                through="medium",
                preflight=False,
            )
            self.assertEqual(result.completed_shards, 4)

    def test_tokenizer_failure_leaves_resumable_part(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "fixture tokenizer failure"):
                build_fineweb(
                    config(root),
                    fake_inventory(),
                    exclusions(),
                    FakeSource(documents()),
                    FailingEncoder(),
                    through="tiny",
                    preflight=False,
                )
            part = root / ".fineweb-build" / "shards" / "fineweb_val_000000.bin.part"
            self.assertTrue(part.is_file())
            result = build_fineweb(
                config(root),
                fake_inventory(),
                exclusions(),
                FakeSource(documents()),
                FakeEncoder(),
                through="tiny",
                preflight=False,
            )
            self.assertEqual(result.completed_shards, 2)

    def test_temporal_filter_is_fail_closed_and_prefix_counts_are_stable(self) -> None:
        rows = documents()
        rows.insert(0, ("excluded", "future", "https://future.test", "2024-04-01"))
        rows.insert(1, ("missing", "missing", "https://missing.test", ""))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_fineweb(
                config(root),
                fake_inventory(),
                exclusions(),
                FakeSource(rows),
                FakeEncoder(),
                through="medium",
                preflight=False,
            )
            tiny = json.loads((root / "tiny" / "manifest.json").read_text())
            medium = json.loads((root / "medium" / "manifest.json").read_text())
            tiny_counts = tiny["source"]["excluded_documents_at_prefix_end"]
            medium_counts = medium["source"]["excluded_documents_at_prefix_end"]
            self.assertEqual(
                tiny_counts, {"date_cutoff": 1, "missing_or_invalid_date": 1}
            )
            self.assertEqual(medium_counts, tiny_counts)

    def test_probe_reports_gpt2_token_retention(self) -> None:
        rows = [
            ("abc", "old", "https://old.test", "2020-01-01"),
            ("defgh", "new", "https://new.test", "2025-01-01"),
        ]
        result = probe_fineweb(
            FakeSource(rows),
            FakeEncoder(),
            exclusions(),
            examined_token_target=10,
            batch_rows=2,
        )
        self.assertEqual(result.examined_gpt2_tokens, 10)
        self.assertEqual(result.accepted_gpt2_tokens, 4)
        self.assertEqual(result.exclusion_documents, {"date_cutoff": 1})

    def test_cache_environment_is_forced_under_selected_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = configure_cache_root(directory)
            root = Path(directory).resolve()
            for path in paths.values():
                self.assertTrue(path.is_relative_to(root))
            self.assertTrue(Path(os.environ["HF_HOME"]).is_relative_to(root))
            self.assertTrue(Path(os.environ["TIKTOKEN_CACHE_DIR"]).is_relative_to(root))

    def test_selected_root_symlink_allowed_but_child_symlink_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "target"
            target.mkdir()
            selected = base / "selected"
            selected.symlink_to(target, target_is_directory=True)
            configure_cache_root(selected)
            shutil.rmtree(target / ".cache")
            outside = base / "outside"
            outside.mkdir()
            (target / ".cache").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(FineWebBuildError, "symlink below"):
                configure_cache_root(selected)

            (target / ".cache").unlink()
            (target / ".fineweb-build").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(FineWebBuildError, "symlink below"):
                ensure_build_work_directory(selected)

    def test_source_read_failure_is_normalized_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / ".cache" / "source"
            inventory = SourceInventory(
                fake_inventory().repository,
                fake_inventory().revision,
                (SourceFile("data/train-00000-of-00100.parquet", 100, "a" * 64),),
            )
            cache = SourceFileCache(source_root, inventory)
            with (
                patch(
                    "speedrun.fineweb_builder.urlopen", return_value=BrokenResponse()
                ),
                self.assertRaisesRegex(FineWebBuildError, "cannot cache source file"),
            ):
                cache.get(0)
            self.assertTrue(cache.part_path.is_file())

    def test_mismatched_resume_does_not_overwrite_saved_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_fineweb(
                config(root),
                fake_inventory(),
                exclusions(),
                FakeSource(documents()),
                FakeEncoder(),
                through="tiny",
                stop_after_shards=1,
                preflight=False,
            )
            plan = root / ".fineweb-build" / "plan.json"
            original = plan.read_bytes()
            with self.assertRaisesRegex(FineWebBuildError, "differ"):
                build_fineweb(
                    config(root),
                    fake_inventory(),
                    exclusions(),
                    FakeSource(documents()),
                    DifferentEncoder(),
                    through="tiny",
                    preflight=False,
                )
            self.assertEqual(plan.read_bytes(), original)

    def test_url_normalization(self) -> None:
        self.assertEqual(
            normalize_url("HTTPS://Example.COM:443/path?z=2&a=1#fragment"),
            "https://example.com/path?a=1&z=2",
        )

    def test_raw_fresh10_hash_is_checked(self) -> None:
        text = "raw fixture"
        import hashlib

        policy = ExclusionPolicy(
            "2024-04-01",
            exclusions().normalized_urls,
            frozenset(),
            frozenset({hashlib.sha256(text.encode()).hexdigest()}),
        )
        self.assertEqual(
            policy.reason(text, "https://unrelated.test", "2020-01-01"),
            "fresh10_raw_sha256",
        )

    def test_publication_manifest_uses_immutable_shard_revision(self) -> None:
        manifest = {
            "source": {},
            "files": [{"path": "fineweb_val_000000.bin", "sha256": "a" * 64}],
        }
        revision = "b" * 40
        result = publication_manifest(manifest, "quintic/example", "2B", revision)
        self.assertIn(
            f"/resolve/{revision}/2B/fineweb_val_000000.bin",
            result["files"][0]["url"],
        )
        self.assertNotIn("url", manifest["files"][0])

    def test_token_file_requires_private_mode_and_exact_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env.hf"
            path.write_text("HF_TOKEN=hf_" + "a" * 30 + "\n")
            path.chmod(0o600)
            self.assertTrue(read_token_file(path).startswith("hf_"))
            path.chmod(0o644)
            with self.assertRaisesRegex(FineWebBuildError, "group/world"):
                read_token_file(path)

    def test_remote_tree_validates_selected_sha256_metadata(self) -> None:
        selected = [{"path": "x.bin", "bytes": 12, "sha256": "a" * 64}]
        tree = [
            {
                "type": "file",
                "path": "2B/x.bin",
                "size": 12,
                "lfs": {"oid": "a" * 64, "size": 12},
            }
        ]
        validate_remote_tree_entries(tree, "2B", selected)
        tree[0]["lfs"]["oid"] = "b" * 64
        with self.assertRaisesRegex(FineWebBuildError, "SHA-256"):
            validate_remote_tree_entries(tree, "2B", selected)

    def test_hub_tree_next_link(self) -> None:
        header = (
            '<https://huggingface.co/api/tree?cursor=abc&limit=100>; rel="next", '
            '<https://huggingface.co/api/tree?cursor=last>; rel="last"'
        )
        self.assertEqual(
            next_link(header),
            "https://huggingface.co/api/tree?cursor=abc&limit=100",
        )
        self.assertIsNone(next_link(""))


if __name__ == "__main__":
    unittest.main()
