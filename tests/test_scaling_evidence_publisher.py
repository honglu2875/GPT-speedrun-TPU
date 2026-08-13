from __future__ import annotations

import json
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.publish_scaling_evidence import (
    LAUNCH_COMMIT,
    MANIFEST_NAME,
    RUN_FILES,
    SEMANTIC_FLAGS,
    EvidenceError,
    anonymous_revalidate,
    canonical_json_bytes,
    deterministic_git_archive,
    discover_runs_source,
    hash_regular_file,
    publication_receipt,
    publish_archive,
    read_token_file,
    scan_regular_tree,
    semantic_verify,
    sha256_bytes,
    snapshot_bundle,
    validate_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN = "c025_n023_lr200"
SELECTION = "runs/learning-rate-selections/c025/n023.json"
SOURCE_PATH = f"provenance/source-{LAUNCH_COMMIT}.tar"


def _base_files() -> dict[str, tuple[bytes, str]]:
    files: dict[str, tuple[bytes, str]] = {
        "README.md": (b"readme\n", "documentation"),
        "verify.py": (b"# verifier\n", "verifier"),
        SOURCE_PATH: (b"tar bytes", "source_archive"),
        "runs/fit.json": (b"{}\n", "final_fit_json"),
        "runs/fit.md": (b"fit\n", "final_fit_markdown"),
        "runs/fits/c025.json": (b"{}\n", "slice_fit"),
        "runs/fits/c050.json": (b"{}\n", "slice_fit"),
        "runs/fits/c100.json": (b"{}\n", "slice_fit"),
        SELECTION: (b"{}\n", "learning_rate_selection"),
    }
    for relative, role in RUN_FILES:
        files[f"runs/{RUN}/{relative}"] = (b"{}\n", role)
    return files


def _manifest(files: dict[str, tuple[bytes, str]] | None = None) -> dict[str, object]:
    files = files or _base_files()
    inventory = [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "role": role,
        }
        for path, (payload, role) in sorted(files.items())
    ]
    archive_id = (
        "current_budget_isoflop_v4-"
        f"{sha256_bytes(canonical_json_bytes(inventory))[:16]}-"
        f"{LAUNCH_COMMIT[:12]}"
    )
    return {
        "schema_version": 1,
        "kind": "gpt_tpu_speedrun_scaling_evidence",
        "archive_id": archive_id,
        "identity": {
            "suite_id": "current_budget_isoflop_v4",
            "suite_sha256": "1" * 64,
            "template_sha256": "2" * 64,
            "execution_fingerprint": "3" * 64,
            "trainer_sha256": "4" * 64,
            "seed": 1337,
            "launch_commit": LAUNCH_COMMIT,
        },
        "publication_target": {
            "repository": "quintic/gpt-tpu-speedrun-scaling-evidence",
            "directory": f"current_budget_isoflop_v4/{archive_id}",
        },
        "source_archive": {
            "path": SOURCE_PATH,
            "commit": LAUNCH_COMMIT,
            "prefix": "source/",
            "bytes": len(files[SOURCE_PATH][0]),
            "sha256": sha256_bytes(files[SOURCE_PATH][0]),
        },
        "study": {
            "runs": [RUN],
            "classifications": {
                "stable": [RUN],
                "suspect": [],
                "rejected": [],
            },
            "learning_rate_selections": [SELECTION],
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
        },
        "inventory": {
            "file_count": len(inventory),
            "total_bytes": sum(item["bytes"] for item in inventory),
            "files": inventory,
        },
    }


def _write_bundle(root: Path) -> dict[str, object]:
    files = _base_files()
    manifest = _manifest(files)
    for relative, (payload, _role) in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root / MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest))
    return manifest


class ScalingEvidencePublisherTests(unittest.TestCase):
    def test_manifest_has_exact_schema_and_exact_fifteen_file_run(self) -> None:
        manifest = _manifest()
        self.assertEqual(validate_manifest(manifest), manifest)

        extra = json.loads(json.dumps(manifest))
        extra["surprise"] = True
        with self.assertRaisesRegex(EvidenceError, "wrong schema"):
            validate_manifest(extra)

        missing = json.loads(json.dumps(manifest))
        missing["inventory"]["files"] = [
            item
            for item in missing["inventory"]["files"]
            if item["path"] != f"runs/{RUN}/artifacts/metrics.json"
        ]
        missing["inventory"]["file_count"] -= 1
        missing["inventory"]["total_bytes"] -= 3
        with self.assertRaisesRegex(EvidenceError, "exact closed v4 archive"):
            validate_manifest(missing)

    def test_snapshot_rejects_extra_file_empty_directory_links_and_hash_drift(self) -> None:
        cases = ("extra", "empty", "link", "hardlink", "drift")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "bundle"
                root.mkdir()
                _write_bundle(root)
                if case == "extra":
                    (root / "extra.log").write_text("extra\n", encoding="utf-8")
                elif case == "empty":
                    (root / "empty").mkdir()
                elif case == "link":
                    target = root / "README.md"
                    target.unlink()
                    target.symlink_to(root / "verify.py")
                elif case == "hardlink":
                    target = root / "README.md"
                    target.unlink()
                    os.link(root / "verify.py", target)
                else:
                    (root / "README.md").write_text("changed\n", encoding="utf-8")
                with self.assertRaises(EvidenceError):
                    snapshot_bundle(root, Path(directory) / "snapshot")

    def test_snapshot_copies_bytes_and_never_hardlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            root.mkdir()
            _write_bundle(root)
            snapshot = Path(directory) / "snapshot"
            snapshot_bundle(root, snapshot)
            self.assertEqual(scan_regular_tree(root).keys(), scan_regular_tree(snapshot).keys())
            self.assertEqual(
                (root / "README.md").read_bytes(), (snapshot / "README.md").read_bytes()
            )
            self.assertNotEqual(
                (root / "README.md").stat().st_ino,
                (snapshot / "README.md").stat().st_ino,
            )

    def test_source_run_inventory_is_strict_and_observes_empty_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative, _role in RUN_FILES:
                path = root / RUN / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            for relative in (
                "fit.json",
                "fit.md",
                "fits/c025.json",
                "fits/c050.json",
                "fits/c100.json",
                "learning-rate-selections/c025/n023.json",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            runs, selections, _inventory = discover_runs_source(root, {RUN})
            self.assertEqual(runs, [RUN])
            self.assertEqual(selections, ["learning-rate-selections/c025/n023.json"])
            (root / "unexpected-empty").mkdir()
            with self.assertRaisesRegex(EvidenceError, "directory tree is not closed"):
                discover_runs_source(root, {RUN})

    def test_deterministic_source_archive_is_pinned_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.tar"
            second = Path(directory) / "second.tar"
            one = deterministic_git_archive(REPOSITORY_ROOT, LAUNCH_COMMIT, first)
            two = deterministic_git_archive(REPOSITORY_ROOT, LAUNCH_COMMIT, second)
            self.assertEqual(one, two)
            self.assertEqual(one, hash_regular_file(first))
            self.assertGreater(one[0], 0)

    def test_token_file_requires_owned_regular_private_single_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_file = Path(directory) / "token"
            token = "hf_" + "A" * 24
            token_file.write_text(f"HF_TOKEN={token}\n", encoding="utf-8")
            token_file.chmod(0o600)
            self.assertEqual(read_token_file(token_file), token)
            token_file.chmod(0o644)
            with self.assertRaisesRegex(EvidenceError, "mode 0600"):
                read_token_file(token_file)
            token_file.unlink()
            target = Path(directory) / "target"
            target.write_text(f"HF_TOKEN={token}\n", encoding="utf-8")
            target.chmod(0o600)
            token_file.symlink_to(target)
            with self.assertRaisesRegex(EvidenceError, "not a link"):
                read_token_file(token_file)

    def test_receipt_pins_manifest_tree_revision_and_all_semantic_flags(self) -> None:
        manifest = _manifest()
        verification = {
            "semantic_verification": {flag: True for flag in SEMANTIC_FLAGS}
        }
        receipt = publication_receipt(
            manifest=manifest,
            revision="a" * 40,
            verification=verification,
        )
        self.assertEqual(receipt["publication"]["revision"], "a" * 40)
        self.assertTrue(receipt["anonymous_verification"]["closed_tree_inventory"])
        self.assertTrue(receipt["anonymous_verification"]["all_file_sha256"])
        self.assertEqual(receipt["inventory"]["files"], manifest["inventory"]["files"])
        self.assertEqual(
            receipt["archive_manifest"]["sha256"],
            sha256_bytes(canonical_json_bytes(manifest)),
        )

    def test_anonymous_revalidation_rejects_remote_extra_before_download(self) -> None:
        manifest = _manifest()
        directory = manifest["publication_target"]["directory"]

        class Api:
            def list_repo_tree(self, **_kwargs):
                entries = []
                for relative in [MANIFEST_NAME, *sorted(_base_files())]:
                    entries.append(
                        SimpleNamespace(
                            path=f"{directory}/{relative}", type="file", size=None
                        )
                    )
                entries.append(
                    SimpleNamespace(path=f"{directory}/extra", type="file", size=1)
                )
                return entries

        with patch("scripts.publish_scaling_evidence.anonymous_download") as download:
            with self.assertRaisesRegex(EvidenceError, "not closed"):
                anonymous_revalidate(
                    Api(),
                    repo_id="quintic/gpt-tpu-speedrun-scaling-evidence",
                    revision="a" * 40,
                    directory=str(directory),
                    manifest=manifest,
                )
            download.assert_not_called()

    def test_publish_verifies_before_token_and_uses_one_public_folder_commit(self) -> None:
        manifest = _manifest()
        events: list[str] = []
        apis: list[object] = []

        class Api:
            def __init__(self, token=None):
                self.token = token
                self.create_kwargs = None
                self.upload_kwargs = None
                apis.append(self)

            def create_repo(self, **kwargs):
                self.create_kwargs = kwargs

            def upload_folder(self, **kwargs):
                self.upload_kwargs = kwargs
                return SimpleNamespace(oid="a" * 40)

        hub = ModuleType("huggingface_hub")
        hub.HfApi = Api

        def verified(_bundle):
            events.append("verify")
            return {}

        def token(_path):
            events.append("token")
            return "hf_" + "A" * 24

        remote_verification = {
            "semantic_verification": {flag: True for flag in SEMANTIC_FLAGS}
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"huggingface_hub": hub}
        ), patch(
            "scripts.publish_scaling_evidence.verify_bundle", side_effect=verified
        ), patch(
            "scripts.publish_scaling_evidence.read_token_file", side_effect=token
        ), patch(
            "scripts.publish_scaling_evidence.snapshot_bundle", return_value=manifest
        ), patch(
            "scripts.publish_scaling_evidence.semantic_verify"
        ), patch(
            "scripts.publish_scaling_evidence.anonymous_revalidate",
            return_value=remote_verification,
        ):
            receipt_path = Path(directory) / "receipt.json"
            receipt = publish_archive(
                bundle=Path(directory) / "bundle",
                manifest=manifest,
                token_file=Path(directory) / "token",
                receipt_output=receipt_path,
            )
            self.assertEqual(
                json.loads(receipt_path.read_text(encoding="utf-8")), receipt
            )
        self.assertEqual(events, ["verify", "token"])
        self.assertEqual(len(apis), 2)
        authenticated, anonymous = apis
        self.assertTrue(authenticated.token.startswith("hf_"))
        self.assertIs(anonymous.token, False)
        self.assertIs(authenticated.create_kwargs["private"], False)
        self.assertEqual(
            authenticated.upload_kwargs["path_in_repo"],
            manifest["publication_target"]["directory"],
        )

    def test_semantic_verifier_recomputes_every_derived_layer(self) -> None:
        fit = {
            "can_estimate_scaling_exponent": True,
            "scaling_law": {"parameter_exponent_a": 0.5},
            "slices": [
                {"slice": "c025", "value": 1},
                {"slice": "c050", "value": 2},
                {"slice": "c100", "value": 3},
            ],
        }
        files = _base_files()
        pretty_fit = (json.dumps(fit, indent=2, sort_keys=True) + "\n").encode()
        files["runs/fit.json"] = (pretty_fit, "final_fit_json")
        files["runs/fit.md"] = (b"rendered fit\n", "final_fit_markdown")
        for item in fit["slices"]:
            files[f"runs/fits/{item['slice']}.json"] = (
                canonical_json_bytes(item),
                "slice_fit",
            )
        manifest = _manifest(files)
        identity = manifest["identity"]
        calls = {"run": 0, "selection": 0, "fit": 0, "write": 0}

        class FakeScaling:
            @staticmethod
            def load_suite(_path):
                return {
                    "suite_id": identity["suite_id"],
                    "suite_sha256": identity["suite_sha256"],
                    "template_sha256": identity["template_sha256"],
                    "execution_fingerprint": identity["execution_fingerprint"],
                    "trainer_source_sha256": identity["trainer_sha256"],
                    "seed": identity["seed"],
                    "all_variants": [{"id": RUN}],
                }

            @staticmethod
            def _read_run(_suite, _point, _runs):
                calls["run"] += 1
                return {
                    "stability_admission": {"classification": "stable"}
                }

            @staticmethod
            def select_learning_rate(_suite, **_kwargs):
                calls["selection"] += 1
                return {}

            @staticmethod
            def fit_results(_suite, _runs):
                calls["fit"] += 1
                return fit

            @staticmethod
            def write_fit(result, output):
                calls["write"] += 1
                output.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                markdown = output.with_suffix(".md")
                markdown.write_bytes(b"rendered fit\n")
                return output, markdown

        @contextmanager
        def fake_archived_module(_bundle, _manifest):
            yield FakeScaling(), Path("/source")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative, (payload, _role) in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            with patch(
                "scripts.publish_scaling_evidence.archived_scaling_module",
                fake_archived_module,
            ), patch(
                "scripts.publish_scaling_evidence.SOURCE_ARCHIVE_BYTES",
                len(files[SOURCE_PATH][0]),
            ), patch(
                "scripts.publish_scaling_evidence.SOURCE_ARCHIVE_SHA256",
                sha256_bytes(files[SOURCE_PATH][0]),
            ):
                flags = semantic_verify(root, manifest)
        self.assertEqual(flags, {flag: True for flag in SEMANTIC_FLAGS})
        self.assertEqual(calls, {"run": 1, "selection": 1, "fit": 1, "write": 1})

    def test_dry_run_cli_never_reads_token_or_publishes(self) -> None:
        from scripts.publish_scaling_evidence import main

        manifest = _manifest()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "scripts.publish_scaling_evidence.build_archive",
                return_value=(Path(directory) / "bundle", manifest),
            ),
            patch("scripts.publish_scaling_evidence.read_token_file") as token,
            patch("scripts.publish_scaling_evidence.publish_archive") as publish,
        ):
            result = main(
                [
                    "build",
                    "--runs",
                    str(Path(directory) / "runs"),
                    "--output",
                    str(Path(directory) / "bundle"),
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        token.assert_not_called()
        publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
