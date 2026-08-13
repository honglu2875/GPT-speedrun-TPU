from __future__ import annotations

import json
from contextlib import contextmanager
import io
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
    MAX_TOKEN_BYTES,
    PUBLISHED_4B_MANIFEST_CANONICAL_SHA256,
    PUBLISHED_4B_MANIFEST_PATH,
    PUBLISHED_4B_MANIFEST_SHA256,
    RUN_FILES,
    SEMANTIC_FLAGS,
    EvidenceError,
    anonymous_download,
    anonymous_revalidate,
    build_archive,
    canonical_json_bytes,
    deterministic_git_archive,
    discover_runs_source,
    hash_regular_file,
    publication_receipt,
    prospective_selection_groups,
    publish_archive,
    read_token_file,
    scan_regular_tree,
    semantic_verify,
    sha256_bytes,
    snapshot_bundle,
    validate_publication_paths,
    validate_published_4b_binding,
    validate_manifest,
    validate_repo_id,
    write_atomic,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN = "c025_n023_lr200"
CONTROL = "c100_n124_control"
SELECTION = "runs/learning-rate-selections/c025/n023.json"
SOURCE_PATH = f"provenance/source-{LAUNCH_COMMIT}.tar"


def _base_files(
    *,
    run_names: tuple[str, ...] = (RUN,),
    selections: tuple[str, ...] = (SELECTION,),
) -> dict[str, tuple[bytes, str]]:
    files: dict[str, tuple[bytes, str]] = {
        "README.md": (b"readme\n", "documentation"),
        "verify.py": (b"# verifier\n", "verifier"),
        SOURCE_PATH: (b"tar bytes", "source_archive"),
        "runs/fit.json": (b"{}\n", "final_fit_json"),
        "runs/fit.md": (b"fit\n", "final_fit_markdown"),
        "runs/fits/c025.json": (b"{}\n", "slice_fit"),
        "runs/fits/c050.json": (b"{}\n", "slice_fit"),
        "runs/fits/c100.json": (b"{}\n", "slice_fit"),
    }
    for selection in selections:
        files[selection] = (b"{}\n", "learning_rate_selection")
    for run_name in run_names:
        for relative, role in RUN_FILES:
            files[f"runs/{run_name}/{relative}"] = (b"{}\n", role)
    return files


def _manifest(
    files: dict[str, tuple[bytes, str]] | None = None,
    *,
    run_names: tuple[str, ...] = (RUN,),
    selections: tuple[str, ...] = (SELECTION,),
) -> dict[str, object]:
    run_names = tuple(sorted(run_names))
    selections = tuple(sorted(selections))
    files = files or _base_files(run_names=run_names, selections=selections)
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
            "runs": list(run_names),
            "classifications": {
                "stable": list(run_names),
                "suspect": [],
                "rejected": [],
            },
            "learning_rate_selections": list(selections),
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


def _write_bundle(
    root: Path,
    *,
    run_names: tuple[str, ...] = (RUN,),
    selections: tuple[str, ...] = (SELECTION,),
) -> dict[str, object]:
    files = _base_files(run_names=run_names, selections=selections)
    manifest = _manifest(files, run_names=run_names, selections=selections)
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
            with self.assertRaisesRegex(EvidenceError, "no linked component"):
                read_token_file(token_file)

    def test_token_read_is_bounded_and_rejects_hardlinks_parent_links_and_swaps(self) -> None:
        token = "hf_" + "A" * 24
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_file = root / "token"
            token_file.write_bytes(b"x" * (MAX_TOKEN_BYTES + 1))
            token_file.chmod(0o600)
            with self.assertRaisesRegex(EvidenceError, "exceeds"):
                read_token_file(token_file)

            token_file.write_text(f"HF_TOKEN={token}\n", encoding="utf-8")
            token_file.chmod(0o600)
            alias = root / "alias"
            os.link(token_file, alias)
            with self.assertRaisesRegex(EvidenceError, "non-hard-linked"):
                read_token_file(token_file)
            alias.unlink()

            real_parent = root / "real"
            real_parent.mkdir()
            nested = real_parent / "token"
            nested.write_text(f"HF_TOKEN={token}\n", encoding="utf-8")
            nested.chmod(0o600)
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(EvidenceError, "linked"):
                read_token_file(linked_parent / "token")

            with patch(
                "scripts.publish_scaling_evidence._directory_entry_signature",
                return_value=None,
            ):
                with self.assertRaisesRegex(EvidenceError, "changed while reading"):
                    read_token_file(nested)

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
        manifest_size = len(canonical_json_bytes(manifest))
        sizes = {
            MANIFEST_NAME: manifest_size,
            **{
                item["path"]: item["bytes"]
                for item in manifest["inventory"]["files"]
            },
        }

        class Api:
            def repo_info(self, **_kwargs):
                return SimpleNamespace(sha="a" * 40)

            def list_repo_tree(self, **_kwargs):
                entries = []
                for relative in [MANIFEST_NAME, *sorted(_base_files())]:
                    entries.append(
                        SimpleNamespace(
                            path=f"{directory}/{relative}",
                            type="file",
                            size=sizes[relative],
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

    def test_publish_verifies_before_token_and_uploads_one_exact_snapshot(self) -> None:
        events: list[str] = []
        apis: list[object] = []
        uploaded: list[tuple[str, bytes, int]] = []
        verified_snapshot: list[Path] = []

        class Add:
            def __init__(self, *, path_in_repo, path_or_fileobj):
                self.path_in_repo = path_in_repo
                self.path_or_fileobj = path_or_fileobj

        class Api:
            def __init__(self, token=None):
                self.token = token
                self.create_kwargs = None
                self.commit_kwargs = None
                apis.append(self)

            def create_repo(self, **kwargs):
                self.create_kwargs = kwargs

            def repo_info(self, **_kwargs):
                return SimpleNamespace(sha="b" * 40)

            def list_repo_files(self, **_kwargs):
                return []

            def create_commit(self, **kwargs):
                self.commit_kwargs = kwargs
                for operation in kwargs["operations"]:
                    uploaded.append(
                        (
                            operation.path_in_repo,
                            operation.path_or_fileobj.read(),
                            operation.path_or_fileobj.fileno(),
                        )
                    )
                return SimpleNamespace(oid="a" * 40)

        hub = ModuleType("huggingface_hub")
        hub.HfApi = Api
        hub.CommitOperationAdd = Add

        def verified(snapshot, _manifest, **_kwargs):
            events.append("verify")
            verified_snapshot.append(snapshot)
            return {flag: True for flag in SEMANTIC_FLAGS}

        def token(_path):
            events.append("token")
            return "hf_" + "A" * 24

        remote_verification = {
            "semantic_verification": {flag: True for flag in SEMANTIC_FLAGS}
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            manifest = _write_bundle(bundle)
            token_path = root / "token"
            token_path.write_text("unused\n", encoding="utf-8")
            token_path.chmod(0o600)
            receipt_path = Path(directory) / "receipt.json"
            with patch.dict(
                sys.modules, {"huggingface_hub": hub}
            ), patch(
                "scripts.publish_scaling_evidence.verify_frozen_snapshot",
                side_effect=verified,
            ), patch(
                "scripts.publish_scaling_evidence.read_token_file", side_effect=token
            ), patch(
                "scripts.publish_scaling_evidence.snapshot_bundle",
                wraps=snapshot_bundle,
            ) as snapshot, patch(
                "scripts.publish_scaling_evidence.anonymous_revalidate",
                return_value=remote_verification,
            ):
                receipt = publish_archive(
                    bundle=bundle,
                    token_file=token_path,
                    receipt_output=receipt_path,
                    expected_manifest=manifest,
                )
                self.assertEqual(snapshot.call_count, 1)
            self.assertEqual(
                json.loads(receipt_path.read_text(encoding="utf-8")), receipt
            )
        self.assertEqual(events, ["verify", "token"])
        self.assertEqual(len(apis), 2)
        authenticated, anonymous = apis
        self.assertTrue(authenticated.token.startswith("hf_"))
        self.assertIs(anonymous.token, False)
        self.assertIs(authenticated.create_kwargs["private"], False)
        self.assertEqual(authenticated.commit_kwargs["parent_commit"], "b" * 40)
        expected_relatives = {
            MANIFEST_NAME,
            *(item["path"] for item in manifest["inventory"]["files"]),
        }
        self.assertEqual(
            {path for path, _payload, _source in uploaded},
            {
                f"{manifest['publication_target']['directory']}/{relative}"
                for relative in expected_relatives
            },
        )
        self.assertEqual(len(verified_snapshot), 1)
        expected_payloads = {
            MANIFEST_NAME: canonical_json_bytes(manifest),
            **{path: payload for path, (payload, _role) in _base_files().items()},
        }
        for path, payload, descriptor in uploaded:
            relative = path.removeprefix(
                f"{manifest['publication_target']['directory']}/"
            )
            self.assertEqual(payload, expected_payloads[relative])
            self.assertIsInstance(descriptor, int)

    def test_semantic_verifier_recomputes_every_derived_layer(self) -> None:
        fit = {
            "can_estimate_scaling_exponent": True,
            "scaling_law": {"parameter_exponent_a": 0.5},
            "controls": [{"id": CONTROL}],
            "slices": [
                {"slice": "c025", "value": 1},
                {"slice": "c050", "value": 2},
                {"slice": "c100", "value": 3},
            ],
        }
        run_names = tuple(sorted((RUN, CONTROL)))
        files = _base_files(run_names=run_names)
        pretty_fit = (json.dumps(fit, indent=2, sort_keys=True) + "\n").encode()
        files["runs/fit.json"] = (pretty_fit, "final_fit_json")
        files["runs/fit.md"] = (b"rendered fit\n", "final_fit_markdown")
        for item in fit["slices"]:
            files[f"runs/fits/{item['slice']}.json"] = (
                canonical_json_bytes(item),
                "slice_fit",
            )
        selection = {"candidates": [{"id": RUN}]}
        files[SELECTION] = (canonical_json_bytes(selection), "learning_rate_selection")
        manifest = _manifest(files, run_names=run_names)
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
                    "all_variants": [{"id": RUN}, {"id": CONTROL}],
                    "controls": [{"id": CONTROL}],
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
                return selection

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
            ), patch(
                "scripts.publish_scaling_evidence.validate_published_4b_binding"
            ), patch(
                "scripts.publish_scaling_evidence.prospective_selection_groups",
                return_value={("c025", "n023")},
            ):
                flags = semantic_verify(root, manifest)
        self.assertEqual(flags, {flag: True for flag in SEMANTIC_FLAGS})
        self.assertEqual(calls, {"run": 2, "selection": 1, "fit": 1, "write": 1})

    def test_atomic_receipt_rejects_leaf_and_parent_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim"
            victim.write_bytes(b"unchanged")
            linked_output = root / "receipt"
            linked_output.symlink_to(victim)
            with self.assertRaisesRegex(EvidenceError, "regular file"):
                write_atomic(linked_output, b"malicious")
            self.assertEqual(victim.read_bytes(), b"unchanged")

            real_parent = root / "real"
            real_parent.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(EvidenceError, "linked"):
                write_atomic(linked_parent / "receipt", b"malicious")
            self.assertFalse((real_parent / "receipt").exists())

            regular = root / "regular"
            regular.write_bytes(b"old")
            write_atomic(regular, b"new")
            self.assertEqual(regular.read_bytes(), b"new")
            self.assertEqual(regular.stat().st_mode & 0o777, 0o600)

    def test_publication_paths_reject_receipt_and_token_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            _write_bundle(bundle)
            token = root / "token"
            token.write_text("HF_TOKEN=hf_" + "A" * 24 + "\n", encoding="utf-8")
            token.chmod(0o600)
            with self.assertRaisesRegex(EvidenceError, "receipt path must be disjoint"):
                validate_publication_paths(
                    bundle=bundle,
                    token_file=token,
                    receipt_output=bundle / "receipt.json",
                )
            with self.assertRaisesRegex(EvidenceError, "must not alias"):
                validate_publication_paths(
                    bundle=bundle,
                    token_file=token,
                    receipt_output=token,
                )

            alias = root / "receipt-alias"
            os.link(bundle / "README.md", alias)
            with self.assertRaisesRegex(EvidenceError, "hard-linked|aliases a bundle"):
                validate_publication_paths(
                    bundle=bundle,
                    token_file=token,
                    receipt_output=alias,
                )

    def test_publication_allows_one_resolved_bundle_root_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            _write_bundle(bundle)
            alias = root / "bundle-alias"
            alias.symlink_to(bundle, target_is_directory=True)
            token = root / "token"
            token.write_text("HF_TOKEN=hf_" + "A" * 24 + "\n", encoding="utf-8")
            token.chmod(0o600)
            validate_publication_paths(
                bundle=alias,
                token_file=token,
                receipt_output=root / "receipt.json",
            )

    def test_disjoint_failure_happens_before_token_or_hub_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            _write_bundle(bundle)
            token = root / "token"
            token.write_text("unused\n", encoding="utf-8")
            token.chmod(0o600)
            with patch("scripts.publish_scaling_evidence.read_token_file") as read:
                with self.assertRaisesRegex(EvidenceError, "receipt path must be disjoint"):
                    publish_archive(
                        bundle=bundle,
                        token_file=token,
                        receipt_output=bundle / "receipt.json",
                    )
            read.assert_not_called()

    def test_publish_collision_is_safe_resume_without_new_commit(self) -> None:
        apis: list[object] = []
        revisions: list[str] = []

        class Add:
            def __init__(self, **_kwargs):
                raise AssertionError("collision resume must not construct upload operations")

        class Api:
            def __init__(self, token=None):
                self.token = token
                apis.append(self)

            def create_repo(self, **_kwargs):
                return None

            def repo_info(self, **_kwargs):
                return SimpleNamespace(sha="c" * 40)

            def list_repo_files(self, **_kwargs):
                return [f"{self.remote_directory}/archive-manifest.json"]

            def create_commit(self, **_kwargs):
                raise AssertionError("collision resume must not create a commit")

        hub = ModuleType("huggingface_hub")
        hub.HfApi = Api
        hub.CommitOperationAdd = Add
        verification = {
            "semantic_verification": {flag: True for flag in SEMANTIC_FLAGS}
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            manifest = _write_bundle(bundle)
            Api.remote_directory = manifest["publication_target"]["directory"]
            token_file = root / "token"
            token_file.write_text("unused\n", encoding="utf-8")
            token_file.chmod(0o600)

            def anonymous(_api, **kwargs):
                revisions.append(kwargs["revision"])
                return verification

            with patch.dict(sys.modules, {"huggingface_hub": hub}), patch(
                "scripts.publish_scaling_evidence.verify_frozen_snapshot",
                return_value={flag: True for flag in SEMANTIC_FLAGS},
            ), patch(
                "scripts.publish_scaling_evidence.read_token_file",
                return_value="hf_" + "A" * 24,
            ), patch(
                "scripts.publish_scaling_evidence.anonymous_revalidate",
                side_effect=anonymous,
            ):
                publish_archive(
                    bundle=bundle,
                    token_file=token_file,
                    receipt_output=root / "receipt.json",
                )
        self.assertEqual(revisions, ["c" * 40])
        self.assertEqual(len(apis), 2)

    def test_anonymous_revision_must_be_exact_oid_and_resolve_to_itself(self) -> None:
        manifest = _manifest()

        class Api:
            def __init__(self, resolved):
                self.resolved = resolved
                self.calls = 0

            def repo_info(self, **_kwargs):
                self.calls += 1
                return SimpleNamespace(sha=self.resolved)

        long_oid = "a" * 64
        api = Api(long_oid)
        with self.assertRaisesRegex(EvidenceError, "40-hex"):
            anonymous_revalidate(
                api,
                repo_id="quintic/gpt-tpu-speedrun-scaling-evidence",
                revision=long_oid,
                directory=manifest["publication_target"]["directory"],
                manifest=manifest,
            )
        self.assertEqual(api.calls, 0)

        api = Api("b" * 40)
        with self.assertRaisesRegex(EvidenceError, "different commit"):
            anonymous_revalidate(
                api,
                repo_id="quintic/gpt-tpu-speedrun-scaling-evidence",
                revision="a" * 40,
                directory=manifest["publication_target"]["directory"],
                manifest=manifest,
            )
        self.assertEqual(api.calls, 1)

    def test_anonymous_remote_requires_sizes_and_stream_cap(self) -> None:
        manifest = _manifest()
        directory = manifest["publication_target"]["directory"]

        class Api:
            def repo_info(self, **_kwargs):
                return SimpleNamespace(sha="a" * 40)

            def list_repo_tree(self, **_kwargs):
                return [
                    SimpleNamespace(
                        path=f"{directory}/{MANIFEST_NAME}", type="file", size=None
                    )
                ]

        with self.assertRaisesRegex(EvidenceError, "invalid size"):
            anonymous_revalidate(
                Api(),
                repo_id="quintic/gpt-tpu-speedrun-scaling-evidence",
                revision="a" * 40,
                directory=directory,
                manifest=manifest,
            )

        with tempfile.TemporaryDirectory() as temp, patch(
            "scripts.publish_scaling_evidence.urlopen",
            return_value=io.BytesIO(b"abc"),
        ):
            destination = Path(temp) / "object"
            with self.assertRaisesRegex(EvidenceError, "exceeds"):
                anonymous_download(
                    "https://example.invalid/object",
                    destination,
                    expected_bytes=2,
                )
            self.assertFalse(destination.exists())

    def test_repo_id_matches_hugging_face_contract(self) -> None:
        self.assertEqual(validate_repo_id("owner/repository_1"), "owner/repository_1")
        for invalid in (
            "missing-namespace",
            "/repository",
            "owner/",
            "owner/.hidden",
            "owner/trailing-",
            "owner/double--dash",
            "owner/double..dot",
            "owner/repository.git",
            "a/" + "b" * 95,
        ):
            with self.subTest(invalid=invalid), self.assertRaises(EvidenceError):
                validate_repo_id(invalid)

    def test_manifest_enforces_per_role_byte_bounds(self) -> None:
        manifest = _manifest()
        oversized = json.loads(json.dumps(manifest))
        item = next(
            value
            for value in oversized["inventory"]["files"]
            if value["path"] == "README.md"
        )
        delta = 2 * 1024 * 1024 + 1 - item["bytes"]
        item["bytes"] += delta
        oversized["inventory"]["total_bytes"] += delta
        with self.assertRaisesRegex(EvidenceError, "documentation byte bound"):
            validate_manifest(oversized)

    @staticmethod
    def _prospective_fixture():
        def point(identifier, rate, *, role="calibration"):
            return {
                "id": identifier,
                "slice": "c025",
                "shape_id": "n023",
                "learning_rate": rate,
                "role": role,
            }

        initial = [
            point("c025_n023_lr200", 0.0002),
            point("c025_n023_lr300", 0.0003),
            point("c025_n023_lr450", 0.00045),
        ]
        lower = [
            point("c025_n023_lr133_adaptive", 0.00013333333333333334),
            point("c025_n023_lr089_adaptive", 0.00008888888888888889),
        ]
        upper_rates = [0.000675, 0.0010125, 0.00151875]
        upper = [
            point(f"c025_n023_upper_{index}", rate)
            for index, rate in enumerate(upper_rates)
        ]
        control = {"id": CONTROL, "slice": "c100", "role": "control"}
        suite = {
            "all_variants": [*initial, *lower, *upper, control],
            "controls": [control],
            "compute_slices": [{"id": "c025"}],
            "fit_shapes": [{"shape_id": "n023"}],
            "optional_extension_shapes": [],
            "calibrations": initial,
            "extension_calibrations": [],
            "adaptive_calibrations": [*lower, *upper],
            "learning_rate_candidates": [
                {"id": "lr200", "value": 0.0002},
                {"id": "lr300", "value": 0.0003},
                {"id": "lr450", "value": 0.00045},
            ],
            "learning_rate_search": {
                "lower": [
                    {"id": "lr133", "value": lower[0]["learning_rate"]},
                    {"id": "lr089", "value": lower[1]["learning_rate"]},
                ],
                "upper": [
                    {"id": f"upper_{index}", "value": rate}
                    for index, rate in enumerate(upper_rates)
                ],
            },
        }
        losses = {
            initial[0]["id"]: 3.0,
            initial[1]["id"]: 1.0,
            initial[2]["id"]: 2.0,
            **{item["id"]: 4.0 + index for index, item in enumerate(lower + upper)},
            CONTROL: 5.0,
        }
        measurements = {
            identifier: {
                "validation_loss": loss,
                "stability_admission": {"classification": "stable"},
            }
            for identifier, loss in losses.items()
        }
        return suite, initial, lower, upper, measurements

    def test_prospective_lr_state_rejects_missing_suffix_and_post_bracket_trial(self) -> None:
        suite, initial, _lower, upper, measurements = self._prospective_fixture()
        exact = [CONTROL, *(point["id"] for point in initial)]
        self.assertEqual(
            prospective_selection_groups(
                suite, run_names=exact, measurements=measurements
            ),
            {("c025", "n023")},
        )
        with self.assertRaisesRegex(EvidenceError, "missing suffix"):
            prospective_selection_groups(
                suite, run_names=exact[:-1], measurements=measurements
            )
        with self.assertRaisesRegex(EvidenceError, "unnecessary"):
            prospective_selection_groups(
                suite,
                run_names=[*exact, upper[0]["id"]],
                measurements=measurements,
            )
        with self.assertRaisesRegex(EvidenceError, "control evidence is missing"):
            prospective_selection_groups(
                suite,
                run_names=[point["id"] for point in initial],
                measurements=measurements,
            )

    def test_prospective_lr_state_requires_next_edge_trial_and_stops_at_frontier(self) -> None:
        suite, initial, _lower, upper, measurements = self._prospective_fixture()
        measurements[initial[2]["id"]]["validation_loss"] = 0.5
        exact_initial = [CONTROL, *(point["id"] for point in initial)]
        with self.assertRaisesRegex(EvidenceError, "necessary upper LR trial is missing"):
            prospective_selection_groups(
                suite, run_names=exact_initial, measurements=measurements
            )
        measurements[upper[0]["id"]]["stability_admission"]["classification"] = "rejected"
        self.assertEqual(
            prospective_selection_groups(
                suite,
                run_names=[*exact_initial, upper[0]["id"]],
                measurements=measurements,
            ),
            {("c025", "n023")},
        )
        with self.assertRaisesRegex(EvidenceError, "unnecessary"):
            prospective_selection_groups(
                suite,
                run_names=[*exact_initial, upper[0]["id"], upper[1]["id"]],
                measurements=measurements,
            )

    def test_every_run_binds_exact_published_4b_manifest_and_shards(self) -> None:
        public = json.loads(
            (REPOSITORY_ROOT / PUBLISHED_4B_MANIFEST_PATH).read_text(encoding="utf-8")
        )
        source = public["source"]
        preparation = public["preparation"]
        suite = {
            "dataset": {
                "id": public["name"],
                "source_repository": source["dataset"],
                "source_revision": source["revision"],
                "source_inventory_sha256": source["inventory_sha256"],
                "exclusion_policy_sha256": source["exclusion_policy_sha256"],
                "tokenizer_version": public["tokenizer"]["implementation_version"],
                "preparation_core_sha256": preparation["core_sha256"],
            }
        }
        provenance = {
            "name": public["name"],
            "manifest_raw_sha256": PUBLISHED_4B_MANIFEST_SHA256,
            "manifest_canonical_sha256": PUBLISHED_4B_MANIFEST_CANONICAL_SHA256,
            "production": {
                "source_inventory_sha256": source["inventory_sha256"],
                "exclusion_policy_sha256": source["exclusion_policy_sha256"],
                "preparation_core_sha256": preparation["core_sha256"],
                "builder_module_sha256": preparation["builder_module_sha256"],
                "entrypoint_sha256": preparation["entrypoint_sha256"],
                "source_date_before": source["source_date_before"],
                "validation_train_document_disjoint": True,
                "validation_boundary_discarded_tokens": preparation[
                    "validation_boundary_discarded_tokens"
                ],
                "validation_boundary_document_id_sha256": preparation[
                    "validation_boundary_document_id_sha256"
                ],
            },
            "shards": [
                {
                    key: item[key]
                    for key in ("path", "split", "tokens", "bytes", "sha256")
                }
                for item in public["files"]
            ],
        }
        validate_published_4b_binding(
            source_root=REPOSITORY_ROOT,
            suite=suite,
            measurements={RUN: {"_dataset_provenance": provenance}},
        )
        changed = json.loads(json.dumps(provenance))
        changed["manifest_canonical_sha256"] = "0" * 64
        with self.assertRaisesRegex(EvidenceError, "identity/shards"):
            validate_published_4b_binding(
                source_root=REPOSITORY_ROOT,
                suite=suite,
                measurements={RUN: {"_dataset_provenance": changed}},
            )

    def test_build_refuses_existing_or_linked_output_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            victim = root / "victim"
            victim.write_bytes(b"unchanged")
            output = root / "output"
            output.symlink_to(victim)
            with self.assertRaisesRegex(EvidenceError, "already exists"):
                build_archive(
                    repository_root=REPOSITORY_ROOT,
                    runs=runs,
                    output=output,
                    repo_id="quintic/gpt-tpu-speedrun-scaling-evidence",
                    prefix="current_budget_isoflop_v4",
                )
            self.assertEqual(victim.read_bytes(), b"unchanged")

            real_parent = root / "real"
            real_parent.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(EvidenceError, "linked"):
                build_archive(
                    repository_root=REPOSITORY_ROOT,
                    runs=runs,
                    output=linked_parent / "output",
                    repo_id="quintic/gpt-tpu-speedrun-scaling-evidence",
                    prefix="current_budget_isoflop_v4",
                )
            self.assertFalse((real_parent / "output").exists())

    def test_upload_detects_mutation_of_open_snapshot_descriptor(self) -> None:
        class Add:
            def __init__(self, *, path_in_repo, path_or_fileobj):
                self.path_in_repo = path_in_repo
                self.path_or_fileobj = path_or_fileobj

        class Api:
            def __init__(self, token=None):
                self.token = token

            def create_repo(self, **_kwargs):
                return None

            def repo_info(self, **_kwargs):
                return SimpleNamespace(sha="b" * 40)

            def list_repo_files(self, **_kwargs):
                return []

            def create_commit(self, **kwargs):
                handle = kwargs["operations"][0].path_or_fileobj
                os.fchmod(handle.fileno(), 0o600)
                os.ftruncate(handle.fileno(), 0)
                return SimpleNamespace(oid="a" * 40)

        hub = ModuleType("huggingface_hub")
        hub.HfApi = Api
        hub.CommitOperationAdd = Add
        verification = {
            "semantic_verification": {flag: True for flag in SEMANTIC_FLAGS}
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            _write_bundle(bundle)
            token = root / "token"
            token.write_text("unused\n", encoding="utf-8")
            token.chmod(0o600)
            with patch.dict(sys.modules, {"huggingface_hub": hub}), patch(
                "scripts.publish_scaling_evidence.verify_frozen_snapshot",
                return_value={flag: True for flag in SEMANTIC_FLAGS},
            ), patch(
                "scripts.publish_scaling_evidence.read_token_file",
                return_value="hf_" + "A" * 24,
            ), patch(
                "scripts.publish_scaling_evidence.anonymous_revalidate",
                return_value=verification,
            ) as anonymous:
                with self.assertRaisesRegex(EvidenceError, "snapshot changed|upload"):
                    publish_archive(
                        bundle=bundle,
                        token_file=token,
                        receipt_output=root / "receipt.json",
                    )
            anonymous.assert_not_called()
            self.assertFalse((root / "receipt.json").exists())

    def test_publish_subcommand_resumes_without_building(self) -> None:
        from scripts.publish_scaling_evidence import main

        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.publish_scaling_evidence.publish_archive",
            return_value={"resumed": True},
        ) as publish, patch(
            "scripts.publish_scaling_evidence.build_archive"
        ) as build:
            result = main(
                [
                    "publish",
                    "--bundle",
                    str(Path(directory) / "bundle"),
                    "--token-file",
                    str(Path(directory) / "token"),
                    "--receipt-output",
                    str(Path(directory) / "receipt"),
                ]
            )
        self.assertEqual(result, 0)
        publish.assert_called_once()
        build.assert_not_called()

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
