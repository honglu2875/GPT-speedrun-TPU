"""Tests for the source attestation that guards multi-host runs."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rig.doctor import check_source_attestation, source_digest


def _tree(root: Path) -> None:
    (root / "rig" / "kernels").mkdir(parents=True)
    (root / "rig" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (root / "rig" / "kernels" / "a.py").write_text("y = 2\n", encoding="utf-8")
    (root / "recipes" / "reference").mkdir(parents=True)
    (root / "recipes" / "reference" / "train.py").write_text("z = 3\n", encoding="utf-8")
    (root / "recipes" / "reference" / "config.yaml").write_text("k: v\n", encoding="utf-8")


class SourceDigestTests(unittest.TestCase):
    def test_digest_is_deterministic_and_counts_every_covered_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _tree(root)
            first, files = source_digest(root)
            second, again = source_digest(root)
            self.assertEqual(first, second)
            self.assertEqual((files, again), (4, 4))

    def test_any_covered_byte_changes_the_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _tree(root)
            before, _ = source_digest(root)
            target = root / "rig" / "kernels" / "a.py"
            target.write_text("y = 3\n", encoding="utf-8")
            self.assertNotEqual(before, source_digest(root)[0])

    def test_entry_program_and_config_are_covered(self) -> None:
        # A peer running a stale trainer is exactly what this must catch.
        for name in ("train.py", "config.yaml"):
            with self.subTest(file=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    _tree(root)
                    before, _ = source_digest(root)
                    path = root / "recipes" / "reference" / name
                    path.write_text(path.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8")
                    self.assertNotEqual(before, source_digest(root)[0])

    def test_renaming_a_file_changes_the_digest(self) -> None:
        # Content alone is not enough: layout is part of the identity.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _tree(root)
            before, _ = source_digest(root)
            (root / "rig" / "kernels" / "a.py").rename(root / "rig" / "kernels" / "b.py")
            self.assertNotEqual(before, source_digest(root)[0])

    def test_unrelated_files_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _tree(root)
            before, files = source_digest(root)
            (root / "runs").mkdir()
            (root / "runs" / "result.json").write_text("{}", encoding="utf-8")
            (root / "README.md").write_text("docs\n", encoding="utf-8")
            (root / "rig" / "notes.txt").write_text("scratch\n", encoding="utf-8")
            after, again = source_digest(root)
            self.assertEqual((before, files), (after, again))


class AttestationCheckTests(unittest.TestCase):
    def test_single_process_reports_its_own_digest(self) -> None:
        result = check_source_attestation()
        self.assertEqual(result.status, "ok")
        self.assertIn("sha256:", result.message)

    def test_disagreement_is_an_error_that_names_the_processes(self) -> None:
        import numpy as np

        mine = bytes.fromhex("aa" * 32)
        theirs = bytes.fromhex("bb" * 32)
        gathered = np.stack(
            [np.frombuffer(b, dtype=np.uint8) for b in (mine, theirs, theirs, theirs)]
        )
        with (
            patch("rig.doctor.source_digest", return_value=("aa" * 32, 23)),
            patch("jax.process_count", return_value=4),
            patch(
                "jax.experimental.multihost_utils.process_allgather",
                return_value=gathered,
            ),
        ):
            result = check_source_attestation()
        self.assertEqual(result.status, "error")
        self.assertIn("different source", result.message)
        self.assertIn("[0]", result.message)
        self.assertIn("[1, 2, 3]", result.message)
        self.assertIsNotNone(result.hint)

    def test_agreement_across_processes_is_ok(self) -> None:
        import numpy as np

        same = np.stack([np.frombuffer(bytes.fromhex("cc" * 32), dtype=np.uint8)] * 4)
        with (
            patch("rig.doctor.source_digest", return_value=("cc" * 32, 23)),
            patch("jax.process_count", return_value=4),
            patch(
                "jax.experimental.multihost_utils.process_allgather",
                return_value=same,
            ),
        ):
            result = check_source_attestation()
        self.assertEqual(result.status, "ok")
        self.assertIn("identical across 4 processes", result.message)


if __name__ == "__main__":
    unittest.main()
