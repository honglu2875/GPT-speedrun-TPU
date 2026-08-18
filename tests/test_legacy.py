"""Gates for the long-form CSV converter.

This code rewrites recorded measurements into a different file format. A bug
here does not crash anything -- it produces a plausible curve that is not the
one that was measured, which is the worst failure mode available. So the tests
are mostly about refusing to convert, rather than about converting.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rig import legacy, logpack, runlog


HEADER = ",".join(legacy.TRAINING_HEADER)
TOKENS_PER_STEP = 4096
FLOPS_PER_TOKEN = 1_000_000


def _rows(count: int, *, tokens_per_step: int = TOKENS_PER_STEP) -> list[str]:
    rows = []
    for step in range(1, count + 1):
        tokens = step * tokens_per_step
        rows.append(
            f"{step},{tokens},{tokens * FLOPS_PER_TOKEN},"
            f"{10.0 - step * 0.1},{0.001},{1.5}"
        )
    return rows


def _write_csv(directory: Path, rows: list[str], header: str = HEADER) -> Path:
    path = directory / "training.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


class ReadTrainingCsvTests(unittest.TestCase):
    def test_recovers_history_and_the_header_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(Path(directory), _rows(5))
            history, tokens_per_step, flops = legacy.read_training_csv(path)
        self.assertEqual(history.shape, (5, 3))
        self.assertEqual(tokens_per_step, TOKENS_PER_STEP)
        self.assertEqual(flops, FLOPS_PER_TOKEN)
        # Loss, learning rate, gradient norm -- in that order, not any other.
        np.testing.assert_allclose(history[0], [9.9, 0.001, 1.5], rtol=1e-6)
        np.testing.assert_allclose(history[-1], [9.5, 0.001, 1.5], rtol=1e-6)

    def test_refuses_an_unrecognized_header(self) -> None:
        """Column order is the whole meaning of a row.

        A file whose columns were emitted in a different order would convert
        without complaint and silently label the learning rate as the loss.
        """

        swapped = HEADER.replace("train_loss,learning_rate", "learning_rate,train_loss")
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(Path(directory), _rows(3), header=swapped)
            with self.assertRaisesRegex(legacy.LegacyError, "expected"):
                legacy.read_training_csv(path)

    def test_refuses_a_gapped_or_unsorted_step_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _rows(4)
            del rows[2]
            path = _write_csv(Path(directory), rows)
            with self.assertRaisesRegex(legacy.LegacyError, "without gaps"):
                legacy.read_training_csv(path)

    def test_refuses_a_drifting_tokens_per_step(self) -> None:
        """The packed header states this once, so it has to be a constant.

        Keeping the first row's value and ignoring later disagreement would
        misplace every subsequent point on the token axis.
        """

        with tempfile.TemporaryDirectory() as directory:
            rows = _rows(4)
            rows[-1] = (
                f"4,{5 * TOKENS_PER_STEP},{5 * TOKENS_PER_STEP * FLOPS_PER_TOKEN},9.6,0.001,1.5"
            )
            path = _write_csv(Path(directory), rows)
            with self.assertRaisesRegex(legacy.LegacyError, "tokens per step"):
                legacy.read_training_csv(path)

    def test_refuses_a_drifting_flops_per_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _rows(4)
            tokens = 4 * TOKENS_PER_STEP
            rows[-1] = f"4,{tokens},{tokens * FLOPS_PER_TOKEN * 2},9.6,0.001,1.5"
            path = _write_csv(Path(directory), rows)
            with self.assertRaisesRegex(legacy.LegacyError, "FLOPs per token"):
                legacy.read_training_csv(path)

    def test_refuses_an_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(legacy.LegacyError, "empty"):
                legacy.read_training_csv(path)
            _write_csv(Path(directory), [])
            with self.assertRaisesRegex(legacy.LegacyError, "no samples"):
                legacy.read_training_csv(path)


class ConvertRunTests(unittest.TestCase):
    def _run(self, directory: Path, *, steps: int = 6, **overrides) -> Path:
        source = directory / "run"
        source.mkdir()
        _write_csv(source, _rows(steps))
        metrics = {
            "training_steps": steps,
            "tokens_processed": steps * TOKENS_PER_STEP,
            "train_loss": 10.0 - steps * 0.1,
            **overrides,
        }
        (source / "result.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "artifacts": {
                        "training_curve": "training.csv",
                        "diagnostics": "diagnostics.csv",
                        "validation_curve": "validation.csv",
                    },
                }
            ),
            encoding="utf-8",
        )
        (source / "validation.csv").write_text("step,loss\n6,1.0\n", encoding="utf-8")
        return source

    def test_round_trips_the_curve_and_repoints_the_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._run(root)
            result = legacy.convert_run(source, root / "out")

            log = logpack.read_log(root / "out" / runlog.TRAINING_LOG_NAME)
            self.assertEqual(len(log), 6)
            np.testing.assert_allclose(log.series("train_loss")[-1], 9.4, rtol=1e-6)
            np.testing.assert_allclose(log.series("learning_rate")[0], 0.001, rtol=1e-6)

            # The report resolves artifacts through result.json, so a converted
            # run that still points at the CSV is a converted run nobody reads.
            self.assertEqual(
                result["artifacts"]["training_curve"], runlog.TRAINING_LOG_NAME
            )
            # Declared-but-absent is an error downstream, so the pointer to the
            # unconverted diagnostics has to be dropped, not left dangling.
            self.assertNotIn("diagnostics", result["artifacts"])
            self.assertTrue((root / "out" / "validation.csv").is_file())

    def test_leaves_every_original_file_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._run(root)
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            legacy.convert_run(source, root / "out")
            after = {path.name: path.read_bytes() for path in source.iterdir()}
        self.assertEqual(before, after)

    def test_refuses_when_the_csv_contradicts_the_result(self) -> None:
        """The two disagreeing means one of them is wrong about this run.

        Converting anyway would carry the disagreement forward into a file
        that no longer has the CSV beside it to be checked against.
        """

        for field, value, message in (
            ("training_steps", 99, "result.json says 99"),
            ("tokens_processed", 12345, "result.json says 12345"),
            ("train_loss", 1.234, "result.json says 1.234"),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = self._run(root, **{field: value})
                    with self.assertRaisesRegex(legacy.LegacyError, message):
                        legacy.convert_run(source, root / "out")

    def test_refuses_a_run_with_no_result_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._run(root)
            (source / "result.json").unlink()
            with self.assertRaisesRegex(legacy.LegacyError, "no result.json"):
                legacy.convert_run(source, root / "out")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
