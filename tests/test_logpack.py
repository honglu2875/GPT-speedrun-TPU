from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from rig import logpack, metrics


def _columns() -> list[logpack.Column]:
    return [
        logpack.column("train_loss"),
        logpack.column("learning_rate"),
        logpack.column("grad.l2_norm", "block", 0, element_count=1_000),
        logpack.column("grad.l2_norm", "block", 1, element_count=1_000),
    ]


def _write(path: Path, samples, columns=None, **header):
    options = {"tokens_per_step": 131_072, "flops_per_token": 1.5e9, **header}
    with logpack.LogWriter(path, columns or _columns(), **options) as writer:
        for step, values in samples:
            writer.append(step, values)


class LogPackTests(unittest.TestCase):
    def test_round_trip_preserves_float32_exactly(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.standard_normal((64, 4), dtype=np.float32) * 1e6
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            _write(path, [(step + 1, row) for step, row in enumerate(values)])
            log = logpack.read_log(path)
        # fp32 in, fp32 out. This is the property that ruled out bf16.
        np.testing.assert_array_equal(log.values, values)
        np.testing.assert_array_equal(log.steps, np.arange(1, 65))
        self.assertEqual(len(log), 64)

    def test_derived_axes_stay_exact_beyond_float32(self) -> None:
        # Real 500M scale: 76,691 steps of 131,072 tokens at 3.06e9 FLOPs/token.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            _write(
                path,
                [(76_691, (2.88, 2.7e-4, 0.23, 0.24))],
                tokens_per_step=131_072,
                flops_per_token=3.0644e9,
            )
            log = logpack.read_log(path)

        tokens = log.axis("tokens_processed")
        self.assertEqual(tokens.dtype, np.dtype(np.int64))
        self.assertEqual(int(tokens[0]), 76_691 * 131_072)

        # The FLOP axis is the one float32 cannot hold: ~3e19 against seven
        # significant digits. Deriving it in float64 keeps the exact product.
        flops = log.axis("cumulative_flops")
        self.assertEqual(flops.dtype, np.dtype(np.float64))
        exact = 76_691 * 131_072 * 3.0644e9
        self.assertEqual(float(flops[0]), exact)
        self.assertNotEqual(float(np.float32(exact)), exact)

        np.testing.assert_array_equal(log.axis("step"), log.steps)
        with self.assertRaises(KeyError):
            log.axis("train_loss")

    def test_a_partial_trailing_record_keeps_every_whole_record(self) -> None:
        """A preempted run is truncated mid-write; the rest must still read."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            _write(path, [(step, (1.0, 2.0, 3.0, 4.0)) for step in range(1, 11)])
            whole = path.read_bytes()
            # 1 byte, a partial field, and one byte short of a whole record.
            for lost in (1, 3, logpack._record_size(4) - 1):
                with self.subTest(bytes_lost=lost):
                    path.write_bytes(whole[:-lost])
                    log = logpack.read_log(path)
                    self.assertEqual(len(log), 9)
                    np.testing.assert_array_equal(log.steps, np.arange(1, 10))

    def test_the_file_is_readable_after_every_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            writer = logpack.LogWriter(
                path, _columns(), tokens_per_step=1_024, flops_per_token=1.0
            )
            for step in range(1, 6):
                writer.append(step, (float(step), 0.0, 0.0, 0.0))
                log = logpack.read_log(path)
                self.assertEqual(len(log), step)
                self.assertAlmostEqual(float(log.series("train_loss")[-1]), step)
            writer.close()

    def test_unknown_columns_are_carried_not_rejected(self) -> None:
        """A file from a later build names metrics this one has never seen."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            _write(path, [(1, (1.0, 2.0, 3.0, 4.0))])
            raw = bytearray(path.read_bytes())
            # Rewrite the first column's metric id to one no build defines.
            offset = len(logpack.MAGIC) + logpack._HEADER_STRUCT.size
            struct.pack_into("<i", raw, offset, 987_654)
            path.write_bytes(bytes(raw))

            log = logpack.read_log(path)
            self.assertEqual(len(log.columns), 4)
            self.assertIsNone(log.columns[0].metric)
            self.assertEqual(log.columns[0].describe(), "overall/metric:987654")
            # The columns this build does understand still resolve.
            np.testing.assert_array_equal(log.series("learning_rate"), [2.0])
            self.assertIsNone(log.series("train_loss"))

    def test_missing_series_answers_none_rather_than_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            _write(path, [(1, (1.0, 2.0, 3.0, 4.0))])
            log = logpack.read_log(path)
        self.assertIsNone(log.series("param.l1_norm"))
        self.assertIsNone(log.series("grad.l2_norm", "block", 9))
        self.assertIsNone(log.index_of("grad.l2_norm", "embeddings"))
        self.assertEqual(log.index_of("grad.l2_norm", "block", 1), 3)
        self.assertEqual(log.columns[2].element_count, 1_000)

    def test_steps_must_increase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            writer = logpack.LogWriter(
                path, _columns(), tokens_per_step=1_024, flops_per_token=1.0
            )
            writer.append(5, (1.0, 2.0, 3.0, 4.0))
            for bad in (5, 4):
                with self.subTest(step=bad):
                    with self.assertRaisesRegex(ValueError, "must increase"):
                        writer.append(bad, (1.0, 2.0, 3.0, 4.0))
            writer.close()

    def test_column_helper_checks_the_registry_and_layering(self) -> None:
        with self.assertRaises(KeyError):
            logpack.column("not_a_metric")
        with self.assertRaises(KeyError):
            logpack.column("train_loss", "not_a_scope")
        with self.assertRaisesRegex(ValueError, "requires a layer"):
            logpack.column("grad.l2_norm", "block")
        with self.assertRaisesRegex(ValueError, "does not take a layer"):
            logpack.column("grad.l2_norm", "overall", 0)
        entry = logpack.column("grad.l2_norm", "block", 4)
        self.assertEqual(entry.metric_id, metrics.metric("grad.l2_norm").id)
        self.assertEqual(entry.scope_id, metrics.scope("block").id)

    def test_a_foreign_file_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            for content, expected in (
                (b"", "bad magic"),
                (b"step,train_loss\n1,2.0\n", "bad magic"),
                (logpack.MAGIC, "truncated inside its header"),
                (logpack.MAGIC + b"\x00" * 8, "truncated inside its header"),
            ):
                with self.subTest(prefix=content[:8]):
                    path.write_bytes(content)
                    with self.assertRaisesRegex(logpack.LogError, expected):
                        logpack.read_log(path)

    def test_header_rejects_impossible_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            for kwargs, message in (
                ({"tokens_per_step": 0}, "tokens_per_step"),
                ({"flops_per_token": 0.0}, "flops_per_token"),
            ):
                with self.subTest(**kwargs):
                    with self.assertRaisesRegex(ValueError, message):
                        logpack.LogWriter(
                            path,
                            _columns(),
                            **{
                                "tokens_per_step": 1_024,
                                "flops_per_token": 1.0,
                                **kwargs,
                            },
                        )
            with self.assertRaisesRegex(ValueError, "at least one column"):
                logpack.LogWriter(path, [], tokens_per_step=1_024, flops_per_token=1.0)

    def test_size_is_the_declared_layout(self) -> None:
        rows, columns = 100, 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            _write(path, [(step, (1.0, 2.0, 3.0, 4.0)) for step in range(1, rows + 1)])
            size = path.stat().st_size
        self.assertEqual(
            size,
            logpack._header_size(columns) + rows * logpack._record_size(columns),
        )
        # Spelled out once, so a layout change has to be deliberate.
        self.assertEqual(logpack._header_size(columns), 8 + 24 + columns * 24)
        self.assertEqual(logpack._record_size(columns), 4 + columns * 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
