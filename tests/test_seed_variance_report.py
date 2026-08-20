from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from rig import logpack
from studies.seed_variance.report import (
    DIAGNOSTICS_LOG,
    TRAINING_LOG,
    StudySpec,
    VarianceReportError,
    _sample_standard_deviation,
    build_seed_variance_report,
)


class SeedVarianceReportTests(unittest.TestCase):
    def test_builds_deterministic_paired_all_metric_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_study(root / "first", validation_base=3.9)
            second = _write_study(root / "second", validation_base=3.5)
            output = root / "variance.html"
            specs = (
                StudySpec("first", "First", first, planned_seeds=5),
                StudySpec("second", "Second", second, planned_seeds=5),
            )

            summary = build_seed_variance_report(specs, output, max_points=3)
            first_build = output.read_bytes()
            payload = _payload(first_build.decode("utf-8"))
            build_seed_variance_report(specs, output, max_points=3)

            self.assertEqual(first_build, output.read_bytes())

        self.assertEqual(summary.studies, (("First", 3), ("Second", 3)))
        self.assertEqual(summary.metrics, 4)
        self.assertEqual(payload["meta"]["maxPoints"], 3)
        self.assertEqual(payload["studies"][0]["seeds"], [10, 11, 12])
        self.assertEqual(payload["studies"][0]["flopsPerStep"], 200.0)
        self.assertAlmostEqual(payload["studies"][0]["finalValidationSd"], 0.01)
        self.assertEqual(
            payload["meta"]["defaultMetric"],
            next(
                metric["id"]
                for metric in payload["metrics"]
                if metric["metric"] == "train_loss"
            ),
        )
        loss = next(
            metric
            for metric in payload["metrics"]
            if metric["metric"] == "train_loss"
        )
        # The stored coordinate is [step, sample SD, finite seed count]. FLOPs
        # are reconstructed exactly as step * the study header constant.
        self.assertEqual(loss["series"][0][0], [1, 1.0, 3])
        self.assertEqual(loss["series"][0][-1], [4, 1.0, 3])
        self.assertLessEqual(len(loss["series"][0]), 3)
        learning_rate = next(
            metric
            for metric in payload["metrics"]
            if metric["metric"] == "learning_rate"
        )
        self.assertTrue(
            all(point[1] == 0.0 for point in learning_rate["series"][0])
        )
        html = first_build.decode("utf-8")
        self.assertIn("102a264672c8453700a02e321495a14c585e58ea", html)
        self.assertNotRegex(html, r"<script[^>]+src=|<link[^>]+href=")

    def test_rejects_runs_that_do_not_share_exact_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_study(root / "first", validation_base=3.9)
            second = _write_study(root / "second", validation_base=3.5)
            _write_run(
                first / "seed-11",
                seed=11,
                validation_loss=3.9,
                training_steps=(1, 2, 3, 5),
            )

            with self.assertRaisesRegex(
                VarianceReportError, "does not log the same optimizer steps"
            ):
                build_seed_variance_report(
                    (
                        StudySpec("first", "First", first),
                        StudySpec("second", "Second", second),
                    ),
                    root / "variance.html",
                    max_points=3,
                )

    def test_sample_sd_uses_available_finite_seeds_and_n_minus_one(self) -> None:
        values = np.asarray(
            [
                [[1.0], [4.0]],
                [[2.0], [float("nan")]],
                [[3.0], [8.0]],
            ]
        )
        deviations, counts = _sample_standard_deviation(values)

        np.testing.assert_array_equal(counts[:, 0], [3, 2])
        np.testing.assert_allclose(deviations[:, 0], [1.0, 2**0.5 * 2])


def _write_study(path: Path, *, validation_base: float) -> Path:
    for offset, seed in enumerate((10, 11, 12)):
        _write_run(
            path / f"seed-{seed}",
            seed=seed,
            validation_loss=validation_base + offset * 0.01,
        )
    return path


def _write_run(
    path: Path,
    *,
    seed: int,
    validation_loss: float,
    training_steps: tuple[int, ...] = (1, 2, 3, 4),
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    offset = seed - 10
    training_columns = (
        logpack.column("train_loss"),
        logpack.column("learning_rate"),
    )
    with logpack.LogWriter(
        path / TRAINING_LOG,
        training_columns,
        tokens_per_step=10,
        flops_per_token=20.0,
    ) as writer:
        for step in training_steps:
            writer.append(step, [10.0 - step + offset, 0.001])

    diagnostic_columns = (
        logpack.column("grad.l2_norm"),
        logpack.column("grad.std", "block", layer=0),
    )
    with logpack.LogWriter(
        path / DIAGNOSTICS_LOG,
        diagnostic_columns,
        tokens_per_step=10,
        flops_per_token=20.0,
    ) as writer:
        writer.append(1, [2.0 + offset, 0.2 + offset])
        writer.append(3, [4.0 + offset, 0.4 + offset])

    result = {
        "status": "ok",
        "seed": seed,
        "metrics": {"validation_loss": validation_loss},
        "system": {
            "device_kinds": ["Test TPU"],
            "process_count": 1,
            "device_count": 2,
        },
    }
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")


def _payload(html: str) -> dict:
    match = re.search(
        r'<script id="payload" type="application/octet-stream">([^<]+)</script>',
        html,
    )
    if match is None:
        raise AssertionError("compressed payload not found")
    return json.loads(gzip.decompress(base64.b64decode(match.group(1))))


if __name__ == "__main__":
    unittest.main()
