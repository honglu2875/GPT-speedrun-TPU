from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from speedrun.report import (
    REPORT_ADMISSION_QUALIFICATION_LOSS,
    ReportError,
    _attach_flops,
    _checkpoint_layer_stats,
    _default_run_selection,
    _diagnostic_metric,
    _lttb,
    _overall_metric_identity,
    build_report,
)


class ReportTests(unittest.TestCase):
    def test_report_strictly_admits_only_complete_official_runs_at_3_76(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            cases = {
                "admitted-boundary": ("sample_efficiency", "official", 20, 3.76),
                "not-admitted": ("sample_efficiency", "official", 20, 3.76001),
                "development": ("sample_efficiency", "dev", 20, 3.0),
                "partial-open": ("open", "official", 20, 3.0),
            }
            for name, (track, profile, tokens, validation_loss) in cases.items():
                run = runs / name
                run.mkdir(parents=True)
                (run / "training.csv").write_text(
                    "step,tokens_processed,train_loss\n1,10,4.5\n2,20,4.0\n",
                    encoding="utf-8",
                )
                _write_result(
                    run,
                    validation_artifact=False,
                    track=track,
                    profile=profile,
                    tokens=tokens,
                    validation_loss=validation_loss,
                )

            summary = build_report(runs, root / "report.html")
            payload = _payload((root / "report.html").read_text(encoding="utf-8"))

        self.assertEqual(REPORT_ADMISSION_QUALIFICATION_LOSS, 3.76)
        self.assertEqual(summary.included, ("admitted-boundary",))
        self.assertIn(
            "does not meet the baseline report admission qualification",
            summary.skipped["not-admitted"],
        )
        self.assertIn(
            "requires profile='official'", summary.skipped["development"]
        )
        self.assertIn("open run is incomplete", summary.skipped["partial-open"])
        self.assertEqual(payload["meta"]["admissionQualificationLoss"], 3.76)

    def test_include_dev_is_a_narrow_diagnostic_escape_hatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            cases = {
                "dev-diagnostic": ("open", "dev", 20, 9.0),
                "official-rejected": ("sample_efficiency", "official", 20, 3.8),
                "official-valid": ("sample_efficiency", "official", 20, 3.7),
                "smoke-run": ("sample_efficiency", "smoke", 20, 3.0),
            }
            for name, (track, profile, tokens, validation_loss) in cases.items():
                run = runs / name
                run.mkdir(parents=True)
                (run / "training.csv").write_text(
                    "step,tokens_processed,train_loss\n1,10,4.5\n2,20,4.0\n",
                    encoding="utf-8",
                )
                _write_result(
                    run,
                    validation_artifact=False,
                    track=track,
                    profile=profile,
                    tokens=tokens,
                    validation_loss=validation_loss,
                )

            summary = build_report(
                runs,
                root / "report.html",
                include_dev=True,
            )
            payload = _payload((root / "report.html").read_text(encoding="utf-8"))

        self.assertEqual(summary.included, ("dev-diagnostic", "official-valid"))
        self.assertIn("profile='official'", summary.skipped["smoke-run"])
        self.assertIn(
            "does not meet the baseline report admission qualification",
            summary.skipped["official-rejected"],
        )
        self.assertTrue(payload["meta"]["includeDev"])
        classifications = {
            run["id"]: run["classification"] for run in payload["runs"]
        }
        self.assertEqual(classifications["dev-diagnostic"], "diagnostic")
        self.assertEqual(classifications["official-valid"], "official")
        self.assertTrue(
            any("not official or qualifying" in notice for notice in payload["notices"])
        )

    def test_long_form_diagnostics_build_all_family_and_final_scope_charts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "diagnostic-run"
            run.mkdir(parents=True)
            (run / "training.csv").write_text(
                "step,tokens_processed,cumulative_estimated_flops,train_loss\n"
                "1,10,1000,4.5\n"
                "2,20,2000,4.0\n",
                encoding="utf-8",
            )
            _write_diagnostics(run / "diagnostics.csv")
            _write_result(
                run,
                validation_artifact=False,
                diagnostics_artifact=True,
            )

            summary = build_report(runs, root / "report.html")
            html = (root / "report.html").read_text(encoding="utf-8")
            payload = _payload(html)

        self.assertEqual(summary.included, (run.name,))
        self.assertEqual(
            {chart["family"] for chart in payload["diagnosticCharts"]},
            {"grad", "update", "param"},
        )
        self.assertEqual(len(payload["diagnosticCharts"]), 18)
        self.assertEqual(len(payload["layerCharts"]), 18)
        parameter_mean = next(
            chart
            for chart in payload["layerCharts"]
            if chart["family"] == "param" and chart["stat"] == "mean"
        )
        self.assertEqual(
            [point[2] for point in parameter_mean["series"][0]["points"]],
            ["embeddings", "block 0", "final norm", "unembedding"],
        )
        self.assertIn('id="family-control"', html)
        self.assertIn('id="focus-dialog"', html)
        self.assertIn('id="smoothing-control"', html)
        self.assertIn('id="x-scale-control"', html)
        self.assertIn('name="x-scale" value="log" checked', html)
        self.assertIn('name="x-scale" value="linear"', html)
        self.assertIn("Math.log10(x)", html)
        self.assertIn('name="smoothing" value="ema"', html)
        self.assertIn('name="smoothing" value="mean"', html)
        self.assertIn('name="smoothing" value="median"', html)
        self.assertIn("function finishBox(e,item)", html)
        self.assertIn("Raw sample:", html)
        self.assertNotIn("onwheel", html)
        self.assertNotIn("function zoom(", html)
        self.assertNotIn("setInterval", html)

    def test_incomplete_diagnostics_grid_excludes_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "truncated-diagnostics"
            run.mkdir(parents=True)
            (run / "training.csv").write_text(
                "step,tokens_processed,cumulative_estimated_flops,train_loss\n"
                "1,10,1000,4.5\n"
                "2,20,2000,4.0\n",
                encoding="utf-8",
            )
            diagnostics = run / "diagnostics.csv"
            _write_diagnostics(diagnostics)
            lines = diagnostics.read_text(encoding="utf-8").splitlines()
            diagnostics.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            _write_result(
                run,
                validation_artifact=False,
                diagnostics_artifact=True,
            )

            summary = build_report(runs, root / "report.html")

        self.assertFalse(summary.included)
        self.assertIn("incomplete diagnostic grid", summary.skipped[run.name])

    def test_standalone_report_includes_sound_run_and_both_axis_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "complete-run"
            run.mkdir(parents=True)
            (run / "training.csv").write_text(
                "step,tokens_processed,train_loss,learning_rate,grad_norm\n"
                "1,10,4.5,0.001,2.0\n"
                "2,20,4.0,0.0001,1.5\n",
                encoding="utf-8",
            )
            (run / "validation.csv").write_text(
                "step,tokens_processed,kind,domain,validation_loss\n"
                "1,10,fineweb_probe,fineweb,4.4\n"
                "2,20,fineweb,fineweb,3.7\n",
                encoding="utf-8",
            )
            _write_result(run)

            summary = build_report(runs, root / "report.html", max_chart_points=64)
            html = (root / "report.html").read_text(encoding="utf-8")
            payload = _payload(html)

        self.assertEqual(summary.included, ("complete-run",))
        self.assertFalse(summary.skipped)
        self.assertEqual(payload["meta"]["defaultXAxis"], "flops")
        self.assertEqual(payload["meta"]["defaultXScale"], "log")
        self.assertEqual(payload["meta"]["maxChartPoints"], 64)
        self.assertTrue(payload["runs"][0]["selected"])
        self.assertEqual(payload["runs"][0]["classification"], "official")
        self.assertEqual(payload["runs"][0]["flopSource"], "derived: result metrics.flops_per_token × tokens_processed")
        train = next(chart for chart in payload["timeCharts"] if chart["key"] == "train_loss")
        self.assertEqual(train["series"][0]["points"][-1], [2.0, 2000.0, 4.0])
        self.assertIn("equi-FLOP", html)
        self.assertIn("equi-step", html)
        self.assertIn("Learning rate", html)
        coverage = next(
            notice
            for notice in payload["notices"]
            if notice.startswith("Overall training diagnostic coverage:")
        )
        self.assertIn("recorded gradient L2 norm", coverage)
        self.assertIn("not recorded: gradient L1 norm", coverage)
        self.assertIn("update fourth moment", coverage)
        self.assertIn("parameter fourth moment", coverage)
        self.assertIn("parameter fourth moment", coverage)
        self.assertNotRegex(html, r'<script[^>]+src=|<link[^>]+href=')
        self.assertNotIn(".slice(0,10)", html)
        self.assertNotIn("Math.min(...xs)", html)
        self.assertIn("filter(r=>r.selected)", html)

    def test_default_selection_distinguishes_official_diagnostic_and_partial(self) -> None:
        budget = 624_984_064
        self.assertEqual(
            _default_run_selection("open", "official", budget),
            (True, "official"),
        )
        self.assertEqual(
            _default_run_selection("open", "official", budget - 1),
            (False, "partial"),
        )
        self.assertEqual(
            _default_run_selection("sample_efficiency", "official", 123),
            (True, "official"),
        )
        self.assertEqual(
            _default_run_selection("open", "dev", budget),
            (False, "diagnostic"),
        )

    def test_explicit_cumulative_flops_take_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "explicit-flops"
            run.mkdir(parents=True)
            (run / "training.csv").write_text(
                "step,tokens_processed,train_loss,cumulative_estimated_flops\n"
                "1,10,4.5,1234\n"
                "2,20,4.0,3456\n",
                encoding="utf-8",
            )
            _write_result(run, validation_artifact=False)
            build_report(runs, root / "report.html", max_chart_points=64)
            payload = _payload((root / "report.html").read_text(encoding="utf-8"))

        self.assertEqual(
            payload["runs"][0]["flopSource"],
            "training.csv cumulative_estimated_flops",
        )
        train = next(chart for chart in payload["timeCharts"] if chart["key"] == "train_loss")
        self.assertEqual(train["series"][0]["points"][-1][1], 3456.0)

    def test_explicit_cumulative_flops_are_strict_and_match_declared_total(self) -> None:
        with self.assertRaisesRegex(ReportError, "positive and increase strictly"):
            _attach_flops(
                [
                    {"cumulative_estimated_flops": 10.0},
                    {"cumulative_estimated_flops": 10.0},
                ],
                {},
            )
        with self.assertRaisesRegex(ReportError, "estimated_total_flops"):
            _attach_flops(
                [
                    {"cumulative_estimated_flops": 10.0},
                    {"cumulative_estimated_flops": 20.0},
                ],
                {"estimated_total_flops": 21.0},
            )
        rows = [
            {"cumulative_estimated_flops": 10.0},
            {"cumulative_estimated_flops": 20.0},
        ]
        self.assertEqual(
            _attach_flops(rows, {"estimated_total_flops": 20.0}),
            "training.csv cumulative_estimated_flops",
        )

    def test_lttb_defaults_to_the_flop_coordinate(self) -> None:
        points = [
            [1, 1, -4],
            [2, 2, 3],
            [3, 3, -10],
            [4, 21, 6],
            [5, 22, -3],
            [6, 35, 4],
        ]
        by_flops = _lttb(points, 4)
        by_steps = _lttb(points, 4, x_index=0)
        self.assertEqual(by_flops, [points[0], points[1], points[4], points[5]])
        self.assertNotEqual(by_flops, by_steps)

    def test_ledger_hash_mismatch_excludes_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "tampered-run"
            run.mkdir(parents=True)
            training = run / "training.csv"
            validation = run / "validation.csv"
            training.write_text(
                "step,tokens_processed,train_loss\n1,10,4.5\n2,20,4.0\n",
                encoding="utf-8",
            )
            validation.write_text(
                "step,tokens_processed,kind,domain,validation_loss\n"
                "2,20,fineweb,fineweb,3.7\n",
                encoding="utf-8",
            )
            _write_result(run)
            record = {
                "run_id": run.name,
                "status": "ok",
                "submission": "reference",
                "track": "sample_efficiency",
                "profile": "official",
                "seed": 1,
                "metrics": {
                    "tokens_processed": 20,
                    "train_seconds": 1.0,
                    "validation_loss": 3.7,
                },
                "artifacts": {
                    "training_curve": {
                        "path": "training.csv",
                        "bytes": training.stat().st_size,
                        "sha256": "0" * 64,
                    },
                    "validation_curve": {
                        "path": "validation.csv",
                        "bytes": validation.stat().st_size,
                        "sha256": _sha(validation),
                    },
                },
            }
            (runs / "records.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            summary = build_report(runs, root / "report.html")

        self.assertFalse(summary.included)
        self.assertIn("SHA-256", summary.skipped["tampered-run"])

    def test_malformed_unledgered_timing_is_skipped_without_aborting_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            good = runs / "good-run"
            bad = runs / "bad-run"
            for run in (good, bad):
                run.mkdir(parents=True)
                (run / "training.csv").write_text(
                    "step,tokens_processed,train_loss\n1,10,4.5\n2,20,4.0\n",
                    encoding="utf-8",
                )
                _write_result(run, validation_artifact=False)
            bad_result = json.loads((bad / "result.json").read_text(encoding="utf-8"))
            bad_result["metrics"]["train_seconds"] = float("nan")
            (bad / "result.json").write_text(json.dumps(bad_result), encoding="utf-8")

            summary = build_report(runs, root / "report.html")

        self.assertEqual(summary.included, ("good-run",))
        self.assertIn("train_seconds is not finite", summary.skipped["bad-run"])

    def test_invalid_qualified_value_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "odd-qualified"
            run.mkdir(parents=True)
            training = run / "training.csv"
            training.write_text(
                "step,tokens_processed,train_loss\n1,10,4.5\n2,20,4.0\n",
                encoding="utf-8",
            )
            _write_result(run, validation_artifact=False)
            record = _record_for_run(run, validation=False)
            record["qualified"] = float("nan")
            (runs / "records.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            summary = build_report(runs, root / "report.html")
            payload = _payload((root / "report.html").read_text(encoding="utf-8"))

        self.assertEqual(summary.included, (run.name,))
        self.assertIsNone(payload["runs"][0]["qualified"])
        self.assertTrue(any("qualified flag was ignored" in n for n in summary.notices))

    def test_duplicate_ledger_run_id_excludes_ambiguous_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "duplicate-run"
            run.mkdir(parents=True)
            (run / "training.csv").write_text(
                "step,tokens_processed,train_loss\n1,10,4.5\n2,20,4.0\n",
                encoding="utf-8",
            )
            _write_result(run, validation_artifact=False)
            record = _record_for_run(run, validation=False)
            (runs / "records.jsonl").write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )

            summary = build_report(runs, root / "report.html")

        self.assertFalse(summary.included)
        self.assertIn("duplicate entries", summary.skipped[run.name])

    def test_checkpoint_stats_group_parameter_arrays_by_logical_layer(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.npz"
            np.savez(
                checkpoint,
                **{
                    "params/blocks/0/a": np.array([1.0, -1.0]),
                    "params/blocks/0/b": np.array([2.0]),
                    "params/blocks/1/a": np.array([3.0, 4.0]),
                    "grads/layers/0/a": np.array([0.5, -0.5]),
                    "updates/h/1/a": np.array([0.25, -0.25]),
                    "metadata.json": np.array([1], dtype=np.uint8),
                },
            )
            stats = _checkpoint_layer_stats(checkpoint)

        self.assertEqual(set(stats), {"param", "grad", "update"})
        self.assertEqual([row["layer"] for row in stats["param"]], [0.0, 1.0])
        self.assertAlmostEqual(stats["param"][0]["l1_norm"], 4.0)
        self.assertAlmostEqual(stats["param"][1]["l2_norm"], 5.0)
        self.assertAlmostEqual(stats["param"][1]["mean"], 3.5)
        self.assertAlmostEqual(stats["grad"][0]["l1_norm"], 1.0)
        self.assertAlmostEqual(stats["update"][0]["l2_norm"], 2**-2 * 2**0.5)

    def test_future_overall_grad_update_and_param_columns_are_recognized(self) -> None:
        expected = {
            "overall_grad_l1_norm": ("grad", "l1_norm"),
            "gradient_norm": ("grad", "l2_norm"),
            "update_std": ("update", "std"),
            "params_third_moment": ("param", "third_moment"),
            "parameter_fourth_moment": ("param", "fourth_moment"),
        }
        for name, identity in expected.items():
            with self.subTest(name=name):
                self.assertTrue(_diagnostic_metric(name))
                self.assertEqual(_overall_metric_identity(name), identity)

    def test_embedded_data_escapes_html_and_cannot_close_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "script-safe"
            run.mkdir(parents=True)
            (run / "training.csv").write_text(
                "step,tokens_processed,train_loss\n1,10,4.5\n2,20,4.0\n",
                encoding="utf-8",
            )
            _write_result(run, validation_artifact=False)
            record = {
                "run_id": run.name,
                "status": "ok",
                "submission": "</script><script>alert(1)</script>",
                "track": "sample_efficiency",
                "profile": "official",
                "seed": 1,
                "metrics": {
                    "tokens_processed": 20,
                    "train_seconds": 1.0,
                    "validation_loss": 3.7,
                },
                "artifacts": {
                    "training_curve": {
                        "path": "training.csv",
                        "bytes": (run / "training.csv").stat().st_size,
                        "sha256": _sha(run / "training.csv"),
                    }
                },
            }
            (runs / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            build_report(runs, root / "report.html")
            html = (root / "report.html").read_text(encoding="utf-8")
            payload = _payload(html)

        self.assertEqual(payload["runs"][0]["submission"], record["submission"])
        self.assertEqual(html.count("<script>"), 1)
        self.assertNotIn(record["submission"], html)


def _write_result(
    run: Path,
    *,
    validation_artifact: bool = True,
    diagnostics_artifact: bool = False,
    track: str = "sample_efficiency",
    profile: str = "official",
    tokens: int = 20,
    validation_loss: float = 3.7,
) -> None:
    artifacts = {"training_curve": "training.csv"}
    if validation_artifact:
        artifacts["validation_curve"] = "validation.csv"
    if diagnostics_artifact:
        artifacts["diagnostics"] = "diagnostics.csv"
    result = {
        "schema_version": 1,
        "status": "ok",
        "track": track,
        "profile": profile,
        "seed": 1,
        "checkpoint": "checkpoint.npz",
        "artifacts": artifacts,
        "metrics": {
            "tokens_processed": tokens,
            "train_seconds": 1.0,
            "train_loss": 4.0,
            "validation_loss": validation_loss,
            "flops_per_token": 100,
        },
    }
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")


def _write_diagnostics(path: Path) -> None:
    families = ("param", "grad", "update")
    statistics = (
        "l1_norm",
        "l2_norm",
        "mean",
        "std",
        "third_moment",
        "fourth_moment",
    )
    scopes = (
        ("overall", "", 8),
        ("embeddings", "", 2),
        ("unembedding", "", 2),
        ("block", "0", 3),
        ("final_norm", "", 1),
    )
    rows = [
        "step,tokens_processed,cumulative_estimated_flops,scope,layer,"
        "family,stat,value,element_count"
    ]
    for step in (1, 2):
        for scope_index, (scope, layer, element_count) in enumerate(scopes):
            for family_index, family in enumerate(families):
                for stat_index, statistic in enumerate(statistics):
                    value = step + scope_index / 10 + family_index / 100 + stat_index / 1000
                    rows.append(
                        f"{step},{step * 10},{step * 1000},{scope},{layer},"
                        f"{family},{statistic},{value},{element_count}"
                    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _record_for_run(run: Path, *, validation: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "run_id": run.name,
        "status": "ok",
        "submission": "reference",
        "track": "sample_efficiency",
        "profile": "official",
        "seed": 1,
        "qualified": True,
        "metrics": {
            "tokens_processed": 20,
            "train_seconds": 1.0,
            "validation_loss": 3.7,
        },
        "artifacts": {
            "training_curve": {
                "path": "training.csv",
                "bytes": (run / "training.csv").stat().st_size,
                "sha256": _sha(run / "training.csv"),
            }
        },
    }
    if validation:
        artifacts = record["artifacts"]
        assert isinstance(artifacts, dict)
        artifacts["validation_curve"] = {
            "path": "validation.csv",
            "bytes": (run / "validation.csv").stat().st_size,
            "sha256": _sha(run / "validation.csv"),
        }
    return record


def _payload(html: str) -> dict[str, object]:
    match = re.search(
        r'<script type="application/json" id="report-data">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
