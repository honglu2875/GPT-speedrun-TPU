from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from speedrun import family_study


REPO = Path(__file__).parents[1]
SUITE = REPO / "studies" / "complete_d_p_lr_v3" / "suite.yaml"
LARGE_SUITE = REPO / "studies" / "complete_d_p_lr_large_v1" / "suite.yaml"
DEPTH_SUITES = {
    "l16": REPO / "studies" / "complete_d_p_depth_l16_lr_v1" / "suite.yaml",
    "l24": REPO / "studies" / "complete_d_p_depth_l24_lr_v1" / "suite.yaml",
}


class FamilyStudyTests(unittest.TestCase):
    def test_checked_in_suite_expands_to_immutable_csv_plan(self) -> None:
        suite = family_study.load_suite(SUITE, REPO)
        rows = family_study.planned_rows(suite)
        self.assertEqual(len(rows), 3 * 7)
        self.assertEqual(rows[0]["point_id"], "60m-lr00-s1337")
        self.assertEqual(rows[-1]["point_id"], "250m-lr06-s1337")
        self.assertEqual(rows[0]["base_learning_rate"], "0.0009765625")
        self.assertEqual(rows[-1]["base_learning_rate"], "0.0625")
        self.assertEqual(rows[0]["planned_steps"], 2286)
        self.assertEqual(rows[0]["planned_train_tokens"], 299_630_592)
        self.assertEqual(rows[-1]["planned_steps"], 9_325)
        self.assertEqual(rows[-1]["planned_train_tokens"], 1_222_246_400)
        self.assertTrue(all(row["suite_sha256"] == suite["suite_sha256"] for row in rows))

    def test_resume_rejects_any_change_to_plan_fields(self) -> None:
        suite = family_study.load_suite(SUITE, REPO)
        planned = family_study.planned_rows(suite)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            family_study._write_csv(path, planned)
            rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
            rows[0]["base_learning_rate"] = "0.9"
            family_study._write_csv(path, rows)
            with self.assertRaisesRegex(family_study.StudyError, "immutable field"):
                family_study._read_existing(path, planned)

    def test_large_suite_covers_both_confirmation_tiers(self) -> None:
        suite = family_study.load_suite(LARGE_SUITE, REPO)
        rows = family_study.planned_rows(suite)
        self.assertEqual(len(rows), 2 * 3)
        self.assertEqual(rows[0]["point_id"], "500m-lr00-s1337")
        self.assertEqual(rows[-1]["point_id"], "1b-lr02-s1337")
        self.assertEqual(rows[1]["point_id"], "500m-lr01-s1337")
        self.assertEqual(rows[4]["point_id"], "1b-lr01-s1337")
        self.assertEqual(rows[1]["base_learning_rate"], "0.00390625")
        self.assertEqual(rows[4]["base_learning_rate"], "0.00390625")
        self.assertGreater(int(rows[-1]["planned_train_tokens"]), 3_900_000_000)

    def test_depth_ablation_suites_reuse_the_reference_lr_protocol(self) -> None:
        expected = {
            "l16": (67_012_992, 2_556, 335_020_032),
            "l24": (81_202_560, 3_098, 406_061_056),
        }
        for name, path in DEPTH_SUITES.items():
            with self.subTest(candidate=name):
                suite = family_study.load_suite(path, REPO)
                rows = family_study.planned_rows(suite)
                parameters, steps, tokens = expected[name]
                self.assertEqual(len(rows), 7)
                self.assertEqual(rows[0]["base_learning_rate"], "0.0009765625")
                self.assertEqual(rows[-1]["base_learning_rate"], "0.0625")
                self.assertEqual(rows[0]["declared_parameters"], parameters)
                self.assertEqual(rows[0]["planned_steps"], steps)
                self.assertEqual(rows[0]["planned_train_tokens"], tokens)

    def test_only_point_rejects_unknown_identity_before_loading_config(self) -> None:
        suite = family_study.load_suite(SUITE, REPO)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(family_study.StudyError, "unknown point"):
                family_study.run_study(
                    suite,
                    repo=REPO,
                    results=Path(directory) / "results.csv",
                    color="never",
                    only_point="not-a-point",
                )

    def test_record_identity_includes_suite_hash(self) -> None:
        record = {
            "provenance": {
                "study": {
                    "study_id": "study",
                    "point_id": "point",
                    "suite_sha256": "a" * 64,
                }
            }
        }
        self.assertIs(
            family_study._record_for_point(
                [record], "study", "point", "a" * 64
            ),
            record,
        )
        self.assertIsNone(
            family_study._record_for_point(
                [record], "study", "point", "b" * 64
            )
        )

    def test_completed_record_must_match_planned_tier_budget_and_lr(self) -> None:
        suite = family_study.load_suite(SUITE, REPO)
        row = family_study.planned_rows(suite)[0]
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            run_id = "reference-run"
            run = artifacts / run_id
            run.mkdir()
            (run / "result.json").write_text("{}\n", encoding="utf-8")
            record = {
                "run_id": run_id,
                "metrics": {
                    "parameter_count": int(row["declared_parameters"]),
                    "tokens_processed": int(row["planned_train_tokens"]),
                    "training_steps": int(row["planned_steps"]),
                    "validation_loss": 3.0,
                    "train_seconds": 2.0,
                    "tokens_per_second": 10.0,
                },
                "implementation": {
                    "configuration": {
                        "sha256": "c" * 64,
                        "resolved": {
                            "training": {"batch_size": int(row["batch_size"])},
                            "model": {"tier": row["tier"]},
                            "optimizer": {
                                "learning_rate": float(row["base_learning_rate"]),
                                "effective": {
                                    "global_peak_learning_rate": float(
                                        row["base_learning_rate"]
                                    )
                                },
                            },
                            "parameterization": {
                                "width_multiplier": 1.0,
                                "depth_multiplier": 1.0,
                                "data_multiplier": 1.0,
                                "batch_multiplier": 1.0,
                                "depth_alpha": 1.0,
                            },
                        },
                    }
                },
                "provenance": {
                    "dataset": {
                        "name": "fineweb-4b-gpt2",
                        "manifest": {"canonical_sha256": "d" * 64},
                    }
                },
            }
            family_study._populate(row, record, artifacts)
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["actual_parameters"], row["declared_parameters"])
            self.assertEqual(row["dataset_name"], "fineweb-4b-gpt2")

            changed = dict(row)
            changed["status"] = "pending"
            changed["planned_steps"] = int(row["planned_steps"]) + 1
            with self.assertRaisesRegex(family_study.StudyError, "changed steps"):
                family_study._populate(changed, record, artifacts)


if __name__ == "__main__":
    unittest.main()
