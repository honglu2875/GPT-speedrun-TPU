from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
import unittest

from rig.console import (
    Console,
    standard_data_rows,
    standard_identity_rows,
    standard_kernel_rows,
    standard_schedule_rows,
    standard_training_rows,
)
from rig.evaluation import DomainEvaluation, EvaluationReport, EvaluationResult


class StandardRunRowsTests(unittest.TestCase):
    def test_row_groups_have_fixed_labels_and_formatting(self) -> None:
        devices = [SimpleNamespace(device_kind="TPU v4") for _ in range(4)]
        rows = (
            *standard_identity_rows(
                config_filename="config.yaml",
                config_profile="dev",
                config_sha256="a" * 64,
                devices=devices,
                process_count=1,
                process_index=0,
            ),
            *standard_data_rows(
                source="79 train + 1 val shard(s)",
                train_tokens=7_900_000_000,
                validation_tokens=100_000_000,
                downstream_domains=0,
                downstream_tokens=0,
            ),
            *standard_training_rows(
                parameterization="completep_fixed_tpp_v1",
                width_multiplier=1.0,
                depth_multiplier=1.0,
                data_multiplier=1.0,
                batch_size=128,
                seq_len=1024,
                sampling="shuffled_epochs",
                usable_tokens_per_epoch=7_899_979_776,
                dtype_name="bfloat16",
            ),
            *standard_kernel_rows(
                attention_backend="tpu_flash",
                attention_rows=(("attention tuning", "heuristic"),),
                loss_backend="tiled",
                semantic_vocab_size=50_304,
                vocab_tile_size=2_048,
            ),
            *standard_schedule_rows(
                diagnostics_every=10,
                final_step=10,
                schedule_steps=2_286,
                early_stopped=True,
                tokens_processed=1_310_720,
                total_flops=447_364_800_184_320,
                flop_breakdown=(("dot_general", "10 (100.0%)"),),
                capture_window=None,
                xprof_destination=None,
            ),
        )

        self.assertEqual(
            [label for label, _ in rows],
            [
                "experiment config",
                "devices",
                "JAX processes",
                "mesh",
                "dataset",
                "train / val tokens",
                "downstream",
                "parameterization",
                "global batch",
                "train sampling",
                "compute",
                "attention",
                "attention tuning",
                "output loss",
                "diagnostics",
                "duration",
                "train tokens",
                "traced FLOPs",
                "FLOP breakdown",
                "XProf",
            ],
        )
        rendered = dict(rows)
        self.assertEqual(rendered["downstream"], "not requested")
        self.assertIn("mN=1", rendered["parameterization"])
        self.assertIn("early stop", rendered["duration"])
        self.assertEqual(rendered["XProf"], "disabled")

    def test_console_renders_domain_results_from_the_report(self) -> None:
        report = EvaluationReport(
            EvaluationResult(loss=1.0, scored_tokens=16, seconds=0.1),
            (
                DomainEvaluation(
                    "books", EvaluationResult(loss=2.0, scored_tokens=8, seconds=0.2)
                ),
            ),
        )
        stderr = StringIO()

        with redirect_stderr(stderr):
            Console("never").evaluations(report)

        output = stderr.getvalue()
        self.assertIn("books", output)
        self.assertIn("fresh10 macro", output)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
