"""Scientific contract for the Complete(d)P token-duration ablation."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("JAX_PLATFORMS", "cpu")


ROOT = Path(__file__).parents[1]


def _load_trainer(name: str, recipe: str):
    path = ROOT / "recipes" / recipe / "train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reference = _load_trainer("reference_duration_control", "reference")
duration = _load_trainer("reference_duration_treatment", "reference_duration")


def _resolve(module, *, tpp: float, batch: int = 128, learning_rate: float = 2**-8):
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--profile",
            "dev",
            "--tier",
            "500m",
            "--context",
            "1k",
            "--tokens-per-parameter",
            str(tpp),
            "--batch-size",
            str(batch),
            "--base-learning-rate",
            str(learning_rate),
        ]
    )
    experiment_config, config_sha256 = module.load_experiment_config("dev")
    return module.resolve_config(
        args,
        "tpu",
        experiment_config=experiment_config,
        config_sha256=config_sha256,
    )


class ReferenceDurationTests(unittest.TestCase):
    def test_default_output_directory_tracks_each_recipe_folder(self) -> None:
        for module, name in (
            (reference, "reference"),
            (duration, "reference_duration"),
        ):
            with self.subTest(recipe=name):
                self.assertEqual(module.RECIPE_NAME, name)
                self.assertEqual(
                    module.build_parser().parse_args([]).output_dir,
                    Path("runs") / name,
                )

    def test_five_tpp_anchor_matches_reanchored_reference(self) -> None:
        control = _resolve(reference, tpp=5)
        treatment = _resolve(duration, tpp=5)

        self.assertEqual(treatment.parameterization, "completedp_duration_v1")
        self.assertEqual(treatment.duration_anchor_tpp, 5.0)
        self.assertEqual(treatment.duration_multiplier, 1.0)
        self.assertEqual(treatment.data_multiplier, control.data_multiplier)
        self.assertEqual(treatment.steps, control.steps)
        self.assertEqual(treatment.warmup_steps, control.warmup_steps)
        self.assertEqual(
            duration.effective_optimizer_metadata(treatment),
            reference.effective_optimizer_metadata(control),
        )

    def test_twenty_tpp_applies_every_duration_scalar_together(self) -> None:
        control = _resolve(reference, tpp=20)
        treatment = _resolve(duration, tpp=20)
        control_effective = reference.effective_optimizer_metadata(control)
        treatment_effective = duration.effective_optimizer_metadata(treatment)

        self.assertEqual(treatment.duration_multiplier, 4.0)
        self.assertEqual(treatment.data_multiplier, control.data_multiplier * 4.0)
        self.assertAlmostEqual(
            treatment_effective["global_peak_learning_rate"],
            control_effective["global_peak_learning_rate"] / 2.0,
        )
        self.assertAlmostEqual(
            treatment_effective["adam_epsilon_horizon_multiplier"],
            control_effective["adam_epsilon_horizon_multiplier"] * 2.0,
        )
        self.assertAlmostEqual(
            treatment_effective["weight_decay_horizon_multiplier"],
            control_effective["weight_decay_horizon_multiplier"] / 2.0,
        )
        for beta in ("beta1", "beta2"):
            self.assertAlmostEqual(
                1.0 - treatment_effective[beta],
                (1.0 - control_effective[beta]) / 4.0,
            )

    def test_twenty_tpp_batch_512_is_the_five_tpp_iso_horizon(self) -> None:
        anchor = _resolve(reference, tpp=5, batch=128)
        iso_horizon = _resolve(duration, tpp=20, batch=512)

        self.assertEqual(iso_horizon.steps, anchor.steps)
        self.assertEqual(iso_horizon.warmup_steps, anchor.warmup_steps)
        self.assertEqual(iso_horizon.batch_multiplier, 4.0)
        self.assertEqual(iso_horizon.duration_multiplier, 4.0)
        self.assertTrue(
            math.isclose(
                iso_horizon.batch_multiplier / iso_horizon.data_multiplier,
                anchor.batch_multiplier / anchor.data_multiplier,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
        self.assertEqual(
            duration.effective_optimizer_metadata(iso_horizon),
            reference.effective_optimizer_metadata(anchor),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
