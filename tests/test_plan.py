from __future__ import annotations

import sys
from pathlib import Path
import unittest

from rig.plan import PlanError, resolve_recipe_plan, validate_recipe_plan


ROOT = Path(__file__).resolve().parents[1]


def resolve(name: str, *arguments: str):
    recipe = ROOT / "recipes" / name
    return resolve_recipe_plan(
        python_executable=sys.executable,
        trainer=recipe / "train.py",
        arguments=arguments,
        cwd=ROOT,
    )


class RecipePlanTests(unittest.TestCase):
    def test_context_presets_and_recipe_fork_share_the_500m_5tpp_horizon(self) -> None:
        common = (
            "--profile",
            "dev",
            "--tier",
            "500m",
            "--tokens-per-parameter",
            "5",
        )
        plans = {
            "dense_1k": resolve("reference", *common),
            "dense_8k": resolve("reference", *common, "--context", "8k"),
            "moe_8k": resolve("reference_moe", *common),
            "moe_1k": resolve("reference_moe", *common, "--context", "1k"),
        }

        expected_tokens = {plan.expected_tokens for plan in plans.values()}
        declared_parameters = {
            plan.payload["declared_parameters"] for plan in plans.values()
        }
        tokens_per_step = {plan.payload["tokens_per_step"] for plan in plans.values()}
        ladder_multipliers = {
            plan.payload["ladder_data_multiplier"] for plan in plans.values()
        }

        self.assertEqual(expected_tokens, {2_513_043_456})
        self.assertEqual(declared_parameters, {502_602_240})
        self.assertEqual(tokens_per_step, {131_072})
        self.assertEqual(len(ladder_multipliers), 1)
        self.assertTrue(all(plan.run_kind == "full" for plan in plans.values()))
        self.assertEqual(plans["dense_1k"].payload["batch_size"], 128)
        self.assertEqual(plans["dense_8k"].payload["batch_size"], 16)
        self.assertEqual(plans["moe_8k"].payload["batch_size"], 16)
        self.assertEqual(plans["moe_1k"].payload["batch_size"], 128)
        self.assertEqual(plans["dense_1k"].payload["sequence_length"], 1024)
        self.assertEqual(plans["dense_8k"].payload["sequence_length"], 8192)
        self.assertFalse(plans["dense_1k"].payload["document_masking"])
        self.assertTrue(plans["dense_8k"].payload["document_masking"])
        self.assertEqual(plans["moe_8k"].payload["context_preset"], "8k")
        self.assertEqual(plans["moe_1k"].payload["context_preset"], "1k")
        self.assertEqual(
            {plan.validation_predictions for plan in plans.values()},
            {1_048_576},
        )

    def test_stop_after_step_is_a_full_schedule_prefix(self) -> None:
        common = (
            "--profile",
            "dev",
            "--tier",
            "500m",
            "--tokens-per-parameter",
            "5",
        )
        full = resolve("reference", *common)
        stopped = resolve("reference", *common, "--stop-after-step", "100")

        self.assertEqual(full.run_kind, "full")
        self.assertEqual(stopped.run_kind, "diagnostic")
        for field in (
            "schedule_steps",
            "planned_tokens",
            "target_tokens_per_parameter",
            "achieved_tokens_per_parameter",
            "ladder_data_multiplier",
            "base_learning_rate",
            "batch_size",
            "tokens_per_step",
        ):
            with self.subTest(field=field):
                self.assertEqual(stopped.payload[field], full.payload[field])
        self.assertEqual(stopped.payload["stop_after_step"], 100)
        self.assertEqual(
            stopped.expected_tokens,
            100 * int(stopped.payload["tokens_per_step"]),
        )

    def test_validator_rejects_an_inconsistent_token_budget(self) -> None:
        valid = dict(resolve("reference", "--profile", "smoke").payload)
        valid["expected_tokens"] = int(valid["expected_tokens"]) + 1

        with self.assertRaisesRegex(PlanError, "expected_tokens"):
            validate_recipe_plan(valid)

    def test_validator_rejects_unversioned_extra_fields(self) -> None:
        valid = dict(resolve("reference", "--profile", "smoke").payload)
        valid["surprise"] = True

        with self.assertRaisesRegex(PlanError, "unknown field"):
            validate_recipe_plan(valid)


if __name__ == "__main__":
    unittest.main()
