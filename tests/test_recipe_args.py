"""Contracts for the explicit argument groups shared by recipes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from rig.recipe_args import (
    add_standard_config_arguments,
    add_standard_data_arguments,
    add_standard_reporting_arguments,
    add_standard_xprof_arguments,
    new_recipe_parser,
    validate_standard_data_arguments,
    validate_standard_reporting_arguments,
    validate_standard_xprof_arguments,
)


def _option_strings(parser: argparse.ArgumentParser, title: str) -> set[str]:
    group = next(group for group in parser._action_groups if group.title == title)
    return {
        option
        for action in group._group_actions
        for option in action.option_strings
        if option != "--help"
    }


def _parser() -> argparse.ArgumentParser:
    parser = new_recipe_parser(description="test recipe")
    run = parser.add_argument_group("run")
    add_standard_config_arguments(
        run,
        default_output_dir=Path("runs/test"),
        profiles=("smoke", "dev", "official"),
    )
    add_standard_xprof_arguments(parser)
    add_standard_data_arguments(parser)
    optimization = parser.add_argument_group("optimization")
    add_standard_reporting_arguments(optimization)
    return parser


class RecipeArgumentDeclarationTests(unittest.TestCase):
    def test_helpers_add_exactly_the_documented_protocol_flags(self) -> None:
        parser = _parser()
        self.assertEqual(
            _option_strings(parser, "run"),
            {
                "--output-dir",
                "--seed",
                "--profile",
                "--color",
                "--print-plan",
            },
        )
        self.assertEqual(
            _option_strings(parser, "profiling"),
            {
                "--xprof-dir",
                "--xprof-start-step",
                "--xprof-steps",
                "--diagnostic-mode",
                "--omit-checkpoint",
            },
        )
        self.assertEqual(
            _option_strings(parser, "data"),
            {
                "--train-data",
                "--val-data",
                "--data-dtype",
                "--dataset-id",
                "--tokenizer-id",
                "--data-format",
                "--downstream-manifest",
                "--downstream-root",
            },
        )
        self.assertEqual(_option_strings(parser, "optimization"), {"--peak-tflops"})

    def test_defaults_match_the_recipe_invocation_protocol(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            args = _parser().parse_args([])
        self.assertEqual(args.output_dir, Path("runs/test"))
        self.assertEqual(args.seed, 1337)
        self.assertIsNone(args.profile)
        self.assertEqual(args.color, "auto")
        self.assertEqual(args.train_data, [])
        self.assertEqual(args.val_data, [])
        self.assertEqual(args.data_dtype, "uint16")
        self.assertEqual(args.data_format, "auto")

    def test_profile_environment_cannot_select_a_scientific_profile(self) -> None:
        with patch.dict(os.environ, {"RIG_PROFILE": "official"}, clear=True):
            self.assertIsNone(_parser().parse_args([]).profile)

    def test_parser_never_abbreviates_protocol_flags(self) -> None:
        with self.assertRaises(SystemExit):
            _parser().parse_args(["--diag"])

    def test_internal_checkpoint_signal_is_not_a_second_user_facing_policy(
        self,
    ) -> None:
        parser = _parser()
        self.assertNotIn("--omit-checkpoint", parser.format_help())
        self.assertTrue(parser.parse_args(["--omit-checkpoint"]).omit_checkpoint)


class RecipeArgumentValidationTests(unittest.TestCase):
    def parse(self, *arguments: str) -> argparse.Namespace:
        return _parser().parse_args(arguments)

    def test_data_validation_accepts_defaults_and_rejects_invalid_combinations(
        self,
    ) -> None:
        validate_standard_data_arguments(self.parse())
        for arguments, message in (
            (("--train-data", "train.bin"), "must be supplied together"),
            (("--val-data", "val.bin"), "must be supplied together"),
            (("--downstream-root", "root"), "requires --downstream-manifest"),
        ):
            with (
                self.subTest(arguments=arguments),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_standard_data_arguments(self.parse(*arguments))

    def test_xprof_validation_pins_the_complete_capture_window(self) -> None:
        validate_standard_xprof_arguments(self.parse(), execution_type="dev")
        valid = self.parse(
            "--xprof-dir",
            "trace",
            "--xprof-start-step",
            "11",
            "--xprof-steps",
            "10",
            "--diagnostic-mode",
        )
        validate_standard_xprof_arguments(valid, execution_type="dev")

        for arguments, execution_type, message in (
            (("--xprof-start-step", "1"), "dev", "require --xprof-dir"),
            (("--xprof-dir", "trace"), "dev", "requires both"),
            (("--diagnostic-mode",), "dev", "requires --xprof-dir"),
            (
                (
                    "--xprof-dir",
                    "trace",
                    "--xprof-start-step",
                    "1",
                    "--xprof-steps",
                    "1",
                    "--diagnostic-mode",
                    "--omit-checkpoint",
                ),
                "dev",
                "mutually exclusive",
            ),
            (("--omit-checkpoint",), "official", "restricted to development"),
        ):
            with (
                self.subTest(arguments=arguments),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_standard_xprof_arguments(
                    self.parse(*arguments), execution_type=execution_type
                )

    def test_diagnostic_mode_rejects_downstream_inputs(self) -> None:
        args = self.parse(
            "--xprof-dir",
            "trace",
            "--xprof-start-step",
            "1",
            "--xprof-steps",
            "1",
            "--diagnostic-mode",
            "--downstream-manifest",
            "manifest.yaml",
        )
        with self.assertRaisesRegex(ValueError, "downstream evaluation data"):
            validate_standard_xprof_arguments(args, execution_type="dev")

    def test_reporting_validation_requires_positive_finite_peak(self) -> None:
        validate_standard_reporting_arguments(self.parse())
        for value in ("0", "-1", "nan", "inf"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "must be positive"),
            ):
                validate_standard_reporting_arguments(
                    self.parse("--peak-tflops", value)
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
