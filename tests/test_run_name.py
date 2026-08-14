"""Tests for run naming: the slug, the run ID, and the prompt."""

from __future__ import annotations

import argparse
import sys
import unittest
from unittest.mock import patch

from rig import cli
from rig.config import ConfigError
from rig.harness import normalize_run_name
from rig.harness.models import MAX_RUN_NAME
from rig.harness.runner import _new_run_id


class _Style:
    enabled = False

    def text(self, value: str, *_: str) -> str:
        return value

    def note(self, value: str) -> None:
        self.noted = value


class NormalizeRunNameTests(unittest.TestCase):
    def test_reduces_human_input_to_a_safe_directory_segment(self) -> None:
        cases = {
            "LR 2^-8 seed 1337": "lr-2-8-seed-1337",
            "  Big Batch!!  ": "big-batch",
            "already-fine": "already-fine",
            "Mixed___Separators   here": "mixed-separators-here",
            "café": "caf",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_run_name(raw), expected)

    def test_unusable_input_becomes_empty_rather_than_mangled(self) -> None:
        for raw in ("", "   ", "!!!", "---", "///"):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_run_name(raw), "")

    def test_output_is_bounded_and_never_ends_in_a_separator(self) -> None:
        name = normalize_run_name("a" * (MAX_RUN_NAME + 20))
        self.assertEqual(len(name), MAX_RUN_NAME)
        # Truncation must not leave a trailing hyphen in a directory name.
        awkward = normalize_run_name("a" * (MAX_RUN_NAME - 1) + " tail")
        self.assertFalse(awkward.endswith("-"))

    def test_rejects_non_strings(self) -> None:
        with self.assertRaises(TypeError):
            normalize_run_name(None)  # type: ignore[arg-type]


class RunIdTests(unittest.TestCase):
    def test_unnamed_id_keeps_the_original_shape(self) -> None:
        run_id = _new_run_id("reference")
        self.assertRegex(run_id, r"^\d{8}T\d{6}\.\d+Z-reference-[0-9a-f]{8}$")

    def test_named_id_embeds_the_label_and_stays_unique(self) -> None:
        first = _new_run_id("reference", "lr-2-8")
        second = _new_run_id("reference", "lr-2-8")
        self.assertRegex(first, r"^\d{8}T\d{6}\.\d+Z-reference-lr-2-8-[0-9a-f]{8}$")
        self.assertNotEqual(first, second)


class ResolveRunNameTests(unittest.TestCase):
    def _resolve(self, name, *, tty=False, worker=False, typed=()):
        with (
            patch.object(sys.stdin, "isatty", return_value=tty),
            patch.object(cli, "_is_cluster_worker", return_value=worker),
            patch("builtins.input", side_effect=list(typed)),
        ):
            return cli._resolve_run_name(argparse.Namespace(name=name), _Style())

    def test_explicit_flag_is_normalized(self) -> None:
        self.assertEqual(self._resolve("LR 2^-8 seed 1337"), "lr-2-8-seed-1337")

    def test_explicit_flag_with_no_usable_characters_is_an_error(self) -> None:
        # Silently ignoring it would attach the wrong label to a real run.
        with self.assertRaises(ConfigError):
            self._resolve("!!!")

    def test_a_terminal_is_prompted_and_enter_keeps_the_default(self) -> None:
        self.assertEqual(self._resolve(None, tty=True, typed=[""]), "")
        self.assertEqual(self._resolve(None, tty=True, typed=["Big Batch"]), "big-batch")

    def test_prompt_repeats_until_usable_or_skipped(self) -> None:
        self.assertEqual(self._resolve(None, tty=True, typed=["!!!", "second"]), "second")
        self.assertEqual(self._resolve(None, tty=True, typed=["!!!", ""]), "")

    def test_non_interactive_callers_are_never_prompted(self) -> None:
        # A study loop, a piped shell, or a pdsh worker cannot answer a prompt,
        # so asking would hang the run instead of naming it.
        self.assertEqual(self._resolve(None, tty=False), "")
        self.assertEqual(self._resolve(None, tty=True, worker=True), "")

    def test_run_parser_exposes_the_flag_and_defaults_to_unset(self) -> None:
        args = cli.build_parser().parse_args(["run", "reference"])
        self.assertIsNone(args.name)
        named = cli.build_parser().parse_args(["run", "reference", "--name", "x"])
        self.assertEqual(named.name, "x")


if __name__ == "__main__":
    unittest.main()
