"""One axis for what happens to model weights."""

from __future__ import annotations

import argparse
from typing import get_args
import unittest

from rig.cli import _CHECKPOINT_POLICIES, _checkpoint_policy
from rig.config import ConfigError
from rig.harness.models import CheckpointPolicy


def _args(**overrides) -> argparse.Namespace:
    base = dict(checkpoint_policy=None)
    base.update(overrides)
    return argparse.Namespace(**base)


class PolicyTests(unittest.TestCase):
    def test_the_three_outcomes(self) -> None:
        self.assertEqual(_CHECKPOINT_POLICIES, ("always", "qualifying", "none"))
        self.assertEqual(get_args(CheckpointPolicy), _CHECKPOINT_POLICIES)

    def test_settings_supply_the_default(self) -> None:
        self.assertEqual(
            _checkpoint_policy(_args(), "qualifying", profile="dev"),
            "qualifying",
        )

    def test_legacy_spellings_are_no_longer_accepted(self) -> None:
        # "all"/"none-after-validation" and --omit-checkpoint are gone rather
        # than aliased: no legacy runs exist on these nodes, and keeping both
        # spellings is what allowed a contradiction to be expressed at all.
        for gone in ("all", "none-after-validation"):
            with self.subTest(value=gone):
                with self.assertRaisesRegex(ConfigError, "unknown checkpoint policy"):
                    _checkpoint_policy(_args(), gone, profile="dev")

    def test_an_unknown_policy_is_refused(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown checkpoint policy"):
            _checkpoint_policy(_args(), "keep-forever", profile="dev")


class DefaultYieldsWhereNoneIsIllegalTests(unittest.TestCase):
    """A saved default of ``none`` must not refuse runs that require weights."""

    @staticmethod
    def _resolve(explicit, profile):
        return _checkpoint_policy(
            _args(checkpoint_policy=explicit), "none", profile=profile
        )

    def test_the_default_yields_where_none_is_not_legal(self) -> None:
        # Sweep points keep no weights; that is the point of the default.
        self.assertEqual(self._resolve(None, "dev"), "none")
        # Official runs require a checkpoint. Without this, flipping the saved
        # default to "none" refused every official run outright -- the guard
        # below rejects the policy the default had just chosen for it.
        self.assertEqual(self._resolve(None, "official"), "qualifying")

    def test_an_explicit_none_is_still_refused_where_it_is_illegal(self) -> None:
        # Yielding covers a default reaching somewhere it was not meant to,
        # never a caller asking for something disallowed.
        self.assertEqual(self._resolve("none", "official"), "none")

    def test_an_explicit_choice_always_wins(self) -> None:
        for policy in ("always", "qualifying", "none"):
            with self.subTest(policy=policy):
                self.assertEqual(self._resolve(policy, "dev"), policy)


class EvaluatorRemovalTests(unittest.TestCase):
    def test_the_unimplemented_evaluator_hook_is_gone(self) -> None:
        # A type alias with one guarded call site and no implementation in 93
        # commits. Its only live effect was forcing a checkpoint to exist so it
        # could be deleted again.
        import rig.harness.models as models
        import rig.harness.runner as runner
        import rig.harness.validation as validation

        self.assertFalse(hasattr(models, "Evaluator"))
        for module in (runner, validation):
            self.assertNotIn("evaluator", inspect_source(module))


def inspect_source(module) -> str:
    import inspect

    return inspect.getsource(module)


if __name__ == "__main__":
    unittest.main()


class PublicSurfaceTests(unittest.TestCase):
    def test_unknown_trainer_arguments_are_not_forwarded(self) -> None:
        from rig.cli import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(["run", "reference", "--layers", "13"])

    def test_supported_research_overrides_are_explicit(self) -> None:
        from rig.cli import build_parser

        args = build_parser().parse_args(
            [
                "run",
                "reference",
                "--base-learning-rate",
                "0.001",
                "--batch-size",
                "128",
            ]
        )
        self.assertEqual(args.base_learning_rate, 0.001)
        self.assertEqual(args.batch_size, 128)
