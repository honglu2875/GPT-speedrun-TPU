"""One axis for what happens to model weights."""

from __future__ import annotations

import argparse
import unittest

from rig.cli import _CHECKPOINT_POLICIES, _checkpoint_policy
from rig.config import ConfigError


def _args(**overrides) -> argparse.Namespace:
    base = dict(checkpoint_policy=None)
    base.update(overrides)
    return argparse.Namespace(**base)


class PolicyTests(unittest.TestCase):
    def test_the_three_outcomes(self) -> None:
        self.assertEqual(_CHECKPOINT_POLICIES, ("always", "qualifying", "none"))

    def test_settings_supply_the_default(self) -> None:
        self.assertEqual(
            _checkpoint_policy(_args(), "qualifying", track="open", profile="dev"),
            "qualifying",
        )

    def test_legacy_spellings_are_no_longer_accepted(self) -> None:
        # "all"/"none-after-validation" and --omit-checkpoint are gone rather
        # than aliased: no legacy runs exist on these nodes, and keeping both
        # spellings is what allowed a contradiction to be expressed at all.
        for gone in ("all", "none-after-validation"):
            with self.subTest(value=gone):
                with self.assertRaisesRegex(ConfigError, "unknown checkpoint policy"):
                    _checkpoint_policy(_args(), gone, track="open", profile="dev")

    def test_an_unknown_policy_is_refused(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown checkpoint policy"):
            _checkpoint_policy(_args(), "keep-forever", track="open", profile="dev")


class DefaultYieldsWhereNoneIsIllegalTests(unittest.TestCase):
    """A saved default of ``none`` must not refuse runs that require weights."""

    @staticmethod
    def _resolve(explicit, track, profile):
        return _checkpoint_policy(
            _args(checkpoint_policy=explicit), "none", track=track, profile=profile
        )

    def test_the_default_yields_where_none_is_not_legal(self) -> None:
        # Sweep points keep no weights; that is the point of the default.
        self.assertEqual(self._resolve(None, "open", "dev"), "none")
        # Official runs require a checkpoint. Without this, flipping the saved
        # default to "none" refused every official run outright -- the guard
        # below rejects the policy the default had just chosen for it.
        self.assertEqual(
            self._resolve(None, "sample_efficiency", "official"), "qualifying"
        )
        self.assertEqual(self._resolve(None, "open", "official"), "qualifying")

    def test_an_explicit_none_is_still_refused_where_it_is_illegal(self) -> None:
        # Yielding covers a default reaching somewhere it was not meant to,
        # never a caller asking for something disallowed.
        self.assertEqual(self._resolve("none", "sample_efficiency", "official"), "none")

    def test_an_explicit_choice_always_wins(self) -> None:
        for policy in ("always", "qualifying", "none"):
            with self.subTest(policy=policy):
                self.assertEqual(self._resolve(policy, "open", "dev"), policy)


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


class RetiredFlagTests(unittest.TestCase):
    """A removed flag must say what replaced it."""

    def _run_cli(self, *argv: str) -> str:
        import contextlib, io
        from rig.cli import main

        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            main(["run", "reference", *argv])
        return err.getvalue()

    def test_removed_checkpoint_flags_name_their_replacement(self) -> None:
        # `rig run` forwards unknown arguments to the trainer, so a retired
        # flag would otherwise surface as an argparse error from train.py
        # about a flag the user had used correctly the day before.
        self.assertIn("--checkpoint-policy", self._run_cli("--checkpoints", "always"))
        self.assertIn("--checkpoint-policy none", self._run_cli("--omit-checkpoint"))

    def test_genuine_trainer_arguments_still_pass_through(self) -> None:
        # The forwarding itself is deliberate; only retired flags are caught.
        from rig.cli import _RETIRED_FLAGS

        self.assertNotIn("--base-learning-rate", _RETIRED_FLAGS)
        self.assertNotIn("--study-batch-size", _RETIRED_FLAGS)
