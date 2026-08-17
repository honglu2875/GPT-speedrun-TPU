"""One axis for what happens to model weights."""

from __future__ import annotations

import argparse
import unittest

from rig.cli import _CHECKPOINT_POLICIES, _checkpoint_policy
from rig.config import ConfigError


def _args(**overrides) -> argparse.Namespace:
    base = dict(checkpoint_policy=None, checkpoints=None, omit_checkpoint=False)
    base.update(overrides)
    return argparse.Namespace(**base)


class PolicyTests(unittest.TestCase):
    def test_the_three_outcomes(self) -> None:
        self.assertEqual(_CHECKPOINT_POLICIES, ("always", "qualifying", "none"))

    def test_settings_supply_the_default(self) -> None:
        self.assertEqual(_checkpoint_policy(_args(), "qualifying"), "qualifying")

    def test_legacy_spellings_map_onto_the_new_axis(self) -> None:
        # "all" read as "every step" rather than "keep it", which is why it
        # was renamed; scripts using it must keep working.
        for legacy, expected in (
            ("all", "always"),
            ("qualifying", "qualifying"),
            ("none-after-validation", "none"),
        ):
            with self.subTest(legacy=legacy):
                self.assertEqual(
                    _checkpoint_policy(_args(checkpoints=legacy), "qualifying"), expected
                )

    def test_legacy_settings_values_are_accepted_as_the_default(self) -> None:
        self.assertEqual(_checkpoint_policy(_args(), "none-after-validation"), "none")
        self.assertEqual(_checkpoint_policy(_args(), "all"), "always")

    def test_omit_checkpoint_alone_still_means_none(self) -> None:
        self.assertEqual(_checkpoint_policy(_args(omit_checkpoint=True), "all"), "none")

    def test_agreeing_legacy_pair_is_accepted(self) -> None:
        # The sweep scripts pass both; they agree, so nothing is ambiguous.
        self.assertEqual(
            _checkpoint_policy(
                _args(checkpoints="none-after-validation", omit_checkpoint=True),
                "qualifying",
            ),
            "none",
        )

    def test_contradiction_is_refused_rather_than_silently_resolved(self) -> None:
        # This combination reads as "keep every checkpoint" and produced none.
        # It cost a 5.7-hour run its weights, so it must not be quietly picked.
        for contradiction in (
            _args(checkpoints="all", omit_checkpoint=True),
            _args(checkpoint_policy="always", omit_checkpoint=True),
            _args(checkpoint_policy="qualifying", omit_checkpoint=True),
        ):
            with self.subTest(args=contradiction):
                with self.assertRaisesRegex(ConfigError, "contradicts"):
                    _checkpoint_policy(contradiction, "qualifying")

    def test_the_new_flag_wins_over_the_legacy_one(self) -> None:
        self.assertEqual(
            _checkpoint_policy(
                _args(checkpoint_policy="always", checkpoints="none-after-validation"),
                "qualifying",
            ),
            "always",
        )

    def test_an_unknown_policy_is_refused(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown checkpoint policy"):
            _checkpoint_policy(_args(), "keep-forever")


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
