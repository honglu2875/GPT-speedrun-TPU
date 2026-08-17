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
        self.assertEqual(_checkpoint_policy(_args(), "qualifying"), "qualifying")

    def test_legacy_spellings_are_no_longer_accepted(self) -> None:
        # "all"/"none-after-validation" and --omit-checkpoint are gone rather
        # than aliased: no legacy runs exist on these nodes, and keeping both
        # spellings is what allowed a contradiction to be expressed at all.
        for gone in ("all", "none-after-validation"):
            with self.subTest(value=gone):
                with self.assertRaisesRegex(ConfigError, "unknown checkpoint policy"):
                    _checkpoint_policy(_args(), gone)

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
