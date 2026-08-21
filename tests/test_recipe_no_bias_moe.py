"""Contracts for the bias-free routed recipe fork."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rig.configfile import read_config_document  # noqa: E402
from rig.nn import linear  # noqa: E402


ROOT = Path(__file__).parents[1]


def _load_trainer(recipe: str):
    path = ROOT / "recipes" / recipe / "train.py"
    spec = importlib.util.spec_from_file_location(f"{recipe}_train", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trainer = _load_trainer("no_bias_moe")
reference = _load_trainer("reference_moe")


def _smoke_config(module):
    experiment, digest = module.load_experiment_config("smoke")
    args = module.build_parser().parse_args(["--profile", "smoke"])
    return module.resolve_config(
        args,
        "cpu",
        experiment_config=experiment,
        config_sha256=digest,
    )


def _dev_config(module, tier: str):
    experiment, digest = module.load_experiment_config("dev")
    args = module.build_parser().parse_args(
        ["--profile", "dev", "--tier", tier]
    )
    return module.resolve_config(
        args,
        "tpu",
        experiment_config=experiment,
        config_sha256=digest,
    )


class NoBiasArchitectureTests(unittest.TestCase):
    def test_recipe_identity_and_scientific_configs_are_inherited(self) -> None:
        self.assertEqual(trainer.RECIPE_NAME, "no_bias_moe")
        self.assertEqual(
            trainer.build_parser().parse_args([]).output_dir,
            Path("runs/no_bias_moe"),
        )
        for filename in ("config.yaml", "dev.yaml", "smoke.yaml"):
            with self.subTest(filename=filename):
                got, _ = read_config_document(
                    ROOT / "recipes" / "no_bias_moe" / filename
                )
                inherited, _ = read_config_document(
                    ROOT / "recipes" / "reference_moe" / filename
                )
                self.assertEqual(got, inherited)

    def test_only_bias_leaves_are_removed_and_common_weights_are_bit_equal(self) -> None:
        no_bias = trainer.flatten_arrays(trainer.init_params(_smoke_config(trainer), 7))
        biasful = reference.flatten_arrays(
            reference.init_params(_smoke_config(reference), 7)
        )

        self.assertEqual(set(no_bias), set(biasful) & set(no_bias))
        removed = set(biasful) - set(no_bias)
        self.assertTrue(removed)
        self.assertEqual(
            {path.rsplit("/", 1)[-1] for path in removed},
            {"qkv_b", "attn_b", "expert_up_b", "expert_down_b"},
        )
        for path, value in no_bias.items():
            with self.subTest(path=path):
                np.testing.assert_array_equal(value, biasful[path])

    def test_active_counter_uses_weight_only_experts(self) -> None:
        config = _smoke_config(trainer)
        params = trainer.init_params(config, 11)
        total = trainer.parameter_count(params)
        per_expert = 2 * config.d_model * config.expert_mult * config.d_model
        expected = total - config.layers * (
            config.experts - config.expert_top_k
        ) * per_expert
        self.assertEqual(trainer.active_parameter_count(params, config), expected)
        self.assertFalse(trainer.contract_model_metadata(config)["linear_bias"])
        runtime = trainer.AttentionRuntime(None, "dense", None, 0.0)
        self.assertEqual(
            trainer.implementation_metadata(config, runtime)["weight_decay_policy"],
            "weights_and_embeddings_only_v3_no_bias",
        )

    def test_declared_ladder_and_materialized_active_count_agree(self) -> None:
        for tier in ("60m", "125m", "250m", "500m"):
            with self.subTest(tier=tier):
                config = _dev_config(trainer, tier)
                expected = config.declared_parameters + config.layers * (
                    config.d_model * config.experts
                    - (config.mlp_mult + 5) * config.d_model
                )
                self.assertEqual(trainer.expected_active_parameters(config), expected)

        # Materialize only the smallest rung; larger tiers exercise the same
        # closed form but would add gigabytes and minutes to a CPU test.
        config = _dev_config(trainer, "60m")
        params = trainer.init_params(config, 12)
        self.assertEqual(
            trainer.active_parameter_count(params, config),
            trainer.expected_active_parameters(config),
        )

    def test_gradients_reach_router_and_every_expert_weight(self) -> None:
        config = _smoke_config(trainer)
        params = trainer.init_params(config, 13)
        tokens = jnp.asarray(
            np.random.default_rng(14).integers(
                0,
                config.semantic_vocab_size,
                size=(1, config.seq_len + 1),
            )
        )
        gradients = jax.grad(
            lambda candidate: trainer.cross_entropy(
                candidate, tokens[:, :-1], tokens[:, 1:], config
            )
        )(params)

        for layer, block in enumerate(gradients["blocks"]):
            with self.subTest(layer=layer):
                self.assertEqual(
                    set(block),
                    {
                        "ln1_scale",
                        "qkv_w",
                        "attn_w",
                        "ln2_scale",
                        "router_w",
                        "expert_up_w",
                        "expert_down_w",
                    },
                )
                for name in ("router_w", "expert_up_w", "expert_down_w"):
                    magnitude = float(jnp.abs(block[name]).max())
                    self.assertTrue(np.isfinite(magnitude))
                    self.assertGreater(magnitude, 0.0)
                per_expert = jnp.abs(block["expert_up_w"]).max(axis=(1, 2))
                self.assertTrue(bool((per_expert > 0).all()))

    def test_decay_policy_rejects_an_accidental_bias(self) -> None:
        params = trainer.init_params(_smoke_config(trainer), 17)
        mask = trainer.weight_decay_mask(params)
        self.assertTrue(mask["token_embedding"])
        self.assertTrue(mask["blocks"][0]["expert_up_w"])
        self.assertFalse(mask["blocks"][0]["ln1_scale"])
        with self.assertRaisesRegex(ValueError, "forbidden"):
            trainer.weight_decay_mask({"accidental_b": np.zeros((8,))})


class SharedLinearTests(unittest.TestCase):
    def test_optional_bias_is_exactly_optional(self) -> None:
        x = jnp.asarray([[1.0, 2.0]], jnp.float32)
        weight = jnp.asarray([[3.0, 4.0], [5.0, 6.0]], jnp.float32)
        bias = jnp.asarray([7.0, 8.0], jnp.float32)
        unbiased = linear(x, weight, None, jnp.float32)
        np.testing.assert_array_equal(unbiased, np.asarray([[13.0, 16.0]]))
        np.testing.assert_array_equal(
            linear(x, weight, bias, jnp.float32),
            np.asarray([[20.0, 24.0]]),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
