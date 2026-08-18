"""Gates for the routed (mixture-of-experts) recipe.

These ran as throwaway scripts while the recipe was being written, which is
exactly why two regressions reached a TPU: the grouped matmul is a Mosaic
kernel and cannot be auto-partitioned, and the balance loss silently degenerates
if it is handed an unreduced probability matrix. Both are checkable on CPU in
under a second, so both are checked here.
"""

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
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402


TRAINER_PATH = Path(__file__).parents[1] / "recipes" / "reference_moe" / "train.py"
SPEC = importlib.util.spec_from_file_location("reference_moe_train", TRAINER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib invariant
    raise RuntimeError(f"could not import {TRAINER_PATH}")
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


EXPERTS = 8
TOP_K = 2
WIDTH = 128
HIDDEN = 256


def _weights(seed: int = 0):
    """Router and expert stacks shaped as init_params lays them out."""

    rng = np.random.default_rng(seed)

    def draw(*shape):
        return jnp.asarray(rng.normal(size=shape) * 0.05, jnp.float32)

    return (
        draw(WIDTH, EXPERTS),
        draw(EXPERTS, WIDTH, HIDDEN),
        draw(EXPERTS, HIDDEN),
        draw(EXPERTS, HIDDEN, WIDTH),
        draw(EXPERTS, WIDTH),
    )


def _routed(x, weights, *, axis_name=None):
    return trainer.routed_mlp_local(
        x,
        *weights,
        experts=EXPERTS,
        top_k=TOP_K,
        dtype=jnp.float32,
        axis_name=axis_name,
    )


def _dense_reference(x, weights):
    """The same computation with no grouping: every expert sees every token.

    Deliberately written the slow, obvious way. If the grouped matmul, the
    argsort, the scatter-add, or the gate normalization is wrong, this disagrees.
    """

    router_w, up_w, up_b, down_w, down_b = weights
    batch, length, width = x.shape
    flat = x.reshape(batch * length, width)

    logits = flat @ router_w
    chosen_logits, chosen = jax.lax.top_k(logits, TOP_K)
    gate = jax.nn.softmax(chosen_logits, axis=-1)

    out = np.zeros((batch * length, width), np.float32)
    for token in range(batch * length):
        for slot in range(TOP_K):
            expert = int(chosen[token, slot])
            hidden = flat[token] @ up_w[expert] + up_b[expert]
            hidden = jax.nn.gelu(hidden, approximate=True)
            out[token] += float(gate[token, slot]) * np.asarray(
                hidden @ down_w[expert] + down_b[expert]
            )
    return jnp.asarray(out).reshape(batch, length, width)


class RoutedMlpTests(unittest.TestCase):
    def test_matches_a_dense_per_expert_reference(self) -> None:
        # tokens * top_k must be a multiple of the grouped matmul's 128 m-tile.
        x = jnp.asarray(
            np.random.default_rng(1).normal(size=(2, 32, WIDTH)) * 0.5, jnp.float32
        )
        weights = _weights()
        got, _, _ = _routed(x, weights)
        want = _dense_reference(x, weights)
        self.assertLess(float(jnp.abs(got - want).max()), 1e-5)

    def test_routing_is_dropless(self) -> None:
        """Every assignment is served, so no capacity factor exists to tune.

        ``group_sizes`` is data while the total row count is static, which is
        what lets the grouped matmul stay dropless. If a token were ever
        dropped the realized load would sum to less than one.
        """

        x = jnp.asarray(
            np.random.default_rng(2).normal(size=(4, 32, WIDTH)) * 3.0, jnp.float32
        )
        _, _, load = _routed(x, _weights())
        self.assertAlmostEqual(float(load.sum()), 1.0, places=5)
        self.assertTrue(bool((load >= 0).all()))

    def test_sharded_wrapper_agrees_with_the_local_body(self) -> None:
        """The regression that reached a TPU: gmm needs an explicit shard_map.

        An outer jit refuses to partition a Mosaic kernel at all -- "Mosaic
        kernels cannot be automatically partitioned" -- so the routed MLP needs
        its own sharded boundary exactly as attention does. This runs the real
        wrapper on a multi-device CPU mesh and requires it to reproduce the
        unsharded answer.

        The CPU symptom differs from the TPU one: gmm runs in interpret mode
        here, so deleting the wrapper trips the unbound ``data`` axis of the
        statistics pmean rather than the Mosaic message. Either way the guard
        fires, which is what makes it worth keeping on a CPU suite.
        """

        devices = jax.devices()
        self.assertGreaterEqual(len(devices), 8, "conftest forces 8 CPU devices")
        mesh = Mesh(np.asarray(devices[:8]).reshape(8), ("data",))

        # One sequence per device, and 64 * top_k = 128 rows locally -- the
        # smallest shape that satisfies the m-tile on every shard.
        x = jnp.asarray(
            np.random.default_rng(3).normal(size=(8, 64, WIDTH)), jnp.float32
        )
        weights = _weights()
        config = trainer.Config.__new__(trainer.Config)
        object.__setattr__(config, "experts", EXPERTS)
        object.__setattr__(config, "expert_top_k", TOP_K)
        object.__setattr__(config, "compute_dtype", jnp.float32)

        sharded = trainer.make_mesh_routed_mlp(config, mesh)
        with jax.set_mesh(mesh):
            got, got_probability, got_load = jax.jit(sharded)(x, *weights)
        want, want_probability, want_load = _routed(x, weights)

        self.assertLess(float(jnp.abs(got - want).max()), 1e-5)
        # The two statistics are pmean'd across the data axis, so the sharded
        # run must see the same global load the unsharded one does -- not one
        # device's view of it.
        self.assertLess(float(jnp.abs(got_load - want_load).max()), 1e-6)
        self.assertLess(float(jnp.abs(got_probability - want_probability).max()), 1e-6)

    def test_dense_mlp_is_untouched_when_experts_is_zero(self) -> None:
        config = trainer.Config.__new__(trainer.Config)
        object.__setattr__(config, "experts", 0)
        self.assertIsNone(trainer.make_mesh_routed_mlp(config, None))


class BalanceLossTests(unittest.TestCase):
    def test_uniform_load_is_the_minimum_and_equals_one(self) -> None:
        uniform = jnp.full((EXPERTS,), 1.0 / EXPERTS, jnp.float32)
        self.assertAlmostEqual(
            float(trainer.load_balance_loss(uniform, uniform)), 1.0, places=5
        )

        collapsed = jnp.zeros((EXPERTS,), jnp.float32).at[0].set(1.0)
        self.assertAlmostEqual(
            float(trainer.load_balance_loss(collapsed, collapsed)),
            float(EXPERTS),
            places=5,
        )
        self.assertGreater(
            float(trainer.load_balance_loss(collapsed, collapsed)),
            float(trainer.load_balance_loss(uniform, uniform)),
        )

    def test_rejects_an_unreduced_probability_matrix(self) -> None:
        """Passing [tokens, E] here is a silent no-op, not a loud error.

        Averaging a per-expert vector over axis 0 gives a scalar, which makes
        the term collapse to a constant 1.0 carrying no gradient to the router.
        The model trains normally and simply never balances, so the shape has
        to be rejected rather than broadcast.
        """

        with self.assertRaisesRegex(ValueError, "already reduced over"):
            trainer.load_balance_loss(
                jnp.full((32, EXPERTS), 1.0 / EXPERTS, jnp.float32),
                jnp.full((EXPERTS,), 1.0 / EXPERTS, jnp.float32),
            )

    def test_gradient_pushes_an_overloaded_expert_down(self) -> None:
        load = jnp.asarray([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01], jnp.float32)
        grad = jax.grad(lambda p: trainer.load_balance_loss(p, load))(
            jnp.full((EXPERTS,), 1.0 / EXPERTS, jnp.float32)
        )
        # d/dP_i = E * f_i, so the busiest expert has the steepest slope and
        # gradient descent lowers its router probability the most.
        self.assertEqual(int(jnp.argmax(grad)), int(jnp.argmax(load)))


class ActiveParameterTests(unittest.TestCase):
    def _config(self, tier: str):
        # platform="tpu" only gets past the guard on tpu_flash; nothing here
        # executes, and parameter counts do not depend on the backend.
        parser = trainer.build_parser()
        return trainer.resolve_config(
            parser.parse_args(["--tier", tier, "--profile", "dev"]), "tpu", 50_304
        )

    def test_active_count_exceeds_the_dense_tier_only_by_the_router(self) -> None:
        """A routed tier is sized by its *active* parameters, not its total.

        The first TPU smoke test failed because the check compared the declared
        count against the total, which a routed model necessarily exceeds. The
        declared number stays the dense tier size so the sparse and dense
        ladders line up; routing then adds exactly two things, the router
        projection and one extra set of expert biases per additional expert a
        token visits. Both are named rather than absorbed into a tolerance, so
        an unaccounted parameter is a failure and not a rounding difference.
        """

        for tier in ("60m", "125m", "250m", "500m"):
            with self.subTest(tier=tier):
                config = self._config(tier)
                declared = config.declared_parameters
                excess = trainer.expected_active_parameters(config) - declared
                self.assertEqual(
                    excess,
                    config.layers
                    * (
                        config.d_model * config.experts
                        + (config.expert_top_k - 1) * config.d_model
                    ),
                )
                # Small enough that the sparse tier is still the tier it claims
                # to be rather than a quietly larger model.
                self.assertLess(excess / declared, 0.001)

    def test_the_counter_agrees_with_the_closed_form(self) -> None:
        # Only the smallest tier is materialized: 500m totals over a billion
        # parameters, which is minutes and gigabytes for no extra coverage.
        config = self._config("60m")
        params = trainer.init_params(config, 1337)
        self.assertEqual(
            trainer.active_parameter_count(params, config),
            trainer.expected_active_parameters(config),
        )
        # Routing is only worth its complexity if total exceeds active.
        self.assertGreater(
            trainer.parameter_count(params),
            trainer.active_parameter_count(params, config),
        )


class RoutedModelTests(unittest.TestCase):
    def _config(self):
        parser = trainer.build_parser()
        config = trainer.resolve_config(
            parser.parse_args(["--profile", "smoke"]), "cpu", 256
        )
        from dataclasses import replace

        return replace(config, layers=2, d_model=128, heads=2, seq_len=64)

    def test_gradients_reach_the_router_and_every_expert(self) -> None:
        """A router that receives no gradient looks exactly like a working one.

        Both produce finite losses that go down, because the experts keep
        learning either way. Only the gradient tells them apart.
        """

        config = self._config()
        params = trainer.init_params(config, 1337)
        tokens = jnp.asarray(
            np.random.default_rng(4).integers(
                0, config.semantic_vocab_size, size=(2, config.seq_len + 1)
            )
        )
        grads = jax.grad(
            lambda p: trainer.cross_entropy(p, tokens[:, :-1], tokens[:, 1:], config)
        )(params)

        for index, block in enumerate(grads["blocks"]):
            with self.subTest(layer=index):
                for name in (
                    "router_w",
                    "expert_up_w",
                    "expert_down_w",
                    "expert_up_b",
                    "expert_down_b",
                ):
                    magnitude = float(jnp.abs(block[name]).max())
                    self.assertTrue(np.isfinite(magnitude), f"{name} not finite")
                    self.assertGreater(magnitude, 0.0, f"{name} gets no gradient")
                # Per-expert, not just in aggregate: one dead expert would
                # otherwise hide inside a healthy stack-wide maximum.
                per_expert = jnp.abs(block["expert_up_w"]).max(axis=(1, 2))
                self.assertTrue(
                    bool((per_expert > 0).all()), f"dead experts: {per_expert}"
                )
        self.assertGreater(float(jnp.abs(grads["token_embedding"]).max()), 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
