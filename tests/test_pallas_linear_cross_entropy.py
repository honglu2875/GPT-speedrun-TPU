from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rig.kernels.pallas_linear_cross_entropy import (
    pallas_tied_cross_entropy_backward,
    pallas_tied_cross_entropy_forward,
)


class PallasLinearCrossEntropyTests(unittest.TestCase):
    def test_interpret_forward_and_explicit_backward_match_dense(self) -> None:
        key = jax.random.key(19)
        hidden = jax.random.normal(key, (128, 128), dtype=jnp.float32)
        embedding = jax.random.normal(
            jax.random.fold_in(key, 1), (256, 128), dtype=jnp.float32
        )
        targets = jnp.arange(128, dtype=jnp.int32) % 250
        cotangent = jnp.linspace(0.1, 1.0, 128, dtype=jnp.float32)

        losses, log_normalizer = pallas_tied_cross_entropy_forward(
            hidden,
            embedding,
            targets,
            semantic_vocab_size=250,
            token_tile_size=128,
            vocab_major_size=128,
            vocab_compute_size=128,
            compute_dtype=jnp.float32,
            interpret=True,
        )
        grad_hidden, grad_embedding = pallas_tied_cross_entropy_backward(
            hidden,
            embedding,
            targets,
            log_normalizer,
            cotangent,
            semantic_vocab_size=250,
            token_tile_size=128,
            vocab_major_size=128,
            vocab_compute_size=128,
            weight_tile_size=128,
            token_major_size=128,
            token_compute_size=128,
            compute_dtype=jnp.float32,
            interpret=True,
        )

        def dense_loss(h: jax.Array, e: jax.Array) -> jax.Array:
            logits = h @ e[:250].T
            per_token = -jax.nn.log_softmax(logits)[jnp.arange(128), targets]
            return jnp.sum(per_token * cotangent)

        dense_losses = -jax.nn.log_softmax(hidden @ embedding[:250].T)[
            jnp.arange(128), targets
        ]
        expected_hidden, expected_embedding = jax.grad(
            dense_loss, argnums=(0, 1)
        )(hidden, embedding)
        np.testing.assert_allclose(losses, dense_losses, rtol=2e-5, atol=3e-5)
        np.testing.assert_allclose(
            grad_hidden, expected_hidden, rtol=3e-5, atol=8e-6
        )
        np.testing.assert_allclose(
            grad_embedding, expected_embedding, rtol=3e-5, atol=8e-6
        )
        np.testing.assert_array_equal(grad_embedding[250:], 0.0)

    def test_tile_contracts_are_checked_before_lowering(self) -> None:
        hidden = jnp.zeros((128, 128), jnp.float32)
        embedding = jnp.zeros((256, 128), jnp.float32)
        targets = jnp.zeros((128,), jnp.int32)
        with self.assertRaises(ValueError):
            pallas_tied_cross_entropy_forward(
                hidden,
                embedding,
                targets,
                semantic_vocab_size=250,
                vocab_compute_size=64,
                interpret=True,
            )

    def test_bfloat16_target_dot_and_invalid_target_gradients(self) -> None:
        rng = np.random.default_rng(31)
        hidden = jnp.asarray(
            rng.normal(0.0, 0.2, (128, 128)).astype(np.float32)
        )
        embedding = jnp.asarray(
            rng.normal(0.0, 0.2, (256, 128)).astype(np.float32)
        )
        targets = jnp.arange(128, dtype=jnp.int32) % 250
        targets = targets.at[0].set(-1).at[1].set(999)
        cotangent = jnp.linspace(0.1, 1.0, 128, dtype=jnp.float32)

        losses, normalizer = pallas_tied_cross_entropy_forward(
            hidden,
            embedding,
            targets,
            semantic_vocab_size=250,
            token_tile_size=128,
            vocab_major_size=128,
            vocab_compute_size=128,
            compute_dtype=jnp.bfloat16,
            interpret=True,
        )
        valid = (targets >= 0) & (targets < 250)
        safe_targets = jnp.clip(targets, 0, 249)
        logits = jnp.einsum(
            "nd,vd->nv",
            hidden.astype(jnp.bfloat16),
            embedding[:250].astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )
        expected_losses = -jax.nn.log_softmax(logits)[
            jnp.arange(128), safe_targets
        ]
        np.testing.assert_allclose(
            losses[valid], expected_losses[valid], rtol=3e-4, atol=3e-4
        )
        self.assertTrue(bool(jnp.all(jnp.isinf(losses[~valid]))))

        gradients = pallas_tied_cross_entropy_backward(
            hidden,
            embedding,
            targets,
            normalizer,
            cotangent,
            semantic_vocab_size=250,
            token_tile_size=128,
            vocab_major_size=128,
            vocab_compute_size=128,
            weight_tile_size=128,
            token_major_size=128,
            token_compute_size=128,
            compute_dtype=jnp.bfloat16,
            interpret=True,
        )

        def dense_loss(h: jax.Array, e: jax.Array) -> jax.Array:
            dense_logits = jnp.einsum(
                "nd,vd->nv",
                h.astype(jnp.bfloat16),
                e[:250].astype(jnp.bfloat16),
                preferred_element_type=jnp.float32,
            )
            selected = -jax.nn.log_softmax(dense_logits)[
                jnp.arange(128), safe_targets
            ]
            return jnp.sum(selected * cotangent * valid)

        expected_gradients = jax.grad(dense_loss, argnums=(0, 1))(
            hidden, embedding
        )
        np.testing.assert_allclose(
            gradients[0], expected_gradients[0], rtol=1e-2, atol=3e-3
        )
        np.testing.assert_allclose(
            gradients[1], expected_gradients[1], rtol=1e-2, atol=3e-3
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
