from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from pathlib import Path

import rig.kernels.tpu_flash_attention as tpu_flash_attention

import jax
import jax.numpy as jnp
import numpy as np

from rig.kernels.autotune import AttentionTilePlan
from rig.kernels.tpu_flash_attention import (
    AttentionConfig,
    attention_tile_candidates,
    make_causal_attention,
    reference_causal_attention,
    select_attention_tiles,
)


class TpuFlashAttentionTests(unittest.TestCase):
    @staticmethod
    def random_qkv(
        shape: tuple[int, int, int, int], dtype: jnp.dtype = jnp.float32
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        keys = jax.random.split(jax.random.key(123), 3)
        return tuple(jax.random.normal(key, shape, dtype=dtype) * 0.15 for key in keys)  # type: ignore[return-value]

    def test_reference_has_causal_prefix_invariance(self) -> None:
        q, k, v = self.random_qkv((1, 2, 16, 8))
        changed_v = v.at[:, :, 9:, :].set(10_000.0)
        original = reference_causal_attention(q, k, v)
        changed = reference_causal_attention(q, k, changed_v)
        np.testing.assert_allclose(original[:, :, :9], changed[:, :, :9])

    def test_reference_is_differentiable(self) -> None:
        q, k, v = self.random_qkv((1, 2, 16, 8))

        def loss(*values: jax.Array) -> jax.Array:
            return jnp.square(reference_causal_attention(*values)).mean()

        grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
        self.assertEqual(tuple(value.shape for value in grads), (q.shape,) * 3)
        self.assertTrue(all(np.isfinite(value).all() for value in grads))

    def test_pallas_interpret_matches_reference(self) -> None:
        # Pallas TPU tiles are 128-aligned even in CPU interpret mode.  Use two
        # query/KV tiles so this exercises online-softmax rescaling and causal
        # tile skipping rather than only the degenerate single-tile path.
        q, k, v = self.random_qkv((1, 1, 256, 8))
        tiles = AttentionTilePlan(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_q_dkv_compute=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            block_q_dq=128,
            block_kv_dq=128,
            block_kv_dq_compute=128,
        )
        actual = make_causal_attention(
            AttentionConfig(backend="tpu_flash", tiles=tiles, interpret=True)
        )(q, k, v)
        expected = reference_causal_attention(q, k, v)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_tpu_flash_backward_matches_reference(self) -> None:
        q, k, v = self.random_qkv((1, 1, 128, 8))
        tiles = AttentionTilePlan(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_q_dkv_compute=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            block_q_dq=128,
            block_kv_dq=128,
            block_kv_dq_compute=128,
        )
        attention = make_causal_attention(
            AttentionConfig(backend="tpu_flash", tiles=tiles, interpret=True)
        )
        cotangent = jax.random.normal(jax.random.key(91), q.shape)
        _, actual_pullback = jax.vjp(attention, q, k, v)
        actual = actual_pullback(cotangent)
        _, expected_pullback = jax.vjp(reference_causal_attention, q, k, v)
        expected = expected_pullback(cotangent)
        for actual_grad, expected_grad in zip(actual, expected, strict=True):
            np.testing.assert_allclose(actual_grad, expected_grad, rtol=3e-5, atol=3e-5)

    @staticmethod
    def _segments(sequence: int, cuts: tuple[int, ...]) -> jax.Array:
        """Documents of uneven length, as EOT positions would produce."""

        starts = np.zeros(sequence, np.int32)
        starts[list(cuts)] = 1
        return jnp.asarray(np.cumsum(starts)[None, :] - 1, jnp.int32)

    def test_document_masking_makes_a_document_independent_of_its_neighbours(
        self,
    ) -> None:
        """The defining property: block-diagonal means no cross-document leak.

        A document scored inside a packed window must give exactly what it
        gives alone. Without this, an 8k context mostly trains the model to
        attend across unrelated documents, since a random FineWeb window of
        that length spans about twelve of them.
        """

        q, k, v = self.random_qkv((1, 2, 256, 8))
        segments = self._segments(256, (0, 61, 130, 131, 200))
        packed = reference_causal_attention(q, k, v, segment_ids=segments)
        alone = reference_causal_attention(
            q[:, :, 61:130], k[:, :, 61:130], v[:, :, 61:130]
        )
        np.testing.assert_allclose(packed[:, :, 61:130], alone, rtol=1e-6, atol=1e-6)

        # And it is a real restriction, not a no-op.
        self.assertFalse(
            np.allclose(packed, reference_causal_attention(q, k, v), atol=1e-4)
        )
        # One segment everywhere must reproduce plain causal attention.
        single = reference_causal_attention(
            q, k, v, segment_ids=jnp.zeros((1, 256), jnp.int32)
        )
        np.testing.assert_allclose(
            single, reference_causal_attention(q, k, v), rtol=1e-6, atol=1e-6
        )

    def test_segmented_kernel_matches_the_reference_forward_and_backward(self) -> None:
        q, k, v = self.random_qkv((1, 2, 256, 8))
        segments = self._segments(256, (0, 61, 130, 131, 200))
        tiles = AttentionTilePlan(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_q_dkv_compute=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            block_q_dq=128,
            block_kv_dq=128,
            block_kv_dq_compute=128,
        )
        attention = make_causal_attention(
            AttentionConfig(backend="tpu_flash", tiles=tiles, interpret=True)
        )
        cotangent = jax.random.normal(jax.random.key(7), q.shape, q.dtype) * 0.1

        def kernel_loss(q_value, k_value, v_value):
            return (attention(q_value, k_value, v_value, segments) * cotangent).sum()

        def oracle_loss(q_value, k_value, v_value):
            out = reference_causal_attention(
                q_value, k_value, v_value, segment_ids=segments
            )
            return (out * cotangent).sum()

        np.testing.assert_allclose(
            attention(q, k, v, segments),
            reference_causal_attention(q, k, v, segment_ids=segments),
            rtol=2e-5,
            atol=2e-5,
        )
        for name, actual, expected in zip(
            ("dq", "dk", "dv"),
            jax.grad(kernel_loss, argnums=(0, 1, 2))(q, k, v),
            jax.grad(oracle_loss, argnums=(0, 1, 2))(q, k, v),
        ):
            with self.subTest(gradient=name):
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)

    def test_segments_are_an_argument_not_a_closure_capture(self) -> None:
        """A traced array closed over by a custom_vjp is hoisted as an operand.

        The first version captured the segment index from the enclosing scope.
        Each of a model's attention calls then compiled to one extra implicit
        input, so a twelve-layer model produced an executable wanting twelve
        more arrays than the caller passes -- "compiled for 385 inputs but
        called with 373", visible only on a real multi-layer run. Compiling a
        two-layer stack and counting the executable's inputs catches it here.
        """

        tiles = AttentionTilePlan(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_q_dkv_compute=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            block_q_dq=128,
            block_kv_dq=128,
            block_kv_dq_compute=128,
        )
        attention = make_causal_attention(
            AttentionConfig(backend="tpu_flash", tiles=tiles, interpret=True)
        )
        q, k, v = self.random_qkv((1, 1, 256, 8))
        segments = self._segments(256, (0, 61, 130))

        def two_layers(q_value, k_value, v_value, segment_value):
            first = attention(q_value, k_value, v_value, segment_value)
            return attention(first, k_value, v_value, segment_value).sum()

        compiled = jax.jit(two_layers).lower(q, k, v, segments).compile()
        # Four arrays in, four expected: nothing was hoisted per call site.
        self.assertEqual(
            len(jax.tree_util.tree_leaves((q, k, v, segments))),
            4,
        )
        self.assertIsInstance(float(compiled(q, k, v, segments)), float)

    def test_omitting_segments_leaves_the_kernel_bit_identical(self) -> None:
        # Document masking must cost nothing when it is not requested: the
        # segment operands are absent from the pallas_call entirely.
        q, k, v = self.random_qkv((1, 1, 256, 8))
        tiles = AttentionTilePlan(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_q_dkv_compute=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            block_q_dq=128,
            block_kv_dq=128,
            block_kv_dq_compute=128,
        )
        attention = make_causal_attention(
            AttentionConfig(backend="tpu_flash", tiles=tiles, interpret=True)
        )
        np.testing.assert_array_equal(attention(q, k, v), attention(q, k, v, None))
        np.testing.assert_allclose(
            attention(q, k, v),
            reference_causal_attention(q, k, v),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_pallas_interpret_right_padding_matches_reference(self) -> None:
        q, k, v = self.random_qkv((1, 1, 129, 8))
        tiles = AttentionTilePlan(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_q_dkv_compute=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            block_q_dq=128,
            block_kv_dq=128,
            block_kv_dq_compute=128,
        )
        actual = make_causal_attention(
            AttentionConfig(backend="tpu_flash", tiles=tiles, interpret=True)
        )(q, k, v)
        expected = reference_causal_attention(q, k, v)
        self.assertEqual(actual.shape, q.shape)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_tpu_flash_right_padding_backward_matches_reference(self) -> None:
        q, k, v = self.random_qkv((1, 1, 129, 8))
        tiles = AttentionTilePlan(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_q_dkv_compute=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            block_q_dq=128,
            block_kv_dq=128,
            block_kv_dq_compute=128,
        )
        attention = make_causal_attention(
            AttentionConfig(backend="tpu_flash", tiles=tiles, interpret=True)
        )
        cotangent = jax.random.normal(jax.random.key(92), q.shape)
        _, actual_pullback = jax.vjp(attention, q, k, v)
        actual = actual_pullback(cotangent)
        _, expected_pullback = jax.vjp(reference_causal_attention, q, k, v)
        expected = expected_pullback(cotangent)
        for actual_grad, expected_grad in zip(actual, expected, strict=True):
            np.testing.assert_allclose(actual_grad, expected_grad, rtol=4e-5, atol=4e-5)

    def test_reference_right_padding_gradient_oracle(self) -> None:
        # This mirrors the pad -> attention -> slice transform used by the TPU
        # backends, including an explicit invalid-key mask, and verifies that
        # padding is transparent in both directions.
        q, k, v = self.random_qkv((1, 1, 129, 8))

        def padded_loss(*values: jax.Array) -> jax.Array:
            padded = tuple(
                jnp.pad(value, ((0, 0), (0, 0), (0, 127), (0, 0))) for value in values
            )
            scores = jnp.einsum(
                "bhqd,bhkd->bhqk", *padded[:2], preferred_element_type=jnp.float32
            ) * (8**-0.5)
            rows = jnp.arange(256)[:, None]
            columns = jnp.arange(256)[None, :]
            mask = jnp.logical_and(columns <= rows, columns < 129)
            probabilities = jax.nn.softmax(jnp.where(mask, scores, -1.0e30), axis=-1)
            output = jnp.einsum("bhqk,bhkd->bhqd", probabilities, padded[2])
            return jnp.square(output[:, :, :129]).mean()

        def direct_loss(*values: jax.Array) -> jax.Array:
            return jnp.square(reference_causal_attention(*values)).mean()

        expected = jax.value_and_grad(direct_loss, argnums=(0, 1, 2))(q, k, v)
        actual = jax.value_and_grad(padded_loss, argnums=(0, 1, 2))(q, k, v)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-6)
        for actual_grad, expected_grad in zip(actual[1], expected[1], strict=True):
            np.testing.assert_allclose(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)

    def test_selector_prefers_measured_canonical_tiles(self) -> None:
        tiles = select_attention_tiles(sequence=1024, head_dim=64, training=True)
        self.assertEqual(
            (tiles.block_q, tiles.block_kv, tiles.block_kv_compute),
            (512, 512, 256),
        )
        self.assertTrue(tiles.has_backward_tiles)

    def test_candidates_are_aligned_and_include_large_tiles(self) -> None:
        candidates = attention_tile_candidates(
            sequence=1024, head_dim=64, training=True
        )
        triples = {
            (item.block_q, item.block_kv, item.block_kv_compute) for item in candidates
        }
        self.assertIn((512, 512, 256), triples)
        for item in candidates:
            self.assertEqual(item.block_q % 128, 0)
            self.assertEqual(item.block_kv % 128, 0)
            self.assertEqual(item.block_kv_compute % 128, 0)
            self.assertEqual(item.block_kv % item.block_kv_compute, 0)

    def test_each_explicit_illegal_tile_field_fails_before_pallas_lowering(
        self,
    ) -> None:
        q, k, v = self.random_qkv((1, 1, 256, 8))
        valid = AttentionTilePlan(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_q_dkv_compute=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            block_q_dq=128,
            block_kv_dq=128,
            block_kv_dq_compute=128,
        )
        for field in valid.to_dict():
            values = valid.to_dict()
            values[field] = 384
            illegal = AttentionTilePlan.from_dict(values)
            attention = make_causal_attention(
                AttentionConfig(backend="tpu_flash", tiles=illegal, interpret=True)
            )
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "tile plan is not legal"),
            ):
                attention(q, k, v)

    def test_compiled_custom_backend_rejects_unvalidated_float32(self) -> None:
        q, k, v = self.random_qkv((1, 1, 128, 8), jnp.float32)
        attention = make_causal_attention(AttentionConfig(backend="tpu_flash"))
        with self.assertRaisesRegex(TypeError, "currently supports bfloat16"):
            attention(q, k, v)

    def test_auto_is_trainable_on_cpu(self) -> None:
        q, k, v = self.random_qkv((1, 1, 16, 8))
        attention = make_causal_attention(AttentionConfig(backend="auto"))
        output, gradients = jax.value_and_grad(
            lambda query: jnp.square(attention(query, k, v)).mean()
        )(q)
        self.assertEqual(output.shape, ())
        self.assertEqual(gradients.shape, q.shape)

    def test_validation_rejects_mismatched_shapes(self) -> None:
        q, k, v = self.random_qkv((1, 1, 16, 8))
        with self.assertRaisesRegex(ValueError, "identical q/k/v"):
            reference_causal_attention(q, k[:, :, :-1], v)


class SegmentBlockSpecTests(unittest.TestCase):
    """Document masking must survive more than one sequence per chip.

    The segment array is the kernel's only two-dimensional operand, so its
    batch axis is one of the two dimensions Pallas constrains, while q/k/v are
    four-dimensional and theirs is not. Blocking that axis at 1 was legal only
    while the array's own batch was also 1 -- one sequence per chip -- which is
    what a 16-chip slice at batch 16 gives, and is every 8k run recorded before
    this. On eight chips the same batch puts two sequences on each and lowering
    was refused outright.

    A batch of one therefore proves nothing here, which is why this asks for
    more than one.
    """

    def test_the_segment_array_is_not_two_dimensional_at_the_kernel(self) -> None:
        """The singleton is what keeps batch out of the constrained window."""

        source = Path(tpu_flash_attention.__file__).read_text(encoding="utf-8")
        # Both call sites add the axis on the way in ...
        self.assertEqual(source.count("segment_ids[:, None, :]"), 4)
        # ... and every spec squeezes batch and the singleton back off.
        for spec in (
            "q_segment_spec",
            "kv_segment_spec",
            "dq_q_segment_spec",
            "dq_kv_segment_spec",
            "dkv_q_segment_spec",
            "dkv_kv_segment_spec",
        ):
            with self.subTest(spec=spec):
                self.assertIn(f"{spec} = pl.BlockSpec((None, None, ", source)

    def test_masking_matches_the_oracle_with_several_sequences(self) -> None:
        """Each sequence must be masked by its own document layout.

        Runs on CPU through the reference path, so it checks the masking
        semantics rather than the Pallas lowering; the lowering is what the
        block spec test above pins. Sequences are given deliberately different
        layouts, so masking every row by the first sequence's boundaries would
        be visible rather than plausible.
        """

        rng = np.random.default_rng(0)
        batch, heads, sequence, head_dim = 3, 2, 32, 8
        shape = (batch, heads, sequence, head_dim)
        q, k, v = (jnp.asarray(rng.normal(size=shape), jnp.float32) for _ in range(3))
        segments = jnp.asarray(
            np.stack(
                [np.cumsum(rng.random(sequence) < rate) for rate in (0.1, 0.3, 0.5)]
            ),
            jnp.int32,
        )
        out = reference_causal_attention(
            q, k, v, softmax_scale=1.0, segment_ids=segments
        )
        rolled = reference_causal_attention(
            q, k, v, softmax_scale=1.0, segment_ids=jnp.roll(segments, 1, axis=0)
        )
        # Rolling the layouts between sequences must change the answer; if it
        # does not, the mask is not per-sequence.
        self.assertGreater(float(jnp.abs(out - rolled).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
