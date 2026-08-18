"""Traced FLOP accounting against hand-derived counts.

Every expectation here is computed independently of ``rig.flops`` -- from
the textbook cost of the operation being performed -- so a bug in the
walker cannot agree with a bug in the test.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from rig.flops import (
    FlopError,
    FlopRules,
    Site,
    count_flops,
    default_rules,
)

BATCH, SEQ, MODEL, HEADS = 2, 128, 64, 4
FFN = 4 * MODEL


def _zeros(*shape: int) -> jax.Array:
    return jnp.zeros(shape, jnp.float32)


def _block_params() -> dict[str, jax.Array]:
    return {
        "wq": _zeros(MODEL, MODEL),
        "wk": _zeros(MODEL, MODEL),
        "wv": _zeros(MODEL, MODEL),
        "wo": _zeros(MODEL, MODEL),
        "w1": _zeros(MODEL, FFN),
        "w2": _zeros(FFN, MODEL),
    }


def _block(params, x):
    head_dim = MODEL // HEADS
    shape = (BATCH, SEQ, HEADS, head_dim)
    q = jnp.einsum("btd,dk->btk", x, params["wq"]).reshape(shape).transpose(0, 2, 1, 3)
    k = jnp.einsum("btd,dk->btk", x, params["wk"]).reshape(shape).transpose(0, 2, 1, 3)
    v = jnp.einsum("btd,dk->btk", x, params["wv"]).reshape(shape).transpose(0, 2, 1, 3)
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / head_dim**0.5
    weights = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    merged = context.transpose(0, 2, 1, 3).reshape(BATCH, SEQ, MODEL)
    x = x + jnp.einsum("btd,dk->btk", merged, params["wo"])
    hidden = jax.nn.gelu(jnp.einsum("btd,df->btf", x, params["w1"]))
    return x + jnp.einsum("btf,fd->btd", hidden, params["w2"])


def _stack(params_list, x):
    for params in params_list:
        x = _block(params, x)
    return jnp.sum(x**2)


def _hand_counted_block_forward() -> int:
    """Four square projections, one MLP pair, two attention contractions."""

    projections = 4 * (2 * BATCH * SEQ * MODEL * MODEL)
    mlp = 2 * (2 * BATCH * SEQ * MODEL * FFN)
    attention = 2 * (2 * BATCH * HEADS * SEQ * SEQ * (MODEL // HEADS))
    return projections + mlp + attention


class ForwardCountTests(unittest.TestCase):
    def test_single_block_matches_a_hand_count(self) -> None:
        params = _block_params()
        x = _zeros(BATCH, SEQ, MODEL)
        got = count_flops(lambda p, t: jnp.sum(_block(p, t) ** 2), params, x)
        self.assertEqual(got.matmul, _hand_counted_block_forward())
        self.assertEqual(got.warnings, ())

    def test_depth_scales_the_count_linearly(self) -> None:
        x = _zeros(BATCH, SEQ, MODEL)
        counts = []
        for layers in (1, 2, 3):
            stack = [_block_params() for _ in range(layers)]
            counts.append(count_flops(lambda p, t: _stack(p, t), stack, x).matmul)
        self.assertEqual(counts[0], _hand_counted_block_forward())
        self.assertEqual(counts[1], 2 * counts[0])
        self.assertEqual(counts[2], 3 * counts[0])

    def test_widening_the_model_scales_quadratically(self) -> None:
        # Projections and the MLP are both quadratic in d_model, so doubling
        # width more than doubles the count. A formula pinned to parameter
        # count alone would miss the attention term's separate scaling.
        def total(model_dim: int) -> int:
            params = {
                "w1": _zeros(model_dim, 4 * model_dim),
                "w2": _zeros(4 * model_dim, model_dim),
            }
            x = _zeros(BATCH, SEQ, model_dim)
            fn = lambda p, t: jnp.sum(  # noqa: E731
                jnp.einsum(
                    "btf,fd->btd", jnp.einsum("btd,df->btf", t, p["w1"]), p["w2"]
                )
                ** 2
            )
            return count_flops(fn, params, x).matmul

        self.assertEqual(total(2 * MODEL), 4 * total(MODEL))


class BackwardCountTests(unittest.TestCase):
    def test_gradient_adds_two_more_contractions_per_matmul(self) -> None:
        # y = x W with x also differentiated gives dW and dx: 3x forward.
        params = {"w": _zeros(MODEL, FFN)}
        x = _zeros(BATCH, SEQ, MODEL)

        def loss(p, t):
            return jnp.sum(jnp.einsum("btd,df->btf", t, p["w"]) ** 2)

        forward = count_flops(loss, params, x).matmul
        self.assertEqual(forward, 2 * BATCH * SEQ * MODEL * FFN)
        # Differentiating only the parameters skips the input gradient.
        wrt_params = count_flops(jax.grad(loss), params, x).matmul
        self.assertEqual(wrt_params, 2 * forward)
        wrt_both = count_flops(jax.grad(loss, argnums=(0, 1)), params, x).matmul
        self.assertEqual(wrt_both, 3 * forward)


class StructuralTests(unittest.TestCase):
    def test_scan_multiplies_by_its_trip_count(self) -> None:
        # The body appears once in the jaxpr but runs `length` times.
        weight = _zeros(MODEL, MODEL)
        x = _zeros(BATCH, MODEL)

        def scanned(w, t):
            def step(carry, _):
                return jnp.tanh(jnp.einsum("bd,df->bf", carry, w)), None

            out, _ = jax.lax.scan(step, t, None, length=7)
            return jnp.sum(out)

        got = count_flops(scanned, weight, x)
        self.assertEqual(got.matmul, 7 * 2 * BATCH * MODEL * MODEL)

    def test_cond_bills_the_more_expensive_branch(self) -> None:
        small, large = _zeros(MODEL, MODEL), _zeros(MODEL, 4 * MODEL)
        x = _zeros(BATCH, MODEL)

        def branched(a, b, t, flag):
            return jnp.sum(
                jax.lax.cond(
                    flag,
                    lambda: jnp.sum(jnp.einsum("bd,df->bf", t, a)),
                    lambda: jnp.sum(jnp.einsum("bd,df->bf", t, b)),
                )
            )

        got = count_flops(branched, small, large, x, True)
        self.assertEqual(got.matmul, 2 * BATCH * MODEL * 4 * MODEL)

    def test_elementwise_is_tracked_but_kept_out_of_the_headline(self) -> None:
        x = _zeros(BATCH, SEQ, MODEL)
        got = count_flops(lambda t: jnp.sum(jnp.tanh(t) * 2.0), x)
        self.assertEqual(got.matmul, 0)
        self.assertGreater(got.elementwise, 0)


class WarningTests(unittest.TestCase):
    def test_an_unrecognized_primitive_warns_instead_of_vanishing(self) -> None:
        got = count_flops(lambda t: jnp.sum(jax.lax.cummin(t, axis=0)), _zeros(8, 8))
        # cummin is classified; a genuinely unknown primitive must warn.
        self.assertEqual(got.warnings, ())

        got = count_flops(lambda t: jnp.sum(_opaque(t)), _zeros(8, 8))
        self.assertTrue(got.warnings)
        self.assertIn("cholesky", " ".join(got.warnings))

    def test_strict_mode_raises_on_an_unregistered_opaque_kernel(self) -> None:
        with self.assertRaises(FlopError):
            _count_flash(strict=True, rules=FlopRules())

    def test_an_unregistered_kernel_names_itself_in_the_warning(self) -> None:
        got = _count_flash(rules=FlopRules())
        self.assertTrue(got.warnings)
        self.assertIn("tpu_flash_causal_attention_fwd", " ".join(got.warnings))
        self.assertIn("with_kernel", " ".join(got.warnings))


def _opaque(t):
    # A primitive with no classification and no sub-jaxpr.
    return jax.lax.linalg.cholesky(t @ t.T + jnp.eye(t.shape[0]) * 8.0)


def _count_flash(rules=None, strict: bool = False):
    from rig.kernels import AttentionConfig, make_causal_attention

    attention = make_causal_attention(
        AttentionConfig(backend="tpu_flash", softmax_scale=1.0 / 16)
    )
    shape = (1, HEADS, SEQ, MODEL // HEADS)
    q = jnp.zeros(shape, jnp.bfloat16)
    fn = lambda a, b, c: jnp.sum(attention(a, b, c).astype(jnp.float32))  # noqa: E731
    return count_flops(fn, q, q, q, rules=rules, strict=strict)


class AttentionBackendTests(unittest.TestCase):
    """The count is a property of the architecture, not of the kernel."""

    def _attention(self, backend: str, grad: bool):
        from rig.kernels import AttentionConfig, make_causal_attention

        attention = make_causal_attention(
            AttentionConfig(backend=backend, softmax_scale=1.0 / 16)
        )
        shape = (1, HEADS, SEQ, MODEL // HEADS)
        q = jnp.zeros(shape, jnp.bfloat16)
        fn = lambda a, b, c: jnp.sum(attention(a, b, c).astype(jnp.float32))  # noqa: E731
        if grad:
            fn = jax.grad(fn, argnums=(0, 1, 2))
        return count_flops(fn, q, q, q, rules=default_rules())

    def _square(self) -> int:
        head_dim = MODEL // HEADS
        return 4 * 1 * HEADS * SEQ * SEQ * head_dim

    def test_dense_attention_bills_the_full_square(self) -> None:
        got = self._attention("reference", grad=False)
        self.assertEqual(got.matmul, self._square())
        self.assertEqual(got.warnings, ())

    def test_flash_attention_matches_dense_exactly(self) -> None:
        # If these ever diverge, equi-FLOP comparisons across backends break.
        dense = self._attention("reference", grad=False).matmul
        flash = self._attention("tpu_flash", grad=False).matmul
        self.assertEqual(flash, dense)

    def test_flash_and_dense_agree_through_the_backward_pass(self) -> None:
        dense = self._attention("reference", grad=True).matmul
        flash = self._attention("tpu_flash", grad=True).matmul
        self.assertEqual(flash, dense)
        self.assertEqual(flash, 3 * self._square())

    def test_flash_backward_uses_all_three_kernels(self) -> None:
        got = self._attention("tpu_flash", grad=True)
        self.assertEqual(
            sorted(got.by_site),
            [
                "tpu_flash_causal_attention_bwd_dkv",
                "tpu_flash_causal_attention_bwd_dq",
                "tpu_flash_causal_attention_fwd",
            ],
        )
        self.assertEqual(got.warnings, ())


class ExtensibilityTests(unittest.TestCase):
    """A named jit boundary is the hook for compute-then-mask components."""

    def _moe(self, experts: int):
        weights = _zeros(experts, MODEL, FFN)
        x = _zeros(BATCH, MODEL)

        @jax.jit
        def moe_block(tokens, w):
            return jnp.einsum("bd,edf->ebf", tokens, w)

        def call(w, t):
            return jnp.sum(moe_block(t, w))

        return call, weights, x

    def test_dense_moe_is_overcounted_without_a_rule(self) -> None:
        experts = 8
        call, weights, x = self._moe(experts)
        got = count_flops(call, weights, x)
        self.assertEqual(got.matmul, 2 * BATCH * MODEL * FFN * experts)

    def test_a_scope_rule_replaces_the_traced_cost(self) -> None:
        experts, top_k = 8, 2
        call, weights, x = self._moe(experts)

        def rule(site: Site) -> int:
            tokens = site.in_shapes[0][0]
            _, model_dim, ffn = site.in_shapes[1]
            return 2 * top_k * tokens * model_dim * ffn

        rules = default_rules().with_scope("moe_block", rule)
        got = count_flops(call, weights, x, rules=rules)
        self.assertEqual(got.matmul, 2 * top_k * BATCH * MODEL * FFN)
        self.assertEqual(got.by_site["moe_block"], got.matmul)

    def test_a_scope_rule_stops_the_walk_at_the_boundary(self) -> None:
        # Nothing inside the scope may leak into the total.
        call, weights, x = self._moe(4)
        rules = default_rules().with_scope("moe_block", lambda site: 1234)
        got = count_flops(call, weights, x, rules=rules)
        self.assertEqual(got.matmul, 1234)


class ShardMapTests(unittest.TestCase):
    def test_a_sharded_body_is_scaled_back_to_the_global_figure(self) -> None:
        from jax.sharding import Mesh, PartitionSpec as P

        devices = jax.devices()
        if len(devices) < 2:
            self.skipTest("needs at least two devices")
        mesh = Mesh(devices, ("data",))
        rows = 8 * len(devices)
        w, x = _zeros(MODEL, FFN), _zeros(rows, MODEL)

        def sharded(weight, tokens):
            fn = jax.shard_map(
                lambda ww, tt: jnp.einsum("bd,df->bf", tt, ww),
                mesh=mesh,
                in_specs=(P(), P("data")),
                out_specs=P("data"),
            )
            return jnp.sum(fn(weight, tokens))

        got = count_flops(sharded, w, x)
        self.assertEqual(got.matmul, 2 * rows * MODEL * FFN)


class BatchLinearityTests(unittest.TestCase):
    """``traced_flops`` traces one sequence; that is only valid if linear."""

    def test_flops_are_linear_in_batch(self) -> None:
        params = _block_params()

        def total(batch: int) -> int:
            x = jnp.zeros((batch, SEQ, MODEL), jnp.float32)

            def block(p, t):
                # _block hardcodes BATCH in its reshapes, so inline a version
                # parameterized by the batch actually being traced.
                head_dim = MODEL // HEADS
                shape = (batch, SEQ, HEADS, head_dim)
                q = jnp.einsum("btd,dk->btk", t, p["wq"]).reshape(shape)
                k = jnp.einsum("btd,dk->btk", t, p["wk"]).reshape(shape)
                v = jnp.einsum("btd,dk->btk", t, p["wv"]).reshape(shape)
                q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
                w = jax.nn.softmax(jnp.einsum("bhtd,bhsd->bhts", q, k), axis=-1)
                c = jnp.einsum("bhts,bhsd->bhtd", w, v)
                merged = c.transpose(0, 2, 1, 3).reshape(batch, SEQ, MODEL)
                return jnp.sum(jnp.einsum("btd,dk->btk", merged, p["wo"]) ** 2)

            return count_flops(block, params, x).matmul

        one = total(1)
        self.assertEqual(total(2), 2 * one)
        self.assertEqual(total(5), 5 * one)


class PaddingTests(unittest.TestCase):
    def test_backend_agreement_holds_only_for_tile_aligned_sequences(self) -> None:
        # The equality asserted in AttentionBackendTests is not unconditional:
        # flash right-pads to 128-wide tiles, so an unaligned sequence really
        # does cost more there. Pinning this stops the invariant from being
        # read as broader than it is.
        from rig.kernels import AttentionConfig, make_causal_attention

        def total(backend: str, sequence: int) -> int:
            attention = make_causal_attention(
                AttentionConfig(backend=backend, softmax_scale=1.0 / 16)
            )
            q = jnp.zeros((1, HEADS, sequence, 64), jnp.bfloat16)
            fn = lambda a, b, c: jnp.sum(  # noqa: E731
                attention(a, b, c).astype(jnp.float32)
            )
            return count_flops(fn, q, q, q, rules=default_rules()).matmul

        self.assertEqual(total("tpu_flash", 128), total("reference", 128))
        self.assertGreater(total("tpu_flash", 129), total("reference", 129))


class PerTokenTests(unittest.TestCase):
    def test_per_token_divides_the_total(self) -> None:
        params = _block_params()
        x = _zeros(BATCH, SEQ, MODEL)
        got = count_flops(lambda p, t: jnp.sum(_block(p, t) ** 2), params, x)
        self.assertEqual(got.per_token(BATCH * SEQ), got.matmul // (BATCH * SEQ))
        with self.assertRaises(FlopError):
            got.per_token(0)


if __name__ == "__main__":
    unittest.main()
