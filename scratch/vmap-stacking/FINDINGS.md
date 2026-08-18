# Can stacked models be bit-exact under vmap?

Verification for the model-stacking feature: run N models on one `model_batch`
axis so a small model saturates the chip better. The requirement was that slice
`i` bit-perfectly reproduce a standalone run with seed `i`.

Probes run on the v6e-8, which is single-host, so one process can hold the TPU.

## Answer: no, not reliably, and not for the reason first assumed

Bit-exactness under `vmap` **depends on tensor shape**, and fails at realistic
shapes even with dense attention, dense loss, float32, and no custom kernels
anywhere. `tpu_shape_probe.py`, all float32, params after one step:

| layers | d_model | seq | batch | default precision | highest precision |
|--:|--:|--:|--:|--:|--:|
| 2 | 64 | 32 | 2 | exact | exact |
| 2 | 64 | 64 | 4 | 8.4e-09 | exact |
| 2 | 128 | 64 | 4 | 3.3e-04 | exact |
| 2 | 128 | 256 | 8 | 1.2e-06 | 3.7e-09 |
| 2 | 256 | 256 | 8 | 4.0e-07 | 2.2e-06 |
| 4 | 256 | 512 | 8 | 3.3e-04 | 1.6e-06 |

Two contributors, in order of size:

1. **TPU matmul precision.** TPU runs float32 matmuls as bf16 passes by
   default, and the pass count can differ between batched and unbatched. This
   is the 3.3e-04 entries. `jax.default_matmul_precision("highest")` removes
   them, at a throughput cost that works against the reason for stacking.
2. **Residual XLA codegen**, ~1e-06 in float32, which survives `highest` and
   has no knob.

## What is *not* the cause

Ruled out by isolation, each measured rather than argued:

- **The flash attention kernel.** Raw and inside its `shard_map`, forward and
  all three gradients are bit-exact under `vmap` (`tpu_kernel_probe.py`).
- **The matmul, gradients, moments, or any scalar** in the optimizer. All exact
  under `vmap` when tested alone.
- **bfloat16 as such.** bf16 enlarges the differences, it does not create them;
  float32 diverges at the same shapes.

One real contributor was found and is worth knowing:

- **The tiled cross-entropy with a single vocab tile.** A one-trip `fori_loop`
  is eliminated and the inlined code optimizes differently under batching. With
  two or more tiles it is bit-exact in isolation (`tpu_ce_probe.py`). The real
  config is 50,304 vocab / 2,048 tile = 25 tiles, so it is not in the hot path,
  but a test that sets `vocab_tile_size == semantic_vocab_size` will see it.

## The CPU result is an artifact and cost most of the investigation

CPU diverges on dense backends where TPU does not. XLA:CPU rewrites `a / b`
into `a * (1/b)` when it sees batched shapes and keeps the division unbatched;
compiled HLO shows 20 divides becoming multiplies. Hoisting the reciprocal
makes CPU exact. XLA:TPU performs no such rewrite.

Do not generalise CPU numerics to TPU. Every conclusion drawn from the CPU run
was wrong about the target hardware.

## Recommendation

Drop bit-exactness as an acceptance criterion and keep the feature. The
differences are ULP-scale for the dtype in use, bounded, and do not compound
(3.294e-04 at step 1, 3.433e-04 at step 8, flat after). Slices do not
interfere: two identical models in one `vmap` stay bit-identical to each other,
and stack size does not matter, N=1 differs from unbatched exactly as much as
N=4 does.

For a seed-variance study none of this matters. Each slice is a valid draw from
the seed distribution; it is a different draw than the unstacked run would have
produced, not a wrong one.

A batched cross-entropy kernel with a correct `custom_vjp` for the model axis
would fix contributor (3) above, but not (1) or (2), so it does not buy
bit-exactness on its own.
