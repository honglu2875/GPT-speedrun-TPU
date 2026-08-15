# FLOP accounting

The training FLOP figure in logs, `metrics.json`, and every equi-FLOP plot is
**traced from the model**, not computed by a formula. `rig.flops` walks the
jaxpr that JAX builds and counts the arithmetic it finds. Change the depth,
width, head count, or the shape of a block and the number follows, because
there is no formula to keep in sync.

## What is being measured

*Algorithmic* compute: the arithmetic the model performs as mathematics, not
the arithmetic the hardware issues. The two differ on purpose.

| | counted as |
|---|---|
| Causal attention | the full `T × T` square, even though the flash kernel skips fully-masked tiles |
| A value recomputed to save memory | once, not twice |
| Elementwise work (GELU, softmax, norms) | tracked separately, excluded from the headline |
| An embedding lookup | free — it is a gather, not arithmetic |

The governing property: **the count is a property of the architecture, not of
the backend.** Running the same model with `dense` or `tpu_flash` attention
must produce the same number, or runs stop being comparable. On the 250m
config both trace to 710,411,520 FLOPs/token.

That equality holds for tile-aligned sequences. Flash right-pads q/k/v to
128-wide tiles, so at `seq_len=129` it genuinely costs more than dense, and
the traced count reflects that. Every tier uses `seq_len=1024`, where the
padding is a no-op.

## How primitives are classified

| bucket | examples | treatment |
|---|---|---|
| counted | `dot_general`, `conv_general_dilated` | computed from shapes |
| structural | `reshape`, `transpose`, `gather`, `stack` | free |
| elementwise | `add`, `exp`, `tanh`, `reduce_sum` | separate total |
| opaque | `pallas_call`, `custom_call` | **needs a rule, else warns** |

Anything unrecognized produces a warning rather than vanishing. A silent
undercount is far worse than a loud one — the warnings appear on the console
at startup and are recorded in `metrics.json` under `flop_accounting`.

Higher-order primitives are entered with the right multiplicity: a `scan`
body is billed `length` times, a `shard_map` body is scaled by the mesh, a
`cond` bills its most expensive branch, and a `while` warns because its trip
count is not knowable statically.

## Adding a component

**Ordinary blocks need nothing.** Anything built from matmuls is counted from
its traced shapes.

### A new opaque kernel

Arithmetic inside a `pallas_call` is invisible to the tracer — XLA's own cost
analysis reports roughly 4,000× too few FLOPs for the flash kernel, and
descending into the kernel body is worse than useless because the body
describes a single tile. Register the cost instead:

```python
def my_kernel_rule(site):
    batch, heads, sequence, head_dim = site.first_rank(4)
    return 4 * batch * heads * sequence * sequence * head_dim

rules = default_rules().with_kernel("my_kernel_fwd", my_kernel_rule)
```

Register the backward kernels too. The three flash kernels each bill
`4·B·H·T²·D`, totalling the textbook `12·B·H·T²·D` for a full step.

### A component whose real cost differs from its traced cost

This is the case no warning can catch, because nothing about the graph looks
unusual. **Sparsity is the usual reason.** A mixture-of-experts written as
"compute every expert, then mask to top-k" contains the full dense work in
its graph; the tracer sees real multiplications and cannot know a mask
discards the results. It will bill all of them.

Wrap the component in a named `jax.jit` and register a scope rule. The walker
applies the rule and does not descend:

```python
@partial(jax.jit, static_argnames=("top_k",))
def moe_block(tokens, expert_weights, *, top_k): ...

def moe_rule(site):
    batch = site.in_shapes[0][0]
    _, model_dim, ffn = site.in_shapes[1]
    return 2 * TOP_K * batch * model_dim * ffn

rules = default_rules().with_scope("moe_block", moe_rule)
```

`jax.named_call` does **not** work for this — it inlines away and leaves no
boundary in the jaxpr. `jax.jit` survives as a `jit` equation carrying its
function name, which is what the scope lookup keys on.

## Checklist for a new component

1. Is it built only from matmuls and elementwise ops? Nothing to do.
2. Does it call a Pallas kernel? Add `with_kernel` rules for the forward
   *and* backward kernels.
3. Does it compute more than it uses (sparsity, masking, early exit)? Wrap it
   in a named `jax.jit` and add a `with_scope` rule.
4. Does it introduce a primitive not in `rig/flops.py`? The warning will name
   it. Classify it as structural or elementwise, or give it a rule.
5. Run `pytest tests/test_flops.py`. Add a case that checks the new component
   against a hand-derived count — every expectation in that file is computed
   independently of the walker, so a bug in one cannot agree with a bug in
   the other.

## Why not XLA's cost analysis

`jit(fn).lower(...).compile().cost_analysis()` returns a `flops` field and is
accurate for pure-XLA graphs — it agreed with the analytic count to 3% on
dense attention. But it cannot see inside a custom call. Measured on a v4
slice, all four hosts agreeing:

| backend | cost_analysis | vs analytic |
|---|---|---|
| `reference` | 6,640,124,928 | 1.031× |
| `tpu_flash` | 1,572,864 | 0.00024× |

It is still useful as an independent cross-check on backends with no opaque
kernels, which is exactly where it agrees.
