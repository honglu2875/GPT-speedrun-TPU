# Mixture of experts: design for `recipes/reference_moe`

Author: Opus 5

A plan, not code. Nothing here is in the tree yet, and it should not be until
the phase that needs it is being built.

## The finding that shapes everything below

**No custom kernels are required.** Both pieces exist in JAX 0.11.0, are
differentiable, and were checked against dense references in this environment:

| piece | where | status |
|---|---|---|
| grouped matmul | `jax.experimental.pallas.ops.tpu.megablox.gmm` | Pallas; `custom_vjp` whose backward calls a Pallas `tgmm` for `dW` and `gmm(transpose_rhs)` for `dX` |
| ragged dispatch | `jax.lax.ragged_all_to_all` | native collective; both a JVP and a transpose rule are registered |

A full grouped MLP takes gradients through `gmm` and matches a per-expert loop
to `4.3e-06`. A complete top-2 routed layer — router, sort, two grouped
matmuls, scatter-combine — matches a naive dense reference on the forward pass
and on all four gradients:

```
forward   8.75e-08      d_x 1.71e-07
d_router  1.07e-06      d_w1 9.54e-07      d_w2 4.17e-07
```

So the work is routing, sharding, parameterization, and accounting — not
kernel authoring. Budget accordingly.

## The ladder

Sparsify **the MLP only**. Attention, embeddings, and norms stay dense, so
active parameters land exactly on the existing dense tiers and every
Complete(d)P multiplier carries over unchanged.

Two ways to hold active parameters fixed while adding `E = 8` experts:

| tier | d | L | dense (active) | top-2, h=2d total | top-1, h=4d total |
|---|--:|--:|--:|--:|--:|
| 60M | 384 | 12 | 59,918,208 | 102,371,328 | 158,994,432 |
| 125M | 640 | 12 | 123,456,640 | 241,397,760 | 398,684,160 |
| 250M | 896 | 16 | 244,444,032 | 552,681,472 | 963,723,264 |
| 500M | 1280 | 19 | 502,602,240 | 1,249,638,400 | 2,245,785,600 |
| 1B | 1792 | 21 | 989,943,808 | 2,608,306,176 | 4,766,275,584 |

**Take top-2 with `h = 2d`.** Each expert is half the dense MLP width, two
fire per token, so active FLOPs are identical to dense — and total parameters
are roughly half the top-1 variant's. Three reasons beyond size:

- **It fits.** Optimizer state is 12 B/param (fp32 param + Adam `m` + `v`).
  Replicated, top-1 at the 1B tier needs 57.2 GB against 32 GB of v4/v6e HBM
  and 16 GB of v5e — it cannot run at all. Top-2 needs 31.3 GB, which is still
  too tight for activations but is reachable with expert parallelism 2.
- **The router gets a comparative gradient.** With top-1 the gate is a single
  softmax entry scaled to 1; with top-2 the relative weighting of two experts
  is differentiable, which is the signal the router actually learns from.
- **`gmm` prefers it.** Top-2 doubles `m` and halves `n`. Larger `m` means more
  full 128-tiles per expert and less tile-boundary waste at the same FLOPs.

Note what `E = 8` does *not* buy: total/active is only **1.7x–2.6x**, not 8x,
because attention and embeddings stay dense. If the goal is a large
total-to-active ratio, the lever is more experts, not a bigger `h`.

## Mesh and when expert parallelism is required

Today the mesh is one axis, `("data",)`, with the model replicated. MoE needs
two:

```python
Mesh(devices.reshape(D, P), ("data", "expert"))       # D * P == device count
```

- Batch shards over **both** axes, so every device does attention work:
  `P(("data", "expert"), None)`.
- Expert weights shard over `expert` only: `P("expert", None, None)` on
  `[E, d, h]`, giving `E_local = E / P`.
- Everything else stays replicated, exactly as now.

`P = 1` is the replicated-expert case: no collectives at all, and the dispatch
code is skipped by a static Python branch rather than degenerating into no-op
all-to-alls. Memory per device at 12 B/param, top-2:

| tier | replicated (P=1) | P=2 | P=4 | P=8 |
|---|--:|--:|--:|--:|
| 60M | 1.2 GB | 0.9 GB | 0.7 GB | 0.6 GB |
| 125M | 2.9 GB | 2.0 GB | 1.5 GB | 1.2 GB |
| 250M | 6.6 GB | 4.2 GB | 2.9 GB | 2.3 GB |
| 500M | 15.0 GB | 9.0 GB | 6.0 GB | 4.5 GB |
| 1B | 31.3 GB | 18.4 GB | 11.9 GB | 8.6 GB |

Non-expert parameters stay replicated, so the sharded columns do not simply
halve. Against 32 GB on v4/v6e and 16 GB on v5e, and leaving room for
activations: **60M and 125M run replicated anywhere; 250M and 500M run
replicated on v4/v6e but want `P ≥ 2` on v5e; 1B needs `P ≥ 2` on v4/v6e and
`P ≥ 4` on v5e.** That is what makes the dispatch path necessary
rather than speculative — and it is also why it can wait for phase 2.

## The routed layer

Validated shape, `P = 1`:

```python
logits = x @ w_router                        # [T, E]
top_w, top_e = jax.lax.top_k(logits, K)      # [T, K]
gate = jax.nn.softmax(top_w, axis=-1)        # over the selected K only

flat_e = top_e.reshape(-1)                   # [T*K]
order  = jnp.argsort(flat_e, stable=True)    # expert-major ordering
counts = jax.nn.one_hot(flat_e, E, dtype=jnp.int32).sum(0)   # [E] group_sizes
rows   = jnp.repeat(jnp.arange(T), K)[order]

hs = megablox.gmm(x[rows], w1, counts)       # [T*K, h]
ys = megablox.gmm(jax.nn.gelu(hs), w2, counts)
out = jnp.zeros_like(x).at[rows].add(ys * gate.reshape(-1)[order][:, None])
```

`.add` and not `.set`: with `K = 2` two assignments land on the same row.

`group_sizes` is a traced array while the total `m = T*K` is static, which is
the point of megablox — **dropless routing with static shapes**. No capacity
factor, no dropped tokens, no wasted padding per expert.

`E` is 8, so a counting sort (`cumsum` over the one-hot already being built for
`counts`) is likely cheaper than the comparison `argsort` above. Worth
measuring; not worth assuming.

## Expert parallelism, `P > 1`

Under `shard_map` over the `expert` axis:

1. Route locally, as above, producing `counts[E]` and expert-major tokens.
2. **All-gather `counts`** across `expert` (an `E`-element int32 vector, free)
   so every device knows the full `[P, E]` traffic matrix and can compute all
   offsets locally with prefix sums — no comparison sort, no extra collective.
3. `ragged_all_to_all(operand=x_sorted, output=recv, input_offsets, send_sizes,
   output_offsets, recv_sizes, axis_name="expert")`.
4. The receive buffer arrives **source-major, expert-minor**: offsets are
   per-peer, so a device gets `P` contiguous chunks, each internally sorted by
   expert. Feeding `gmm` needs expert-major, so apply the block transpose of
   the `P x E_local` count matrix — a gather with indices derived from step 2,
   not a sort.
5. `gmm(..., group_offset=first_local_expert, ...)` — `group_offset` exists
   precisely for a sharded expert dimension.
6. `ragged_all_to_all` back with the send/recv roles swapped, then unsort and
   combine.

Both collectives differentiate natively, so the backward pass needs no
special handling.

### There is no capacity factor, and no token is ever dropped

`ragged_all_to_all` writes into a statically sized buffer, which invites a
capacity factor. **Do not add one.** Size the buffer for the provable worst
case instead — every peer in the expert group sending everything to one
device, `P * T_local * K` rows — and the question disappears.

That worst case is affordable at every configuration the ladder needs:

| tier | P | worst-case rows | buffer, all layers | optimizer state | total |
|---|--:|--:|--:|--:|--:|
| 500M | 1 | 16,384 | — | 15.0 GB | 15.0 GB |
| 500M | 2 | 32,768 | 1.6 GB | 9.0 GB | 10.6 GB |
| 1B | 1 | 16,384 | — | 31.3 GB | 31.3 GB |
| 1B | 2 | 32,768 | 2.5 GB | 18.4 GB | **20.8 GB** |
| 1B | 4 | 65,536 | 4.9 GB | 11.9 GB | 16.8 GB |

(bf16 activations, 32 GB of HBM.) Sizing for the worst case costs 2.5 GB at
the tier that needs sharding at all, against 13.6 GB of headroom — and note
the 1B row where `P = 2` with worst-case buffers beats `P = 1` outright,
because sharding the optimizer state saves more than the buffers cost.

**Why this matters more than the memory.** Dropping breaks causality. Token
choice is per-token independent: token `i` picks its experts from its own
hidden state and nothing else, so the routed model factorizes exactly as the
autoregressive loss assumes. A capacity factor destroys that. Whether token
`i` is served depends on how many *other* tokens chose the same expert — which
in the usual position-ordered implementation means tokens at later positions
in the same sequence, and always means other sequences in the batch. Two
distinct failures follow:

- **Position leakage.** The output at position `i` becomes a function of
  positions `j > i`. The loss still factorizes as if it were not.
- **Batch dependence.** The same token in the same sequence gets a different
  answer depending on what it was batched with, so training and inference
  disagree, and nothing is reproducible across a batch-size change.

Both are silent. Neither raises, neither shows up in the loss curve as
anything but slightly worse numbers, and a capacity factor tuned until
`dropped_fraction` reads zero on average still drops on the tail steps that
matter. The only safe amount of dropping is none.

### Why not enforce balance algorithmically

The tempting fix is a routing rule that balances by construction — Expert
Choice, where each expert takes its top-`c` tokens, or BASE/Sinkhorn, which
solve an assignment problem. All of them make group sizes exactly `T*K/E`,
static, with no buffer question at all.

**They break causality in a subtler place.** If expert `e` ranks the sequence
and keeps its top `c`, whether token `i` is selected depends on the scores of
tokens `j > i`. That is the same leak as dropping, moved from the capacity
check into the routing rule, and it is harder to see. These schemes are sound
for encoders and masked objectives; they are not sound for a decoder-only LM.

Under token choice, **exact balance and causality are incompatible** — any
rule that couples tokens to equalize loads is a rule that lets one token's
assignment depend on another's. So balance is *encouraged* by the auxiliary
loss and never *enforced*, imbalance is measured rather than clipped, and the
dispatch buffer simply absorbs whatever the router produces.

## Constraints found by testing

- **`m` must be divisible by the m-tile (128).** `T_local * K` at bs128,
  seq1024, 16 devices is `8192 * 2 = 16384 = 128 * 128`, so production shapes
  are fine — but this bites immediately at odd test sizes. It is why the
  validation above uses `T = 128`, not `96`.
- `k` and `n` want to be multiples of 128 too. Every tier width already is
  (`384, 640, 896, 1280, 1792`), and `h = 2d` inherits that.
- `group_sizes` must be `int32`; `lhs` is strictly 2-D, so `(batch, seq)`
  flattens before the call.
- `tiling` accepts a lookup function, the same shape of interface as the
  existing flash-attention tile plan in `rig/kernels`. Reuse that machinery for
  tuned tilings rather than inventing a second one.

## Complete(d)P for the routed layer

The experts are ordinary hidden matrices of shape `[d, h]` and `[h, d]` with
`h = 2d`, so **every existing rule applies unchanged** — that is the main
reason to sparsify the MLP only.

The router `[d, E]` is the one new object, and it is not a hidden layer: `E`
does not scale with width. It behaves like a readout, so the starting position
is the unembedding's rules — LR scaling `1/m_N`, `epsilon` unscaled — with
initialization small enough that routing starts near-uniform.

**This is an assumption, not a result.** Whether the base LR `2^-8` optimum
transfers to a routed model is exactly the kind of thing this repo measures
rather than asserts, and it should be re-swept at 60M before the ladder is
trusted. Note the prior is weak in a specific way: the router sees a different
loss surface than any dense tensor, because its gradient arrives through a
discrete selection.

## Load balancing

Switch-style auxiliary loss, added to the training loss with coefficient
`alpha`:

```
aux = E * sum_i (f_i * P_i)
```

where `f_i` is the fraction of assignments routed to expert `i` and `P_i` the
mean router probability for it. Add the router z-loss
(`mean(logsumexp(logits)^2)`) if logits drift. Both need to be **logged
separately from the training loss**, or a run that balances well and models
badly looks identical to the reverse.

## FLOP accounting

`rig/flops.py` bills algorithmic compute, and MoE is the special case its
docstring already anticipates. Two rules are needed:

- `gmm`/`tgmm` appear as opaque `pallas_call`s, like the flash kernels. The
  count is **exact and static** despite dynamic groups, because every token
  reaches exactly `K` experts:

  ```
  routed MLP FLOPs = 2 * (T * K) * d * h   per grouped matmul
                   = 4 * T * K * d * h     for the pair
  ```

  With `K = 2, h = 2d` that is `16 * T * d^2` — identical to the dense
  `4d`-wide MLP's `16 * T * d^2`. **The MoE ladder is equi-FLOP with the dense
  ladder by construction**, which is what makes the two directly comparable on
  the equi-FLOP axis the report already defaults to.
- The router adds `2 * T * d * E`: 0.26% of the routed MLP at 60M, falling
  to 0.06% at 1B. Small, but counted rather than waved away.

The counting must bill `K` experts, never `E`. A rule that billed the whole
weight tensor would inflate every MoE run by `E/K = 4x` and silently destroy
the comparison the ladder exists to make. This deserves a test asserting the
traced count equals `4 * T * K * d * h + 2 * T * d * E` exactly.

Dispatch collectives move bytes and do no arithmetic: zero.

## Registry additions

New ids, appended to `rig/metrics.py` and `rig/registry.txt` (see
[RIGLOG_FORMAT.md](RIGLOG_FORMAT.md) — ids are permanent, so these are worth
getting right once):

| metric | why |
|---|---|
| `router.load_balance_loss` | separate from train loss, per above |
| `router.z_loss` | logit drift |
| `router.entropy` | mean routing entropy; collapse detector |
| `router.max_load_fraction` | worst expert's share of assignments |
| `router.max_recv_fraction` | peak dispatch-buffer occupancy, as a skew diagnostic — never a safety margin, since the buffer is sized for the worst case |

Per-expert diagnostics need a **second index**: statistics are per (layer,
expert) and the column table carries only one `layer` field. The `reserved`
int32 already in each 24-byte entry is exactly the slot for it, and old files
carry 0 there — so this extends cleanly without a format break. Add an
`expert` scope id at the same time.

## Phasing

Each phase ends at a gate that must pass before the next begins.

**Phase 1 — routed layer, replicated experts (`P = 1`), 60M and 125M.**
No collectives, no mesh change. Delivers the router, the sort, `gmm` wiring,
the aux loss, the FLOP rule, and the registry ids.
*Gate:* equi-FLOP with the dense tier to within 1%, loss curve tracking dense
within seed noise, a re-swept base LR, and every routed token accounted for —
`counts.sum() == T * K` exactly, asserted, not sampled.

**Phase 2 — expert parallelism.** The 2D mesh, `ragged_all_to_all` dispatch,
`group_offset`, worst-case dispatch buffers. Needed for 1B, and for 500M on
v5e.
*Gate:* bit-identical loss to `P = 1` on 60M at the same seed — the dispatch is
a permutation and must change nothing. This is the single most valuable test in
the plan.

**Phase 3 — the ladder.** 60M through 1B, sweeping what phase 1 showed does
not transfer.

Building phase 2 before phase 1's gate passes would mean debugging routing and
collectives at the same time, which is the main way this goes wrong.

## What I would measure first

1. **Does the dense LR optimum transfer to routed models?** One 60M LR sweep,
   three seeds. Everything downstream assumes it.
2. **Is the sort the bottleneck?** `argsort` versus counting sort at
   `m = 16384`, measured, before optimizing either.
3. **How skewed does routing actually get?** `max_load_fraction` over a 60M
   run says whether the auxiliary loss is doing its job. Nothing depends on the
   answer for correctness — the buffers hold regardless — but a router that
   collapses onto two experts is wasting six, and the loss curve alone will not
   say so.

## Open questions

- Sparsify every layer, or every other layer? Interleaving is common and halves
  the parameter cost; it also breaks the clean "active == dense tier" identity.
- Does `E = 8` stay fixed across the ladder, or scale with width? Fixing it is
  the simpler experiment and the one the current tiers support.

**Settled: no shared expert.** An always-on expert alongside the routed ones is
cheap and usually helps, which is exactly why it does not belong in a baseline
— it adds active FLOPs and so breaks the equi-FLOP identity against the dense
ladder, and it confounds "did sparsity help" with "did extra dense capacity
help". If it is worth measuring later it is worth measuring as its own arm.
