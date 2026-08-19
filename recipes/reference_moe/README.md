# reference_8k — the reference family at 8,192-token context

A fork of [`reference`](../reference/) that changes **two** things and nothing
else: the context length, and document masking.

| | `reference` | `reference_8k` |
|---|---|---|
| `seq_len` | 1,024 | **8,192** |
| document masking | off | **on** |
| `batch_size` (sequences) | 128 | **16** |
| tokens per optimizer step | 131,072 | 131,072 |
| optimizer steps at 60M / 5 TPP | 2,286 | 2,286 |
| tiers, parameter counts, schedule | | identical |

The tiers are unchanged — parameter counts do not depend on `seq_len`, because
positions are RoPE and carry no learned embedding.

## Why batch 16 and not 128

Holding `batch_size` at 128 would have multiplied tokens per step by eight,
which multiplies the token batch by eight and divides the optimizer step count
by eight — 286 steps at 60M / 5 TPP instead of 2,286. A comparison against
`reference` would then confound three changes at once.

Batch 16 holds **tokens per step and step count identical to `reference`**, so
the ladders differ only in how those tokens are arranged into sequences and in
whether attention crosses documents.

Note the consequence for the parameterization: `m_B` counts *sequences*
(`batch_size / 128`), which is the convention the scaling-literature uses, so
this family runs at `m_B = 0.125` and its Complete(d)P learning-rate scaling
differs from `reference` accordingly. **The base-LR optimum therefore does not
transfer from `reference` and must be swept here.** That is why this family
exists as its own ladder rather than as a flag.

## Why masking is on here and off there

A window is cut live from a flat token stream, so it can span several
documents. Measured on a 100M-token FineWeb shard:

| | 1,024 | 8,192 |
|---|--:|--:|
| documents a random window spans | ~1.5 | **~11.8** |
| tokens in documents at least that long | 53.0% | **8.5%** |

At 1k the cross-document surface is about one boundary per window and
`reference` leaves it unmasked, deliberately, to keep its recorded results
bit-reproducible. At 8k a window averages twelve documents and only 8.5% of
tokens come from documents that long — unmasked, most of the added context
would be attention across unrelated text, which is not what the extra compute
is meant to buy.

Masking is `segment_ids` in
[`rig/kernels/tpu_flash_attention.py`](../../rig/kernels/tpu_flash_attention.py):
a position may attend only within its own document, subject to causality. The
segment index is derived on-device from the input tokens as a running count of
EOT boundaries, so nothing extra crosses the host boundary.

## What this costs

Traced, not estimated — attention is a minority of FLOPs at these widths, so
eight times the quadratic term is under twice the total:

| tier | FLOPs/token at 1k | at 8k | ratio | attention share |
|---|--:|--:|--:|--:|
| 125M | 710M | 1,371M | 1.93x | 13% → 55% |
| 500M | 3,064M | 5,156M | 1.68x | 10% → 46% |

## Comparing against `reference`

Validation loss is **not** directly comparable between the families. Validation
windows are cut at `seq_len`, so at 8k every scored token gets up to eight
times more context, and this family will post a lower loss for that reason
alone regardless of model quality.

A defensible comparison needs both families evaluated at a **common** context.
Report each at its native context as its own number, and the 8k model
additionally evaluated at 1,024 — RoPE handles the shorter sequence — for the
comparable one. Loss against position within the window is the measurement that
actually shows whether the added range is being used.

## The honest expectation

Only 8.5% of tokens live in documents long enough to fill an 8k window, so a
masked 8k window is largely short documents sitting beside masked-off
neighbours. Masking makes 8k *correct*; it does not make FineWeb *long*. If
this family does not beat `reference` at equal FLOPs, the corpus is the first
place to look, not the recipe.

## What is not built yet

Experts are **replicated**: every device holds all eight and routes only its own
tokens, so there are no expert collectives. That is what the recorded ladder
ran, and it is sufficient while the experts fit in memory.

Expert *parallelism* — sharding experts across devices — is designed but
deliberately not implemented. It becomes necessary when the experts stop
fitting, which is a bigger-model problem, not one this ladder has. The design,
for whoever needs it: under `shard_map` over an `expert` axis, all-gather the
`[E]` counts so every device can derive the full `[P, E]` traffic matrix by
prefix sum; `ragged_all_to_all` the expert-major tokens; block-transpose the
receive buffer, which arrives source-major and expert-minor, into expert-major
order using indices from the counts rather than a sort; `gmm` with
`group_offset` set to the first local expert, which is what that argument is
for; then the inverse `all_to_all` and unsort.

## Decisions this recipe has settled

**Every layer is routed.** All twelve, not alternating. Interleaving dense and
routed layers is common and halves the parameter cost, but it breaks the clean
"active parameters == dense tier" identity, and that identity is what makes the
routed and dense ladders equi-FLOP and therefore comparable. A cheaper model
that cannot be compared is not cheaper for this purpose.

**`E = 8` is fixed across the ladder**, not scaled with width. It is the
simpler experiment and the one the tiers are sized for.

**No shared expert here.** An always-on expert alongside the routed ones is
cheap and usually helps, which is exactly why it does not belong in the
baseline: it adds active FLOPs, so it breaks the equi-FLOP identity against the
dense ladder, and it confounds "did sparsity help" with "did extra dense
capacity help". It is worth measuring — as an ablation forked from this recipe,
with its own arm, so the two questions stay separable.
