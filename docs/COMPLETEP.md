# Fixed-TPP CompleteP hybrid

This repository uses an opinionated parameterization called
`completep_fixed_tpp_v1`. It starts from **CompleteP, α = 1**, adopts a small
set of useful corrections and batch/duration formulas from Complete(d)P, and
reanchors those formulas independently inside every fixed-TPP ladder.

It is deliberately not presented as a complete implementation of
Complete(d)P. In particular, changing a run from 5 TPP to 20 TPP does not add
the paper's cross-horizon `TPP / TPP₀` factor. The 5-TPP and 20-TPP sweeps are
separate empirical anchors.

## Two papers, not one

These are distinct publications, and the difference decides which scaling
axes are covered:

| | paper | axes |
|---|---|---|
| **CompleteP** | [Don't be lazy: CompleteP enables compute-efficient deep transformers](https://arxiv.org/abs/2505.01618) (Dey et al.) | width, **depth** |
| **Complete(d)P** | [Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration](https://arxiv.org/abs/2512.22382) (Mlodozeniec, Ablin, Béthune, Busbridge, Klein, Ramapuram, Cuturi) | width, depth, **batch**, **duration** |

The `(d)` marks the added axes — token **d**uration and batch size — reading
as both "CompleteP" and "Complete**d**P". It is *not* depth; CompleteP already
had depth. That is why the second paper writes `m_D` for a data/duration
multiplier and uses `sqrt(m_B / m_D)`. This repository uses corrections of the
same shape but defines both ratios locally, as described below.

In its own words: *"we propose the Complete(d) Parameterisation that unifies
scaling in width & depth — using an adaptation of CompleteP (Dey et al. 2025) —
as well as in batch-size and training duration."*

muP established a useful width-scaling framework and a practical
proxy-to-target workflow. Later work found that transfer is approximate at
finite width, that multiple parameterizations can
transfer with the right layerwise rules, and that the original transformer
recipe did not cover depth, Adam epsilon, weight decay, batch size, or
training duration completely.

Here we observe a stable learning-rate and batch-size optimum under the local,
fixed-TPP rules. That is evidence for this recipe, not evidence that the full
cross-horizon Complete(d)P rule is necessary or correct for these runs.

## What is borrowed from Complete(d)P

The paper makes three modifications. Two are implemented and covered by
`tests/test_trainer_static.py::test_fixed_tpp_completep_hybrid_tensor_and_ladder_multipliers`;
the third is deliberately out of scope for this reproduction.

| # | Change | Here |
|---|---|---|
| 1 | Extends the parameterization to **QK-normalization** layers, which CompleteP did not cover | **not implemented** — this family uses no QK norm |
| 2 | Corrects CompleteP's **AdamW epsilon for input embeddings** | **implemented**: `epsilon["token_embedding"] = 1 / m_N` |
| 3 | **Reparameterizes the output layer**, removing the explicit unembedding forward multiplier by absorbing it into learning rate and initialization | **implemented**: `gpt_logits` applies no multiplier; the unembedding gets `lr = 1/m_N`, `epsilon = 1`, and init `init_std / m_N` |

The runtime also uses corrections shaped like `sqrt(m_B / m_D)` and its
reciprocals. The important local choice is what those symbols mean here:

- `m_B = batch_size / configured_batch_size`. Each recipe/profile owns its
  batch anchor. The 1,024-token reference anchors at 128 sequences; the 8,192
  and routed recipes anchor at 16.
- `m_D_ladder = declared_parameters / base_parameters`. This is the data growth
  induced by holding TPP fixed while moving up one model ladder.
- There is no additional `TPP / TPP₀` term. At a given model size,
  `m_D_ladder` is the same in a 5-TPP and 20-TPP run.

These definitions are a project choice. They should not be attributed to the
Complete(d)P paper without the `fixed-TPP` qualification.

The isolated [`reference_duration`](../recipes/reference_duration/) fork adds
the omitted `TPP / 5` factor so the two policies can be compared without
changing this default contract. Its completed 60M/125M ablation does not
support importing that factor into the default: the compensated learning-rate
move loses at 125M and batch 512 provides no improvement. See the
[full report](reports/duration-ablation.md) for the results and limitations.

Primary sources:

- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [An Empirical Study of µP Learning Rate Transfer](https://arxiv.org/abs/2404.05728)
- [Scaling Exponents Across Parameterizations and Optimizers](https://proceedings.mlr.press/v235/everett24a.html)
- [u-µP: The Unit-Scaled Maximal Update Parametrization](https://arxiv.org/abs/2407.17465)
- [Don't be lazy: CompleteP enables compute-efficient deep transformers](https://arxiv.org/abs/2505.01618)
- [Official nanoGPT-mup CompleteP implementation](https://github.com/EleutherAI/nanoGPT-mup/tree/completep)
- [How to set AdamW's weight decay as you scale model and dataset size](https://proceedings.mlr.press/v267/wang25b.html)
- [Completed Hyperparameter Transfer across Modules, Width, Depth, Batch & Duration](https://arxiv.org/abs/2512.22382)
- [Weight Decay may matter more than µP for Learning Rate Transfer in Practice](https://arxiv.org/abs/2510.19093)

## Implementation details

The 60M tier is the base discretization. Let `mN = width / 384`,
`mL = layers / 12`, `mD_ladder = parameters / base_parameters`, and
`mB = global_batch / configured_batch`. A diagnostic is an early-stopped
prefix of the same fixed-TPP schedule, so it retains all four multipliers and
the original warmup/cosine horizon.

The model uses pre-RMSNorm, RoPE, GELU, a 4× MLP, untied embeddings, and a
fixed 64-dimensional attention head. Scaling width therefore adds heads rather
than changing head dimension. Attention follows the official nanoGPT-mup
implementation and divides each head's QK contraction by its own head dimension.

| Quantity | Tensor multiplier used here |
|---|---:|
| attention and MLP residual branches | `mL^-1` |
| input embedding init std | `1` |
| hidden matrix init variance | `mN^-1` |
| unembedding init variance | `mN^-2` |
| attention logits | `1 / d_head` |
| input embedding learning rate | `1` |
| hidden matrix learning rate | `mN^-1 mL^(α-1)` |
| hidden vector learning rate | `mL^(α-1)` |
| unembedding learning rate | `mN^-1` |
| hidden Adam epsilon | `mN^-1 mL^-α` |
| input embedding Adam epsilon | `mN^-1` |
| output/final-norm Adam epsilon | `1` |
| hidden/unembedding matrix weight decay | `mN` |
| other weight decay | `1` (vectors remain masked from decay) |

The unembedding has no explicit forward multiplier. Its old `mN^-1` forward
factor is absorbed into its initialization and learning rate, which keeps the
memory-bounded tiled cross entropy straightforward. AdamW follows the PyTorch
form, `parameter -= lr * (adam_update + weight_decay * parameter)`.

Across the recipe-local batch ratio and fixed-TPP model ladder, the runtime
applies the following Complete(d)P-inspired corrections:

| Quantity | Global multiplier |
|---|---:|
| learning rate | `sqrt(mB / mD_ladder)` |
| Adam epsilon | `sqrt(mD_ladder / mB)` |
| weight decay | `sqrt(mB / mD_ladder)` |
| `1 - beta1`, `1 - beta2` | `mB / mD_ladder` |
| optimizer steps within one fixed-TPP ladder | proportional to `mD_ladder / mB` |

These are approximate finite-step transfer rules. They do not imply identical
training trajectories, and a fitted optimum at 60M is not accepted as proof of
transfer. They also make no prediction here about how the optimum should move
when TPP itself changes; that remains an explicit experimental question.

## Deliberately held fixed

- Global gradient clipping is disabled. Clipping introduces another
  scale-dependent axis unless its threshold is parameterized and validated.
- QK normalization is not mixed into this first reproduction — this is
  Complete(d)P change 1 above, left out. That paper reports improved stability
  with it but also that removing it does not break transfer. It can be tested
  later as a family-level candidate.
- The fixed-head attention temperature follows nanoGPT-mup's executable
  `1 / d_head` rule. The conflicting `1 / d_model` sentence in the CompleteP
  paper is not used; with width scaled by adding heads it would introduce an
  extra inverse-head-count factor.
- We do not combine this hybrid with u-µP, Adam-atan2, GQA-specific µP, Muon,
  or per-module learning-rate tuning in the baseline. (Per-module tuning is one
  of the second paper's results — hyperparameters transfer even when tuned per
  module — but it is a tuning protocol, not a rule this family has to apply.) Each is a meaningful
  alternative, not a free patch that can be layered on without a control.
- Warmup is 10% of each run and cosine decay ends at 10% of peak. Schedule shape,
  weight decay, and batch remain fixed during the learning-rate study.

## Tiers and the learning-rate protocol

| Tier | Layers | Width | Heads | Exact parameters | 5-TPP steps | 5-TPP tokens |
|---|---:|---:|---:|---:|---:|---:|
| 60M | 12 | 384 | 6 | 59,918,208 | 2,286 | 299,630,592 |
| 125M | 12 | 640 | 10 | 123,456,640 | 4,709 | 617,218,048 |
| 250M | 16 | 896 | 14 | 244,444,032 | 9,325 | 1,222,246,400 |
| 500M | 19 | 1,280 | 20 | 502,602,240 | 19,173 | 2,513,043,456 |
| 1B | 21 | 1,792 | 28 | 989,943,808 | 37,763 | 4,949,671,936 |

The first two tiers share the 12-layer base depth, so `mL = 1` and CompleteP
reduces to ordinary muP there. Only 250M, at 16 layers, exercises the
depth-specific rules. This is therefore primarily a width-transfer test with one
modest depth point, not a reproduction of the paper's 2-to-128-layer experiment.

### Sweep protocol

**Sweep the base LR, never the effective one.** The runtime applies
`sqrt(mB / mD_ladder)`, so a fixed base LR is already a `sqrt(batch)` schedule
on the effective LR relative to that recipe's configured batch. Grids are
powers of two, wide enough that the winner has measured neighbours; an edge
optimum is unbracketed, not a winner.

**Ranking requires seeds.** One seed locates a region; ranking needs three and
Welch's `t >= 2.5`. Never rank points within ~0.05 nats from single seeds — seed
noise peaks at the optimum, and one such comparison produced a spurious
transfer break.

**Batch and LR are swept jointly**, because the `sqrt(mB)` rule is the
hypothesis under test. Over batch 32-512 at 5 TPP the base-LR optimum proved
batch-invariant, confirming the rule and justifying a one-dimensional LR sweep
at batch 128 — empirically, and only at this horizon.

Measurements are in
[HYPERPARAMETER_TRANSFER.md](HYPERPARAMETER_TRANSFER.md); this page is the
contract and method.
