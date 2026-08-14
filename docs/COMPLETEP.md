# Complete(d)P model-family contract

This repository uses a conservative **Complete(d)P, α = 1** contract for the
reference family. It is a post-µP engineering baseline, not a claim that every
result in the original µP work is false.

The distinction matters. Tensor Programs V established a useful width-scaling
framework and a practical proxy-to-target workflow. Later work found that
transfer is approximate at finite width, that multiple parameterizations can
transfer with the right layerwise rules, and that the original transformer
recipe did not cover depth, Adam epsilon, weight decay, batch size, or training
duration completely. CompleteP adds depth scaling; the August 11, 2026 revision
of Complete(d)P corrects implementation details and unifies width/depth rules
with approximate SDE rules for batch and token horizon.

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

## Implemented rule

The 60M tier is the base discretization. Let `mN = width / 384`,
`mL = layers / 12`, `mD = training tokens / base-tier training tokens`, and
`mB = global batch / 128`. In a fixed-TPP ladder, `mD` is the tier's parameter
count divided by the 60M parameter count. A fixed-step diagnostic sets `mD=1`
so a profiling override does not silently alter the optimizer.

The model uses pre-RMSNorm, RoPE, GELU, a 4× MLP, untied embeddings, and a
fixed 64-dimensional attention head. Scaling width therefore adds heads rather
than changing head dimension. Attention follows the official nanoGPT-mup
implementation and divides each head's QK contraction by its own head dimension.

| Quantity | Complete(d)P multiplier |
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

Across batch and token horizon, the runtime additionally applies the current
Complete(d)P SDE prescription:

| Quantity | Global multiplier |
|---|---:|
| learning rate | `sqrt(mB / mD)` |
| Adam epsilon | `sqrt(mD / mB)` |
| weight decay | `sqrt(mB / mD)` |
| `1 - beta1`, `1 - beta2` | `mB / mD` |
| optimizer steps | proportional to `mD / mB` |

These are approximate finite-step transfer rules. They do not imply identical
training trajectories, and a fitted optimum at 60M is not accepted as proof of
transfer. Complete(d)P itself reports a small penalty when transferring from a
58M proxy and nearly stabilized optima from about 136M upward. This is why the
admission decision uses 60M, 125M, and 250M trends. The first-pass protocol
stops there; 500M and 1B are reserved for later reproduction or hero runs.

## Deliberately held fixed

- Global gradient clipping is disabled. Clipping introduces another
  scale-dependent axis unless its threshold is parameterized and validated.
- QK normalization is not mixed into this first reproduction. The latest paper
  reports improved stability with it but also reports that removing it does not
  break transfer. It can be tested later as a family-level candidate.
- The fixed-head attention temperature follows nanoGPT-mup's executable
  `1 / d_head` rule. The conflicting `1 / d_model` sentence in the CompleteP
  paper is not used; with width scaled by adding heads it would introduce an
  extra inverse-head-count factor.
- We do not combine Complete(d)P with u-µP, Adam-atan2, GQA-specific µP, Muon,
  or per-module learning-rate tuning in the baseline. Each is a meaningful
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

The learning-rate sweep is intentionally one-dimensional: the seven normalized
base LRs `2^-10`, `2^-9`, `2^-8`, `2^-7`, `2^-6`, `2^-5`, and `2^-4`, one
seed, batch 128, and 5 TPP (0.25× the 20-TPP ladder). This matches the
powers-of-two LR grid visible in the paper's Figures 2 and 3 while retaining our
compute-limited horizon. The same grid runs independently and in sequence at
60M, 125M, and 250M.

The first two tiers both have the 12-layer base depth, so `mL = 1` and
CompleteP is identical to ordinary muP there. Only 250M, at 16 layers, exercises
the depth-specific rules: both residual branches and residual-block AdamW
epsilon receive `mL^-1 = 3/4`. Consequently this is primarily a width-transfer
test with one modest depth-transfer point, not a reproduction of the paper's
2-to-128-layer experiment.

Collect every point before any chart is designed. An optimum on a grid edge is
an unbracketed result, not a winner. Once a common interior neighborhood
appears, it should be rerun with at least three seeds. Only then should batch
size vary, using the SDE rules above and keeping the selected normalized base
LR fixed.

The one-off suites that produced the results above have been removed along with
the study runner; the rules on this page are the durable part. To repeat this
kind of measurement, loop `rig run` over the tiers and learning rates you want
and read the recorded runs back out of `runs/records.jsonl` — every run already
carries the resolved parameterization, so a study is a query over run records
rather than a separate execution framework.
