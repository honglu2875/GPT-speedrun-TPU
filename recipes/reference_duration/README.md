# reference_duration — Complete(d)P duration ablation

An experimental fork of [`reference`](../reference/) for one controlled
question: does Complete(d)P's additional token-horizon factor improve transfer
from 5 to 20 tokens per parameter?

This is intentionally a recipe fork, not a `reference` flag. The reference
family keeps its opinionated fixed-TPP reanchoring, while this arm is free to
change the optimizer parameterization without expanding the public CLI.

## The only scientific difference

The architecture, contexts, initialization, schedule shape, data stream, and
optimizer implementation are cloned from `reference`. The duration anchor is
declared as 5 TPP and the optimizer uses

```text
m_D_total = (parameters / 60M_parameters) * (target_TPP / 5)
```

where `reference` deliberately omits the second factor. The resulting global
Complete(d)P scalars are applied together:

| quantity | multiplier |
|---|---:|
| learning rate | `sqrt(m_B / m_D_total)` |
| weight decay | `sqrt(m_B / m_D_total)` |
| Adam epsilon | `sqrt(m_D_total / m_B)` |
| `1 - beta1`, `1 - beta2` | `m_B / m_D_total` |

At 5 TPP, `target_TPP / 5 = 1`, so this fork resolves the same tensor and
optimizer values as `reference`. At 20 TPP the duration multiplier is four.
For batch 128, that halves LR and weight decay, doubles epsilon, and divides
both Adam `1 - beta` values by four relative to the reanchored arm.

The special batch-512 point is the paper's SDE iso-horizon construction:
`m_B = 4` cancels the 4× duration factor. It processes four times the tokens in
the same number of optimizer steps as that tier's 5-TPP/batch-128 anchor, with
the same effective optimizer hyperparameters and improved gradient
signal-to-noise.

## Scope

- The duration anchor is explicit in `config.yaml`; this is not a general
  claim that 5 TPP is universal.
- Whole-step rounding determines the achieved token count, but the multiplier
  uses the requested TPP so the 5-TPP anchor remains exactly identical.
- The completed v4-32 experiments use `reference` as the reanchored control and
  this recipe as the duration-scaled treatment, with paired seeds at 60M and
  125M.
- As in the paper, the claim is tied to the fixed 10%-warmup cosine schedule.

## Result

The 42-run 20-TPP matrix does not support adding the cross-horizon factor to
this family. The duration treatment's best measured base LR remains `2^-8`,
not the compensated `2^-7`. At 125M, compensation costs 0.02683 nats and the
duration treatment at `2^-8` costs 0.03964 nats against `reference`; both are
separated across three seeds. At 60M the means point the same way but the
corresponding comparisons are noise-limited. Batch 512 is substantially worse
at 60M and tied with duration batch 128 at 125M, where both trail reference.

See the full [duration-ablation report](../../docs/reports/duration-ablation.md)
for the grid, mean ± SD tables, statistical comparisons, limitations, and raw
log links. This fork remains useful as an explicit experimental control; its
rules are not the default recommendation.

Inspect a point without training:

```bash
uv run --frozen --no-sync recipes/reference_duration/train.py \
  --profile dev --tier 125m --context 1k --tokens-per-parameter 20 \
  --batch-size 512 --print-plan
```

Run through the harness:

```bash
uv run --frozen --no-sync rig run reference_duration \
  --cluster v4-32 --profile dev --context 1k --tier 125m \
  --tokens-per-parameter 20 --batch-size 512 \
  --base-learning-rate 0.00390625 --seed 1337 \
  --checkpoint-policy none --name 125m-d20-iso-bs512-lr2e-8-s1337
```
