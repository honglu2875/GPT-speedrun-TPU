# Complete(d)P token-horizon ablation at 60M and 125M

This study tests whether adding Complete(d)P's cross-horizon
duration factor to this repository's fixed-TPP CompleteP hybrid improves
20-TPP training. Note that I do not aim to refute the claims in Complete(d)P,
and there are minor setups not being equivalent to what the paper
did. But a useful scientific findings are those that generalize, and I am
testing whether the cross-horizon optimal transfer applies to my simple
ladder or not.

In the `recipes/reference` baseline, I calculated
```text
m_D = (parameters / 60M_parameters)
```
whereas in Complete(d)P paper, this should have involved a ration between
token-horizon against some baseline:
```text
m_D_total = (parameters / 60M_parameters) * (target_TPP / 5)
```
This adjustment is implemented in `recipes/reference_duration` clone.

This started off as an implementation mistake because I did intend to make `reference`
baseline consistent with Complete(d)P. If this were indeed an error, I should have
felt a big impact because many of my experiments were run
using `TPP = 5`, with a smaller amount of runs promoted to `TPP = 20`. A factor
of `4` in training horizon does make a big difference in hyperparameter choices, yet
I find that the optimal transfer tends to happen better without this
TPP/base_TPP adjustment.

This study performs a 21-run sweep for each of the 60M and 125M tiers, and concluded
that in my case, **without token-horizon adjustment**, the optimal hyperparameter transfers
better. For example, the optimal learning rate for the `reference` baseline is `2^-8`, and
the base learning rate remains `2^-8`. Doubling the
base learning rate — the move that compensates for its 4x duration factor —
makes validation loss worse. At 60M the conclusion is the same, although three
seeds do not separate the two duration learning rates. The batch-512
iso-horizon construction is not better at either tier.

The original Complete(d)P's conclusion might be affected by many things such as architecture,
scale, etc. But I do not have a good explanatin about the counter-evidence in my case. Scale
issue feels slightly more likely, as Complete(d)P paper compares training runs up to 32B tokens
which my v4-32 pod cannot afford.

(below are the experiment details documented by GPT-5.6)

## Design

All cells use 1,024-token context, 20 tokens per parameter, FineWeb-Edu,
seeds 1337–1339, and a TPU v4 slice with 4 processes / 16 chips. The runs use
the dev validation profile, so the within-study rankings are useful but the
absolute losses are not qualification results.

The reference parameterization uses

```text
m_D = parameters / 60M_parameters
```

and reanchors that ratio inside the 20-TPP ladder. The duration fork instead
uses

```text
m_D_total = (parameters / 60M_parameters) * (target_TPP / 5)
```

so the added factor is four. At batch 128 it multiplies LR and weight decay by
`1/2`, epsilon by `2`, and each Adam `1 - beta` by `1/4` relative to the
reference. The base-LR `2^-7` duration point therefore restores the reference
`2^-8` effective peak LR; it is the LR-compensated point. Batch 512 supplies
`m_B = 4`, cancelling the added duration factor for all four optimizer
corrections and reducing the number of optimizer steps by four.

| parameterization | batch | base learning rates | purpose | runs / tier |
|---|---:|---|---|---:|
| fixed-TPP reference | 128 | `2^-7`, `2^-8`, `2^-9` | reanchored control and LR bracket | 9 |
| duration-scaled | 128 | `2^-7`, `2^-8`, `2^-9` | duration treatment and LR bracket | 9 |
| duration-scaled | 512 | `2^-8` | SDE iso-horizon point | 3 |

## Final validation loss

Values are mean ± sample standard deviation over three seeds; lower is better.

| arm | batch | base LR | 60M | 125M |
|---|---:|---:|---:|---:|
| fixed-TPP reference | 128 | `2^-7` | 3.71733 ± 0.00167 | 3.41684 ± 0.00147 |
| fixed-TPP reference | 128 | **`2^-8`** | **3.70294 ± 0.00446** | **3.40848 ± 0.00100** |
| fixed-TPP reference | 128 | `2^-9` | 3.70922 ± 0.00740 | 3.42462 ± 0.00100 |
| duration, LR-compensated | 128 | `2^-7` | 3.71935 ± 0.01171 | 3.47495 ± 0.00428 |
| duration | 128 | **`2^-8`** | **3.70921 ± 0.01018** | **3.44812 ± 0.00858** |
| duration | 128 | `2^-9` | 3.74885 ± 0.00367 | 3.48800 ± 0.00361 |
| duration, iso-horizon | 512 | `2^-8` | 3.76860 ± 0.00441 | 3.44688 ± 0.00587 |

The table below reports the second cell minus the first. Positive deltas mean
the second cell is worse. `|t|` is the absolute Welch statistic used by the
repository's sweep protocol; `|t| >= 2.5` is treated as separated. Pairing by
seed gives the same decisions for every comparison discussed below.

| comparison | 60M delta | 60M `|t|` | 125M delta | 125M `|t|` |
|---|---:|---:|---:|---:|
| reference: `2^-7` minus `2^-8` | +0.01439 | 5.24 | +0.00836 | 8.15 |
| reference: `2^-9` minus `2^-8` | +0.00628 | 1.26 | +0.01614 | 19.71 |
| duration: compensated `2^-7` minus `2^-8` | +0.01013 | 1.13 | +0.02683 | 4.85 |
| duration: `2^-9` minus `2^-8` | +0.03964 | 6.34 | +0.03988 | 7.42 |
| duration: iso batch 512 minus batch 128 | +0.05939 | 9.27 | -0.00124 | 0.21 |
| duration `2^-8` minus reference `2^-8` | +0.00627 | 0.98 | +0.03964 | 7.95 |

## Interpretation

1. **The fixed-TPP base-LR anchor survives 20 TPP.** Reference `2^-8` wins at
   both tiers. The low side is noise-limited at 60M, but the 125M bracket is
   separated on both sides. This agrees with the older 75-run
   [60M batch × LR sweep](batch-size-sweep-60M.html) at 5 TPP and with the
   longer-horizon 500M observations.

2. **The predicted LR move is not observed.** With the added fourfold duration
   factor, a `2^-7` base LR is the compensated counterpart of reference
   `2^-8`. Its mean is worse at both tiers and is clearly worse at 125M. The
   directly reused `2^-8` base LR remains the best measured duration point.

3. **The full duration package hurts at 125M.** At the same base LR and batch,
   the duration arm is `+0.03964` nats worse than reference. The 60M difference
   is unresolved. Since the fork moves four optimizer quantities together,
   this result cannot assign the loss to `m_D` alone.

4. **There is no evidence for moving the batch anchor to 512.** The
   iso-horizon point is `+0.05939` nats worse than duration batch 128 at 60M.
   They tie at 125M, but both remain about `0.04` nats behind the reference.

For this family, the practical choice remains the opinionated one already used
by `reference`: keep TPP horizons empirically reanchored, keep base LR `2^-8`,
and do not import the extra `TPP / TPP_0` factor. A larger-tier experiment may
still be informative, but these data do not justify paying for 500M on the
current machine.

## Scope and limitations

- Three seeds can resolve the strong 125M effects but leave two 60M
  comparisons noise-limited.
- Only 60M and 125M were tested; both have 12 layers, so this is primarily a
  width comparison rather than a strong depth-scaling test.
- The duration recipe bundles LR, weight decay, epsilon, and beta scaling. A
  clean causal study of `m_D` would ablate those corrections separately.
- The batch-512 point is an iso-horizon construction, not a full batch sweep.
- No QK normalization is used, and the schedule is this repository's fixed
  10%-warmup cosine schedule.

## Logs and provenance

The full-resolution studies are
[`duration-ablation-60M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/duration-ablation-60M)
and
[`duration-ablation-125M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/duration-ablation-125M).
Each contains 21 run folders, the immutable ledger, and a compact curve
snapshot. Their cards carry the exact grid and current reproduction command.

All 42 runs record Git head
`553b394f3d80b418719416d6f34195360a3781df`. The only dirty-tree entry was the
unrelated, untracked `HANDOVER.md`; recipe, config, shared source, lockfile, and
dataset checksums are recorded independently in every ledger entry.
