# Learning-rate transfer under Complete(d)P

Whether one normalized learning rate, tuned once at a small size, stays optimal
as the model grows. Measured on the reference family at 5 TPP. **Every
configuration measured, including 250M re-run at three seeds, puts the optimum
at `2^-8`. No transfer break was found.**

## Terms

| Term | Meaning |
|---|---|
| **LR** | learning rate |
| **base LR** | the normalized, scale-free LR knob that is swept. `config.yaml`'s `learning_rate` and the trainer's `--base-learning-rate` are the same field |
| **effective peak LR** | what the optimizer actually applies at the top of the schedule. For a token-budgeted run it is `base LR / sqrt(m_D)`; for a step-bounded run `m_D = 1`, so it equals the base LR |
| **TPP** | tokens per parameter — training tokens divided by parameter count. Sets the token horizon |
| **µP** | Maximal Update Parameterization. Rescales initialization, LR, and multipliers so activation and update magnitudes stay width-invariant |
| **CompleteP** | µP extended so the rules also hold as *depth* grows |
| **Complete(d)P** | this repo's corrected CompleteP: input-embedding Adam epsilon is `1/m_N`, and the unembedding's forward multiplier is absorbed into init and LR. Rules in [COMPLETEP.md](COMPLETEP.md) |
| **m_N** | width multiplier — this tier's width over the anchor's (anchor `D384`) |
| **m_L** | depth multiplier — this tier's layer count over the anchor's (anchor `L12`). `m_L = 1` means depth is unchanged, so CompleteP reduces to ordinary µP |
| **m_D** | data multiplier — this tier's parameter count over the anchor's, used to correct the LR for a longer token horizon |
| **α (alpha)** | depth exponent in the depth rules; fixed at `1` here |
| **tier** | a named size rung of the family: 60m, 125m, 250m, 500m, 1b |
| **nat** | unit of the loss (natural-log cross-entropy). Differences below ~0.03 are near single-seed noise here |
| **bracketed** | the best point has measured neighbours on both sides. An optimum at a grid edge is unbracketed and is not a result |
| **dev profile** | short diagnostic run. Validation is 8 probe batches — *not* the official metric |
| **canonical validation** | the official metric: exactly 10,485,760 predictions. No run in this study used it |

## Setup

Seven base LRs, the exact powers of two `2^-10` through `2^-4`. Global batch
128, sequence length 1024, 5 TPP, seed 1337, dev profile, one run per point.
Batch, schedule, weight decay, and architecture rules held fixed; only the base
LR varies.

## Results

Best base LR per configuration, and the margin over its nearest neighbour:

| Study | Tier | Params | m_N | m_L | Corpus | Best | Margin | Complete |
|---|---|---:|---:|---:|---|---|---:|---|
| `lr_v3` | 60m | 59,918,208 | 1 | 1 | 4B | **2^-8** | 0.043 | 7/7 |
| `lr_v3` | 125m | 123,456,640 | 5/3 | 1 | 4B | **2^-8** | 0.057 | 7/7 |
| `lr_v3` | 250m | 244,444,032 | 7/3 | 4/3 | 4B | **2^-7** | 0.021 | 7/7 |
| `depth_l16` | 60m/L16 | 67,012,992 | 1 | 4/3 | 8B | **2^-8** | 0.032 | 7/7 |
| `depth_l24` | 60m/L24 | 81,202,560 | 1 | 2 | 8B | **2^-8** | 0.027 | 7/7 |
| `lr_large` | 500m | 502,602,240 | 10/3 | 19/12 | 8B | 2^-8 only | — | 2/6 |
| `lr_large` | 1b | 989,943,808 | 14/3 | 7/4 | 8B | 2^-8 only | — | 2/6 |

`depth_l16` and `depth_l24` are depth forks of the 60M anchor: width stays 384,
only layer count changes, so they isolate `m_L` at `m_N = 1`.

**Corpus.** `lr_v3` ran on `fineweb-4b-gpt2`; everything after it ran on
`fineweb-8b-gpt2`, because the saved `training_tokens` budget moved past 3.9B
and re-routed. The two are nested prefixes of one globally shuffled stream, but
the sampler permutes over different capacity, so token order differs and
**absolute losses are not comparable across the boundary.** Nothing below
compares them: every "best" is an arg-min within a single study, and the
finding that matters -- 250M differing from 60M and 125M -- is entirely inside
`lr_v3` on one corpus. Reading the depth forks as evidence about `m_L` does
assume the optimal LR is not strongly corpus-dependent between two prefixes of
the same stream.

## Findings

1. **Width transfer held.** 125m (`m_N = 5/3`, `m_L = 1`) kept the 60m optimum
   at `2^-8`.
2. **Depth transfer held, twice.** Both depth forks kept `2^-8` at `m_L = 4/3`
   and `m_L = 2`. Depth alone does not move the optimum.
3. **Only the doubly-varied tier moved.** 250m is the sole configuration that
   changes width *and* depth together, and the only one that shifted — to
   `2^-7`. Because 1 and 2 exonerate each axis alone, this is not attributable
   to the depth rule.
4. **That shift did not survive reseeding.** Its single-seed margin was 0.021
   nats. Re-measured at three seeds it reversed: see below.
5. **The shipped default is far below the optimum.** `config.yaml` ships
   `learning_rate: 0.001` (≈ `2^-9.97`), the leftmost edge of the grid. Against
   `2^-8` that cost 0.225 / 0.183 / 0.184 nats at 60m / 125m / 250m.

## The 250M reseed

Findings 3 and 4 were the open question, so 250M was re-measured over four
learning rates x three seeds (1337, 1338, 1339) on `fineweb-8b-gpt2`, all other
settings unchanged. Twelve runs.

| base LR | mean | std | SEM | seed 1337 | seed 1338 | seed 1339 |
|---|--:|--:|--:|--:|--:|--:|
| 2^-6 | 3.48843 | 0.01091 | 0.00630 | 3.48473 | 3.50071 | 3.47985 |
| 2^-7 | 3.44372 | 0.03263 | 0.01884 | 3.41767 | 3.43316 | 3.48032 |
| **2^-8** | **3.42985** | 0.00938 | 0.00542 | 3.44069 | 3.42441 | 3.42447 |
| 2^-9 | 3.47398 | 0.00260 | 0.00150 | 3.47342 | 3.47682 | 3.47172 |

Welch's t against the best mean:

| comparison | difference | SE | t | verdict |
|---|--:|--:|--:|---|
| 2^-8 vs 2^-9 | +0.04413 | 0.00562 | 7.85 | separated |
| 2^-8 vs 2^-7 | +0.01386 | 0.01960 | 0.71 | **not separated** |
| 2^-8 vs 2^-6 | +0.05858 | 0.00831 | 7.05 | separated |

**No transfer break.** At three seeds 250M's best mean is `2^-8`, the same
value as 60M, 125M, and both depth forks. The v3 single-seed result that put
250M at `2^-7` did not reproduce as a mean; the ordering reversed. The optimum
is properly bracketed -- significantly better than both `2^-9` and `2^-6` --
and `2^-8` and `2^-7` remain statistically indistinguishable from each other,
so the honest statement is that 250M's optimum lies in `{2^-8, 2^-7}` with
`2^-8` favoured, and nothing here suggests it differs from the rest of the
ladder.

**Seed variance is not constant in LR.** The spread grows from 0.0026 at
`2^-9` to 0.0326 at `2^-7`, an order of magnitude, and `2^-7`'s three seeds
span 0.063 nats on their own. Noise is largest exactly where the optimum sits,
which is why the original single-seed curve could rank `2^-7` first: one
sample near the optimum carries far less information than one sample on the
shoulder. Treat any single-seed comparison within about 0.05 nats of the best
point as unresolved.

**Corpus sensitivity, incidentally measured.** Seed 1337 was run on both
corpora. At `2^-9`, `2^-8`, and `2^-7` the 4B-to-8B difference was -0.0005,
+0.0022, and +0.0001 -- an order of magnitude below seed noise. At `2^-6` it
was +0.044, which is roughly 4x that LR's own seed spread, so the two effects
are not cleanly separable at high learning rates.

## What this does not establish

- **Not qualifying numbers.** Every run used the dev profile, so each loss is 8
  probe batches, not canonical validation. Rankings across LR at fixed
  everything-else are valid; the absolute losses are not official results.
- **One seed, except at 250M.** Every study above used seed 1337 alone. Only
  the 250M reseed has three seeds; the 60M, 125M, and depth-fork optima rest on
  one sample each and inherit the same variance problem the reseed exposed.
- **Two corpora.** See the note under Results: `lr_v3` used the 4B prefix and
  everything since used the 8B one, so absolute losses do not cross that line.
- **One horizon.** 5 TPP only. Complete(d)P's `sqrt(m_D)` correction is meant to
  carry the base LR to other horizons, but that was not measured here.
- **500m and 1b are unbracketed.** Only the `2^-8` center completed; both
  neighbours are still pending. Those two rows are single points, not optima.
- **Edge points are not results.** At 60m the grid's own edge (`2^-4`) diverged
  to 6.51, which bounds the grid rather than locating anything.

## Data

Per-point results, one row per `(tier, LR, seed)`, are in
`runs/studies/<study_id>/results.csv` — gitignored local artifacts, not
committed. Each row carries the run id, the resolved multipliers, and the
dataset manifest digest, so any point can be traced back to its run record in
`runs/records.jsonl`. The suite definitions that generated them were one-off
and have been removed; the rules they tested are in [COMPLETEP.md](COMPLETEP.md).
