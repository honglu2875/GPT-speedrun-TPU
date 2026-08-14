# Learning-rate transfer under Complete(d)P

Whether one normalized learning rate, tuned once at a small size, stays optimal
as the model grows. Measured on the reference family at 5 TPP. **Width transfer
and depth transfer each held at `2^-8`. The single tier that changed both at
once landed one grid notch away, by a margin too small to call.**

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
4. **That shift is not established.** Its margin is 0.021 nats, the flattest in
   the study but close behind `depth_l24` (0.027) and `depth_l16` (0.032),
   which both kept `2^-8`. One grid notch at that margin, on a single seed,
   does not distinguish a real break from noise.
5. **The shipped default is far below the optimum.** `config.yaml` ships
   `learning_rate: 0.001` (≈ `2^-9.97`), the leftmost edge of the grid. Against
   `2^-8` that cost 0.225 / 0.183 / 0.184 nats at 60m / 125m / 250m.

## What this does not establish

- **Not qualifying numbers.** Every run used the dev profile, so each loss is 8
  probe batches, not canonical validation. Rankings across LR at fixed
  everything-else are valid; the absolute losses are not official results.
- **One seed.** Seed 1337 only. Finding 4 needs at least three seeds near the
  optimum before the 250m shift means anything.
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
