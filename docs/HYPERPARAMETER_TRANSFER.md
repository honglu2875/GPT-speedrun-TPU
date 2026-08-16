# Hyperparameter transfer under Complete(d)P

Whether hyperparameters tuned once at a small size stay optimal as the model
grows. Two knobs measured on the reference family at 5 TPP: base learning rate
and global batch size.

**Both transfer.** Every configuration measured puts the optimum at base LR
`2^-8` and global batch `128`, across a 4x parameter range (60M → 250M) and a
16x batch range. The one apparent break was a seed artifact with an identified
cause.

The sharpness does *not* transfer: exceeding the optimal batch costs 0.45 nats
at 60M and 0.016 nats at 250M, a ~28x reduction. The location is stable; the
penalty for being wrong shrinks fast with scale.

Supersedes the former `LR_TRANSFER.md`.

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
| **m_N / m_L / m_D** | width, depth, and data multipliers relative to the 60M anchor (`D384`, `L12`) |
| **tier** | a named size rung of the family: 60m, 125m, 250m, 500m, 1b |
| **nat** | unit of the loss (natural-log cross-entropy) |
| **separated** | Welch's t between two 3-seed means exceeds 2.5 in magnitude. Anything below that is reported as unresolved, not as a ranking |
| **bracketed** | the best point has measured neighbours on both sides. An optimum at a grid edge is not a result |
| **dev profile** | short diagnostic run. Validation is 8 probe batches — *not* the official metric |

## Setup

Sequence length 1024, dev profile, seeds 1337/1338/1339 unless noted, on a
v4-32. Corpus `fineweb-8b-gpt2` except `lr_v3` (see the corpus note). Batch,
schedule, weight decay, and architecture rules held fixed except for the knob
under study. `grad_clip: 0.0` — clipping is disabled in every profile, which
matters for the mechanism section below.

## Study 1 — learning rate at fixed batch 128

Seven base LRs, exact powers of two `2^-10` through `2^-4`, one seed.

| Study | Tier | Params | m_N | m_L | Corpus | Best | Margin |
|---|---|---:|---:|---:|---|---|---:|
| `lr_v3` | 60m | 59,918,208 | 1 | 1 | 4B | **2^-8** | 0.043 |
| `lr_v3` | 125m | 123,456,640 | 5/3 | 1 | 4B | **2^-8** | 0.057 |
| `lr_v3` | 250m | 244,444,032 | 7/3 | 4/3 | 4B | 2^-7 | 0.021 |
| `depth_l16` | 60m/L16 | 67,012,992 | 1 | 4/3 | 8B | **2^-8** | 0.032 |
| `depth_l24` | 60m/L24 | 81,202,560 | 1 | 2 | 8B | **2^-8** | 0.027 |
| `lr_large` | 500m | 502,602,240 | 10/3 | 19/12 | 8B | 2^-8 only | — |
| `lr_large` | 1b | 989,943,808 | 14/3 | 7/4 | 8B | 2^-8 only | — |

`depth_l16` and `depth_l24` are depth forks of the 60M anchor: width stays 384,
only layer count changes, so they isolate `m_L` at `m_N = 1`. Width transfer
held at 125M; depth transfer held twice. 250M — the only tier varying width and
depth together — was the sole apparent break, and it did not survive reseeding.

### The 250M reseed

Four LRs x three seeds on `fineweb-8b-gpt2`.

| base LR | mean | SEM | seed 1337 | seed 1338 | seed 1339 |
|---|--:|--:|--:|--:|--:|
| 2^-6 | 3.48843 | 0.00630 | 3.48473 | 3.50071 | 3.47985 |
| 2^-7 | 3.44372 | 0.01884 | 3.41767 | 3.43316 | 3.48032 |
| **2^-8** | **3.42985** | 0.00542 | 3.44069 | 3.42441 | 3.42447 |
| 2^-9 | 3.47398 | 0.00150 | 3.47342 | 3.47682 | 3.47172 |

| comparison | difference | t | verdict |
|---|--:|--:|---|
| 2^-8 vs 2^-9 | +0.04413 | 7.85 | separated |
| 2^-8 vs 2^-7 | +0.01386 | 0.71 | **not separated** |
| 2^-8 vs 2^-6 | +0.05858 | 7.05 | separated |

At three seeds 250M's best mean is `2^-8`, matching every other configuration.
The optimum is bracketed against `2^-9` and `2^-6`; `2^-8` and `2^-7` remain
indistinguishable, so the honest statement is that 250M's optimum lies in
`{2^-8, 2^-7}` with `2^-8` favoured.

### Why the v3 250M point moved

With clipping disabled, a large gradient lands in the weights at full size.
Taking each run's peak gradient norm over its own median as a spike ratio:

| base LR | seed 1337 | seed 1338 | seed 1339 | loss spread |
|---|--:|--:|--:|--:|
| 2^-9 | 12x | 13x | 13x | 0.005 |
| **2^-8** | **90x** | 18x | 18x | 0.016 |
| **2^-7** | **25x** | 125x | 1633x | 0.063 |
| 2^-6 | 748x | 2294x | 114x | 0.021 |

Seed 1337 is the outlier at both learning rates that decided v3, in opposite
directions: at `2^-8` it spikes 90x where its siblings spike 18x and is the
worst of three; at `2^-7` it is the mildest at 25x and the best. v3 ran that
seed alone, comparing a depressed `2^-8` against a flattered `2^-7`.

It reproduces quantitatively: restricted to seed 1337 the reseed puts `2^-7`
ahead by **+0.023** where v3 reported **+0.021**; on three-seed means the sign
flips and `2^-8` leads by 0.014.

**How far to trust this.** Within a learning rate, spike ordering matches loss
ordering exactly at `2^-7` and `2^-6`; at `2^-8` the largest spike belongs to
the worst run while the other two tie; at `2^-9` nothing spikes and nothing
separates. Pooled across learning rates the rank correlation is only **+0.17**,
because spike magnitude grows with LR while loss excess is centred within it —
the effect is within-LR and vanishes if the groups are mixed. Three seeds per
point cannot establish a distribution. The supported claim is narrow: that
single v3 seed was unrepresentative for an identifiable reason.

## Study 2 — batch size x learning rate

Full grid, three seeds per cell, 123 runs.

### 60M — 5 batches x 5 LRs

| batch | steps | 2^-6 | 2^-7 | 2^-8 | 2^-9 | 2^-10 |
|---|--:|--:|--:|--:|--:|--:|
| 32 | 9,143 | 4.1081 | 4.1274 | **4.0664** | 4.0792 | 4.1915 |
| 64 | 4,571 | 4.1106 | 4.0641 | **4.0264** | 4.0596 | 4.1938 |
| 128 | 2,286 | 4.1161 | 4.1469 | **4.0143** | 4.0546 | 4.2304 |
| 256 | 1,143 | 4.4324 | 4.4725 | 4.4638 | **4.3343** | 4.5049 |
| 512 | 571 | 5.5062 | 5.2964 | **5.2473** | 5.4108 | 5.6684 |

bs32 vs bs64 t=+1.38 **not separated** · bs64 vs bs128 t=+1.25 **not
separated** · bs128 vs bs256 t=−7.95 separated · bs256 vs bs512 t=−14.52
separated.

At 60M the optimum is a **plateau over {32, 64, 128}**, not a peak. Only the
collapse above 128 is significant.

### 125M — 3 batches x 3 LRs

| batch | steps | 2^-7 | 2^-8 | 2^-9 |
|---|--:|--:|--:|--:|
| 64 | 9,419 | 3.7677 | **3.7340** | 3.7591 |
| 128 | 4,709 | 3.7135 | **3.6823** | 3.7269 |
| 256 | 2,355 | 3.8406 | **3.7346** | 3.7942 |

bs64 vs bs128 t=+9.22 separated · bs128 vs bs256 t=−7.06 separated.

### 250M — 4 batches x 2 LRs

| batch | steps | 2^-7 | 2^-8 |
|---|--:|--:|--:|
| 64 | 18,650 | 3.4872 | **3.4831** |
| 128 | 9,325 | 3.4437 | **3.4299** |
| 256 | 4,662 | 3.4897 | **3.4457** |
| 512 | 2,331 | — | **3.4895** |

bs64 vs bs128 t=+9.41 separated · bs128 vs bs256 t=−2.89 separated · bs256 vs
bs512 t=−34.62 separated. `2^-7` vs `2^-8` at bs128: t=+0.71, **not
separated** — independently reproducing Study 1's finding.

### The optimum transfers; its sharpness does not

At `2^-8`:

| tier | bs64 | **bs128** | bs256 | bs512 | Δ(256) | Δ(512) |
|---|--:|--:|--:|--:|--:|--:|
| 60m | 4.0264 | **4.0143** | 4.4638 | 5.2473 | **+0.4495** | **+1.2330** |
| 125m | 3.7340 | **3.6823** | 3.7346 | — | +0.0523 | — |
| 250m | 3.4831 | **3.4299** | 3.4457 | 3.4895 | **+0.0159** | **+0.0597** |

Batch 128 wins at all three tiers, bracketed on both sides at 125M and 250M.
The penalty for exceeding it falls ~28x from 60M to 250M.

### It is batch size, not step count

The obvious alternative — that a minimum number of optimizer steps is what
matters — is contradicted. Indexed by steps, the optimum sits at 2,286 (60M),
4,709 (125M), 9,325 (250M): doubling with each tier while batch stays pinned.
If steps were binding, 250M would peak at batch 512 (2,331 steps, matching
60M's optimum). Instead 512 is 250M's *worst* point.

### Wall clock

Larger batches are faster, so the flattening basin has a practical consequence.

| tier | batch | loss | train s | speedup |
|---|--:|--:|--:|--:|
| 60m | 128 | 4.0143 | 84 | 1.00x |
| 60m | 256 | 4.4638 | 74 | 1.14x |
| 250m | 128 | 3.4299 | 1147 | 1.00x |
| 250m | 256 | 3.4457 | 1035 | **1.11x** |
| 250m | 512 | 3.4895 | 990 | 1.16x |

At 60M, batch 256 buys 14% wall clock for 0.45 nats — never worth it. At 250M
the same trade costs 0.016 nats. The right answer is already scale-dependent
and trending toward the larger batch.

## Reproducibility

The 250M/bs128 cells of Study 2 duplicate Study 1's reseed exactly. All six
runs reproduced **bit-identically** (Δ = 0.00e+00 on validation loss) across
the FLOP-accounting refactor, the `submissions` → `recipes` rename, and
different sessions. Seed variance reported here is therefore a real property of
initialization and data order, not run-to-run nondeterminism.

## What this does not establish

- **Not qualifying numbers.** Every run used the dev profile, so each loss is 8
  probe batches, not canonical validation. Rankings at fixed everything-else
  are valid; the absolute losses are not official results.
- **One horizon.** 5 TPP only. Complete(d)P's `sqrt(m_D)` correction is meant
  to carry the base LR to other horizons; that was not measured. The batch
  result is especially horizon-bound — at 20 TPP every batch gets 4x the steps,
  and the large-batch collapse should move.
- **250M's LR is a two-point row.** Study 2 measured only `2^-7` and `2^-8` at
  250M, so its LR optimum is bracketed only by Study 1.
- **60M's lower edge is unresolved.** {32, 64, 128} are not separated there, so
  "batch 128 at 60M" is a plateau statement.
- **Batch below 32 and above 512 untested**, as is batch 512 at 125M.
- **Two corpora.** `lr_v3` used the 4B prefix and everything since used the 8B
  one; the two are nested prefixes of one shuffled stream but the sampler
  permutes over different capacity, so absolute losses do not cross that line.
  Every "best" above is an arg-min within a single study. Measured incidentally
  at 250M, the 4B-to-8B difference was −0.0005/+0.0022/+0.0001 at `2^-9`,
  `2^-8`, `2^-7` — an order of magnitude below seed noise — but +0.044 at
  `2^-6`, so the effects are not cleanly separable at high LR.
- **500M and 1B are unbracketed.** Only the `2^-8` centre completed.
- **The spike mechanism rests on three seeds.** It explains the v3 result and
  reproduces its margin, but cannot establish the distribution it describes.

## Data

Run records live in `runs/records.jsonl` alongside per-run `metrics.json` and
`training.csv`. Completed studies are moved to `~/rig-run-archive/<date>-<name>/`
with their ledger lines, each carrying a README. Dashboards for individual
sweeps are committed under `docs/reports/`. FLOP figures recorded before commit
`21eab99` come from the former hand-maintained formula and are not comparable
with the traced figures that replaced them — see [FLOPS.md](FLOPS.md).
