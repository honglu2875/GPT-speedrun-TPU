# Hyperparameter transfer under Complete(d)P

Whether hyperparameters tuned once at a small size stay optimal as the model
grows. Two knobs measured on the reference family at 5 TPP: base learning rate
and global batch size.

**Both transfer.** Every configuration measured puts the optimum at base LR
`2^-8` and global batch `128`, across a 4x parameter range (60M → 250M) and a
16x batch range. The one apparent break was a seed artifact with an identified
cause. Note that a fixed *base* LR is a `sqrt(batch)` schedule on the
*effective* LR, since the runtime applies `sqrt(m_B / m_D)` — so the
batch-invariant base LR confirms that correction rather than showing LR is
independent of batch.

The penalty for missing it does *not* transfer: exceeding the optimal batch
costs 0.45 nats at 60M, 0.016 at 250M, and by 500M is no longer resolvable
against seed noise (t=0.86). Where the optimum sits is stable across scale;
what it costs to be wrong is not, so the practical recommendation inverts
before the optimum does.

Supersedes the former `LR_TRANSFER.md`.

## Terms

| Term | Meaning |
|---|---|
| **LR** | learning rate |
| **base LR** | the normalized, scale-free LR knob that is swept. `config.yaml`'s `learning_rate` and the trainer's `--base-learning-rate` are the same field |
| **effective peak LR** | what the optimizer actually applies at the top of the schedule: `base LR * sqrt(m_B / m_D)`. Both factors matter here — sweeping batch moves the effective LR even at a fixed base LR |
| **m_B** | batch multiplier — global batch over the 128 baseline |
| **TPP** | tokens per parameter — training tokens divided by parameter count. Sets the token horizon |
| **µP** | Maximal Update Parameterization. Rescales initialization, LR, and multipliers so activation and update magnitudes stay width-invariant |
| **CompleteP** | µP extended so the rules also hold as *depth* grows |
| **Complete(d)P** | this repo's corrected CompleteP: input-embedding Adam epsilon is `1/m_N`, and the unembedding's forward multiplier is absorbed into init and LR. Rules in [COMPLETEP.md](COMPLETEP.md) |
| **m_N / m_L / m_D** | width, depth, and data multipliers relative to the 60M anchor (`D384`, `L12`) |
| **tier** | a named size rung of the family: 60m, 125m, 250m, 500m, 1b. Measured here: 60m through 500m |
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

### What the gradient spikes actually are

Across all 144 runs, 53 contain a spike above 100x their own median gradient.
They are neither seed-intrinsic nor data-driven:

- **Not a bad seed.** Seed 1337 spikes 90x at 250M/`2^-8` and is the worst of
  its cohort, yet at `2^-7` it is the mildest at 25x and the best. No seed is
  reliably fragile.
- **Not a bad batch.** `ShuffledEpochBatchStream` is seeded `args.seed + 1`, so
  every seed sees a different token order. A specific offending batch would
  therefore surface at different steps per seed. Instead the peak step clusters
  within about ±2 inside a cell — 25/23/24 at 60M bs128 `2^-7`, 31/31/33 at
  250M bs256 `2^-7` — which a shared batch cannot explain when the orders
  differ.

At 60M–250M what predicts them is position on the warmup ramp. Spiking runs
peak at a median **10.5% of their own peak LR**, against 0.4% for non-spiking
runs, and the step moves earlier as the target LR rises (60M bs128: step ~24 at
`2^-7`, ~19 at `2^-6`). So this is an early-training instability that every run
passes through, whose severity is set by how high the LR has ramped by the time
it arrives — and with `grad_clip: 0.0`, whatever lands goes into the weights at
full size.

**500M does not fit that story.** Both of its spikes land 25–50% into the run,
long past warmup, at 68–91% of peak LR. The 10.5% median above is a small-tier
number that those two runs sit far outside of. Whether this is the same
instability displaced by scale or a second mechanism is open on two runs; the
per-run index is in [GRADIENT_SPIKES.md](GRADIENT_SPIKES.md).

That reframes the seed variance throughout this note. It is not luck in the
initialization so much as luck in how hard one unavoidable fragile phase hits,
which is why the spread grows with LR and why it is largest exactly at the
optimum.

## Study 2 — batch size x learning rate

Full grid, three seeds per cell, 144 runs across four tiers.

**What "the LR optimum does not move" means here.** The runtime already applies
`sqrt(m_B / m_D)`, so holding the base LR fixed *is* a `sqrt(batch)` schedule on
the effective LR. At 60M:

| batch | base LR | effective peak LR |
|---|--:|--:|
| 32 | 0.00390625 | 0.00195312 |
| 128 | 0.00390625 | 0.00390625 |
| 512 | 0.00390625 | 0.00781250 |

A 4x effective swing across the 16x batch range. So the finding below —
`2^-8` optimal at every batch — is not evidence that LR is independent of
batch. It is evidence that **Complete(d)P's `sqrt(m_B)` rule is the right
correction**, because applying it makes the remaining knob batch-invariant.
Sweeping base LR at each batch is what tests the rule rather than assuming it.

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

### 250M — 4 batches × 3 LRs

| batch | steps | 2^-7 | 2^-8 | 2^-9 |
|---|--:|--:|--:|--:|
| 64 | 18,650 | 3.4872 | **3.4831** | 3.5113 |
| 128 | 9,325 | 3.4437 | **3.4299** | *3.4740* |
| 256 | 4,662 | 3.4897 | **3.4457** | 3.5099 |
| 512 | 2,331 | 3.6029 | **3.4895** | 3.5631 |

*Italic* is Study 1's measurement, included because those runs reproduce
bit-identically (see [Reproducibility](#reproducibility)) and are therefore the
same experiment, not a comparable one.

`2^-8` wins at every batch and every optimum is now bracketed on both sides:
`2^-9` is worse by t=+12.3 / +56.0 / +60.8 at bs64 / bs256 / bs512. Across
batches, bs64 vs bs128 t=+9.41 · bs128 vs bs256 t=−2.89 · bs256 vs bs512
t=−34.62, all separated. `2^-7` vs `2^-8` at bs128 remains t=+0.71, **not
separated** — independently reproducing Study 1's finding.

### 500M — 2 batches at the optimal LR

| batch | steps | 2^-8 |
|---|--:|--:|
| 128 | 19,173 | **3.2175** ±0.0042 |
| 256 | 9,586 | 3.2354 ±0.0204 |

Only `2^-8` was measured, on the strength of it winning every cell at every
smaller tier; the LR optimum at 500M is therefore assumed rather than
bracketed. Throughput was 620K tok/s at bs256 against 489K on a v6e-8, which
is chip count (16 v4 vs 8 v6e), not per-chip speed.

### Learning rate gets less forgiving as batch grows

The cost of being one octave off the optimum, at 250M:

| batch | 2^-8 | +1 octave | −1 octave | worst |
|---|--:|--:|--:|--:|
| 64 | 3.4831 | +0.0041 | +0.0282 | **+0.028** |
| 128 | 3.4299 | +0.0139 | +0.0441 | **+0.044** |
| 256 | 3.4457 | +0.0440 | +0.0642 | **+0.064** |
| 512 | 3.4895 | +0.1134 | +0.0735 | **+0.113** |

A 4× increase from batch 64 to 512, monotonic. This runs *opposite* to the
batch-size penalty, which shrinks with model scale. Both optima transfer, but
tuning the learning rate matters more at large batch while tuning the batch
matters less at large model. The practical consequence: adopting a larger batch
to buy wall clock also narrows the LR window it has to be paired with.

### The optimum transfers; the penalty for missing it does not

At `2^-8`:

| tier | bs64 | **bs128** | bs256 | bs512 | Δ(256) | t |
|---|--:|--:|--:|--:|--:|--:|
| 60m | 4.0264 | **4.0143** | 4.4638 | 5.2473 | **+0.4495** | +15.25 |
| 125m | 3.7340 | **3.6823** | 3.7346 | — | +0.0523 | +7.06 |
| 250m | 3.4831 | **3.4299** | 3.4457 | 3.4895 | +0.0159 | +2.89 |
| 500m | — | **3.2175** | 3.2354 | — | +0.0179 | **+0.86** |

Batch 128 is never beaten. But the cost of exceeding it collapses — 0.45
nats at 60M, 0.016 at 250M — and by 500M it is no longer resolvable:
`t = 0.86`, well inside noise.

The 500M point estimate is not smaller than 250M's; what changed is the
spread. One seed carries it:

| 500M bs256 | loss | peak/median gradient |
|---|--:|--:|
| seed 1337 | 3.2124 | 21x |
| seed 1338 | 3.2176 | 21x |
| **seed 1339** | **3.2761** | **1011x** |

Two of three seeds sit level with bs128; the third took a 1011x gradient
spike and lost 0.06 nats. This is the same unclipped-gradient mechanism
identified at 250M in Study 1, now reproducing at a third scale, and it is
why the honest statement is "indistinguishable" rather than "equal" —
n=3 cannot separate a 0.018 difference against that variance.

**Practically the recommendation inverts before the optimum does.** At 60M,
batch 256 costs 0.45 nats to save 14% wall clock: never worth it. At 500M it
costs nothing measurable and finishes ~10% sooner. Batch 128 remains the
safe default; batch 256 becomes defensible at 500M and above, on the
understanding that it is a wash on loss rather than an improvement.

### It is batch size, not step count

The obvious alternative — that a minimum number of optimizer steps is what
matters — is contradicted. Indexed by steps, the optimum sits at 2,286 (60M),
4,709 (125M), 9,325 (250M): doubling with each tier while batch stays pinned.
If steps were binding, 250M would peak at batch 512 (2,331 steps, matching
60M's optimum). Instead 512 is 250M's *worst* point.

### Wall clock

Larger batches are faster, so a shrinking penalty has a practical consequence.

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
- **250M's `2^-6` shoulder is untested in Study 2.** The optimum is bracketed
  by `2^-7` and `2^-9` at every batch, but the far upper shoulder rests on
  Study 1's batch-128 measurement alone.
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
- **500M's LR is assumed, not measured.** Only `2^-8` ran there, and only at
  batches 128 and 256. Its batch optimum is bracketed on neither side.
- **1B is untouched.** Only Study 1's single `2^-8` centre exists.
- **The 500M batch comparison is noise-limited.** A 0.018 difference against a
  seed spread that one 1011x gradient spike widened to ±0.020 cannot be
  resolved at n=3. "Indistinguishable" is the claim; "equal" is not.
- **The spike mechanism rests on three seeds.** It explains the v3 result and
  reproduces its margin, but cannot establish the distribution it describes.

## Data

Run records live in `runs/records.jsonl` alongside per-run `metrics.json` and
`training.csv`. Completed studies are moved to `~/rig-run-archive/<date>-<name>/`
with their ledger lines, each carrying a README. Dashboards for individual
sweeps are committed under `docs/reports/`. FLOP figures recorded before commit
`21eab99` come from the former hand-maintained formula and are not comparable
with the traced figures that replaced them — see [FLOPS.md](FLOPS.md).
