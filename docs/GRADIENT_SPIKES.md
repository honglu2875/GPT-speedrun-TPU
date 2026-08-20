# Gradient spikes: an index of reproducible cases

Working notes for studying gradient spikes, not a result. Every run below is
recorded, and training is bit-reproducible, so each row can be regenerated
exactly on demand.

Of 144 runs across 60M–500M, **53 contain a spike of 100x or more** over that
run's own median gradient norm. `grad_clip: 0.0` in every profile, so whatever
spikes goes into the weights at full size.

## Two populations, not one

They do not share a mechanism, and the distinction matters before drawing any
conclusion from the tables below.

| | when | LR at the spike | tiers |
|---|---|--:|---|
| **warmup** | step 9–160, during the ramp | ~10% of peak | 60M, 125M, 250M |
| **mid-training** | 25–50% through the run | 68–91% of peak | 500M |

Every 60M/125M/250M spike is a warmup event: the peak lands early, and the step
moves *earlier* as the target LR rises (60M bs128: step ~24 at `2^-7`, ~19 at
`2^-6`). Both 500M spikes are the opposite — well past warmup, deep in cosine
decay, at near-peak LR.

Whether these are one phenomenon at different scales or two unrelated ones is
open. The 500M evidence is two runs.

## What they are not

- **Not a fragile seed.** Seed 1337 spikes 90x at 250M/`2^-8` and is the worst
  of its cohort; at `2^-7` it is the mildest at 25x and the best.
- **Not a bad batch.** `ShuffledEpochBatchStream` is seeded `args.seed + 1`, so
  each seed sees a different token order. A specific offending batch would
  surface at different steps per seed; instead the peak step clusters within
  about ±2 inside a cell (25/23/24 at 60M bs128 `2^-7`).

## Dataset

Every run below used the same corpus.

| | |
|---|---|
| dataset id | `fineweb-8b-gpt2` |
| manifest | `data/manifests/fineweb-scaled-gpt2/8B.json` |
| manifest sha256 | `06dea754546bdfec4c77fba9afd4540c93bd2b0dc8cd1f958656a74f44b63034` |
| recorded dataset sha256 | `64473be07689abbf926027ef96fa7b679206510e29bae19403d94890c4fa1a7c` |
| tokenizer | `gpt2`, vocab 50,304 |
| shards | 80 (79 train + 1 validation), 7.9B train tokens |
| cache | `/dev/shm/.speedrun-cache/fineweb-scaled/8B` |

Common to every row: sequence length 1024, `dev` profile, 5 TPP, `grad_clip
0.0`, `warmup_ratio 0.1`, cosine decay to `min_lr_ratio 0.1`, on v4-32.

## Reproducing a case

Training is deterministic — the 250M/bs128 cells reproduced **bit-identically**
across a code refactor and a package rename — so re-running a row lands the
spike on exactly the listed step.

```bash
uv --cache-dir /tmp/uv-cache run --frozen --no-sync rig run reference \
  --cluster v4-32 --profile dev --tier 250m \
  --tokens-per-parameter 5 --batch-size 128 \
  --base-learning-rate 0.0078125 --seed 1339 \
  --name "spike-250m-bs128-lr2e-7-s1339" \
  --checkpoint-policy none --timeout 14400
```

That is the 250M / bs128 / `2^-7` / seed 1339 row: a 1633x spike at step 60.

LR values: `2^-6` = `0.015625`, `2^-7` = `0.0078125`, `2^-8` = `0.00390625`,
`2^-9` = `0.001953125`, `2^-10` = `0.0009765625`.

### Stopping at the spike

Raw `--steps` and `--train-tokens` horizons are intentionally not available.
Use `--stop-after-step N`. It leaves `steps`, `warmup_steps`, and `m_D` resolved
from the full fixed-TPP horizon and simply exits after step `N`, so the
trajectory is the untruncated run's prefix, step for step.

```bash
uv --cache-dir /tmp/uv-cache run --frozen --no-sync rig run reference \
  --cluster v4-32 --profile dev --tier 250m \
  --tokens-per-parameter 5 --batch-size 128 \
  --base-learning-rate 0.0078125 --seed 1339 \
  --stop-after-step 61 \
  --name "spike-250m-bs128-lr2e-7-s1339-at-spike" \
  --checkpoint-policy none --timeout 3600
```

Stop one step *past* the listed step to keep the post-spike update: the row
records the step whose gradient norm peaked, and its effect on the weights only
appears in the next optimizer state. Diagnostics fire on the stopping step, so
`diagnostics.riglog` carries the per-layer picture there whatever the cadence.
Drop `--checkpoint-policy none` when you want the weights at that point.

Paying for the full run is also fine for most rows: the cheapest 1882x spike
costs **83 seconds** end to end (60M / bs128 / `2^-7` / seed 1337, spike at step
25). Only 500M is expensive at ~4,500s, and that is where truncation earns its
keep — its spikes sit 25–50% into the run, so stopping there still saves half.

## Cheapest reproduction per tier

| tier | config | spike | ratio | cost |
|---|---|--:|--:|--:|
| 60M | bs128 `2^-7` seed 1337 | step 25 | **1882x** | 83s |
| 125M | bs256 `2^-7` seed 1338 | step 24 | 1878x | 271s |
| 250M | bs512 `2^-7` seed 1339 | step 19 | 1310x | 991s |
| 500M | bs256 `2^-8` seed 1339 | step 4444 | 1011x | 4052s |

## 60M — 12 layers, d 384

| batch | LR | seed | spike step | of total | ratio | peak grad | median | val loss | run time |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 32 | 2^-6 | 1337 | **68** | 9,143 | 110x | 11.0 | 0.100 | 4.0980 | 155s |
| 32 | 2^-6 | 1338 | **64** | 9,143 | 2547x | 259.6 | 0.102 | 4.1310 | 155s |
| 32 | 2^-6 | 1339 | **43** | 9,143 | 252x | 25.7 | 0.102 | 4.0954 | 156s |
| 32 | 2^-7 | 1337 | **88** | 9,143 | 796x | 113.4 | 0.142 | 4.1194 | 155s |
| 32 | 2^-7 | 1338 | **83** | 9,143 | 1587x | 237.7 | 0.150 | 4.1568 | 156s |
| 32 | 2^-7 | 1339 | **83** | 9,143 | 203x | 29.2 | 0.144 | 4.1059 | 155s |
| 32 | 2^-8 | 1337 | **111** | 9,143 | 711x | 165.8 | 0.233 | 4.1215 | 155s |
| 64 | 2^-6 | 1337 | **36** | 4,571 | 1743x | 150.8 | 0.087 | 4.1293 | 108s |
| 64 | 2^-6 | 1338 | **32** | 4,571 | 691x | 59.5 | 0.086 | 4.1124 | 109s |
| 64 | 2^-6 | 1339 | **37** | 4,571 | 1098x | 93.6 | 0.085 | 4.0902 | 108s |
| 64 | 2^-7 | 1337 | **46** | 4,571 | 114x | 15.6 | 0.137 | 4.0730 | 108s |
| 64 | 2^-7 | 1338 | **51** | 4,571 | 111x | 13.2 | 0.118 | 4.1019 | 108s |
| 64 | 2^-8 | 1339 | **69** | 4,571 | 115x | 22.7 | 0.197 | 4.0437 | 108s |
| 128 | 2^-6 | 1337 | **20** | 2,286 | 205x | 16.9 | 0.082 | 4.0956 | 84s |
| 128 | 2^-6 | 1338 | **18** | 2,286 | 1551x | 132.2 | 0.085 | 4.1239 | 83s |
| 128 | 2^-6 | 1339 | **19** | 2,286 | 871x | 70.2 | 0.081 | 4.1289 | 83s |
| 128 | 2^-7 | 1337 | **25** | 2,286 | 1882x | 218.8 | 0.116 | 4.1611 | 83s |
| 128 | 2^-7 | 1338 | **23** | 2,286 | 113x | 12.5 | 0.111 | 4.1285 | 83s |
| 128 | 2^-7 | 1339 | **24** | 2,286 | 1613x | 186.0 | 0.115 | 4.1510 | 85s |
| 256 | 2^-6 | 1337 | **13** | 1,143 | 2091x | 237.5 | 0.114 | 4.4789 | 73s |
| 256 | 2^-6 | 1338 | **15** | 1,143 | 406x | 44.7 | 0.110 | 4.4602 | 75s |
| 256 | 2^-7 | 1337 | **15** | 1,143 | 988x | 116.5 | 0.118 | 4.4571 | 73s |
| 256 | 2^-7 | 1339 | **14** | 1,143 | 1719x | 183.7 | 0.107 | 4.5289 | 76s |
| 256 | 2^-8 | 1337 | **24** | 1,143 | 119x | 18.5 | 0.156 | 4.4533 | 73s |
| 256 | 2^-8 | 1338 | **21** | 1,143 | 246x | 38.0 | 0.154 | 4.5188 | 73s |
| 512 | 2^-6 | 1337 | **9** | 571 | 575x | 93.1 | 0.162 | 5.4811 | 74s |
| 512 | 2^-6 | 1338 | **10** | 571 | 255x | 44.5 | 0.174 | 5.5766 | 74s |
| 512 | 2^-7 | 1337 | **13** | 571 | 378x | 54.0 | 0.143 | 5.3224 | 74s |
| 512 | 2^-7 | 1338 | **12** | 571 | 110x | 15.1 | 0.137 | 5.3438 | 75s |
| 512 | 2^-7 | 1339 | **16** | 571 | 386x | 60.5 | 0.157 | 5.2230 | 76s |
| 512 | 2^-8 | 1339 | **11** | 571 | 261x | 53.3 | 0.204 | 5.2180 | 74s |
| 512 | 2^-9 | 1338 | **19** | 571 | 124x | 29.7 | 0.239 | 5.6108 | 75s |
| 512 | 2^-10 | 1337 | **19** | 571 | 127x | 37.6 | 0.295 | 5.7509 | 74s |
| 512 | 2^-10 | 1338 | **24** | 571 | 507x | 153.2 | 0.302 | 5.6758 | 74s |
| 512 | 2^-10 | 1339 | **26** | 571 | 264x | 85.3 | 0.323 | 5.5785 | 74s |

Note the bs512 rows: spikes appear at *every* LR including `2^-10`, the
smallest measured. At 571 total steps warmup is only 57, so this tier reaches
its peak LR almost immediately.

## 125M — 12 layers, d 640

| batch | LR | seed | spike step | of total | ratio | peak grad | median | val loss | run time |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 64 | 2^-7 | 1338 | **81** | 9,419 | 151x | 21.8 | 0.145 | 3.7588 | 390s |
| 64 | 2^-7 | 1339 | **86** | 9,419 | 380x | 54.8 | 0.144 | 3.7972 | 390s |
| 64 | 2^-8 | 1338 | **104** | 9,419 | 116x | 26.0 | 0.224 | 3.7275 | 391s |
| 64 | 2^-8 | 1339 | **108** | 9,419 | 317x | 71.2 | 0.224 | 3.7404 | 390s |
| 128 | 2^-7 | 1337 | **38** | 4,709 | 164x | 22.0 | 0.134 | 3.7230 | 309s |
| 128 | 2^-7 | 1339 | **46** | 4,709 | 139x | 18.6 | 0.133 | 3.7391 | 309s |
| 256 | 2^-7 | 1338 | **24** | 2,355 | 1878x | 224.2 | 0.119 | 3.8582 | 271s |
| 256 | 2^-7 | 1339 | **25** | 2,355 | 736x | 88.7 | 0.121 | 3.8515 | 271s |

## 250M — 16 layers, d 896

| batch | LR | seed | spike step | of total | ratio | peak grad | median | val loss | run time |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 64 | 2^-7 | 1339 | **110** | 18,650 | 721x | 122.2 | 0.169 | 3.5378 | 1411s |
| 128 | 2^-7 | 1338 | **55** | 9,325 | 125x | 18.4 | 0.148 | 3.4332 | 1147s |
| 128 | 2^-7 | 1339 | **60** | 9,325 | 1633x | 232.8 | 0.143 | 3.4803 | 1147s |
| 256 | 2^-7 | 1338 | **31** | 4,662 | 1301x | 163.7 | 0.126 | 3.5116 | 1034s |
| 256 | 2^-7 | 1339 | **33** | 4,662 | 1300x | 168.1 | 0.129 | 3.5204 | 1035s |
| 512 | 2^-7 | 1337 | **27** | 2,331 | 149x | 18.6 | 0.125 | 3.5884 | 993s |
| 512 | 2^-7 | 1338 | **18** | 2,331 | 832x | 97.0 | 0.117 | 3.6145 | 990s |
| 512 | 2^-7 | 1339 | **19** | 2,331 | 1310x | 161.0 | 0.123 | 3.6058 | 991s |

Every 250M spike is at `2^-7`, one octave above the optimum. At `2^-8` and
`2^-9` the largest ratio in the tier is 90x.

## 500M — 19 layers, d 1280

| batch | LR | seed | spike step | of total | ratio | peak grad | median | val loss | run time |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 2^-8 | 1339 | **5444** | 19,173 | 267x | 59.1 | 0.222 | 3.2259 | 4480s |
| 256 | 2^-8 | 1339 | **4444** | 9,586 | 1011x | 185.9 | 0.184 | 3.2761 | 4052s |

Both are mid-training, not warmup, and both are seed 1339 at the *optimal* LR —
the only tier where spikes appear at `2^-8`. The bs256 case cost 0.06 nats and
is what makes the 500M batch comparison noise-limited.

## Caveats

- Ratios are peak over that run's own median, so they are not comparable in
  absolute gradient magnitude across tiers; the `peak grad` and `median`
  columns are given for that.
- The training log records every step, so the listed step is exact rather than
  binned. Three 250M bs512 `2^-7` runs have truncated diagnostics
  (starting at step 1920) from an unrelated tooling fault; their training curves
  and the numbers here are unaffected.
- Runs live in `~/rig-run-archive/2026-08-15-batch-sweep-60M/` and
  `~/rig-run-archive/2026-08-17-batch-sweep-125M-250M-500M/`, with their ledger
  lines. Nothing here depends on those directories surviving: every row can be
  regenerated from the command above.
