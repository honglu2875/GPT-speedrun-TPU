# Reports — what each one shows and how to reproduce it

Every dashboard in this directory, where its runs came from, and the command
that produces that set of runs. The commands are **demonstrative**: they use the
current CLI and reproduce the *design*, not the exact invocation that ran at the
time (several predate flags that now exist, and one predates the artifact format
entirely). Seeds, tiers, and grids are exact.

Two things are worth knowing before reading any of these.

**The run archives are not in this repository.** They live outside it, under
`~/rig-run-archive/`, and are not pushed anywhere. That is why these HTML files
are committed: for most of these studies the rendered dashboard is the only copy
of the curves that travels with the repo. Treat them as records, not as build
output.

**Two artifact formats.** Runs recorded before commit `75f0b22` wrote
`training.csv` and `diagnostics.csv` in long form; everything after writes the
packed `.riglog` format. [`rig/legacy.py`](../../rig/legacy.py) converts a
long-form *training curve* into the packed format, so those runs can be replotted
today. It does **not** convert diagnostics, so a report rebuilt from a legacy
archive today has loss curves but no per-layer statistics — see
[Rebuilding](#rebuilding) below.

## Contents

| report | runs | tier(s) | what varies |
|---|--:|---|---|
| [batch-size-sweep-60M](batch-size-sweep-60M.html) | 75 | 60M | batch x LR x seed |
| [batch-size-sweep-125M](batch-size-sweep-125M.html) | 27 | 125M | batch x LR x seed |
| [lr-sweep-125M](lr-sweep-125M.html) | 27 | 125M | same runs, LR view |
| [batch-size-sweep-250M](batch-size-sweep-250M.html) | 36 | 250M | batch x LR x seed |
| [batch-size-sweep-125M-250M-500M](batch-size-sweep-125M-250M-500M.html) | 69 | 125M/250M/500M | all three at once |
| [batch-size-sweep-500M](batch-size-sweep-500M.html) | 12 | 500M | batch x LR x seed, 5 and 20 TPP |
| [500M-20tpp-v6e](500M-20tpp-v6e.html) | 3 | 500M | LR, 20 TPP |
| [500M-20tpp-diagnostics](500M-20tpp-diagnostics.html) | 3 | 500M | per-layer diagnostics |
| [3-seed-gradient-spike](3-seed-gradient-spike.html) | 12 | 250M | LR x seed |
| [8k-lr-sweep-60M](8k-lr-sweep-60M.html) | 15 | 60M | LR x seed at 8k context |
| [transfer-charts](transfer-charts.html) | — | — | derived figures, not a run dashboard |

---

## batch-size-sweep-60M.html

75 runs: **5 batches x 5 learning rates x 3 seeds** at the 60M tier, 5 tokens
per parameter, 1,024 context. The widest grid in the collection, and the one
study 2 leans on hardest.

Source archive: `2026-08-15-batch-sweep-60M` (runs dated 2026-08-15).

```bash
for bs in 32 64 128 256 512; do
  for lr in 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625; do
    for seed in 1337 1338 1339; do
      rig run reference --cluster v4-32 --profile dev --track open \
        --tier 60m --tokens-per-parameter 5 \
        --study-batch-size "$bs" --base-learning-rate "$lr" --seed "$seed" \
        --name "60m-bs${bs}-lr${lr}-s${seed}"
    done
  done
done
```

## batch-size-sweep-125M.html

27 runs: **3 batches (64/128/256) x 3 learning rates (2^-7/2^-8/2^-9) x 3
seeds** at 125M, 5 TPP, 1,024 context.

Source archive: `2026-08-17-batch-sweep-125M-250M-500M`, the 125M subset (runs
dated 2026-08-15 — the archive is named for when it was *archived*, not when the
runs happened).

```bash
for bs in 64 128 256; do
  for lr in 0.0078125 0.00390625 0.001953125; do
    for seed in 1337 1338 1339; do
      rig run reference --cluster v4-32 --profile dev --track open \
        --tier 125m --tokens-per-parameter 5 \
        --study-batch-size "$bs" --base-learning-rate "$lr" --seed "$seed" \
        --name "125m-bs${bs}-lr${lr}-s${seed}"
    done
  done
done
```

## lr-sweep-125M.html

**The same 27 runs as `batch-size-sweep-125M.html`**, rendered earlier and read
as a learning-rate comparison rather than a batch one. It is a second view, not
a second experiment — the grid is a batch x LR product, so either axis can be
the subject. Reproduce it with the loop above.

Kept because it is the artifact study 1 was written against. If you want one
file for the 125M tier, prefer `batch-size-sweep-125M.html`.

## batch-size-sweep-250M.html

36 runs: **4 batches (64/128/256/512) x 3 learning rates x 3 seeds** at 250M,
5 TPP, 1,024 context.

Source archive: `2026-08-17-batch-sweep-125M-250M-500M`, the 250M subset (runs
dated 2026-08-15 through 2026-08-17).

```bash
for bs in 64 128 256 512; do
  for lr in 0.0078125 0.00390625 0.001953125; do
    for seed in 1337 1338 1339; do
      rig run reference --cluster v4-32 --profile dev --track open \
        --tier 250m --tokens-per-parameter 5 \
        --study-batch-size "$bs" --base-learning-rate "$lr" --seed "$seed" \
        --name "250m-bs${bs}-lr${lr}-s${seed}"
    done
  done
done
```

## batch-size-sweep-125M-250M-500M.html

All 69 runs of the `2026-08-17-batch-sweep-125M-250M-500M` archive in one
dashboard: the 27 above at 125M, the 36 above at 250M, and 6 at 500M
(2 batches x 3 seeds at `2^-8`, 5 TPP).

**Largely superseded.** Its 125M and 250M halves are the same runs as the two
per-tier reports, and its 500M half is now inside
`batch-size-sweep-500M.html` alongside six further runs. It is still the only
file that shows all three tiers on shared axes.

## batch-size-sweep-500M.html

12 runs, and the only report assembled from more than one source: six at 5 TPP
from `2026-08-17-batch-sweep-125M-250M-500M`, three at 20 TPP from
`2026-08-18-500m-20tpp-v6e`, and three at 20 TPP from `runs/`. The first nine
are long-form CSV converted by `rig/legacy.py`; the last three were recorded
packed.

Run names carry the token budget (`500m-5tpp-...` against `500m-20tpp-...`)
because the two budgets are different experiments whose losses are not
comparable to each other.

This is study 3's dashboard. Per-layer diagnostics are deliberately excluded:
only three of the twelve runs have them in a readable format, and a report where
a quarter of the runs carry extra series invites reading a gap in coverage as a
gap in behaviour. They are in `500M-20tpp-diagnostics.html` instead.

```bash
# 5 TPP arm, batch bracket at the optimal LR
for bs in 128 256; do
  for seed in 1337 1338 1339; do
    rig run reference --cluster v4-32 --profile dev --track open \
      --tier 500m --tokens-per-parameter 5 \
      --study-batch-size "$bs" --base-learning-rate 0.00390625 --seed "$seed" \
      --name "500m-5tpp-bs${bs}-s${seed}"
  done
done

# 20 TPP arm on the v6e-8: batch bracket, then the LR bracket at batch 128
for bs in 64 128 256; do
  rig run reference --cluster v6e-8 --profile dev --track open \
    --tier 500m --tokens-per-parameter 20 --checkpoint-policy none \
    --study-batch-size "$bs" --base-learning-rate 0.00390625 --seed 1337 \
    --name "500m-20tpp-bs${bs}-s1337"
done
for lr in 0.0078125 0.001953125; do
  rig run reference --cluster v6e-8 --profile dev --track open \
    --tier 500m --tokens-per-parameter 20 --checkpoint-policy none \
    --study-batch-size 128 --base-learning-rate "$lr" --seed 1337 \
    --name "500m-20tpp-bs128-lr${lr}-s1337"
done
```

## 500M-20tpp-v6e.html

The three 20-TPP runs of `2026-08-18-500m-20tpp-v6e` on their own, rendered
before the packed-format series began.

**Superseded** by `batch-size-sweep-500M.html`, which contains these three plus
nine more. Kept only as the record that existed when study 3 was first written.

## 500M-20tpp-diagnostics.html

The three 500M runs recorded in the packed format, **with** their per-layer
parameter, gradient, and update statistics — batch 128 at `2^-8` (seeds 1337 and
1338) and batch 64 at `2^-8` (seed 1337).

These are the only 500M runs whose diagnostics are readable by the current
tooling, which is the whole reason this file is separate from the sweep report.
At 68 MB it is by far the largest file here; the diagnostics are ~99% of that.

```bash
for bs in 64 128; do
  rig run reference --cluster v6e-8 --profile dev --track open \
    --tier 500m --tokens-per-parameter 20 --checkpoint-policy none \
    --study-batch-size "$bs" --base-learning-rate 0.00390625 --seed 1337 \
    --name "500m-20tpp-bs${bs}-s1337"
done
```

## 3-seed-gradient-spike.html

12 runs: **4 learning rates x 3 seeds** at 250M, batch 128, 5 TPP. Built to
settle the 250M reseed described in study 1, and the evidence base for
[GRADIENT_SPIKES.md](../GRADIENT_SPIKES.md).

Source archive: `2026-08-15-lr-transfer-5tpp` (runs dated 2026-08-14/15).

```bash
for lr in 0.015625 0.0078125 0.00390625 0.001953125; do
  for seed in 1337 1338 1339; do
    rig run reference --cluster v4-32 --profile dev --track open \
      --tier 250m --tokens-per-parameter 5 \
      --study-batch-size 128 --base-learning-rate "$lr" --seed "$seed" \
      --name "250m-lr${lr}-s${seed}"
  done
done
```

## 8k-lr-sweep-60M.html

15 runs: **5 learning rates x 3 seeds** of [`reference_8k`](../../recipes/reference_8k/)
— 60M at 8,192 context with document masking, batch 16 so that tokens per step
and step count match the 1,024-context ladder exactly. This is study 4.

These runs are still live in `runs/`, not archived, and were recorded packed.

```bash
for lr in 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625; do
  for seed in 1337 1338 1339; do
    rig run reference_8k --cluster v4-32 --profile dev --track open \
      --tier 60m --tokens-per-parameter 5 \
      --base-learning-rate "$lr" --seed "$seed" --checkpoint-policy none \
      --name "60m-bs16-lr${lr}-s${seed}"
  done
done
rig report --runs runs --select reference_8k --output docs/reports/8k-lr-sweep-60M.html
```

## transfer-charts.html

Not a run dashboard. Derived figures built from recorded results by
[`make_transfer_charts.py`](make_transfer_charts.py), which is committed beside
it. Regenerate with:

```bash
uv run --frozen --no-sync python docs/reports/make_transfer_charts.py
```

---

## Rebuilding

`rig report --runs <dir> --output <file>` renders any directory of runs, and
`--select <regex>` narrows it to one family. A legacy archive needs its training
curves converted first — `rig/legacy.py` does that without touching the
originals.

Two limits are worth stating plainly, because both are easy to trip over:

- **Diagnostics do not convert.** Only training curves do. A report rebuilt
  today from a pre-`75f0b22` archive will have loss and validation curves but no
  per-layer series, so it is **not** byte- or content-equivalent to the file
  committed here. The committed HTML remains the only complete rendering of
  those diagnostics. Do not overwrite one with a rebuild expecting a smaller
  file to be the same file.
- **The archives are not in this repository.** Anything rebuilt from
  `~/rig-run-archive/` cannot be rebuilt by someone who only has the repo. This
  README, the committed HTML, and the tables in
  [HYPERPARAMETER_TRANSFER.md](../HYPERPARAMETER_TRANSFER.md) are what survive
  without it.
