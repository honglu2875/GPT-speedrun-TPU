# Reports — what each one shows and how to rebuild it

Every dashboard here, the runs behind it, and the command that reproduces it.
Commands are **demonstrative**: they use the current CLI and reproduce the
*design*, not the exact invocation from the time. Seeds, tiers, and grids are
exact.

## The logs live on HuggingFace

**[huggingface.co/datasets/quintic/rig-logs](https://huggingface.co/datasets/quintic/rig-logs)**
— 177 runs across six studies, laid out as `<study>/<run-name>/`, at full
recorded resolution. That is the archive of record; its
[dataset card](https://huggingface.co/datasets/quintic/rig-logs/blob/main/README.md)
is a copy of this file.

The dashboards committed here are **summaries** of those logs, thinned so they
stay portable. Nothing in them is a substitute for the logs: they are one
rendering at one fidelity, and a thinned curve is indistinguishable on screen
from a complete one. When a number matters, read it from the `.riglog`.

```python
from huggingface_hub import hf_hub_download
from rig import logpack

path = hf_hub_download(
    "quintic/rig-logs",
    "batch-sweep-60M/60m-5tpp-bs128-lr2e-8-s1337/training.riglog",
    repo_type="dataset",
)
log = logpack.read_log(path)
log.series("train_loss")          # every optimizer step
```

## What "summary" means here

Every series is thinned to at most **1,440 points**. Per-layer diagnostic
charts additionally keep a bounded number of step frames — 8 for most studies,
and more for the two where the per-layer behaviour is the subject rather than a
by-product:

| report | curve points | layer frames | size |
|---|--:|--:|--:|
| batch-size-sweep-60M | 1,440 | 400 | 44.3 MB |
| batch-size-sweep-500M | 1,440 | 1,440 | 44.3 MB |
| batch-size-sweep-250M | 1,440 | 8 | 15.4 MB |
| lr-batch-sweep-125M | 1,440 | 8 | 8.2 MB |
| 3-seed-gradient-spike | 1,440 | 8 | 6.6 MB |
| 8k-lr-sweep-60M | 1,440 | 8 | 2.4 MB |
| moe-lr-sweep-8k | 1,440 | 8 | 7.2 MB |

The two large ones carry layer detail because gradient spikes are visible in
it, and studying them is the point. This is deliberate discretion, not a
default: keep it to a couple of files so the repository stays clonable.

Charts resample against the visible span as you zoom, keeping each pixel
bucket's minimum and maximum rather than one representative point — so a spike
inside the embedded data stays visible at every zoom level. It cannot recover a
sample that thinning already dropped.

Charts are per-metric, and a metric no selected run recorded is not drawn at
all — the panel is hidden rather than left as an empty frame. Routed runs
record routing series a dense run never will, so most reports carry charts that
do not apply to part of the selection, and a grid of empty frames would bury
the ones that do.

Which metrics get charted is a declared list in `rig/report.py`, separate from
the metric registry, because how a quantity should be drawn is a judgement the
registry cannot make. Everything so far is a line against the time axis; a
distribution rather than a scalar — a routing histogram, say — wants bars
against expert index and would arrive as a new chart kind rather than being
bent into a timeline.

## The study browser

[`study-browser.html`](study-browser.html) carries no data at all — 53 KB. It
lists the studies, renders each one's card from the dataset, and fetches only
that study's overview (0.05–0.30 MB) when you pick one. The full logs are a
second, separately labelled click that states the size before it starts:
6.4 MB for the 8k sweep, 138 MB for the 500M one. Nothing downloads on load.

Everything it fetches is an ordinary report payload, so the page never needs to
understand the packed log format — the two only have to agree about JSON.

## Contents

| report | runs | tier(s) | what varies | logs |
|---|--:|---|---|---|
| [batch-size-sweep-60M](batch-size-sweep-60M.html) | 75 | 60M | batch × LR × seed | [`batch-sweep-60M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/batch-sweep-60M) |
| [lr-batch-sweep-125M](lr-batch-sweep-125M.html) | 27 | 125M | batch × LR × seed | [`lr-batch-sweep-125M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/lr-batch-sweep-125M) |
| [batch-size-sweep-250M](batch-size-sweep-250M.html) | 36 | 250M | batch × LR × seed | [`batch-sweep-250M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/batch-sweep-250M) |
| [batch-size-sweep-500M](batch-size-sweep-500M.html) | 12 | 500M | batch × LR × seed, 5 and 20 TPP | [`batch-sweep-500M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/batch-sweep-500M) |
| [3-seed-gradient-spike](3-seed-gradient-spike.html) | 12 | 250M | LR × seed | [`lr-transfer-250M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/lr-transfer-250M) |
| [8k-lr-sweep-60M](8k-lr-sweep-60M.html) | 15 | 60M | LR × seed at 8k context | [`lr-sweep-8k-60M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/lr-sweep-8k-60M) |
| [moe-lr-sweep-8k](moe-lr-sweep-8k.html) | 18 | 60M/125M | LR × seed, top-2 of 8 experts | [`moe-lr-sweep-8k`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/moe-lr-sweep-8k) |
| [transfer-charts](transfer-charts.html) | — | — | derived figures, not a run dashboard | — |

Each study also carries a `snapshot.json.gz` (loss curves only, 0.05–0.30 MB)
and, for the two above, a `snapshot-diagnostics.json.gz` (1.0–3.5 MB). These
are what the study browser loads before you ask it for anything larger.

---

## batch-size-sweep-60M.html

75 runs: **5 batches × 5 learning rates × 3 seeds** at 60M, 5 tokens per
parameter, 1,024 context. The widest grid here, and what study 2 leans on.

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
rig report --runs <batch-sweep-60M> --max-points 1440 --layer-snapshots 400 \
  --output docs/reports/batch-size-sweep-60M.html
```

## lr-batch-sweep-125M.html

27 runs: **3 batches (64/128/256) × 3 learning rates (2^-7/2^-8/2^-9) × 3
seeds** at 125M, 5 TPP, 1,024 context.

The grid is a batch × LR product, so either axis can be read as the subject.
This replaces the former `batch-size-sweep-125M.html` and `lr-sweep-125M.html`,
which were two renderings of these same 27 runs.

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

## batch-size-sweep-250M.html

36 runs: **4 batches (64/128/256/512) × 3 learning rates × 3 seeds** at 250M,
5 TPP, 1,024 context.

Three runs — `250m-5tpp-bs512-lr2e-7`, all three seeds — recorded diagnostics
only from step 1920 onward. A report refuses a diagnostics log that does not
start at step 1, because its axes would not line up with the training curve, so
those three carry their partial series as `diagnostics-partial.riglog`: kept
beside the run, not declared, read by nothing automatically. The runs still
plot from their training curves rather than being dropped over it.

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

## batch-size-sweep-500M.html

12 runs at two token budgets. Run names carry the budget
(`500m-5tpp-…` against `500m-20tpp-…`) because the two are different
experiments whose losses are not comparable to each other.

This is study 3's dashboard. It replaces both the former `500M-20tpp-v6e.html`
(three of these twelve) and `500M-20tpp-diagnostics.html`, which existed only
because those three were once the only 500M runs whose diagnostics could be
read. All twelve can now.

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

## 3-seed-gradient-spike.html

12 runs: **4 learning rates × 3 seeds** at 250M, batch 128, 5 TPP. Built to
settle the 250M reseed in study 1, and the evidence base for
[GRADIENT_SPIKES.md](../GRADIENT_SPIKES.md).

Its diagnostics were unreadable long-form CSV until they were converted, so
for a while the dashboard about gradient spikes contained no gradient
statistics at all.

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

15 runs: **5 learning rates × 3 seeds** of
[`reference_8k`](../../recipes/reference_8k/) — 60M at 8,192 context with
document masking, batch 16 so tokens per step and step count match the
1,024-context ladder exactly. This is study 4.

```bash
for lr in 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625; do
  for seed in 1337 1338 1339; do
    rig run reference_8k --cluster v4-32 --profile dev --track open \
      --tier 60m --tokens-per-parameter 5 \
      --base-learning-rate "$lr" --seed "$seed" --checkpoint-policy none \
      --name "60m-bs16-lr${lr}-s${seed}"
  done
done
```

## moe-lr-sweep-8k.html

18 runs of [`reference_moe`](../../recipes/reference_moe/) — top-2 of 8
experts at 8,192 context, forked from the dense 8k ladder. 60M at five learning
rates × three seeds, plus 125M spot runs at three learning rates.

The routed ladder peaks at `2^-8`, the same learning rate the dense one does,
and beats it at every learning rate by 0.07–0.12 nats at equal *active*
parameters and matched compute, for about 1.7x the memory. No expert in any of
the 12 layers finished below 1% of assignments in any of the 18 runs.

This report carries six routing series the dense reports do not have: balance
loss, busiest and idlest expert share, routing entropy, mean top-1 gate, and
router logit RMS. They are recorded model-wide and per layer, with per-expert
load for all 8 experts in all 12 layers, at every step.

```bash
for lr in 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625; do
  for seed in 1337 1338 1339; do
    rig run reference_moe --cluster v4-32 --profile dev --track open \
      --tier 60m --tokens-per-parameter 5 \
      --base-learning-rate "$lr" --seed "$seed" --checkpoint-policy none \
      --name "60m-moe-lr${lr}-s${seed}"
  done
done
```

## transfer-charts.html

Not a run dashboard. Derived figures built from recorded results by
[`make_transfer_charts.py`](make_transfer_charts.py), committed beside it.

```bash
uv run --frozen --no-sync python docs/reports/make_transfer_charts.py
```

---

## Rebuilding

Download a study from the dataset and point `rig report` at it:

```bash
huggingface-cli download quintic/rig-logs --repo-type dataset \
  --include 'batch-sweep-60M/*' --local-dir /tmp/rig-logs
rig report --runs /tmp/rig-logs/batch-sweep-60M \
  --max-points 1440 --layer-snapshots 400 \
  --output docs/reports/batch-size-sweep-60M.html
```

`--max-points 0 --layer-snapshots 0` embeds every recorded sample. That is what
the dataset holds; it makes a much larger file than anything committed here.

## Two runs that are not in the dataset

- `20260816T213609.122328Z-…-37299d66` — a 500M run whose `stdout.log` was
  deleted while the process still held the descriptor, so no `result.json` was
  ever written. Its curves survive in the original archive but nothing records
  what it measured, so it cannot be placed on a chart.
- A `studies` directory inside the 60M archive, which is not a run.
