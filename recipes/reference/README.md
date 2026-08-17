# Reference model family

`train.py` is a readable pure-JAX GPT family, not a claimed record. Model,
Complete(d)P AdamW, batching, sharding, timing, evaluation, and checkpoint logic
remain visible in this one entry file. The strict sibling `config.yaml` defines
five architecture tiers and three execution profiles.

The non-smoke tiers are:

| Tier | Layers | Width | Heads | Exact parameters |
|---|---:|---:|---:|---:|
| 60M | 12 | 384 | 6 | 59,918,208 |
| 125M | 12 | 640 | 10 | 123,456,640 |
| 250M | 16 | 896 | 14 | 244,444,032 |
| 500M | 19 | 1,280 | 20 | 502,602,240 |
| 1B | 21 | 1,792 | 28 | 989,943,808 |

Every tier uses 64-wide heads, base-10,000 RoPE, pre-RMSNorm, a 4× GELU
MLP, and untied input/output embeddings. The family is identified as
`reference-gpt-v3-family`. Width, depth, initialization, residual, attention,
per-tensor AdamW, batch, and token-horizon scaling are specified in
[the Complete(d)P contract](../../docs/COMPLETEP.md).

`make run` selects 125M by default. Use `make run TIER=60m`,
`TIER=250m`, `TIER=500m`, or `TIER=1b` to select another tier. The official
profile trains for approximately 20 tokens per parameter, rounded to a complete
global step. The dev profile is bounded to 100 steps unless a research command
supplies `--tokens-per-parameter`; smoke keeps a tiny standard-parameterized CPU
wiring test.

## Data parallelism

Dev and official training use `shuffled_epochs`. A deterministic keyed
permutation traverses non-overlapping windows without allocating a global index
array. Every JAX process constructs the same global order and consumes only its
rank-local slice, so hosts do not duplicate examples within a global batch.
The rank comes from `jax.process_index()` after `jax.distributed.initialize()`,
not from an assumed `RANK` environment variable or hostname suffix. Model and
optimizer state remain replicated; this is data sharding, not model sharding.

The stream starts a newly seeded shuffled epoch if a requested horizon exceeds
the prepared distinct-token capacity. Prepare a corpus large enough for the
horizon you want when a study needs to stay no-replacement.

## Optimizer and kernels

The family uses Complete(d)P with α=1, PyTorch-form AdamW, 10% warmup, cosine
decay to 10% of peak, and no global gradient clipping. The custom trainable
Pallas attention backend and tiled output cross entropy are selected for dev
and official TPU profiles. Smoke uses dense FP32 kernels.

For non-dense attention, the trainer resolves a static ten-field tile plan
before compiling the real step. Resolution checks an exact runtime-fingerprinted
cache, a source-pinned shipped entry, and then a deterministic shape heuristic.
An explicit synthetic autotuner is available; it never reads real data or runs
inside timed training. See [the kernel notes](../../docs/KERNELS.md).

## Artifacts

Every accepted run writes:

- `training.riglog`: every optimizer step's train loss, effective global LR, and
  gradient norm;
- `validation.csv`: deterministic probes and canonical final validation;
- `diagnostics.riglog`: sparse parameter, gradient, and update statistics for
  every supported model scope and statistic;
- `checkpoint.npz`, except for explicit open/dev study runs using
  `--checkpoint-policy none`;
- `metrics.json`, with the resolved tier, exact parameter count, Complete(d)P
  multipliers, data-sharding rule, system topology, and result protocol.

Both logs are packed binary, not CSV: a header naming each column by its
permanent id from `rig/metrics.py`, then fixed-width `float32` records. A 500M
run's diagnostics are 6.4 MB instead of 144 MB, and the report reads them in
milliseconds. Fixed stride also makes the file append-only, so a preempted run
keeps every sample already written. Read one with `rig.logpack.read_log`, which
returns the column table plus one `(samples x columns)` array.

Curves accumulate on device and move to the host after synchronized training,
so per-step capture does not add a host synchronization. Dev and official
runs enable all diagnostics by default, every 10 and 500 steps respectively
(plus the first and final steps); smoke runs keep them disabled. Official probes
run every 500 steps and count inside `train_seconds`; canonical final FineWeb
and Fresh10 evaluation run outside it.

`--checkpoint-policy none` is restricted to open/dev research. It preserves
metrics, curves, immutable records, and re-verification while never writing
hundreds of megabytes of weights at every sweep point. `qualifying` keeps them
only at or below the target loss; `always` keeps them regardless. Official and
sample-efficiency runs still require a checkpoint.

Use the harness rather than invoking `train.py` directly:

```bash
uv run --frozen --no-sync rig run reference --profile smoke
make run                         # official, 125M, 20 TPP
make run TIER=250m
```
