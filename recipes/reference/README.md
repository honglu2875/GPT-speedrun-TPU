# Reference model family

`train.py` is a readable pure-JAX GPT family, not a claimed record. Model,
fixed-TPP CompleteP-hybrid AdamW, batching, sharding, timing, evaluation, and checkpoint logic
remain visible in this one entry file. Three strict standalone documents select
the execution contract: `config.yaml` is official, `dev.yaml` is development,
and `smoke.yaml` is the tiny CPU wiring test. Each document is complete; no YAML
inheritance or runtime profile overlay is involved.

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
per-tensor AdamW and recipe-local batch/data scaling are specified in
[the fixed-TPP parameterization contract](../../docs/COMPLETEP.md).

`make run` selects 125M by default. Use `make run TIER=60m`,
`TIER=250m`, `TIER=500m`, or `TIER=1b` to select another tier. The official
profile trains for approximately 20 tokens per parameter, rounded to a complete
global step. The dev profile uses 5 TPP. `--stop-after-step` reproduces a prefix
without changing either schedule; smoke keeps a tiny standard-parameterized CPU
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

## Context presets and document boundaries

The dense implementation is shared. `reference` defaults to the established
1k baseline; `--context 8k` selects the long-context contract:

| preset | sequence length | reference batch | document masking | tokens/step |
|---|---:|---:|---|---:|
| `1k` (default) | 1,024 | 128 | off | 131,072 |
| `8k` | 8,192 | 16 | on | 131,072 |

The batch is part of the preset because it is the recipe-local anchor used by
`m_B`. Holding it at 128 for 8k would multiply tokens per optimizer step by
eight and divide the fixed-TPP schedule by eight. An explicit `--batch-size`
still overrides the selected preset's anchor for a batch study.

The corpus is a flat token stream — whole FineWeb documents, each prefixed by
the GPT-2 EOT token `50256`, concatenated with no offset index. Windows are cut
live at arbitrary positions (`shard[start : start + seq_len + 1]`), so a window
can begin mid-document and run through several.

At the default 1k preset, attention is causal only: a token can attend across
an EOT boundary into the preceding document. This preserves every recorded 1k
baseline exactly.

Measured on a 100M-token shard (143,857 documents, median length 405 tokens):

| | at `seq_len` 1024 |
|---|--:|
| documents a random window spans | ~1.5 |
| documents that fit whole (< 1024 tokens) | 83.7% |
| **tokens living in documents ≥ 1024 tokens** | **53.0%** |

So most *documents* are shorter than the window and sit inside it entirely,
while most *tokens* come from documents long enough that a 1024-token window
truncates them. Either way the cross-document surface is small: about one
boundary per window.

Two reasons this stays off:

- **Bit-identical reproducibility.** Every result in
  [the transfer note](../../docs/HYPERPARAMETER_TRANSFER.md) was measured
  without masking. Enabling it would silently redefine the baseline and make
  new 1k runs incomparable with the recorded ones — the 250M cells reproduced
  bit-identically across a refactor and a package rename, and that property is
  worth more than a marginal correction.
- **The baseline should stay basic.** 1024 tokens is a short context; masking
  buys little at ~1.5 documents per window and adds a data-dependent term to
  the attention mask.

It is not free of consequence, and the honest statement is that the 1k preset
trains with a small amount of cross-document attention.

The `8k` preset **does** mask, because an 8,192-token window spans about 11.8
documents and only 8.5% of tokens live in documents that long. Without masking,
most of the added range would be attention across unrelated text. The preset
derives segment ids on-device from EOT boundaries; implementation is in
[`rig/kernels/tpu_flash_attention.py`](../../rig/kernels/tpu_flash_attention.py).

Native validation losses at 1k and 8k are not directly comparable: each token
in the 8k evaluation may receive much more context. Report native loss for each
contract, and evaluate the 8k model at 1k as well when the question requires a
common-context comparison.

## Optimizer and kernels

The family uses the fixed-TPP CompleteP hybrid with α=1, PyTorch-form AdamW,
10% warmup, cosine
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
- `checkpoint.npz`, except for explicit development study runs using
  `--checkpoint-policy none`;
- `metrics.json`, with the resolved tier, exact parameter count, parameterization
  multipliers, data-sharding rule, system topology, and result protocol.

Both logs are packed binary, not CSV: a header naming each column by its
permanent id from `rig/metrics.py`, then fixed-width `float32` records. A 500M
run's diagnostics are 6.4 MB instead of 144 MB, and the report reads them in
milliseconds. Fixed stride also makes the file append-only, so a preempted run
keeps every sample already written. Read one with `rig.logpack.read_log`, which
returns the column table plus one `(samples x columns)` array; the byte
layout and the metric-id registry are in
[docs/RIGLOG_FORMAT.md](../../docs/RIGLOG_FORMAT.md).

Curves accumulate on device and move to the host after synchronized training,
so per-step capture does not add a host synchronization. Dev and official
runs enable all diagnostics by default, every 10 and 500 steps respectively
(plus the first and final steps); smoke runs keep them disabled. Official probes
run every 500 steps and count inside `train_seconds`; canonical final FineWeb
and Fresh10 evaluation run outside it.

`--checkpoint-policy none` is restricted to development research. It preserves
metrics, curves, immutable records, and re-verification while never writing
hundreds of megabytes of weights at every sweep point. `qualifying` keeps them
only at or below the target loss; `always` keeps them regardless. Official runs
still require a checkpoint.

Use the harness rather than invoking `train.py` directly:

```bash
uv run --frozen --no-sync rig run reference --profile smoke
make run                         # official, 125M, 20 TPP
make run TIER=250m
uv run --frozen --no-sync rig run reference --context 8k --tier 60m --profile dev
```
