# The `.riglog` container

The on-disk format for a run's training curve and its per-scope diagnostics.
Both are the same shape of data — a growing list of samples, each holding one
value per column, indexed by optimizer step — so both use one container.

Implemented in [`rig/logpack.py`](../rig/logpack.py); column identities come
from [`rig/metrics.py`](../rig/metrics.py).

## Why it is not CSV

A 500M run's `diagnostics.csv` was 143.8 MB, of which **21.8% was
measurement**. A byte census of the real file:

| field | share | what it was |
|---|--:|---|
| `value` | 21.8% | the only payload |
| `cumulative_estimated_flops` | 21.7% | a 20-digit integer, derivable from the step |
| `tokens_processed` | 10.9% | derivable from the step |
| `stat` + `scope` + `family` | 20.4% | a fixed 414-way cross product, re-spelled every row |
| commas + newlines | 9.9% | framing |
| `element_count` | 8.7% | constant for the whole run |
| `step` | 5.2% | repeated 414x per sample |

Packing the same data lands at 6.4 MB — **22.5x** — with every one of the
1,588,104 values bit-identical, because they were already `float32` on the
accelerator. Reading went from 47.8s to 3.0ms, and the report's peak RSS from
2.3 GB to 214 MB, because a million-plus rows no longer become a million-plus
Python dictionaries before anything is plotted.

## Byte layout

Little-endian throughout. Offsets are absolute, `n` is the column count.

```
offset  size          field
------  ------------  ---------------------------------------------------
0       8             MAGIC = b"RIGLOG\x00\x01"
8       4             column_count           int32
12      4             (padding)
16      8             tokens_per_step        int64
24      8             flops_per_token        float64
32      n x 24        column table           see below
32+24n  m x (4+4n)    records                see below
```

### Column table entry — 24 bytes

```
+0      4             metric_id              int32   rig/metrics.py
+4      4             scope_id               int32   rig/metrics.py
+8      4             layer                  int32   -1 when not layered
+12     4             (reserved, written 0)
+16     8             element_count          int64   scalars in this scope
```

### Record — `4 + 4n` bytes

```
+0      4             step                   int32
+4      4n            values                 n x float32, in table order
```

### Alignment

`MAGIC` is eight bytes rather than seven precisely so the table starts at 32
and every field lands on its natural offset. The table entry is 24 bytes, so
the record block begins 8-aligned for any `n`; the record is `4 + 4n`, so every
`step` and every value stays 4-aligned however many columns a run declares.

This is not decoration. At a seven-byte magic the table began at 31, which put
every `int64` at offset ≡ 7 (mod 8) and made a `memmap` view of the value block
impossible. The layout was corrected before any run was recorded in it.

## What the header carries, and why

`tokens_per_step` and `flops_per_token` are run constants, so they live in the
header and the two axes derived from them are never stored per sample:

```
tokens_processed = step x tokens_per_step            exact, int64
cumulative_flops = step x tokens_per_step x flops_per_token   float64
```

That deletes a third of the old file and fixes a precision bug on the way out:
`cumulative_flops` reaches ~3e19, far past `float32`'s seven significant
digits, so a stored `float32` column would have rounded it. Deriving in
`float64` keeps the exact product. `Log.axis()` materializes both on demand.

`element_count` is likewise constant for a run — parameter shapes do not change
— so it belongs in the table, not on 3,836 repetitions of every row. It matters
because two of the six diagnostic statistics are **unnormalized**: `l1_norm` is
`Σ|x|` and `l2_norm` is `√(Σx²)`, both scaling with scope size, while `mean`,
`std`, and the third/fourth moments are already divided by the count. Without
it a block's `l1_norm` cannot be compared against the model total.

## The schema is the column table

There is no fixed set of columns. A log declares its own: `column_count`
entries, each naming a metric, a scope, and a layer by permanent integer id.
A 12-layer 60M diagnostics log and a 21-layer 1B one are the same format with
different tables.

Column identity is **never positional and never textual**. That is what lets
the set of columns change without breaking old files: a reader looks up the
ids it understands and ignores the rest. `metric_by_id()` and `scope_by_id()`
return `None` rather than raising for an id this build has never heard of, so a
file written by a later version still opens and plots whatever overlaps.

## Where the ids live

[`rig/metrics.py`](../rig/metrics.py) is the single source of truth.

```python
METRICS = (
    Metric(1, "step"),
    Metric(2, "tokens_processed"),
    Metric(3, "cumulative_flops"),
    Metric(4, "train_loss"),
    Metric(5, "learning_rate"),
    Metric(6, "grad_norm"),
    # 100/200/300 + offset: {param, grad, update} x six statistics
)

SCOPES = (
    Scope(1, "overall"),   Scope(2, "embeddings"), Scope(3, "unembedding"),
    Scope(4, "block", layered=True),               Scope(5, "final_norm"),
)
```

A metric id names the *kind* of measurement — `grad.l2_norm` — while scope and
layer are separate columns in the table. Folding position into the id would
mean a 12-layer tier and a 21-layer tier shared no ids at all.

`step`, `tokens_processed`, and `cumulative_flops` hold ids even though they are
never stored as columns. They are addressed by id like anything else; the
reader computes them instead of loading them, and a consumer cannot tell.

### The ids are add-only

**An id, once assigned, is permanent.** Never renumber one, never rename one,
and never reuse the id of a metric that was removed — retire it instead.
Renumbering silently reinterprets every artifact already on disk.

[`rig/registry.txt`](../rig/registry.txt) is a checked-in snapshot of every
assignment, and `tests/test_metrics_registry.py` fails on any line that changes
or disappears. Adding a metric is therefore two deliberate edits — the entry in
`metrics.py`, and its line appended to the snapshot. That friction is the point.
The test distinguishes the four ways this goes wrong (renumbered, renamed,
removed, added-but-not-snapshotted) and names which one it caught.

## How the code implements it

### Writing

`LogWriter` seals the header on first `append` and then writes one fixed-width
record per sample, flushing each. Because the stride is fixed, **the file is
valid after every append** — a run killed at any point leaves a readable log
with every record that reached disk, and nothing needs repair.

That collapsed a distinction the CSV era needed. There used to be two writers:
a best-effort one appending coarse rows during the run so a preempted job
salvaged something, and an authoritative one rewriting everything at the end.
They had to agree on layout by hand. Now both build their columns from the same
helper, so a partial file is a prefix of the complete one by construction.

`recipes/reference/train.py` opens two writers before the loop:

- `training_log_columns()` — three columns, `overall` scope: `train_loss`,
  `learning_rate`, `grad_norm`. Appended at `log_every` cadence from host
  metrics the progress line already pulled, so it costs no extra device sync.
- `diagnostic_log_columns(scope_metadata)` — the `[scope, family, stat]` grid
  flattened in place, so a captured point becomes a record with no per-value
  bookkeeping.

At the end, `write_training_log` and `write_diagnostics_log` write the full
history to a temporary path and `os.replace` it over the coarse file.

### Reading

`read_log(path)` is one `read_bytes`, one `struct.unpack_from` for the header,
one `np.frombuffer` for the table, and a reshape of the remainder. It returns
the column table plus a single `(samples x columns)` `float32` array. There is
no per-record work at any point.

A partial trailing record — what a preempted run leaves — is dropped and every
whole record before it is kept, rather than the file being refused.

```python
from rig import logpack

log = logpack.read_log(Path("runs/<id>/diagnostics.riglog"))
log.steps                                    # int32, one per sample
log.values                                   # (samples, columns) float32
log.axis("tokens_processed")                 # int64, derived
log.series("grad.l2_norm")                   # overall scope, or None
log.series("param.l1_norm", "block", 7)      # one layer, or None
[c.describe() for c in log.columns]          # 'block[7]/param.l1_norm'
```

`series()` answers `None` for a column that was not recorded. Absent is an
ordinary answer — diagnostics may be disabled, or an older build may not have
had the metric — not an error.

### Measured

On the real 500M diagnostics artifact, 3,836 samples x 414 columns:

| | long-form CSV | `.riglog` |
|---|--:|--:|
| file | 143.8 MB | 6.38 MB |
| read | 47.8 s | 3.0 ms |
| write | 5.78 s | 0.24 s |
| `rig report`, one run | 32.2 s | 8.3 s |
| report peak RSS | 2,319 MB | 214 MB |

## Why `float32` and not something narrower

`bfloat16` would halve the file again, and it was measured and rejected. The
values span **34.6 decades**, from `4.8e-28` to `1.9e+07`:

| | max relative error | inf/nan | flushed to zero |
|---|--:|--:|--:|
| `float32` | 0 | 0 | 0 |
| `bfloat16` | 0.39% | 0 | 0 |
| `float16` | ∞ | 84,392 | 477,976 |

`float16` is disqualified outright: 5.3% of values exceed its 65,504 ceiling —
essentially all of `param.l1_norm` — and the `third_moment`/`fourth_moment`
families live at `1e-28..1e-10`, entirely beneath its subnormal floor.

`bfloat16` survives the range but costs precision on quantities that feed
ratios and spike detection, and these values arrive as `float32` from the
device — so `float32` is lossless rather than a trade. 6.4 MB is small enough
that the remaining 2x is not worth buying.

## Compatibility

Runs recorded before commit `75f0b22` wrote `training.csv` and
`diagnostics.csv` in long form. **Nothing converts them.** Reading one needs a
checkout at or before `8936b51`, where the old reader still exists — and note
its `_MAX_CSV_BYTES` guard skips artifacts over 128 MB, which the 500M
diagnostics exceed. The archived dashboards under `docs/reports/` are the
durable record of those studies.
