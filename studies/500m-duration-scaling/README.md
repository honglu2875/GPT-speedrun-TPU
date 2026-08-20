# 500M duration-scaling discrimination study

This study asks whether the 4× token-horizon change from 5 to 20 TPP needs the
additional Complete(d)P duration factor, or whether this repository's
fixed-TPP reanchoring is the better description of the observed optimum.

All new runs use the same TPU v4-32 topology, 1,024-token context, FineWeb-Edu
stream, 500M architecture, cosine schedule, and seeds 1337/1338/1339. The old
20-TPP v6e-8 points are useful reconnaissance but are excluded from hypothesis
tests because topology shifts validation loss on the scale being measured.

## Arms

| arm | recipe | TPP | `m_D` policy | new runs |
|---|---|---:|---|---:|
| `r5` | `reference` | 5 | fixed-TPP reanchored | 12 |
| `r20` | `reference` | 20 | fixed-TPP reanchored | 18 |
| `d20` | `reference_duration` | 20 | includes `TPP / 5` | 18 |

Six existing v4-32 `r5` runs are reused: batch 128 and 256 at base LR `2^-8`,
three seeds each. Together with the 48 queued runs, every arm has 18 v4-32
runs.

### `r5`: finish the 500M/5-TPP bracket

- LR sweep at batch 128: `2^-7`, `2^-8`, `2^-9` × 3 seeds.
- Batch sweep at base LR `2^-8`: 64, 128, 256, 512 × 3 seeds.
- Only the missing 12 cells are queued.

### `r20`: reanchored 20-TPP control

- The same LR and batch grid as `r5`, all on v4-32.
- This replaces the mixed-topology, mostly single-seed evidence as the
  scientific control.

### `d20`: Complete(d)P duration treatment

- LR sweep at batch 128: `2^-6`, `2^-7`, `2^-8`, `2^-9` × 3 seeds.
  Complete(d)P predicts that the transferable base optimum remains `2^-8`.
  If the duration factor overcorrects by its exact 2× LR multiplier, the
  observed base optimum should move to `2^-7`; `2^-6` brackets that outcome.
- Batch sweep at base LR `2^-8`: 128, 256, 512 × 3 seeds.
  Batch 512 is the 4×-batch/4×-duration iso-horizon point.

## Pre-registered interpretation

Use three-seed means and the repository's existing separation rule,
`|Welch t| >= 2.5`. Matching-seed deltas are also reported, but do not replace
the declared rule.

- **Evidence against the added duration factor:** `r20` retains a bracketed
  `2^-8` optimum while `d20` moves to a separated `2^-7` optimum, or the
  `d20` iso-horizon cell is materially worse than the best reanchored cell.
- **Evidence for it:** `d20` keeps `2^-8` bracketed and the batch-512
  iso-horizon point is competitive with or better than its batch-128/256
  neighbours.
- **Inconclusive:** the relevant three-seed differences do not meet the
  separation threshold. An unresolved result is not reported as transfer or
  contradiction.

The primary comparisons are within the 20-TPP/v4-32 cohort. Cross-budget loss
differences are descriptive only; they are not used to choose a policy.

## Queue

[`queue_v4_32.sh`](queue_v4_32.sh) is an ordinary sequence of `rig run`
commands. It front-loads twelve decisive bridge runs, then completes the 5-TPP
bracket before filling the outer LR and batch shoulders. It holds a process
lock and skips any named point that already has a `result.json`, so rerunning
it resumes rather than duplicates completed work.

Resolve and validate all 48 plans without training:

```bash
RIG_QUEUE_PLAN_ONLY=1 bash studies/500m-duration-scaling/queue_v4_32.sh
```

```bash
nohup bash studies/500m-duration-scaling/queue_v4_32.sh \
  > /tmp/500m-duration-v4-32.log 2>&1 &
```

Progress:

```bash
tail -f /tmp/500m-duration-v4-32.log
```
