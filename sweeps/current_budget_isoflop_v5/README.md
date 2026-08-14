# Current-budget IsoFLOP v5

V5 is a fresh, fail-closed rerun of the local equal-FLOP study. It independently
calibrates learning rate for every `(compute slice, model shape)` pair. A
selected calibration run is the equal-FLOP model-size measurement, so no
short-horizon learning rate is reused at a longer horizon.

V4 stopped without a scaling-law claim after its first `c050/n023` trial at
`2e-4` and its first lower recovery trial at `1.3333333333333334e-4` were both
rejected by the frozen stability gate. Its next declared rate, `8.888888888888889e-5`,
was never launched. V4 is now an immutable incomplete archive. V5 was designed
after observing that outcome, so its longer lower table is prospective only for
v5; it is not a preregistration independent of v4. V5 has no lineage declaration
and reuses no v4 measurement, selection, or fit.

This study targets the pre-Complete(d)P GPT-2 baseline: 12 layers, width 768,
and exactly 124,475,904 parameters under the schema-1 trainer's tied-embedding
parameter count. It does **not** target the promoted schema-2 Complete(d)P
reference baseline.

`suite.yaml` pins the regular same-directory `trainer.py` byte-for-byte. The
snapshot is identical to v4 and to `submissions/reference/train.py` from commit
`a3dd3f25bd1b29e30d61764221a6d0eda0791b82`, SHA-256
`b99dd38fe3a47b6b82e8d2c82649b53070f40de99838545ea18a5c2bccd7aea7`.
The loader rejects path escape, symlinks, or hash drift. The reviewed v5
execution fingerprint is
`695013df27e2aad51de13cea79e140c08218fc7ef7dd0f40bc4b738ee594bbb1`.

## Frozen stability admission

V5 keeps v4's stability policy byte-for-byte. Every curve is parsed in full.
Headers, rows, optimizer steps, token and FLOP coordinates, float32 LR schedule,
fixed-prefix probes, final FineWeb/Fresh10 rows, diagnostic cadence and grid,
scope element counts, finiteness, and cross-file identities must match. A
complete run is `rejected` if any frozen temporal rule fires:

- two consecutive fixed-prefix probes are each at least 0.05 above the best
  preceding probe;
- any trailing 512-step window has at least 50% of raw gradient norms above 10;
- any trailing 512-step mean train loss is at least 0.20 above the best prior
  512-step mean.

It is `suspect` if the final maximum block parameter L2 norm divided by the
median block norm exceeds 2.0. Only `stable` runs enter LR selection or fitting.
Suspect and rejected runs remain required raw evidence. Admission is recomputed
from `training.csv`, `validation.csv`, and `diagnostics.csv`; the immutable
admission record binds those curves, the suite, execution fingerprint, point,
config, policy, and bounded witnesses.

## Directional LR frontier

The initial grid is launched low-to-high: `2e-4`, `3e-4`, `4.5e-4`. The high
side is unchanged from v4: the first suspect or rejected point at or above
`2e-4` is an impenetrable frontier. No higher LR may be launched after it.

If `2e-4` is ineligible, the runner moves only toward safer rates, one exact
1.5x geometric step at a time. Unlike v4, an ineligible point on this descending
recovery path is evidence rather than a barrier to a still-lower rate. The
bounded lower table, in launch order, is:

| ID | Peak learning rate |
|---|---:|
| `lr133` | 0.00013333333333333334 |
| `lr089` | 0.00008888888888888889 |
| `lr059` | 0.00005925925925925926 |
| `lr040` | 0.00003950617283950617 |
| `lr026` | 0.000026337448559670783 |
| `lr018` | 0.00001755829903978052 |

Recovery does not hand control to selection until two **adjacent** lower-table
points are stable. Stable points separated by an ineligible point do not meet
that condition. Selection then remains exactly as strict as v4: choose the
lowest canonical FineWeb validation loss among all stable completed points,
with an immediately adjacent stable lower-LR neighbor and an immediately
adjacent completed upper boundary. The upper boundary may be stable with worse
loss or ineligible. A globally best stable point is not skipped merely because
it lacks a stable immediate lower neighbor; the runner descends further. If the
bounded table is exhausted without a valid bracket, the group fails closed and
no dependent run or scaling-law claim is emitted.

Completion must be a contiguous prefix in each launch direction. Resume rejects
holes, a higher run beyond an ineligible high frontier, edited selections, or
local evidence whose full semantic admission changes.

Before any v5 scaling-law claim is released, every raw curve, manifest, result,
config, trainer snapshot, admission, selection, and fit must be placed in one
closed immutable archive and verified again from its exact public revision. The
existing evidence publisher is intentionally v4-specific and must not be used
under a v5 label; its versioned v5 successor is a separate release-gate change
after this launch source commit is known.

## Model-size frontier and budget

Each slice starts with the five base shapes and adds `n082` and then `n102`,
with fresh per-slice LR calibration, only while the current model-size fit is
unbracketed on the high-model-size side. The extension stopping rule is
unchanged from v4.

The suite declares 315 calibration configurations (seven possible shapes,
three slices, and fifteen LR values) plus one control, but the staged state
machine cannot traverse both LR tails for one group. Its reachable ceiling,
if both optional shapes are warranted in all slices, is 190 runs and 111.25 C0:
nine trials per group across 21 groups, plus the control. The three-point base
grid for the five required shapes plus the control is 27.25 C0. These are
prospective bounds, not a promise that every point will run.

## Run

Inspect and record the fingerprint before using TPU compute:

```bash
uv run --frozen --no-sync python -m speedrun.scaling plan
```

Then use a new v5 output root:

```bash
uv run --frozen --no-sync python -m speedrun.scaling run --staged \
  --data-path /dev/shm/fineweb-scaled/4B \
  --downstream-manifest data/manifests/fresh10.json \
  --downstream-root /dev/shm \
  --runs runs/scaling/current-budget-isoflop-v5 \
  --confirm-execution-fingerprint DIGEST_FROM_PLAN \
  --resume --color always
```

This remains a one-seed local law, not a universal Chinchilla estimate. The same
canonical FineWeb validation set is reused across LR candidates for selection
and for the model-size outcome, so fit-only uncertainty excludes selection and
multiple-comparison optimism.
