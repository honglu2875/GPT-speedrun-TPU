# Current-budget IsoFLOP v4

V4 is a fresh, fail-closed rerun. It independently calibrates learning rate for
every `(compute slice, model shape)` pair. A selected calibration run is itself
the equal-FLOP model-size measurement, so no short-horizon learning rate is
reused at a longer horizon.

V2 and v3 remain immutable archives. V4 deliberately has no lineage declaration:
their lineage manifests pin `run-manifest.json` and `result.json`, but not the
full `training.csv`, `validation.csv`, and `diagnostics.csv` evidence now required
for stability admission. Copying or relabeling those mutable curves would not be
a reproducible scientific input. Development evidence includes divergent c025
trials n023 at `1.51875e-3` and n030 at `6.75e-4`, the finite-divergent v3
`c050_n030`, and the suspect v3 `c050_n023`; none is reused.

## Stability admission

These thresholds were chosen retrospectively after inspecting v2/v3 development
evidence. They are prospective only for the fresh v4 results and are frozen in
this suite before any v4 run; they are not an independent preregistration with
respect to the archived studies.

Every curve is parsed in full. Expected headers, rows, optimizer steps, token and
FLOP coordinates, float32 learning-rate schedule, fixed-prefix probes, final
FineWeb/Fresh10 rows, diagnostic cadence, unique diagnostic grid, scope element
counts, finiteness, and cross-file identities must all match exactly (apart from
the pinned LR numeric tolerance). Structural drift is an error.

A complete run is classified `rejected` if any frozen temporal rule fires:

- two consecutive fixed-prefix probes are each at least 0.05 above the best
  preceding probe;
- any trailing 512-step window has at least 50% of raw gradient norms above 10;
- any trailing 512-step mean train loss is at least 0.20 above the best prior
  512-step mean.

It is classified `suspect` if the final maximum block parameter L2 norm divided
by the median block norm exceeds 2.0. Only `stable` runs enter LR selection or
the model-size fit. Suspect and rejected runs remain evidence but are quarantined.
Each read recomputes an immutable `artifacts/stability-admission.json`, binding the
suite, execution fingerprint, point/config, policy, curve hashes, byte/row/header
counts, summaries, bounded witnesses, and classification. Float32 LR replay uses
the pinned tolerance `rtol=2e-6`, `atol=1e-12`; inclusive threshold comparisons
use `atol=1e-12` to make decimal boundary semantics reproducible.

Admission hashes do not make missing raw curves independently auditable. Before
any scaling-law claim, all v4 raw CSVs, run manifests, results, configs, and
admission records will be archived together in an immutable Hugging Face or
release bundle whose object/revision identity is recorded in the final report.

The LR grid is launched low-to-high. No trial may be launched beyond the first
suspect/rejected high-LR frontier. Selection requires an immediately adjacent
stable lower-LR neighbor and an immediately adjacent completed upper boundary;
the upper boundary may be stable with worse loss or ineligible. Adaptive geometric
expansion remains bounded by the declared table.

Model-size extension is likewise prospective and slice-local. Each slice starts
with the five base shapes and adds `n082` then `n102`, with fresh per-slice LR
calibration, only while its current fit remains unbracketed on the high-model-size
side. It stops at a bracketed or non-high-side fit. An interrupted warranted next
shape is an incomplete fit error; a no-law result is allowed only after that
stopping rule fires or the declared `n102` grid is exhausted.

## Run

```bash
uv run --frozen --no-sync python -m speedrun.scaling plan
uv run --frozen --no-sync python -m speedrun.scaling run --staged --data-path /dev/shm/fineweb-scaled/4B --downstream-manifest data/manifests/fresh10.json --downstream-root /dev/shm --runs runs/scaling/current-budget-isoflop-v4 --confirm-execution-fingerprint DIGEST_FROM_PLAN --resume --color always
```

This remains a one-seed local law, not a universal Chinchilla estimate. The same
canonical FineWeb validation set is reused across the many LR candidates for
selection and again as the model-size-fit outcome, so fit-only uncertainty does
not include selection or multiple-comparison optimism.
