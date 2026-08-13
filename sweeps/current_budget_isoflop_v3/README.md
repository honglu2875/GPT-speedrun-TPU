# Archived current-budget IsoFLOP v3 continuation

The promotion of `reference-gpt-v2` makes this `reference-gpt-v1` continuation
read-only. It remains available for exact historical analysis, but new work must
start in a versioned suite whose architecture contract matches the promoted
reference.

This study continues `current_budget_isoflop_v2` after the 51.50M-parameter
calibration continued improving through the old upper bound, `2.278125e-3`.
It does not change, move, copy, or relabel any v2 run. Instead,
`lineage-v2.json` is a checked-in allowlist of the exact run-manifest and result
SHA-256 values for the 24 completed v2 measurements.

Every historical measurement is admitted through two independent gates:

1. The lineage manifest, historical suite, historical template, historical
   execution fingerprint, and complete historical source snapshot are pinned.
   All historical execution sources are pinned to their v2 Git bytes. The
   current reference trainer may advance, in which case this suite rejects new
   launches while retaining exact read access to its historical measurements.
2. The historical run manifest and result must match their individual allowlist
   hashes. The runner then applies the existing full point/config/work snapshot,
   dataset/Fresh10/runtime, no-replacement, result schema, and metric checks to
   those bytes. A local v3 directory with the same point ID is rejected as a
   shadow rather than preferred over lineage.

Selections, measurements, fit JSON, and `plan --json` expose the lineage ID,
origin identities, and artifact hashes. Exactly the 24 original v2
`run-manifest.json` and `result.json` pairs are selectively tracked under
`lineage-v2-artifacts/` as immutable scientific inputs (about 672 KiB total),
rather than as active run logs. Their curves, diagnostics, copied work trees,
and checkpoints remain discarded. Absence or byte drift fails closed, and the
original v2 identities are never relabeled as v3 outputs.

## Learning-rate boundary

The next geometric point, `3.4171875e-3`, is mandatory for n051 because its v2
best was the upper endpoint. `5.12578125e-3` is preregistered as a reserve and
runs only if the mandatory point is again best. We intentionally do not include
`7.688671875e-3`: two new 1.5× points are enough to find an interior neighbor
under the present hypothesis. Continued improvement through both would be new
evidence deserving another versioned, reviewed bound, rather than an implicit
open-ended search.

Other shapes reuse their already bracketed v2 searches. The n065 calibration,
dependent 0.50C0/1.00C0 runs, controls, and model-size extensions retain the v2
policy. The data, runtime, statistical caveats, and fitting method remain as
documented in [`../current_budget_isoflop/README.md`](../current_budget_isoflop/README.md).

## Commands

Inspect the new fingerprint and explicit lineage:

```bash
uv run --frozen --no-sync python -m speedrun.scaling plan
```

New launches are intentionally disabled now that the default trainer is
`reference-gpt-v2`. Use the new 20-TPP suite for further scaling work; this
directory exists only to audit the earlier measurements.
