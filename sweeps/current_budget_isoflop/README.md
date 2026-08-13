# Current-budget IsoFLOP study

This is a diagnostic, non-competition study of model/data allocation on the
local TPU v4-8 stack. It keeps the completed baseline's analytic compute budget
as `C0 = 537,549,813,420,392,448` training FLOPs and measures three slices:
`0.25 C0`, `0.50 C0`, and `1.00 C0`.

The five constant-aspect fit shapes are 22.59M, 30.36M, 39.90M, 51.50M, and
65.47M parameters. Every shape uses 64-wide attention heads, MLP multiplier 4,
sequence length 1,024, global batch 32, BF16, TPU Flash attention, and the dense
loss. The 124.48M completed-baseline shape is retained as a `1.00 C0` control.
Bounded 82.09M and 101.66M points are available to every compute slice and run
only when that slice's fit is unbracketed on the high-model-size side.

## Statistical policy

The `0.25 C0` slice doubles as bounded learning-rate calibration. Each shape is
trained at `2e-4`, `3e-4`, and `4.5e-4`; canonical loss over exactly 99,975,168
validation targets selects the learning rate propagated to its `0.50 C0` and
`1.00 C0` runs. An edge winner expands geometrically, one point at a time, down
through `1.333e-4`/`8.889e-5` or up through `6.75e-4`, `1.0125e-3`,
`1.51875e-3`, and `2.278125e-3`. The two higher points were preregistered after
the first bounded pilot remained monotone through `1.0125e-3`; all pilot runs
were archived and the fingerprinted study restarted rather than importing
measurements across search policies. If the winner is still an edge at the new
bound, the runner refuses selection and does not launch dependent runs. Warmup,
validation cadence, and logging cadence preserve the completed baseline's
fraction of total steps. All runs share seed 1337 because three-seed replication
does not fit the current time budget.

Training uses the YAML-only `shuffled_epochs` sampler. It gives a deterministic
pseudorandom permutation of non-overlapping shard windows, needs no giant index
array, assigns disjoint global-batch slices to multi-host ranks, and does not
repeat data in this suite. The official reference configuration explicitly
continues to use `random_windows` and is unchanged as a competition baseline.

The fit uses a quadratic in log model size for each IsoFLOP curve. It refuses to
report an interpolated optimum if the quadratic minimum is outside the measured
range or the lowest observed loss is a grid endpoint. Only if all three minima
are bracketed does it fit:

```text
N_opt(C) = N0 * (C / C0)^a
D_opt(C) = D0 * (C / C0)^b
```

This is a one-seed local empirical law, not a universal Chinchilla law. It is
conditional on this tokenizer, FineWeb build, model aspect family,
initialization, global batch, and optimizer schedule. Its fit-only uncertainty
does not include seed noise or hyperparameter uncertainty beyond the three-way
learning-rate selection.
The learning rate chosen at `0.25 C0` may itself cease to be optimal at the
longer `0.50 C0` and `1.00 C0` horizons; this bounded calibration is a practical
compromise rather than a full per-shape, per-budget optimizer sweep.

## Commands

Inspect the complete step/token/FLOP plan without JAX initialization:

```bash
uv run --frozen --no-sync python -m speedrun.scaling plan
```

After the 4B dataset exists locally, run the gated sequence:

```bash
uv run --frozen --no-sync python -m speedrun.scaling run --staged --data-path /dev/shm/fineweb-scaled/4B --downstream-manifest data/manifests/fresh10.json --downstream-root /dev/shm --confirm-execution-fingerprint DIGEST_FROM_PLAN --resume --color always
```

The data gate requires the builder's sibling `manifest.json`, `source.json`,
`exclusions.json`, and `BUILD_PLAN.json`, the exact ordered
`fineweb-4b-gpt2` 39-train-plus-1-validation shard set, and the pinned source and
tokenizer identity. It recomputes the pinned source-inventory, exclusion-policy,
and builder-core digests and enforces the pre-2024 cutoff and document-disjoint
validation split. Before the first stage it reads all 8 GB once to verify every
manifest SHA-256; later stages reuse that verified inventory. The canonical
manifest digest and all selected shard hashes are copied into every immutable
run manifest and the final fit. Missing, extra, reordered, mixed, or altered
shards stop the study before a result can enter the fit. Fresh10 is mandatory:
the checked-in manifest and all ten domain files are hash-verified, and every
result must contain the ten per-domain losses plus their 81,920-token macro.

The runtime gate requires a single Python 3.12 process with JAX/JAXlib 0.11.0,
libtpu 0.0.44.1, and exactly four visible TPU v4 devices (a v4-8). Prelaunch and
trainer-reported runtime identities must match exactly.

The execution fingerprint covers the suite, template, trainer, shared kernel
sources, and scaling runner. A source change invalidates it and forces another
review before TPU work. No long run is started while writing or testing the
framework. Resolved
`config.yaml`, copied `train.py`, `run-manifest.json`, result, curves, and final
layer diagnostics are saved beneath
`runs/scaling/current-budget-isoflop-v1/<point>/`. The runner deliberately uses
the open/dev-only checkpoint-omission mode: final validation and metrics remain,
but redundant FP32 parameter archives do not consume the 29 GB root disk. If
retained, the base checkpoints would occupy about 4.70 GB (7.64 GB if both
optional shapes run); this policy reduces that to zero. Learning-rate choices
live beneath `learning-rate-selections/`. Derived slice fits are replaced safely
beneath `fits/`; raw configs and run manifests are immutable and a byte mismatch
stops resumption.

If every required run is already complete, rebuild the fit with:

```bash
uv run --frozen --no-sync python -m speedrun.scaling fit
```

The base plan costs 12.25 completed-baseline compute equivalents. At the
observed baseline training time of roughly 28.7 minutes this is approximately
5.85 hours of ideal training time. The 26 base runs are 15 `0.25 C0`
learning-rate trials, five `0.50 C0` fits, five `1.00 C0` fits, and the one
`1.00 C0` 124M control. The five selected calibration runs are the `0.25 C0`
fit points themselves; they are not trained again. Thus compute is exactly
`15×0.25 + 5×0.50 + 5×1.00 + 1×1.00 = 12.25 C0` apart from nearest-step
rounding. There are 26 run-level compilations (with possible persistent-cache
reuse), while 26 full 100M-token validations add roughly 30–40 minutes at the
measured validation rate. Budget 7–9 hours on the v4-8.

Adaptive work is paid only when needed. Each shape can add at most two lower-side
or four upper-side LR trials. Each extension shape costs `0.75 C0` for its
initial LR grid, with its selected trial reused as the c025 fit point, then at
most `0.50 C0` and `1.00 C0` for later slices. The absolute bounded maximum is
64 runs and `23.75 C0`: about 11.36 ideal training hours plus validation and
compilation overhead, so this rare path may exceed 12 hours. The expected
no-expansion path remains 26 runs/`12.25 C0`; both model extensions across all
slices without LR-edge expansion give 36 runs/`16.75 C0`. Larger models are not
launched for a low-side or non-directional failure; the final artifact then
explicitly reports no scaling law.
