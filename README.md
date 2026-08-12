# GPT Speedrun TPU

A collaborative GPT training speedrun for Cloud TPU v4 slices, from a v4-8 to
larger multi-host slices. Every algorithm is a polished single-file JAX program
named `train.py`; shared code handles
reproducible data, machine checks, run capture, protocol validation, and
leaderboards.

This is intentionally a checkout-based benchmark, not a relocatable Python
library: manifests, submissions, Git state, and persistent run records are all
repository resources. Run the commands from a cloned checkout (they remain
independent of the caller's current directory after installation by `uv sync`).

The initial goal is a mean next-token validation loss at or below **3.28** on
the fixed GPT-2-tokenized FineWeb validation prefix. We retain the complete
metrics needed to push the target lower after calibrating this hardware.

## Start here

The top-level Makefile is the user interface and a set of copyable command
templates:

```bash
make prepare
make baseline
make profile
make report
```

| Target | Purpose |
|---|---|
| `make prepare` | synchronize the frozen uv environment, then open the interactive setup wizard |
| `make baseline` | verify every configured TPU VM and its data, then run the fully explicit reference baseline |
| `make profile` | run a validation-free 100-step diagnostic, capture XProf steps 11–20, then serve the trace on port 8791 |
| `make report` | integrity-check completed runs and rebuild the standalone `report.html` dashboard |

`make prepare` runs these two commands:

```bash
uv --cache-dir /tmp/uv-cache sync --frozen
uv --cache-dir /tmp/uv-cache run --frozen --no-sync speedrun prepare
```

It asks for the data-cache root, data and run profiles, persistent artifact
directory, TPU VM host count, default track, checkpoint retention, colors, and a
smoke/development loss target. It can then probe JAX/TPU health and prepare the
selected dataset. Personal choices are stored in the gitignored
`.speedrun.toml`; official constants remain versioned in Git. The personal
target applies only to smoke/development work; the official target is fixed at
3.28 and may only be tightened explicitly.

On this node, `shm/` is a symlink to the 201 GiB RAM filesystem and is an ideal
ephemeral data cache:

```bash
uv run --frozen --no-sync speedrun prepare --path shm/ --profile official
```

The cache path is always explicit. The tool supports symlinks, resumable
downloads, free-space checks, and exact header/length/SHA-256 validation. It
never deletes unrelated files. Because `/dev/shm` is ephemeral, preparation may
need to restore the data after a reboot.

### Multi-host TPU slices

A Cloud TPU v4-32 is one slice spread across four TPU VMs, with four TPU v4
chips attached to each VM. It is not four independent v4-8 slices. Run
`make prepare` on worker 0, answer `4` for the TPU VM host count, and accept the
inferred expression when the hostnames follow Cloud TPU's usual convention:

```text
t1v-n-a09f5679-w-[0-3]
```

The expression is ordinary `pdsh` host-list syntax; a comma-separated explicit
list also works. The controller needs `pdsh`, `scp`, and non-interactive SSH to
itself and every peer. Speedrun tests that access, but it never creates, copies,
or modifies SSH keys. If the probe fails, add this controller's public key to the
same user's `~/.ssh/authorized_keys` on the TPU VMs, verify
`pdsh -R ssh -w HOSTS hostname`, and rerun preparation.

After the SSH probe succeeds, preparation archives the current checkout
(including dirty and untracked experiment files but excluding Git metadata,
the virtual environment, caches, and run artifacts), copies it to the same
absolute path on every peer, installs `uv` there if needed with Astral's
official installer, synchronizes the frozen environment, and prepares each
VM's local data cache. `make baseline` repeats the source synchronization and
automatically launches the trainer on all configured hosts through `pdsh`.

Every trainer process calls `jax.distributed.initialize()` before its first
device query. JAX's runtime rank is `jax.process_index()`; there is no launcher
`RANK` variable to trust. Training and evaluation loaders produce a distinct
rank-local portion of each global batch, while JAX arrays span the whole TPU
mesh. Cloud TPU's JAX rank order need not match the `-w-N` hostname suffix, so
the launcher explicitly marks its controller hostname; only that VM emits
human logs, checkpoints, metrics, and the final result consumed by the harness.

This path targets a single multi-host TPU slice such as v4-32. Cloud TPU
Multislice (multiple separately provisioned slices connected over DCN) also
needs a hybrid ICI/DCN mesh and is a distinct scaling mode.

For automation on a four-host v4-32, bypass the questions:

```bash
uv run --frozen --no-sync speedrun prepare \
  --non-interactive --path shm/ --profile official \
  --run-profile official --track open --checkpoints qualifying \
  --tpu-vm-count 4 --tpu-vm-hosts 't1v-n-a09f5679-w-[0-3]'
```

## Data profiles

All profiles use one well-defined token format; downloading and tokenization
are never timed.

| Profile | Data | Intended use |
|---|---:|---|
| `smoke` | generated locally | CPU/CI end-to-end checks |
| `dev` | 100M train + 100M validation tokens (~400 MB cached) | quick TPU iteration |
| `official` | 900M train + 100M validation tokens, plus Fresh10 (~2.0 GB cached) | record attempts |

Official evaluation covers exactly the first 10,485,760 validation predictions.
The harness requires official result events to report that exact coverage;
smoke and development profiles deliberately use shorter diagnostic evaluation.
The source is the pinned `kjj0/fineweb10B-gpt2` revision used by
[Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt). See
[data/README.md](data/README.md) for provenance and the binary contract.

## Run an algorithm

```bash
# Exact reference workflow, including machine/data preflight
make baseline

# Fast end-to-end check
uv run --frozen --no-sync speedrun prepare --non-interactive \
  --path shm/ --profile smoke --run-profile smoke --no-doctor
uv run --frozen --no-sync speedrun run reference --profile smoke

# Initial v4-8 calibration (use the same topology for like-for-like timing)
uv run --frozen --no-sync speedrun run reference \
  --track open --profile official

# Trainer-specific overrides follow --
uv run --frozen --no-sync speedrun run reference --profile dev -- \
  --steps 100 --batch-size 32
```

The harness creates a unique persistent run directory, captures stdout/stderr,
validates the final result event and checkpoint, hashes artifacts, and appends a
JSONL record. The trainer's synchronized accelerator time is the open-track
score; cold process wall time is recorded separately. Human progress is streamed
live while the machine-readable result remains isolated on stdout.

Every successful reference run also writes `training.csv` inside its run
directory. It contains one row per optimizer step with step number, cumulative
training tokens, cumulative analytic estimated FLOPs, loss, learning rate, and
gradient norm. Scalars accumulate on the TPU and transfer only after timed
training, so retaining the complete curve does not add a synchronization to
every step. Token, learning-rate, and FLOP columns are deterministic bookkeeping;
they require no additional device logging. The harness records the CSV's
SHA-256 for later collation across runs.

The reference also writes `validation.csv`. On the official profile it probes
the first eight validation batches every 250 optimizer steps by default, then
records the exact canonical validation as its final FineWeb row. Fresh10 rows
may follow it. Probe synchronization
and evaluation are included in `train_seconds`; the final canonical evaluation
is not. Both training and evaluation executables compile once on synthetic
zero-valued inputs before timing. Use `--val-every 0` to disable probes, or
`--val-every N --val-probe-batches M` to change their cadence and prefix size.
Smoke and development runs do not probe unless explicitly enabled.

The reference is intentionally readable rather than target-capable. The full
19,073-step calibration on this TPU v4-8 processed exactly **624,984,064**
training tokens in **1,716.01 synchronized seconds** (28m36s, compilation
excluded), sustaining about **364k tokens/s**, **313 analytic TFLOP/s**, and
**28.5% analytic MFU**. It reached FineWeb validation loss **3.75788** and
Fresh10 macro loss **3.95959**. That exact token budget is now fixed for
official open-track comparisons while we improve the architecture, initialization,
optimizer, and kernels.

That calibration is a v4-8 result. A v4-32 run is validly recorded with its
16-device/four-process system identity, but its wall-clock score is not a
like-for-like hardware comparison with the original v4-8 number.

### TPU kernel experiments

The calibrated `make baseline` command keeps dense attention and dense output
loss explicit. The reference trainer also exposes two opt-in systems paths:

```bash
# Custom trainable Pallas attention; preserve the baseline output objective.
uv run --frozen --no-sync speedrun run reference --profile official -- \
  --attention-backend tpu_flash --loss-backend dense

# Bounded-memory tied output projection + cross entropy.
uv run --frozen --no-sync speedrun run reference --profile official -- \
  --attention-backend tpu_flash --loss-backend tiled
```

The custom attention kernel includes forward, dQ, and dK/dV kernels, uses a
shape-aware static tile plan, and safely pads non-128-aligned sequence lengths.
An exact runtime/source lookup table supplies measured seeds; an explicit
synthetic autotuner can populate a local cache before real compilation. The
tiled loss streams vocabulary blocks and recomputes them in backward instead of
materializing full logits. Switching loss implementations keeps all 50,304
storage classes by default; reducing `--semantic-vocab-size` changes the model
objective and is not a mere kernel toggle.

The canonical full-step benchmark improved from 93.196 ms to 75.191 ms with
custom attention and dense loss (435.79k tokens/s, +23.9%). The tiled loss was
77.048 ms, so it is retained for its memory bound rather than enabled by
default. These are synchronized short-step measurements, not a new full-run
score. Details, APIs, numerical checks, and tuning policy are in
[docs/KERNELS.md](docs/KERNELS.md).

### Profiling and reports

`make profile` uses the same four-chip data-parallel model shape and the first
100 updates of the 715-step warmup schedule. Compilation happens before capture;
canonical validation, Fresh10, checkpointing, and leaderboard recording are
disabled. The trace includes host sampling, host-to-device transfer, TPU
execution, synchronization, and collectives for ten steady-state steps. After
capture it starts an isolated, version-pinned XProf viewer at
`http://localhost:8791`; Ctrl-C stops the viewer. Override paths or the capture
window with Make variables such as `DATA_PATH`, `PROFILE_OUTPUT`,
`XPROF_START_STEP`, and `XPROF_STEPS`.

`make report` scans completed folders beneath `runs/`, checks their recorded
artifact hashes when an immutable record is available, and writes one
self-contained static `report.html` with no CDN dependency. A multi-select run
sidebar controls overlays. The baseline report admission qualification includes
complete official open runs only when they use the exact 624,984,064-token
budget and have validation loss at or below **3.76**; complete official
sample-efficiency runs must meet the same qualification. Smoke, development,
partial, and runs that do not meet it are omitted with a reported reason. The
baseline report admission qualification is intentionally not configurable and
is distinct from the official competition qualification of 3.28.

One global selector switches every time-series chart
between **equi-FLOP** (the default, using analytic cumulative estimated FLOPs)
and **equi-step**. Official reference runs record `diagnostics.csv` at step 1,
every 250 steps, and the final step. The report exposes separate
**Gradient / Update / Parameter** buttons, with one chart for each L1/L2 norm,
mean, standard deviation, and centered third/fourth moment. Timeline charts use
the whole-model values; final-snapshot charts show embeddings, every transformer
block, and the final normalization. Because the final parameter point is
post-update, it exactly describes the saved checkpoint even when qualifying-only
retention later removes that file. Compatible retained checkpoints are used only
as a legacy fallback for runs that predate `diagnostics.csv`.

Charts render into bounded, downsampled canvases only after an input or resize
event—there is no animation loop or idle redraw. Hover inspects the nearest
curve, the wheel zooms, pointer dragging pans, and reset/double-click restores
the full extent. The expand button opens any chart in a temporary full-panel
dialog; Escape closes it. All code and data remain inside the static HTML.

### Fresh-domain diagnostic

The canonical FineWeb validation loss remains the only qualification metric. A
separate `fresh10` diagnostic covers ten tiny, temporally fresh domains
with exactly 8,192 scored GPT-2 tokens each: science, medicine, software,
history, open-licensed fiction, government, legal, economics, climate, and
education. Source documents
are published after the pinned FineWeb snapshot, license-audited, cleaned by
a versioned deterministic recipe, and frozen by URL/revision plus SHA-256.

`fresh10` reports each domain independently and a macro-average beside FineWeb.
It does not change qualification, and “fresh” means a strong temporal contamination
control rather than a proof that no equivalent passage ever appeared online.
The one-file trainer accepts the frozen set with
`--downstream-manifest data/manifests/fresh10.json --downstream-root PATH`.
It reuses one fixed-shape masked evaluation executable, excludes padding and
cross-document targets, and writes the domain rows plus `fresh10_macro` to
`validation.csv`. Repeat `--downstream-data DOMAIN=PATH` for small standalone
pretokenized documents outside the canonical set.

Useful commands:

```bash
uv run --frozen --no-sync speedrun doctor --require-tpu
uv run --frozen --no-sync speedrun settings
uv run --frozen --no-sync speedrun verify RUN_ID
uv run --frozen --no-sync speedrun leaderboard --track open
uv run --frozen --no-sync speedrun leaderboard --track sample_efficiency
```

## Create an algorithm

Every entry has the same path contract:

```text
submissions/<algorithm>/train.py
```

Clone the current reference without overwriting anything:

```bash
uv run --frozen --no-sync speedrun clone reference my_experiment
```

Keep the interesting model, optimizer, schedule, and training logic visible in
that file. Fundamental shared data/protocol/UI utilities are welcome when they
make entries shorter and easier to compare.

## Tracks

- **Open:** reach the target in the least synchronized training time using the
  fixed 624,984,064-token official budget. Model, optimizer, batching,
  precision, kernels, and systems choices may all change.
- **Sample efficiency:** reach the target with the fewest predicted training
  tokens. The reference model/data/sequence contract is fixed; time breaks ties.

Both tracks deliberately permit systems work. There is no composite score.
Parameter count, FLOP estimates, throughput, MFU estimate, compilation time,
tokens, and loss are emitted as diagnostics in the live trainer and run record.

The full timing, qualification, checkpoint, and human-review rules are in
[docs/RULES.md](docs/RULES.md).

## Repository map

```text
data/manifests/          pinned datasets and hashes
harness/                 execution, validation, records, scoring
speedrun/                CLI, wizard, doctor, and shared data preparation
speedrun/kernels/        shared TPU attention, loss, and autotuning primitives
submissions/reference/   single-entry JAX reference trainer
tests/                   CPU-only infrastructure tests
runs/                    gitignored persistent run artifacts
```

The project is licensed under Apache-2.0.
