# GPT Speedrun TPU

A collaborative GPT training speedrun for Cloud TPU v4 slices, from a v4-8 to
larger multi-host slices. Every algorithm is a polished JAX entry program named
`train.py`, with a sibling `config.yaml`; shared code handles
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
make run
make baseline
make sweep-lr
make profile
make report
```

| Target | Purpose |
|---|---|
| `make prepare` | synchronize the frozen uv environment, then open the interactive setup wizard |
| `make run` | verify every configured TPU VM and its data, then run the 125M reference tier using the saved profile |
| `make baseline` | compatibility alias for `make run` |
| `make run TARGET=name TIER=250m` | run one tier from the model family in `submissions/name/` |
| `make sweep-lr` | run or resume the CSV-first 60M–500M Complete(d)P LR study |
| `make profile` | run a distributed, validation-free 100-step diagnostic, capture worker 0 steps 11–20, then serve XProf on port 8791 |
| `make report` | integrity-check completed runs and rebuild the standalone `report.html` dashboard |

`make prepare` runs these two commands:

```bash
uv --cache-dir /tmp/uv-cache sync --frozen
uv --cache-dir /tmp/uv-cache run --frozen --no-sync speedrun prepare \
  --training-tokens 2600000000
```

It asks for the data-cache root, data and run profiles, persistent artifact
directory, TPU VM host count, default track, checkpoint retention, colors, a
smoke/development loss target, and an immutable corpus-capacity budget. The
saved budget selects the dataset used by every non-smoke run; the default 2.6B
request routes to the 4B corpus and covers the 500M × 5-TPP transfer point.
Training duration is independently specified by each family profile (20 TPP in
the reference official profile) or by an explicit research study.
The wizard can then probe JAX/TPU health and prepare the selected dataset;
only the explicit CLI flag `--no-download` skips data preparation. Personal
choices are stored in the gitignored `.speedrun.toml`; official constants
remain versioned in Git. The personal loss target applies only to
smoke/development work; the official target is fixed at 3.28 and may only be
tightened explicitly.

`make run` requires that saved file to contain a default profile; otherwise it
stops and asks for `make prepare`. `TARGET` defaults to `reference`. A custom
target must be a folder beneath `submissions/` containing regular, non-symlink
`train.py` and `config.yaml` files. New candidates use schema 2 family configs
with 60M, 125M, 250M, 500M, and 1B tiers; `TIER` defaults to `125m`.

For a multi-host run using the conventional `shm` cache, preparation requires
`/dev/shm` to be a writable `tmpfs` or `ramfs` on every VM and creates
`shm -> /dev/shm/.speedrun-cache` in each checkout. It uses `sudo -n` to make
that dedicated directory and completed entries root-owned but writable by the
caller's primary group. This is necessary because systemd-logind defaults to
`RemoveIPC=yes` and recursively removes user-owned `/dev/shm` entries after the
last SSH session for that user ends. Speedrun does not alter that host-wide
policy. If any VM lacks the mount or non-interactive cache ownership access,
preparation stops with a short instruction before downloading data:

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
list also works. The controller needs `pdsh`, `rsync`, and non-interactive SSH to
itself and every peer. Speedrun tests that access, but it never creates, copies,
or modifies SSH keys. If the probe fails, add this controller's public key to the
same user's `~/.ssh/authorized_keys` on the TPU VMs, verify
`pdsh -R ssh -w HOSTS hostname`, and rerun preparation.

After the SSH and RAM-cache probes succeed, preparation attempts a
non-interactive `apt-get` installation wherever `rsync` is missing. It then
incrementally copies the current checkout—including dirty and untracked
experiment files, but excluding Git metadata, the virtual environment,
`shm/`, data, caches, profiles, and run artifacts—to the same absolute path on
every peer. It installs `uv` there if needed, synchronizes the frozen
environment, then uses one `pdsh` launch to prepare the selected dataset and
validations concurrently in every VM's local protected RAM cache. Before each
worker's SSH command exits, it protects the completed entries from logout
cleanup. `make run` repeats the source synchronization and launches the trainer
on all configured hosts.

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

For official preparation, `training_tokens` selects the smallest corpus whose
nominal training capacity fits the requested budget:

| Requested preparation budget | Corpus | Training capacity | Cache root |
|---:|---|---:|---|
| up to 900M | classic | 900M | `<data-path>/` |
| 900M+1 through 1.9B | scaled `2B` | 1.9B | `<data-path>/fineweb-scaled/2B/` |
| 1.9B+1 through 3.9B | scaled `4B` | 3.9B | `<data-path>/fineweb-scaled/4B/` |
| 3.9B+1 through 7.9B | scaled `8B` | 7.9B | `<data-path>/fineweb-scaled/8B/` |
| 7.9B+1 through 74.9B | scaled `hero` | 74.9B | `<data-path>/fineweb-scaled/hero/` |

The saved `training_tokens` route is used by preparation, doctor, profiling,
and every non-smoke run, so a run cannot silently fall back to the classic
dataset after a scaled preparation. Scaled preparation is fail-closed and
starts working only after the corresponding immutable, URL-bearing publication
manifest is checked into `data/manifests/fineweb-scaled-gpt2/`; no placeholder
manifest is accepted. `--check-only` verifies the same routed manifest and
dedicated folder without mutation. Smoke and development profile selection is
unchanged.

Official evaluation covers exactly the first 10,485,760 validation predictions.
The harness requires official result events to report that exact coverage;
smoke and development profiles deliberately use shorter diagnostic evaluation.
The source is the pinned `kjj0/fineweb10B-gpt2` revision used by
[Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt). See
[data/README.md](data/README.md) for provenance and the binary contract.

## Run an algorithm

```bash
# Reference workflow: 125M by default, 20 TPP in the official profile
make run

# Select another size tier
make run TIER=250m

# Run a variant from submissions/dense_control/train.py
make run TARGET=dense_control TIER=125m

# Fast end-to-end check
uv run --frozen --no-sync speedrun prepare --non-interactive \
  --path shm/ --profile smoke --run-profile smoke --no-doctor
uv run --frozen --no-sync speedrun run reference --profile smoke

# Run the versioned official configuration (compare timings on like hardware)
uv run --frozen --no-sync speedrun run reference \
  --track open --profile official

# Short diagnostic overrides follow --; experiment settings stay in config.yaml
uv run --frozen --no-sync speedrun run reference --profile dev -- \
  --steps 100
```

The first family study is `make sweep-lr`. It holds batch, schedule, weight
decay, architecture rules, and 5 TPP fixed while sweeping only the normalized
base learning rate for 60M, 125M, 250M, and 500M. It writes and atomically
updates `runs/studies/complete_d_p_lr_v1/results.csv`; rerunning the command
resumes pending points. No chart is rendered yet. See
[the Complete(d)P contract](docs/COMPLETEP.md) and the
[versioned suite](studies/complete_d_p_lr_v1/suite.yaml).

The legacy-baseline v4 IsoFLOP study is an immutable incomplete archive: its
bounded lower-LR recovery stopped after 42 trials before a three-slice law could
be fitted. Its dedicated fail-closed publisher archives every raw curve and
derived decision, repeats semantic verification from an anonymous immutable
download, and records the result as negative evidence without making a
scaling-law claim. The fresh
[v5 continuation](sweeps/current_budget_isoflop_v5/README.md) extends the
prospective lower bound without reusing v4 measurements. See the
[v4 scaling-evidence release gate](docs/SCALING_EVIDENCE.md).

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
the first eight validation batches every 500 optimizer steps by default, then
records the exact canonical validation as its final FineWeb row. Fresh10 rows
may follow it. Probe synchronization
and evaluation are included in `train_seconds`; the final canonical evaluation
is not. Both training and evaluation executables compile once on synthetic
zero-valued inputs before timing. Use `--val-every N` for a temporary diagnostic
cadence override; change `val_probe_batches` in a cloned YAML profile when the
prefix size is part of the experiment. Smoke and development runs do not probe
unless explicitly enabled.

The reference is intentionally readable. Its v3 model family uses RoPE,
pre-RMSNorm, a 4× GELU MLP, untied embeddings, fixed 64-wide heads, and the
Complete(d)P rules documented in [docs/COMPLETEP.md](docs/COMPLETEP.md). The
60M/125M/250M tiers determine candidate-admission trends; 500M and 1B are
larger confirmation and hero tiers. The historical v1
19,073-step calibration on a TPU v4-8 processed exactly **624,984,064**
training tokens in **1,716.01 synchronized seconds** (28m36s, compilation
excluded), sustaining about **364k tokens/s**, **313 analytic TFLOP/s**, and
**28.5% analytic MFU**. It reached FineWeb validation loss **3.75788** and
Fresh10 macro loss **3.95959**. Those measurements remain labeled as v1 rather
than being attributed to the promoted architecture. The old token budget
remains a historical like-hardware calibration, not the training contract for
the new 20-TPP family ladder.

That calibration is a v4-8 result. A v4-32 run is validly recorded with its
16-device/four-process system identity, but its wall-clock score is not a
like-for-like hardware comparison with the original v4-8 number.

### TPU kernel baseline

The reference [`config.yaml`](submissions/reference/config.yaml) pins the custom
trainable Pallas attention with the memory-bounded tiled output loss. It also
preserves the family shapes, Complete(d)P contract, objective, schedule,
validation cadence, and TPP rule beside
the entry script. `make run` supplies only saved machine/run policy and lets the
trainer read that versioned file. To create a dense control, clone the reference
and change the clone's `attention_backend` field instead of hiding an algorithm
change in a long launch command:

```bash
uv run --frozen --no-sync speedrun clone reference dense_control
# edit submissions/dense_control/config.yaml
uv run --frozen --no-sync speedrun run dense_control --profile official
```

The custom attention kernel includes forward, dQ, and dK/dV kernels, uses a
shape-aware static tile plan, and safely pads non-128-aligned sequence lengths.
An exact runtime/source lookup table supplies measured seeds; an explicit
synthetic autotuner can populate a local cache before real compilation. The
tiled loss streams vocabulary blocks and recomputes them in backward instead of
materializing full logits. Switching loss implementations keeps all 50,304
storage classes by default; reducing `semantic_vocab_size` in a cloned YAML
profile changes the model objective and is not a mere kernel toggle.

The canonical full-step benchmark improved from 93.196 ms to 75.191 ms with
custom attention and dense loss (435.79k tokens/s, +23.9%). The tiled loss was
77.048 ms in that historical shape and is now the memory-safe family default.
Dense attention's completed 3.75788 run remains the historical quality
control until the promoted baseline has completed its full validation. Details,
APIs, numerical checks, and tuning policy are in
[docs/KERNELS.md](docs/KERNELS.md).

### Profiling and reports

`make profile` uses the profile saved by `make prepare` and launches the trainer
on every configured TPU VM, so the measured computation and collectives use the
real global topology. Compilation happens before capture; canonical validation,
Fresh10, checkpointing, and leaderboard recording are disabled. Worker 0 alone
records its host activity and four local TPU chips, including their participation
in global collectives. This avoids trying to merge independent local trace
sessions from four VM filesystems. After capture, worker 0 starts an isolated,
version-pinned XProf viewer at `http://localhost:8791`; Ctrl-C stops it. Override
the output or capture window with `PROFILE_OUTPUT`, `XPROF_START_STEP`, and
`XPROF_STEPS`.

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
The entry trainer accepts the frozen set with
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

Every entry has the same two-file path contract:

```text
submissions/<algorithm>/train.py
submissions/<algorithm>/config.yaml
```

Clone the current reference without overwriting anything:

```bash
uv run --frozen --no-sync speedrun clone reference my_experiment
```

The clone copies both files byte-for-byte. Keep the implementation visible in
`train.py` and the experiment-defining model, optimizer, schedule, objective,
kernel, and validation settings in `config.yaml`. Runtime locations, run
identity, seed, and profiling destinations remain command-line concerns.
Fundamental shared data/protocol/UI utilities are welcome when they make entries
shorter and easier to compare.

The YAML is schema-versioned and contains complete `smoke`, `dev`, and
`official` profiles. The trainer resolves it relative to its own file—not the
caller's working directory—and rejects duplicate/unknown keys, unsafe YAML
features, type/range errors, symlinks, and attempts to replace static settings
with hidden launch flags. The harness records both the source SHA-256 and the
fully resolved profile. Bounded operational overrides—duration and
instrumentation cadence—are recorded explicitly; fold any setting that defines
a new experiment back into the cloned YAML before publishing a result.

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
submissions/reference/   JAX entry trainer + versioned experiment config
sweeps/                  versioned non-competition IsoFLOP study definitions
tests/                   CPU-only infrastructure tests
runs/                    gitignored persistent run artifacts
```

The project is licensed under Apache-2.0.
