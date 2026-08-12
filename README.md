# GPT Speedrun TPU

A collaborative GPT training speedrun for one Cloud TPU v4-8. Every algorithm
is a polished single-file JAX program named `train.py`; shared code handles
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

Install the locked Python 3.12/JAX/libtpu dependency environment once:

```bash
uv sync --frozen
```

Then launch the interactive preparation wizard:

```bash
uv run --frozen --no-sync speedrun prepare
```

It asks for the data-cache root, data and run profiles, persistent artifact
directory, default track, checkpoint retention, colors, and loss target. It can
then probe JAX/TPU health and prepare the selected dataset. Personal choices are
stored in the gitignored `.speedrun.toml`; official constants remain versioned
in Git. The personal target applies to smoke/development work; the official
target is fixed at 3.28 and may only be tightened explicitly.

On this node, `shm/` is a symlink to the 201 GiB RAM filesystem and is an ideal
ephemeral data cache:

```bash
uv run --frozen --no-sync speedrun prepare --path shm/ --profile official
```

The cache path is always explicit. The tool supports symlinks, resumable
downloads, free-space checks, and exact header/length/SHA-256 validation. It
never deletes unrelated files. Because `/dev/shm` is ephemeral, preparation may
need to restore the data after a reboot.

For automation, bypass the questions:

```bash
uv run --frozen --no-sync speedrun prepare \
  --non-interactive --path shm/ --profile official \
  --run-profile official --track open --checkpoints qualifying
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
# Fast end-to-end check
uv run --frozen --no-sync speedrun prepare --non-interactive \
  --path shm/ --profile smoke --run-profile smoke --no-doctor
uv run --frozen --no-sync speedrun run reference --profile smoke

# Initial v4-8 calibration
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
training tokens, loss, learning rate, and gradient norm. Scalars accumulate on
the TPU and transfer only after timed training, so retaining the complete curve
does not add a synchronization to every step. The harness records its SHA-256
for later collation across runs.

The reference also writes `validation.csv`. On the official profile it probes
the first eight validation batches every 250 optimizer steps by default, then
records the exact canonical validation as its final row. Probe synchronization
and evaluation are included in `train_seconds`; the final canonical evaluation
is not. Both training and evaluation executables compile once on synthetic
zero-valued inputs before timing. Use `--val-every 0` to disable probes, or
`--val-every N --val-probe-batches M` to change their cadence and prefix size.
Smoke and development runs do not probe unless explicitly enabled.

The reference is intentionally a readable baseline. A 20-step, official-shape
calibration on this TPU v4-8 sustained about **351k tokens/s** after compilation
(roughly 302 estimated TFLOP/s and 27% analytic MFU). Its current 19,073-step
schedule processes about 625M tokens and is not yet claimed to reach 3.28; use
it to measure the machine while optimized entries and a target-capable schedule
are developed.

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

- **Open:** reach the target in the least synchronized training time. Model,
  optimizer, batching, precision, kernels, and systems choices may all change.
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
submissions/reference/   self-contained JAX baseline
tests/                   CPU-only infrastructure tests
runs/                    gitignored persistent run artifacts
```

The project is licensed under Apache-2.0.
