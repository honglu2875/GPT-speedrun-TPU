# GPT Speedrun TPU rules

This benchmark asks a deliberately simple question: how quickly, or with how
few training tokens, can a single-file JAX trainer reach the target loss on one
Cloud TPU v4-8?

The project is collaborative. Submissions are short enough to review by hand;
the harness is intended to make honest experiments reproducible, not to be a
security boundary.

## Hardware and data

Official runs use one TPU v4-8 (four TPU v4 chips) and the repository's locked
Python environment. Training and validation consume the GPT-2-tokenized
FineWeb shards named by the official data manifest. Official validation is the
mean next-token cross-entropy over exactly the first 10,485,760 predictions;
the result protocol records that coverage. Smoke and development profiles use
shorter deterministic diagnostic evaluation and never enter this leaderboard.

The initial quality target is a validation loss at or below **3.28**. Results
below lower milestones such as 3.27 and 3.26 are retained in the run record so
the target can become more ambitious after the baseline is calibrated.

Downloading, hashing, and tokenizing data are never part of a timed run.
Training data may be staged in RAM before timing, but real host-to-device input
transfer is part of training time. Validation is outside training time.

## Tracks

### Open

The open track ranks qualifying runs by synchronized training seconds, lowest
first. Architecture, parameter count, optimizer, schedule, precision, batch
size, sequence length, sharding, kernels, data order, and other systems choices
may change. A submission must train from its declared initialization during the
run and may use only the official training split.

The leaderboard displays validation loss and tokens consumed alongside the
score. Parameter count, throughput, compilation time, and FLOP estimates remain
available in each run record as diagnostics, not terms in a composite score.

### Sample efficiency

The sample-efficiency track ranks qualifying runs by the number of predicted
training tokens consumed by forward/backward passes, lowest first. Synchronized
training time breaks an exact tie.

This track fixes the reference model contract, tokenizer, selected training
shards, validation prefix, and sequence length. Optimizer, schedule, precision,
batching, sampling order, regularization, and implementation may change. Every
predicted token participating in a forward/backward training loss counts,
including repetitions or overlapping sampled windows. Tokens used only for the
untimed synthetic compilation warm-up do not count.

## Timing

Submissions may compile every static training shape on synthetic data before
the timed region. Warm-up must use disposable model and optimizer state and may
not inspect real training or validation tokens.

Immediately before timing, the submission restores its declared initial state
and random generator. The timed region covers every real training step and its
input transfer. It ends only after the final device result has been synchronized
with the host (for example, using `jax.block_until_ready`). Evaluation and
checkpoint serialization follow outside the timed region.

Each official run receives a fresh persistent compilation-cache directory.
Reports include both synchronized training time and cold process wall time.

## Submission shape

Each algorithm is a directory containing the same entry filename:

```text
submissions/<algorithm>/train.py
```

Fundamental repository utilities—data validation, result protocol, terminal
presentation, and run recording—may be imported. The model, optimizer, schedule,
and novel training logic should remain visible in `train.py`. Avoid vendored
frameworks, generated code, or hidden algorithm-specific modules.

A completed run emits the versioned machine-readable result required by the
harness and writes a portable parameter checkpoint beneath its assigned output
directory. Human-oriented colored output must not corrupt the result record.
The last non-empty stdout line is `SPEEDRUN_RESULT=<json>`; human output belongs
on stderr. Version one requires `track`, `profile`, `seed`, a contained relative
`checkpoint`, and finite `metrics.train_seconds`, `metrics.tokens_processed`,
and `metrics.validation_loss`. Official results must additionally report
`metrics.validation_tokens = 10485760` and the exact single-host four-device
TPU v4 system identity. Extra finite JSON diagnostics are retained verbatim.
Optional named artifacts are contained within the run directory and recorded
with their size and SHA-256; the reference uses this for its per-step CSV curve.

## Qualification and records

Smoke and development profiles never enter an official leaderboard. An
official attempt records its source hash, repository state, lockfile and data
manifest hashes, selected shard names, seed, device/runtime information,
captured output, checkpoint hash, and declared metrics.

The version-one harness validates result structure, identities, finite metrics,
artifact containment, and hashes. It does not yet reload arbitrary submission
checkpoints or independently recompute loss; qualification is therefore
provisional and based on the submission's deterministic evaluation. Human
review checks the one-file evaluator and can rerun the captured command before
accepting a record. A future checkpoint-evaluator protocol can promote this to
automatic independent validation without changing the two scores.

Successful official attempts are appended to the immutable JSONL record.
Failed or timed-out processes retain their run directory and logs for review,
although version one does not yet append them to the leaderboard record. A
single qualifying run is **provisional**; confirmation across a fixed seed set
is planned once the reference schedule has been calibrated.

## Human review

Reviewers may reject results that violate the spirit of training from scratch,
use validation information for optimization, miscount tokens, exploit numerical
errors, or make the submitted result impractical to reproduce. New techniques
are welcome when their behavior is clear in the one-file implementation.
