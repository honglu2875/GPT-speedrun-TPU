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

The report has a separate, versioned baseline report admission qualification:
validation loss at or below **3.76**. Generated reports include only complete
official runs that meet this qualification; excluded runs and their immutable
artifacts remain on disk. The baseline report admission qualification is not
user-configurable and never changes the official competition qualification of
3.28, checkpoint retention, leaderboard eligibility, or ranking.

Downloading, hashing, and tokenizing data are never part of a timed run.
Training data may be staged in RAM before timing, but real host-to-device input
transfer is part of training time. Periodic validation probes are also timed;
only the canonical final validation runs outside the training interval.

## Tracks

### Open

The open track ranks qualifying runs by synchronized training seconds, lowest
first. Every official open attempt predicts exactly **624,984,064 training
tokens**, preserving the calibrated `19,073 × 32 × 1,024` reference budget.
Architecture, parameter count, optimizer, schedule, precision, batch size,
sequence length, sharding, kernels, data order, and other systems choices may
change. The selected global batch and sequence length must currently divide the
fixed budget exactly. A submission must train from its declared initialization
during the run and may use only the official training split.

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
with the host (for example, using `jax.block_until_ready`). Any periodic
validation performed between optimizer steps is part of the timed region;
submissions must synchronize the preceding training work before timing such a
probe. The canonical final evaluation and checkpoint serialization follow
outside the timed region.

Official reference runs also evaluate the versioned `fresh10` diagnostic after
training: ten domains with exactly 8,192 scored GPT-2 tokens apiece. Document
boundaries are reset and masked so no target crosses sources. Reports preserve
each domain's loss, perplexity, scored-token count, and evaluation time, plus
the arithmetic mean of domain losses (and its exponentiated perplexity). These
scores appear beside FineWeb for diagnosis only; they never affect the 3.28
qualification threshold or either track's ordering.

Each official run receives a fresh persistent compilation-cache directory.
Reports include both synchronized training time and cold process wall time.
XProf captures are separate diagnostic runs: profiling starts only after
compilation and synchronization, and no profiled timing enters the leaderboard.

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
Its curve records steps, predicted tokens, analytic cumulative estimated FLOPs,
learning rate, loss, and gradient norm. Estimated FLOPs are a versioned analytic
model diagnostic rather than a hardware-counter score.

The reference additionally records sparse optimizer diagnostics at step 1,
every 250 steps, and the final step. Each point contains post-update parameter,
pre-clipping gradient, and signed actual-update L1/L2 norms, mean, population
standard deviation, and centered third/fourth moments for the whole model and
each logical scope. Diagnostic compilation is outside `train_seconds`; sampled
device computation and synchronization are inside it. These measurements never
affect qualification or ranking.

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
