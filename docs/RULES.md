# GPT TPU rig rules

This benchmark asks a deliberately simple question: how quickly can a
single-entry JAX trainer reach the target loss on a declared Cloud TPU v4
slice?

The project is collaborative. Recipes are short enough to review by hand;
the harness is intended to make honest experiments reproducible, not to be a
security boundary.

## Hardware and data

Official runs use one Cloud TPU v4 slice with four TPU v4 chips per host and
the repository's locked Python environment. The supported reference topologies
range from one v4-8 host upward; the exact global device and process counts are
part of every result. Timing comparisons are meaningful only among matching
hardware topologies. Training and validation consume the GPT-2-tokenized
FineWeb shards named by the official data manifest. Official validation is the
mean next-token cross-entropy over exactly the first 10,485,760 predictions;
the result protocol records that coverage. Smoke and development profiles use
shorter deterministic diagnostic evaluation and never enter this leaderboard.

Here, "multi-host" means one TPU slice whose workers share the TPU ICI fabric
(for example, a four-host v4-32). Cloud TPU Multislice joins multiple slices
over DCN and is outside this version of the reference topology contract.

The initial quality target is a validation loss at or below **3.28**. Results
below lower milestones such as 3.27 and 3.26 are retained in the run record so
the target can become more ambitious after the baseline is calibrated.

Downloading, hashing, and tokenizing data are never part of a timed run.
Training data may be staged in RAM before timing, but real host-to-device input
transfer is part of training time. Periodic validation probes are also timed;
only the canonical final validation runs outside the training interval.

## Comparable cohorts

Qualifying full runs rank by synchronized training seconds, lowest first,
within one explicit content-addressed cohort. The cohort fixes the profile,
model tier and declared parameter anchor, target tokens per parameter and
complete-step rounding rule, selected training and validation data, tokenizer,
validation prefix, target loss, and exact accelerator topology. Optional
downstream evaluation identity is fixed when present. The full cohort object
and its SHA-256 identity are stored in every new rankable record.

Recipe, architecture (including dense versus MoE), optimizer, schedule,
precision, batch size, sequence length, learning rate, seed, sharding, kernels,
and data order remain experimental dimensions. A recipe must train from its
declared initialization during the run and may use only the cohort's immutable
training split. The leaderboard shows validation loss and tokens consumed next
to elapsed time; parameter count, throughput, compilation time, and FLOP
estimates are diagnostics rather than terms in a composite score.

Smoke runs and intentionally truncated diagnostic runs have no cohort and are
never ranked. Historical records without an explicit cohort remain inspectable
and may appear in historical reports, but the leaderboard does not infer a
cohort from incomplete legacy fields or mix different cohort identities.

## Timing

Recipes may compile every static training shape on synthetic data before
the timed region. Warm-up must use disposable model and optimizer state and may
not inspect real training or validation tokens.

Immediately before timing, the recipe restores its declared initial state
and random generator. The timed region covers every real training step and its
input transfer. It ends only after the final device result has been synchronized
with the host (for example, using `jax.block_until_ready`). Any periodic
validation performed between optimizer steps is part of the timed region;
recipes must synchronize the preceding training work before timing such a
probe. The canonical final evaluation and checkpoint serialization follow
outside the timed region.

Official reference runs also evaluate the versioned `fresh10` diagnostic after
training: ten domains with exactly 8,192 scored GPT-2 tokens apiece. Document
boundaries are reset and masked so no target crosses sources. Reports preserve
each domain's loss, perplexity, scored-token count, and evaluation time, plus
the arithmetic mean of domain losses (and its exponentiated perplexity). These
scores appear beside FineWeb for diagnosis only; they never affect the 3.28
qualification threshold or a cohort's ordering.

Each official run receives a fresh persistent compilation-cache directory.
Reports include both synchronized training time and cold process wall time.
XProf captures are separate diagnostic runs: profiling starts only after
compilation and synchronization, and no profiled timing enters the leaderboard.

## Recipe shape

Each algorithm is a directory containing the same entry and configuration
filenames:

```text
recipes/<algorithm>/train.py
recipes/<algorithm>/config.yaml
```

A schema-3 candidate directory defines a family, not one isolated shape. Its
entry must resolve the 60M, 125M, 250M, 500M, and 1B ballparks through `--tier`;
`make run` defaults to 125M. Candidate admission is based on the 60M, 125M, and
250M scaling trend. The 500M and 1B tiers are confirmation/hero runs and are not
required for every idea. Historical schema-1 folders remain provenance for old
runs and should be cloned from the current reference before new experiments.

Fundamental repository utilities—data validation, result protocol, terminal
presentation, and run recording—may be imported. The model, optimizer, schedule,
and novel training logic should remain visible in `train.py`; static experiment
choices belong in the strict, versioned sibling YAML. The harness owns the
configuration path, records its hash, and clones it with the entry script.
Runtime paths, seed, profile identity, and profiling destinations may stay as
arguments. Avoid vendored frameworks, generated code, or hidden
algorithm-specific modules.

A completed run emits the versioned machine-readable result required by the
harness and writes a portable parameter checkpoint beneath its assigned output
directory. Human-oriented colored output must not corrupt the result record.
The last non-empty stdout line is `RIG_RESULT=<json>`; human output belongs
on stderr. Version one requires `profile`, `seed`, a contained relative
`checkpoint`, and finite `metrics.train_seconds`, `metrics.tokens_processed`,
and `metrics.validation_loss`. Official results must additionally report
`metrics.validation_tokens = 10485760` and the configured TPU v4 system
identity: four local devices per JAX process and four times the process count
globally. Extra finite JSON diagnostics are retained verbatim.
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
artifact containment, and hashes. It does not yet reload arbitrary recipe
checkpoints or independently recompute loss; qualification is therefore
provisional and based on the recipe's deterministic evaluation. Human
review checks the entry-file evaluator and can rerun the captured command before
accepting a record. A future checkpoint-evaluator protocol can promote this to
automatic independent validation without changing cohort ranking.

Successful official attempts are appended to the immutable JSONL record.
Failed or timed-out processes retain their run directory and logs for review,
although version one does not yet append them to the leaderboard record. A
single qualifying run is **provisional**; confirmation across a fixed seed set
is planned once the reference schedule has been calibrated.

## Human review

Reviewers may reject results that violate the spirit of training from scratch,
use validation information for optimization, miscount tokens, exploit numerical
errors, or make the submitted result impractical to reproduce. New techniques
are welcome when their behavior is clear in the entry-file implementation and
its sibling experiment config.
