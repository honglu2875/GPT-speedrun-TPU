# Reference

`train.py` is a readable pure-JAX GPT baseline, not a claimed record. It keeps
the model, AdamW optimizer, batching, sharding, timing, evaluation, and
checkpoint logic in one entry file. The sibling `config.yaml` is the versioned
experiment definition: each profile records its model shape, token budget,
precision, kernel choices, optimizer schedule, and validation cadence.

The checkpoint contains model parameters and versioned model metadata. It is an
evaluation artifact, not a resumable training snapshot: Adam moments and the
input RNG are deliberately omitted from this compact baseline.

Every optimizer step is retained in `training.csv` with cumulative tokens,
analytic cumulative estimated FLOPs, training loss, learning rate, and gradient
norm. The history is accumulated on the device and copied after `train_seconds`,
so curve capture does not introduce a host synchronization on every step. The
token, schedule, and FLOP columns are derived without device-side logging.

Official runs also write sparse, long-form `diagnostics.csv` points at step 1,
every 250 steps, and the final step. Set `--diagnostics-every N` to change the
cadence or `--diagnostics-every 0` to disable them; smoke and development
profiles default to disabled. Each point covers the whole model, embeddings,
every zero-based transformer block, and the final normalization, with
`param`/`grad`/`update` families and L1 norm, L2 norm, mean, standard deviation,
third centered moment, and fourth centered moment. Parameters are observed after
the sampled update, so the final parameter point exactly matches the checkpoint;
gradients are observed before global clipping, and updates are the signed actual
parameter delta after clipping, AdamW, and weight decay. Centered moments use a
two-pass calculation.

The diagnostic step is a separately compiled version of the same optimizer
update. Ordinary steps continue to use the compact baseline executable, and an
exact CPU regression test checks that both executables produce identical model
and optimizer state. Compilation remains outside `train_seconds`; all diagnostic
device computation and synchronization are inside it. Values are transferred
and the CSV is atomically written only after the synchronized training clock.

`validation.csv` records periodic fixed-prefix probes plus the canonical final
evaluation. Official runs default to eight batches every 250 optimizer steps;
smoke and development runs default to no probes. Each probe synchronizes the
preceding training update, and the entire probe pause counts toward
`train_seconds`. The canonical final evaluation remains outside that clock.
The evaluation executable is compiled once on synthetic zeros and reused; no
real validation tokens are inspected during compilation. Temporarily override
the cadence with `--val-every` (including `--val-every 0`); change
`val_probe_batches` in a cloned config when the prefix size is experimental.

When the harness supplies the pinned Fresh10 manifest, the same compiled masked
evaluation executable scores ten post-FineWeb domains after canonical
validation. `validation.csv` places their individual loss, perplexity, exact
target count, and time beside the FineWeb rows and finishes with a Fresh10 macro
row. Context never crosses a manifest document boundary. These diagnostic
scores do not affect qualification or `train_seconds`.

The direct trainer interface is `--downstream-manifest MANIFEST
--downstream-root SHARD_DIRECTORY`. For ad-hoc checks, repeat
`--downstream-data DOMAIN=PATH`; every path is one document, so its first token
provides context but is not scored. Omitting both forms skips downstream
evaluation while preserving the FineWeb result contract.

The `official` YAML profile uses the GPT-2-small shape (12 layers, width 768, 12
heads), sequence length 1024, global batch 32, BF16 compute, and an exact
624,984,064-token budget. This resolves to 19,073 optimizer steps. The full
v4-8 calibration trained in 1,716.01 synchronized seconds (compilation excluded),
at about 364k tokens/s and 28.5% analytic MFU, then reached FineWeb loss 3.75788
and Fresh10 macro loss 3.95959. It is the calibrated systems/reference baseline,
not yet a claim of reaching the 3.28 target.

## TPU kernel baseline

The official YAML profile selects `attention_backend: tpu_flash` and
`loss_backend: dense`. The top-level `make baseline` reads those settings
without repeating them as flags. This promotes the
hardware-validated attention improvement without changing the output objective,
model, schedule, or token budget. The smoke and development YAML profiles stay
dense. A cloned config can select any of three attention implementations for
controlled comparisons:

- `attention_backend: tpu_flash` selects the trainable custom Pallas causal
  attention forward, dQ, and dK/dV kernels. It currently requires TPU BF16,
  supports head dimensions up to 128 (divisible by 8), and safely right-pads
  arbitrary sequence lengths to 128-wide tiles. `jax_flash` is retained as a
  JAX-provided control.
- `attention_backend: dense` retains the readable materialized-attention
  control used for the completed 3.75788 calibration.
- `loss_backend: tiled` streams the tied output projection and cross entropy
  over vocabulary tiles, with an online FP32 log-sum-exp and a recomputing
  custom VJP. It never constructs the complete token-by-vocabulary logits.

Both loss backends use the full storage vocabulary in the reference config, so
switching the implementation alone preserves the objective. Changing
`semantic_vocab_size` to 50,257 in a cloned open-track config is a deliberate
model-semantic change; sample-efficiency pins the calibrated value of 50,304.

For non-dense attention, the trainer resolves a static ten-field tile plan
before compiling the real step. Resolution checks an exact runtime-fingerprinted
cache, then a source-pinned shipped entry, then a deterministic shape heuristic.
Pass `--attention-tuning-cache PATH --autotune-attention` to benchmark bounded
synthetic candidates explicitly. Tuning never reads the dataset or occurs
inside `jax.jit`; its time is reported separately from `train_seconds`.

On this v4-8, an identical full-step benchmark measured 93.196 ms for the
dense reference, 75.191 ms for custom TPU FlashAttention with dense loss, and
77.048 ms with the tiled loss. The custom-attention/dense-loss variant reached
435.79k tokens/s, a 23.9% throughput increase in the isolated step benchmark.
The tiled loss remains optional because its bounded-logit memory behavior cost
2.47% in that complete-step comparison. See [the kernel design and benchmark
notes](../../docs/KERNELS.md) for APIs, correctness scope, autotuning policy,
and microbenchmarks.

For a bounded XProf diagnostic, pass `--xprof-dir`, `--xprof-start-step`, and
`--xprof-steps`. Combining those with `--no-final-validation --no-checkpoint`
skips evaluation compilation, all validation, checkpointing, and the competition
result event; it writes only `training.csv` and the trace. The top-level
`make profile` target reads the same official YAML and overrides only its
bounded diagnostic duration/instrumentation before starting the viewer. That
template explicitly passes
`--diagnostics-every 0` so sparse reductions do not pollute the XProf window.

Use the harness instead of invoking this file directly so data and results are
validated and recorded:

```bash
uv run --frozen --no-sync speedrun run reference --profile smoke
make baseline
```
