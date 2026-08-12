# Reference

`train.py` is a readable pure-JAX GPT baseline, not a claimed record. It keeps
the model, AdamW optimizer, cosine schedule, batching, sharding, timing,
evaluation, and checkpoint logic in one file.

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
real validation tokens are inspected during compilation. Override the defaults
with `--val-every` and `--val-probe-batches`, or pass `--val-every 0` to disable
periodic probes.

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

The `official` defaults use the GPT-2-small shape (12 layers, width 768, 12
heads), sequence length 1024, global batch 32, BF16 compute, and an exact
624,984,064-token budget. This resolves to 19,073 optimizer steps. The full
v4-8 calibration trained in 1,716.01 synchronized seconds (compilation excluded),
at about 364k tokens/s and 28.5% analytic MFU, then reached FineWeb loss 3.75788
and Fresh10 macro loss 3.95959. It is the calibrated systems/reference baseline,
not yet a claim of reaching the 3.28 target.

For a bounded XProf diagnostic, pass `--xprof-dir`, `--xprof-start-step`, and
`--xprof-steps`. Combining those with `--no-final-validation --no-checkpoint`
skips evaluation compilation, all validation, checkpointing, and the competition
result event; it writes only `training.csv` and the trace. The top-level
`make profile` target supplies the complete 100-step reference command and
starts the viewer after capture. That template explicitly passes
`--diagnostics-every 0` so sparse reductions do not pollute the XProf window.

Use the harness instead of invoking this file directly so data and results are
validated and recorded:

```bash
uv run --frozen --no-sync speedrun run reference --profile smoke
uv run --frozen --no-sync speedrun run reference --profile official
```
