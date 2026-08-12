# Reference

`train.py` is a readable pure-JAX GPT baseline, not a claimed record. It keeps
the model, AdamW optimizer, cosine schedule, batching, sharding, timing,
evaluation, and checkpoint logic in one file.

The checkpoint contains model parameters and versioned model metadata. It is an
evaluation artifact, not a resumable training snapshot: Adam moments and the
input RNG are deliberately omitted from this compact baseline.

Every optimizer step is retained in `training.csv` with cumulative tokens,
training loss, learning rate, and gradient norm. The history is accumulated on
the device and copied after `train_seconds`, so curve capture does not introduce
a host synchronization on every step.

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
heads), sequence length 1024, global batch 32, and BF16 compute on TPU. The
initial 19,073-step schedule is deliberately marked **uncalibrated**: the first
v4-8 calibration measured about 351k tokens/s at the official shape, but this
625M-token schedule is not claimed to reach 3.28. The schedule and lower loss
milestones should be tuned from longer-run evidence.

Use the harness instead of invoking this file directly so data and results are
validated and recorded:

```bash
uv run --frozen --no-sync speedrun run reference --profile smoke
uv run --frozen --no-sync speedrun run reference --profile official
```
