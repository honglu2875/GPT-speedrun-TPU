# Batch-64 RoPE + RMSNorm + GELU

This accepted open-track candidate combines the 12×768/12-head RoPE, RMSNorm,
and parameter-matched GELU transformer with a global batch of 64. Its AdamW
peak learning rate is square-root scaled to `8.5e-4`, its floor remains exactly
`3e-5`, and warmup/probe cadence are token-matched to the batch-32 control. The
model has 123.67M parameters.

The official 624,984,064-token budget is 9,536½ global batches at batch 64.
`train.py` therefore compiles both the ordinary 64-sequence executable and an
exact 32-sequence final executable. The final optimizer step closes the budget
without padding, dropping tokens, or changing the measured-token contract;
training and diagnostic CSV counters also record the exact partial step.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
in **449.792 synchronized seconds** and reached canonical FineWeb loss
**3.6117** plus Fresh10 macro loss **3.7291**. The matched batch-32 GELU run
measured 602.819 seconds, 3.6455, and 3.8517 respectively, so batch 64 is both
153.0 seconds faster and substantially better on both validation suites. It
does not yet meet the 3.28 qualification target.

The recorded run is
`20260813T033713.195769Z-modern_rope_gelu_b64_lr85e4_min0353-a9afab88`.

```bash
make run TARGET=modern_rope_gelu_b64_lr85e4_min0353
```
