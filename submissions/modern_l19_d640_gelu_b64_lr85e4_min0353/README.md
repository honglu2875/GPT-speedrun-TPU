# 19-layer batch-64 RoPE + RMSNorm + GELU

This accepted open-track candidate is the canonical-quality-oriented companion
to `modern_rope_gelu_b64_lr85e4_min0353`. It retains the exact-budget batch-64
training implementation, RoPE, RMSNorm, GELU, and `8.5e-4` to `3e-5` AdamW
schedule, while changing the model shape from 12×768/12 heads/GELU-3072 to
19×640/10 heads/GELU-2560. The resulting model has 125.72M parameters.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
624,984,064 tokens in 497.221 synchronized seconds and reached canonical
FineWeb loss **3.6056** plus Fresh10 macro loss **3.7363**. The 12-layer
batch-64 candidate measured 449.792 seconds, 3.6117, and 3.7291 respectively.
This is therefore a narrow canonical-loss Pareto point: 0.0061 better on the
official metric for 47.4 additional seconds, while Fresh10 is 0.0072 worse. It
does not yet meet the 3.28 qualification target.

The recorded run is
`20260813T034851.094212Z-modern_l19_d640_gelu_b64_lr85e4_min0353-45a4bc08`.

```bash
make run TARGET=modern_l19_d640_gelu_b64_lr85e4_min0353
```
