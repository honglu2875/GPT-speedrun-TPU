# Batch-128 RoPE + RMSNorm + GELU

This accepted open-track candidate raises the global batch from 64 to 128 for
the 12×768/12-head RoPE, RMSNorm, GELU-3072 transformer. It uses a `1.2e-3`
peak learning rate, a `3e-5` floor, and the exact official token budget. Since
624,984,064 tokens are not divisible by 128×1,024, the compiled final step uses
a global batch of 32 rather than silently dropping or adding tokens.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
624,984,064 tokens in **393.940 synchronized seconds** and reached canonical
FineWeb loss **3.6162** plus Fresh10 macro loss **3.7051**. The batch-64
candidate measured 449.792 seconds, 3.6117, and 3.7291 respectively. Batch 128
is therefore a strong speed/quality Pareto point: 55.9 seconds faster for only
0.0045 canonical-loss regression, while Fresh10 improves by 0.0240. It does
not yet meet the 3.28 qualification target.

The recorded run is
`20260813T040252.915518Z-modern_rope_gelu_b128_lr12e3_min025-8b8bcad6`.

```bash
make run TARGET=modern_rope_gelu_b128_lr12e3_min025
```
