# 19-layer batch-128 RoPE + RMSNorm + GELU

This accepted open-track candidate combines the validated 19×640/10-head
GELU-2560 shape with the exact-budget batch-128 schedule. It uses a `1.2e-3`
peak learning rate, a `3e-5` floor, and a compiled final global batch of 32 so
the run processes exactly 624,984,064 tokens.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
the full token budget in **441.317 synchronized seconds** and reached canonical
FineWeb loss **3.6125** plus Fresh10 macro loss **3.6973**. The 12-layer
batch-128 candidate measured 393.940 seconds, 3.6162, and 3.7051 respectively.
The deeper shape is therefore a quality Pareto point: 0.0037 better canonical
loss and 0.0078 better Fresh10 for 47.4 additional seconds. It does not yet
meet the 3.28 qualification target.

The recorded run is
`20260813T044852.863172Z-modern_l19_d640_gelu_b128_lr12e3_min025-014f7e60`.

```bash
make run TARGET=modern_l19_d640_gelu_b128_lr12e3_min025
```
