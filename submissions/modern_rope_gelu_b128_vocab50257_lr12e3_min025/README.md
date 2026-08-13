# Batch-128 semantic-vocabulary RoPE + RMSNorm + GELU

This accepted open-track candidate retains 50,304 padded embedding/storage rows
for TPU-friendly tiling while restricting cross-entropy to GPT-2's 50,257
semantic tokens. It otherwise matches the exact-budget batch-128 candidate,
including its `1.2e-3` to `3e-5` schedule and compiled final global batch of 32.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
624,984,064 tokens in **383.614 synchronized seconds** and reached canonical
FineWeb loss **3.6144** plus Fresh10 macro loss **3.7076**. The padded-denominator
control measured 393.940 seconds, 3.6162, and 3.7051 respectively. Excluding
unreachable classes therefore creates a stronger speed frontier: 10.3 seconds
faster and 0.0018 better on the official metric, while Fresh10 regresses by
0.0025. It does not yet meet the 3.28 qualification target.

The recorded run is
`20260813T052208.157345Z-modern_rope_gelu_b128_vocab50257_lr12e3_min025-24cb5599`.

A seed-1338 replication trained in 384.080 seconds and reached canonical loss
3.6161 plus Fresh10 3.6985, confirming the throughput result while bounding
the small quality differences between neighboring candidates. Its run is
`20260813T053346.958152Z-modern_rope_gelu_b128_vocab50257_lr12e3_min025-1fba6aca`.

```bash
make run TARGET=modern_rope_gelu_b128_vocab50257_lr12e3_min025
```
