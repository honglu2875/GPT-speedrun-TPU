# Batch-128 semantic-vocabulary LR 1.5e-3 / floor 9e-5

This accepted canonical-frontier candidate keeps the accepted 50,257-class
batch-128 GELU model's `1.5e-3` peak and raises its cosine floor from `6e-5`
to `9e-5`, retaining the exact 624,984,064-token budget and compiled final
global batch of 32.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
in **383.326 synchronized seconds** and reached canonical FineWeb loss
**3.6000** plus Fresh10 macro loss **3.6899**. The seed-matched `6e-5` floor
candidate measured 383.712 seconds, 3.6029, and 3.6881 respectively. The
higher floor therefore improves canonical loss by 0.0029 and saves 0.386
seconds, while Fresh10 regresses by 0.0018. That downstream delta is much
smaller than the observed cross-seed Fresh10 variation, so this candidate is
retained as the canonical-speed frontier rather than as a strict all-metrics
improvement. It does not yet meet the 3.28 qualification target.

The recorded run is
`20260813T065901.850451Z-modern_rope_gelu_b128_vocab50257_lr15e3_min0600-2c446191`.

```bash
make run TARGET=modern_rope_gelu_b128_vocab50257_lr15e3_min0600
```
