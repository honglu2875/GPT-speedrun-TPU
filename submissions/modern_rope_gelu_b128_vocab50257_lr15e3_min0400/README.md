# Batch-128 semantic-vocabulary LR 1.5e-3 / floor 6e-5

This accepted open-track candidate keeps the accepted 50,257-class
batch-128 GELU model's `1.5e-3` peak and raises its cosine floor from `3e-5`
to `6e-5`, retaining the exact 624,984,064-token budget and compiled final
global batch of 32.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
in **383.712 synchronized seconds** and reached canonical FineWeb loss
**3.6029** plus Fresh10 macro loss **3.6881**. The seed-matched `3e-5` floor
control measured 383.432 seconds, 3.6049, and 3.6906 respectively. The higher
floor therefore improves canonical loss by 0.0020 and Fresh10 by 0.0025 for
0.280 additional seconds. It does not yet meet the 3.28 qualification target.

The recorded run is
`20260813T064753.833959Z-modern_rope_gelu_b128_vocab50257_lr15e3_min0400-da23811b`.

```bash
make run TARGET=modern_rope_gelu_b128_vocab50257_lr15e3_min0400
```
