# Batch-128 semantic-vocabulary LR 1.5e-3

This accepted open-track candidate raises the peak learning rate of the
50,257-class batch-128 GELU model from `1.4e-3` to `1.5e-3`, while retaining
the `3e-5` floor, exact 624,984,064-token budget, and compiled final global
batch of 32.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
in **383.432 synchronized seconds** and reached canonical FineWeb loss
**3.6049** plus Fresh10 macro loss **3.6906**. The seed-matched `1.4e-3`
candidate measured 383.825 seconds, 3.6095, and 3.6956 respectively. The
higher peak therefore improves canonical loss by 0.0046 and Fresh10 by 0.0050
while being 0.393 seconds faster. It does not yet meet the 3.28 qualification
target.

The recorded run is
`20260813T062057.933780Z-modern_rope_gelu_b128_vocab50257_lr15e3_min0200-8e6ce590`.

```bash
make run TARGET=modern_rope_gelu_b128_vocab50257_lr15e3_min0200
```
