# Batch-128 semantic-vocabulary LR 1.4e-3

This accepted open-track candidate raises the peak learning rate of the
50,257-class batch-128 GELU model from `1.3e-3` to `1.4e-3`, while retaining
the `3e-5` floor, exact 624,984,064-token budget, and compiled final global
batch of 32.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
in **383.825 synchronized seconds** and reached canonical FineWeb loss
**3.6095** plus Fresh10 macro loss **3.6956**. The seed-matched `1.3e-3`
candidate measured 383.842 seconds, 3.6116, and 3.7041 respectively. The
higher peak therefore improves canonical loss by 0.0021 and Fresh10 by 0.0085
while being 0.017 seconds faster. It does not yet meet the 3.28 qualification
target.

The recorded run is
`20260813T060943.104934Z-modern_rope_gelu_b128_vocab50257_lr14e3_min0214-be48fe2a`.

```bash
make run TARGET=modern_rope_gelu_b128_vocab50257_lr14e3_min0214
```
