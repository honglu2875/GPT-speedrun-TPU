# Batch-128 semantic-vocabulary LR 1.3e-3

This accepted open-track candidate raises the peak learning rate of the
50,257-class batch-128 GELU model from `1.2e-3` to `1.3e-3`, retaining the
`3e-5` floor, exact 624,984,064-token budget, and compiled final global batch
of 32.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
in **383.842 synchronized seconds** and reached canonical FineWeb loss
**3.6116** plus Fresh10 macro loss **3.7041**. The seed-matched `1.2e-3`
control measured 383.614 seconds, 3.6144, and 3.7076 respectively. The higher
peak therefore improves canonical loss by 0.0028 and Fresh10 by 0.0035 for
0.23 additional seconds. It does not yet meet the 3.28 qualification target.

The recorded run is
`20260813T055234.601300Z-modern_rope_gelu_b128_vocab50257_lr13e3_min0231-7c4cc4f1`.

```bash
make run TARGET=modern_rope_gelu_b128_vocab50257_lr13e3_min0231
```
