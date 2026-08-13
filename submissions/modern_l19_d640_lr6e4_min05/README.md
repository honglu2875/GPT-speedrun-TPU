# 19-layer modern transformer

This accepted open-track candidate is the quality-oriented point from a
parameter-matched depth/width sweep. It uses the same RoPE, RMSNorm, SwiGLU,
TPU FlashAttention, dense output loss, token budget, and `6e-4` to `3e-5` AdamW
schedule as `modern_rope_swiglu_lr6e4_min05`, while changing the official shape
from 12×768/12 heads/SwiGLU-2048 to 19×640/10 heads/SwiGLU-1664.

The resulting model has 124.18M parameters. On four TPU v4-8 VMs (16 devices,
four JAX processes), it trained 624,984,064 tokens in 657.731 synchronized
seconds and reached canonical FineWeb loss **3.6315** plus Fresh10 macro loss
**3.8658**. The 12-layer modern candidate measured 605.257 seconds, 3.6445, and
3.8697 respectively, making this a slower but consistently better quality
Pareto point. It does not yet meet the 3.28 qualification target.

The recorded run is
`20260813T020252.981894Z-modern_l19_d640_lr6e4_min05-e5f33ad6`.

```bash
make run TARGET=modern_l19_d640_lr6e4_min05
```
