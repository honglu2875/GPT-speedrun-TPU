# RoPE + RMSNorm + GELU

This accepted open-track candidate isolates the activation choice in the
modern transformer. It keeps RoPE, RMSNorm, the 12×768/12-head shape, TPU
FlashAttention, dense output loss, official token budget, and the `6e-4` to
`3e-5` AdamW schedule, while using a parameter-matched GELU MLP of width 3,072
instead of SwiGLU-2,048. The resulting model has 123.67M parameters.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
624,984,064 tokens in 602.819 synchronized seconds and reached canonical
FineWeb loss **3.6455** plus Fresh10 macro loss **3.8517**. The matched SwiGLU
candidate measured 605.257 seconds, 3.6445, and 3.8697 respectively. GELU is
therefore a useful Pareto point: 2.4 seconds faster, essentially tied on
canonical loss, and 0.018 better on Fresh10. It does not yet meet the 3.28
qualification target.

The recorded run is
`20260813T022906.226975Z-modern_rope_gelu_lr6e4_min05-0393c947`.

```bash
make run TARGET=modern_rope_gelu_lr6e4_min05
```
