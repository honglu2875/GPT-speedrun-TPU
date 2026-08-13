# RoPE + RMSNorm + SwiGLU

This accepted open-track candidate modernizes the reference transformer while
keeping its official token budget, global batch, sequence length, vocabulary,
attention kernel, dense output loss, and approximately 124M-parameter scale.

The official profile replaces learned absolute positions with base-10,000
rotary positions, LayerNorm with bias-free RMSNorm, and the GELU MLP with a
parameter-matched SwiGLU MLP of width 2,048. It also uses the validated AdamW
schedule control: a `6e-4` peak learning rate followed by cosine decay to
`3e-5` (`min_lr_ratio: 0.05`). The versioned `config.yaml` spells out the smoke,
development, and official profiles, while `train.py` records the added model
fields in both result and checkpoint metadata.

On four TPU v4-8 VMs (16 devices, four JAX processes), the full official run
trained 624,984,064 tokens in 605.257 synchronized seconds and reached canonical
FineWeb loss **3.6445** plus Fresh10 macro loss **3.8697**. The same-machine
reference run measured 581.925 seconds, 3.7531 FineWeb loss, and 3.9283 Fresh10
loss. This candidate therefore improves both validation suites with a 4.0%
training-time cost, but it does not yet meet the 3.28 qualification target.

The recorded run is
`20260813T014414.632621Z-modern_rope_swiglu_lr6e4_min05-08ea6762`.

Run it through the harness:

```bash
make run TARGET=modern_rope_swiglu_lr6e4_min05
```
