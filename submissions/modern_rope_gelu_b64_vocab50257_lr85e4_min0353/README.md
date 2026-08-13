# Batch-64 semantic-vocabulary RoPE + RMSNorm + GELU

This accepted open-track candidate retains 50,304 padded embedding/storage rows
for TPU-friendly tiling while restricting the cross-entropy denominator to the
50,257 tokens that GPT-2 can actually emit. Everything else matches the
exact-budget batch-64 GELU candidate, including its compiled final global batch
of 32.

On four TPU v4-8 VMs (16 devices, four JAX processes), the official run trained
624,984,064 tokens in **449.326 synchronized seconds** and reached canonical
FineWeb loss **3.6100** plus Fresh10 macro loss **3.7354**. The padded-denominator
control measured 449.792 seconds, 3.6117, and 3.7291 respectively. Excluding
unreachable classes therefore improves the official metric by 0.0017 and is
0.47 seconds faster, while Fresh10 regresses by 0.0063. It does not yet meet
the 3.28 qualification target.

The recorded run is
`20260813T050906.558498Z-modern_rope_gelu_b64_vocab50257_lr85e4_min0353-fd9b89ab`.

```bash
make run TARGET=modern_rope_gelu_b64_vocab50257_lr85e4_min0353
```
