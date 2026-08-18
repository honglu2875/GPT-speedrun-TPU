# Can stacked models be bit-exact under vmap?

Verification for the model-stacking feature: run N models on one `model_batch`
axis so a small model saturates the chip better. The requirement was that slice
`i` bit-perfectly reproduce a standalone run with seed `i`.

Run these on the v6e-8, which is single-host, so one process can hold the TPU:

```bash
scp scratch/vmap-stacking/tpu_stack_probe.py <v6e-host>:/tmp/
ssh <v6e-host> '/home/cubic27/GPT-speedrun-TPU/.venv/bin/python3 /tmp/tpu_stack_probe.py'
```

## Answer

| configuration | gradients | params after 1 step | after 8 steps |
|---|---|---|---|
| dense attention + dense loss, **TPU** | 22/22 exact | 22/22 exact | 22/22 exact |
| `tpu_flash` + `tiled`, **TPU** | 0/22, 2.4e-04 | 1/22, 3.3e-04 | 0/22, 3.4e-04 |
| dense + dense, **CPU** | 22/22 exact | 5/22, 3.7e-09 | 4/22, 6.0e-08 |

**On TPU with dense backends, stacking is already bit-exact.** Nothing needs to
change for that path.

**With the Pallas flash kernel it is not.** The forward pass is exact -- the
loss matches bitwise -- and the gradients do not, which places it in the flash
*backward* kernels (`bwd_dq`, `bwd_dkv`). `vmap` over a `shard_map`-wrapped
`pallas_call` does not preserve the kernel's blocking.

The error is bounded and does not compound: 3.294e-04 after one step, 3.433e-04
after eight, flat thereafter. bf16 epsilon is 7.8e-03, so for a parameter of
typical magnitude (init_std 0.02) this is roughly two bf16 ULP.

## The CPU result is an artifact, and cost real time

CPU diverges even on dense backends, and chasing that is what most of this
investigation went into. The cause is XLA:CPU strength reduction: it rewrites
`a / b` into `a * (1/b)` when it sees the batched shapes and leaves the division
alone when unbatched. Those differ in the last ULP. Compiled HLO op counts:

```
op          unbatched  vmapped
divide            44       24     <- 20 divisions rewritten
multiply         251      273     <- into multiplies
maximum/minimum   22/22    0/0    <- clip fused to clamp (harmless)
clamp              0       22
```

Writing the reciprocal explicitly (`1.0/c1` hoisted, then multiply) makes CPU
bit-exact, 22/22. **This is not needed on TPU** -- the same probe shows XLA:TPU
does not perform the rewrite, and both forms are already bit-identical there.

Do not generalise CPU numerics to TPU. Every conclusion here that came from the
CPU run was wrong about the target hardware.

## What still needs deciding

The dev and official profiles use `tpu_flash` + `tiled`, so the real training
config is the one that is not bit-exact. Options, in rough order of cost:

1. Accept ~2 bf16 ULP, bounded and non-compounding, and state it in the recipe.
2. Thread the model axis through the flash kernel explicitly instead of
   `vmap`-ing over it, so the kernel sees it as a real batch dimension.
3. Use dense attention for stacked runs. Cheap at 1k context, expensive at 8k.
