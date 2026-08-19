# Known issues

## Document masking fails when a chip holds more than one sequence

**Status:** fixed. Kept as the record of why it hid for so long.

Every 8k run compiled and trained fine until the seed-variance study was queued
on the v6e-8, where every run died before the first step:

```
The Pallas TPU lowering currently requires that the last two dimensions of your
block shape are divisible by 8 and 128 respectively, or be equal to the
respective dimensions of the overall array. Block spec for args[3] in
pallas_call tpu_flash_causal_attention_fwd ... has block shape
(Blocked(block_size=1), Blocked(block_size=512)), array shape (2, 8192)
```

`args[3]` is `segment_ids`, shaped `(local_batch, sequence)`. Its block spec is
`pl.BlockSpec((1, block), ...)` — one sequence at a time. Pallas accepts a block
dimension only when it divides the layout tile (8, for the second-to-last
dimension) **or** equals the array's own dimension.

### Why it stayed hidden

Every 8k run so far used batch 16 on the v4-32 — 16 chips, so exactly one
sequence per chip. There the block `1` equals the array dimension `1` and the
spec is legal. The v6e-8 has 8 chips, so batch 16 puts two sequences on each,
and `1` is neither 8-divisible nor equal to `2`.

`reference_8k` and `reference_moe` had only ever run on the v4-32, so nothing
had exercised a local batch above one.

### Where

`rig/kernels/tpu_flash_attention.py`, six specs across the three kernels
(forward, `bwd_dq`, `bwd_dkv`):

```
q_segment_spec      = pl.BlockSpec((1, tiles.block_q),      q_segment_index)
kv_segment_spec     = pl.BlockSpec((1, tiles.block_kv),     kv_segment_index)
dq_q_segment_spec   = pl.BlockSpec((1, tiles.block_q_dq),   dq_q_segment_index)
dq_kv_segment_spec  = pl.BlockSpec((1, tiles.block_kv_dq),  dq_kv_segment_index)
dkv_q_segment_spec  = pl.BlockSpec((1, tiles.block_q_dkv),  dkv_q_segment_index)
dkv_kv_segment_spec = pl.BlockSpec((1, tiles.block_kv_dkv), dkv_kv_segment_index)
```

### The fix

`segment_ids` now carries a singleton at the kernel boundary --
`segment_ids[:, None, :]` -- so the array is three-dimensional and the batch
axis sits outside the two Pallas constrains. The specs became
`pl.BlockSpec((None, None, block))`, squeezing batch and the singleton back off
so the kernel body still reads a one-dimensional `[block]` ref.

Squeezing alone does **not** work: `pl.BlockSpec((None, block))` over a
two-dimensional array still puts batch in the checked window, and Pallas
rejects it as `(Squeezed(), Blocked(...))` against array `(2, 1024)`. The array
itself has to gain the axis.

### Verifying a fix

Run it where the local batch is genuinely greater than one. The existing
coverage — `tests/test_tpu_flash_attention.py` and the gradient-against-oracle
check document masking landed with — passes today for the same reason the bug
hid, so a fix verified only there proves nothing.

### Verified

On the v6e-8 at one, two, and four sequences per chip. Against the module's own
dense oracle the error is 3.4-3.9e-03, sub-ULP for bfloat16, while against the
same oracle with the sequences' document layouts rolled between them it is 1.02
to 1.26 -- about 250x larger. That gap is what shows each sequence is masked by
its own boundaries rather than by a shared layout, which a compile-only check
could not distinguish.

`tests/test_tpu_flash_attention.py::SegmentBlockSpecTests` pins both the block
spec shape and the per-sequence masking semantics.
