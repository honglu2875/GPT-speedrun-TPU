"""Is the tiled cross-entropy bit-exact under vmap, forward and backward?"""
import sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/cubic27/GPT-speedrun-TPU")
from rig.kernels.linear_cross_entropy import tiled_tied_cross_entropy

B, S, D, V, TILE = 8, 1024, 128, 256, 256
rng = np.random.default_rng(0)
h = jnp.asarray(rng.normal(size=(B, S, D)) * 0.1, jnp.bfloat16)
e = jnp.asarray(rng.normal(size=(V, D)) * 0.02, jnp.bfloat16)
y = jnp.asarray(rng.integers(0, V, size=(B, S)), jnp.int32)
hs, es, ys = jnp.stack([h, h]), jnp.stack([e, e]), jnp.stack([y, y])

f = lambda hh, ee, yy: tiled_tied_cross_entropy(
    hh, ee, yy, semantic_vocab_size=V, vocab_tile_size=TILE, compute_dtype=jnp.bfloat16)

one = jax.jit(f)(h, e, y)
vm = jax.jit(jax.vmap(f))(hs, es, ys)
print(f"  forward          exact={bool(jnp.array_equal(one, vm[0]))}  "
      f"worst={float(jnp.abs(one - vm[0]).max()):.3e}")

g1 = jax.jit(jax.grad(f, argnums=(0, 1)))(h, e, y)
gv = jax.jit(jax.vmap(jax.grad(f, argnums=(0, 1))))(hs, es, ys)
for name, s, m in zip(("grad wrt hidden", "grad wrt embedding"), g1, gv):
    d = float(jnp.abs(s.astype(jnp.float32) - m[0].astype(jnp.float32)).max())
    print(f"  {name:<18} exact={bool(jnp.array_equal(s, m[0]))}  worst={d:.3e}")

# Does the tile count matter? A single tile removes the fori_loop accumulation.
for tile in (V, V // 2, V // 4):
    g = lambda hh, ee, yy: tiled_tied_cross_entropy(
        hh, ee, yy, semantic_vocab_size=V, vocab_tile_size=tile, compute_dtype=jnp.bfloat16)
    a = jax.jit(jax.grad(g, argnums=(0, 1)))(h, e, y)
    b = jax.jit(jax.vmap(jax.grad(g, argnums=(0, 1))))(hs, es, ys)
    ex = [bool(jnp.array_equal(u, w[0])) for u, w in zip(a, b)]
    print(f"  vocab_tile_size={tile:<4} ({V//tile} tiles)  dHidden exact={ex[0]}  dEmbed exact={ex[1]}")
