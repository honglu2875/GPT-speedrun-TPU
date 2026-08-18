"""Does XLA:TPU rewrite divide -> reciprocal*multiply under vmap?"""
import re
from collections import Counter
import jax, jax.numpy as jnp, numpy as np

print("backend:", jax.default_backend(), "devices:", len(jax.devices()))
rng = np.random.default_rng(0)
p = jnp.asarray(rng.normal(size=(64, 256)) * 0.02, jnp.float32)
g = jnp.asarray(rng.normal(size=(64, 256)) * 1e-3, jnp.float32)
m0 = jnp.zeros_like(p); v0 = jnp.zeros_like(p)
b1, b2 = 0.9, 0.95

def make(reciprocal):
    def upd(p, m, v, g, step):
        s = step.astype(jnp.float32)
        lr = 1.65e-4 * jnp.clip(1.0 - 0.5 * s / 1000.0, 0.1, 1.0)
        m2 = b1 * m + (1.0 - b1) * g
        v2 = b2 * v + (1.0 - b2) * jnp.square(g)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        if reciprocal:
            i1, i2 = 1.0 / c1, 1.0 / c2
            adam = (m2 * i1) * (1.0 / (jnp.sqrt(v2 * i2) + 1e-8))
        else:
            adam = (m2 / c1) / (jnp.sqrt(v2 / c2) + 1e-8)
        return p - lr * (adam + 0.1 * p)
    return upd

st = jnp.asarray(1, jnp.int32)
args = (p, m0, v0, g, st)
sargs = tuple(jnp.stack([q, q]) for q in args)
ops = lambda s: Counter(re.findall(r"=\s*\S+\s+(\w+)\(", s))

for label, rec in (("as written  a/b", False), ("explicit reciprocal", True)):
    f = make(rec)
    one = jax.jit(f)(*args)
    vm = jax.jit(jax.vmap(f))(*sargs)
    same = bool(jnp.array_equal(one, vm[0]))
    worst = float(jnp.abs(one - vm[0]).max())
    A = jax.jit(f).lower(*args).compile().as_text()
    B = jax.jit(jax.vmap(f)).lower(*sargs).compile().as_text()
    ca, cb = ops(A), ops(B)
    d = {k: (ca.get(k, 0), cb.get(k, 0)) for k in ("divide", "multiply", "rsqrt", "sqrt")
         if ca.get(k, 0) or cb.get(k, 0)}
    print(f"\n{label}:  bit-identical={same}  worst={worst:.3e}")
    for k, (a, b) in d.items():
        flag = "  <-- differs" if a != b else ""
        print(f"    {k:<10} unbatched {a:>3}   vmapped {b:>3}{flag}")
