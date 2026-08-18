"""Which part of the flash path breaks bit-exactness under vmap?

Compares three levels: the raw kernel, the kernel inside its shard_map, and
each gradient separately (dq / dk / dv).
"""
import importlib.util, sys
from dataclasses import replace
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P

ROOT = Path("/home/cubic27/GPT-speedrun-TPU")
spec = importlib.util.spec_from_file_location("ref", ROOT/"recipes/reference/train.py")
t = importlib.util.module_from_spec(spec); sys.modules["ref"]=t; spec.loader.exec_module(t)
parser = t.build_parser(); args = parser.parse_args(["--profile","smoke"])
base = t.resolve_config(args, "tpu", 256)
devices = jax.devices(); mesh = Mesh(np.asarray(devices).reshape(len(devices)), ("data",))
cfg = replace(base, layers=2, d_model=128, heads=2, seq_len=1024, batch_size=8,
              attention_backend="tpu_flash", loss_backend="tiled",
              vocab_size=256, semantic_vocab_size=256, vocab_tile_size=256,
              compute_dtype=jnp.bfloat16, dtype_name="bfloat16")
tiles = t.prepare_attention_runtime(args, cfg, devices).tiles
B, S, H, Dh = 8, cfg.seq_len, cfg.heads, cfg.d_model // cfg.heads
rng = np.random.default_rng(0)
mk = lambda: jnp.asarray(rng.normal(size=(B,H,S,Dh))*0.1, jnp.bfloat16)  # BHSD
q, k, v = mk(), mk(), mk()
qs, ks, vs = (jnp.stack([z,z]) for z in (q,k,v))

raw = t.make_causal_attention(t.AttentionConfig(
    backend=cfg.attention_backend, tiles=tiles,
    softmax_scale=t.attention_softmax_scale(cfg)))
wrapped = t.make_mesh_attention(cfg, mesh, tiles)

def report(label, fn, a, b, c, sa, sb, sc):
    out1 = jax.jit(fn)(a,b,c)
    outv = jax.jit(jax.vmap(fn))(sa,sb,sc)
    fwd = bool(jnp.array_equal(out1, outv[0]))
    print(f"\n  {label}")
    print(f"    forward            exact={fwd}  worst="
          f"{float(jnp.abs(out1.astype(jnp.float32)-outv[0].astype(jnp.float32)).max()):.3e}")
    loss = lambda x,y,z: jnp.sum(fn(x,y,z).astype(jnp.float32)**2)
    g1 = jax.jit(jax.grad(loss, argnums=(0,1,2)))(a,b,c)
    gv = jax.jit(jax.vmap(jax.grad(loss, argnums=(0,1,2))))(sa,sb,sc)
    for name, s, m in zip(("d/dq","d/dk","d/dv"), g1, gv):
        d = float(jnp.abs(s.astype(jnp.float32)-m[0].astype(jnp.float32)).max())
        print(f"    {name:<18} exact={bool(jnp.array_equal(s,m[0]))}  worst={d:.3e}")

# The raw kernel must run without an active mesh, or auto-partitioning refuses it.
report("raw kernel, single device", raw, q,k,v, qs,ks,vs)
with jax.set_mesh(mesh):
    report("kernel inside shard_map", wrapped, q,k,v, qs,ks,vs)
