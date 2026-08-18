"""Does the real config -- tpu_flash attention + tiled loss -- survive vmap?"""
import importlib.util, sys, traceback
from dataclasses import replace
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh

ROOT = Path("/home/cubic27/GPT-speedrun-TPU")
spec = importlib.util.spec_from_file_location("ref", ROOT / "recipes/reference/train.py")
t = importlib.util.module_from_spec(spec); sys.modules["ref"] = t; spec.loader.exec_module(t)
print("backend:", jax.default_backend(), "devices:", len(jax.devices()))

parser = t.build_parser()
base = t.resolve_config(parser.parse_args(["--profile", "smoke"]), "tpu", 256)
cfg = replace(base, layers=2, d_model=128, heads=2, seq_len=1024, batch_size=8,
              attention_backend="tpu_flash", loss_backend="tiled",
              vocab_size=256, semantic_vocab_size=256, vocab_tile_size=256,
              compute_dtype=jnp.bfloat16, dtype_name="bfloat16")
devices = jax.devices()
mesh = Mesh(np.asarray(devices).reshape(len(devices)), ("data",))
args = parser.parse_args(["--profile", "smoke"])
runtime = t.prepare_attention_runtime(args, cfg, devices)
tiles = getattr(runtime, "tiles", None)
print("attention tiles:", tiles)

pa, pb = t.init_params(cfg, 1337), t.init_params(cfg, 1338)
o  = jax.tree_util.tree_map(jnp.asarray, t.init_optimizer(pa, cfg.steps))
ob = jax.tree_util.tree_map(jnp.asarray, t.init_optimizer(pb, cfg.steps))
rng = np.random.default_rng(0)
tok = jnp.asarray(rng.integers(0, cfg.semantic_vocab_size, size=(8, cfg.seq_len + 1)))
x, y = tok[:, :-1], tok[:, 1:]
sp = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), pa, pb)
so = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), o, ob)

with jax.set_mesh(mesh):
    attn = t.make_mesh_attention(cfg, mesh, tiles)
    f = lambda p, oo, a, b: t._apply_training_update(p, oo, a, b, cfg, None, attn)
    print("\n-- unbatched (tpu_flash + tiled) --")
    p1, o1, m1, g1 = jax.jit(f)(pa, o, x, y)
    print(f"   ok, loss {float(m1['loss']):.6f}")
    print("-- vmapped --")
    try:
        pv, ov, mv, gv = jax.jit(jax.vmap(f))(sp, so, jnp.stack([x, x]), jnp.stack([y, y]))
        def cmp(label, a, b):
            ls = jax.tree_util.tree_leaves(a); lv = [q[0] for q in jax.tree_util.tree_leaves(b)]
            ex = sum(bool(jnp.array_equal(u, v)) for u, v in zip(ls, lv))
            w = max(float(jnp.abs(u - v).max()) for u, v in zip(ls, lv))
            print(f"   {label:<24} {ex}/{len(ls)} exact  worst {w:.3e}")
        cmp("gradients", g1, gv)
        cmp("PARAMS after 1 step", p1, pv)
        print(f"   loss exact={bool(jnp.array_equal(m1['loss'], mv['loss'][0]))}")
    except Exception:
        print("   VMAP FAILED:")
        traceback.print_exc(limit=6)
