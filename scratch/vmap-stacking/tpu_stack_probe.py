"""Is the real reference train_step bit-exact under vmap on TPU?

Dumps gradients, optimizer moments, and parameters for an unbatched run and a
vmapped stack, and compares slice 0 against the unbatched result.
"""
import importlib.util, sys
from dataclasses import replace
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

ROOT = Path("/home/cubic27/GPT-speedrun-TPU")
spec = importlib.util.spec_from_file_location("ref", ROOT / "recipes/reference/train.py")
t = importlib.util.module_from_spec(spec); sys.modules["ref"] = t; spec.loader.exec_module(t)

print("backend:", jax.default_backend(), "devices:", len(jax.devices()))
parser = t.build_parser()
base = t.resolve_config(parser.parse_args(["--profile", "smoke"]), "tpu", 256)

def compare(label, a_tree, b_tree, slice_index=0):
    ls = jax.tree_util.tree_leaves(a_tree)
    lv = [q[slice_index] for q in jax.tree_util.tree_leaves(b_tree)]
    ex = sum(bool(jnp.array_equal(x, y)) for x, y in zip(ls, lv))
    worst = max(float(jnp.abs(x - y).max()) for x, y in zip(ls, lv))
    print(f"    {label:<26} {ex}/{len(ls)} exact   worst {worst:.3e}")
    return ex == len(ls)

for tag, over in (("dense attention + dense loss", {}),):
    cfg = replace(base, layers=2, d_model=128, heads=2, seq_len=64, **over)
    print(f"\n== {tag}: layers={cfg.layers} d_model={cfg.d_model} seq={cfg.seq_len} "
          f"attn={cfg.attention_backend} loss={cfg.loss_backend}")
    pa, pb = t.init_params(cfg, 1337), t.init_params(cfg, 1338)
    o = jax.tree_util.tree_map(jnp.asarray, t.init_optimizer(pa, cfg.steps))
    ob = jax.tree_util.tree_map(jnp.asarray, t.init_optimizer(pb, cfg.steps))
    rng = np.random.default_rng(0)
    tok = jnp.asarray(rng.integers(0, cfg.semantic_vocab_size, size=(2, cfg.seq_len + 1)))
    x, y = tok[:, :-1], tok[:, 1:]
    sp = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), pa, pb)
    so = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), o, ob)
    xs, ys = jnp.stack([x, x]), jnp.stack([y, y])

    f = lambda p, oo, a, b: t._apply_training_update(p, oo, a, b, cfg, None)
    p1, o1, m1, g1 = jax.jit(f)(pa, o, x, y)
    pv, ov, mv, gv = jax.jit(jax.vmap(f))(sp, so, xs, ys)
    ok = []
    ok.append(compare("gradients", g1, gv))
    ok.append(compare("optimizer m", o1["m"], ov["m"]))
    ok.append(compare("optimizer v", o1["v"], ov["v"]))
    ok.append(compare("PARAMS after 1 step", p1, pv))
    print(f"    loss  single {float(m1['loss']):.10f}  vmap {float(mv['loss'][0]):.10f}  "
          f"exact={bool(jnp.array_equal(m1['loss'], mv['loss'][0]))}")
    print(f"    => fully bit-exact: {all(ok)}")

    # multi-step, since single-step agreement can hide compounding drift
    step = jax.jit(lambda p, oo, a, b: t.train_step(p, oo, a, b, cfg, None))
    vstep = jax.jit(jax.vmap(lambda p, oo, a, b: t.train_step(p, oo, a, b, cfg, None)))
    P, O, SP, SO = pa, o, sp, so
    for i in range(1, 9):
        tk = jnp.asarray(rng.integers(0, cfg.semantic_vocab_size, size=(2, cfg.seq_len + 1)))
        P, O, _ = step(P, O, tk[:, :-1], tk[:, 1:])
        SP, SO, _ = vstep(SP, SO, jnp.stack([tk[:, :-1]] * 2), jnp.stack([tk[:, 1:]] * 2))
    compare("params after 8 steps", P, SP)
