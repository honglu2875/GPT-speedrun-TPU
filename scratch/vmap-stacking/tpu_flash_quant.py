import importlib.util, sys
from dataclasses import replace
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh
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
pa, pb = t.init_params(cfg,1337), t.init_params(cfg,1338)
o  = jax.tree_util.tree_map(jnp.asarray, t.init_optimizer(pa,cfg.steps))
ob = jax.tree_util.tree_map(jnp.asarray, t.init_optimizer(pb,cfg.steps))
sp = jax.tree_util.tree_map(lambda a,b: jnp.stack([a,b]), pa, pb)
so = jax.tree_util.tree_map(lambda a,b: jnp.stack([a,b]), o, ob)
rng = np.random.default_rng(0)
with jax.set_mesh(mesh):
    attn = t.make_mesh_attention(cfg, mesh, tiles)
    step  = jax.jit(lambda p,oo,a,b: t.train_step(p,oo,a,b,cfg,None,attn))
    vstep = jax.jit(jax.vmap(lambda p,oo,a,b: t.train_step(p,oo,a,b,cfg,None,attn)))
    P,O,SP,SO = pa,o,sp,so
    print(f"bf16 epsilon = {float(jnp.finfo(jnp.bfloat16).eps):.3e}")
    print(f"{'step':>5} {'exact':>8} {'max abs':>11} {'max rel':>11}")
    for i in range(1,9):
        tk = jnp.asarray(rng.integers(0,cfg.semantic_vocab_size,size=(8,cfg.seq_len+1)))
        P,O,_ = step(P,O,tk[:,:-1],tk[:,1:])
        SP,SO,_ = vstep(SP,SO,jnp.stack([tk[:,:-1]]*2),jnp.stack([tk[:,1:]]*2))
        if i in (1,2,4,8):
            ls=jax.tree_util.tree_leaves(P); lv=[q[0] for q in jax.tree_util.tree_leaves(SP)]
            ex=sum(bool(jnp.array_equal(a,b)) for a,b in zip(ls,lv))
            aw=max(float(jnp.abs(a.astype(jnp.float32)-b.astype(jnp.float32)).max()) for a,b in zip(ls,lv))
            rel=max(float((jnp.abs(a.astype(jnp.float32)-b.astype(jnp.float32))/(jnp.abs(a.astype(jnp.float32))+1e-30)).max()) for a,b in zip(ls,lv))
            print(f"{i:>5} {ex:>3}/{len(ls):<4} {aw:>11.3e} {rel:>11.3e}")
