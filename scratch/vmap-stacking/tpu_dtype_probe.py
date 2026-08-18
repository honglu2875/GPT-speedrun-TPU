"""Is it the kernels, or just bfloat16?  dense+dense in fp32 vs bf16."""
import importlib.util, sys
from dataclasses import replace
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
ROOT = Path("/home/cubic27/GPT-speedrun-TPU")
spec = importlib.util.spec_from_file_location("ref", ROOT/"recipes/reference/train.py")
t = importlib.util.module_from_spec(spec); sys.modules["ref"]=t; spec.loader.exec_module(t)
parser = t.build_parser()
base = t.resolve_config(parser.parse_args(["--profile","smoke"]), "tpu", 256)

def probe(label, **over):
    cfg = replace(base, layers=2, d_model=128, heads=2, seq_len=256, batch_size=8, **over)
    pa, pb = t.init_params(cfg,1337), t.init_params(cfg,1338)
    o  = jax.tree_util.tree_map(jnp.asarray, t.init_optimizer(pa,cfg.steps))
    ob = jax.tree_util.tree_map(jnp.asarray, t.init_optimizer(pb,cfg.steps))
    rng = np.random.default_rng(0)
    tok = jnp.asarray(rng.integers(0,cfg.semantic_vocab_size,size=(8,cfg.seq_len+1)))
    x,y = tok[:,:-1], tok[:,1:]
    sp = jax.tree_util.tree_map(lambda a,b: jnp.stack([a,b]), pa, pb)
    so = jax.tree_util.tree_map(lambda a,b: jnp.stack([a,b]), o, ob)
    f = lambda p,oo,a,b: t._apply_training_update(p,oo,a,b,cfg,None)
    p1,o1,m1,g1 = jax.jit(f)(pa,o,x,y)
    pv,ov,mv,gv = jax.jit(jax.vmap(f))(sp,so,jnp.stack([x,x]),jnp.stack([y,y]))
    def cmp(a,b):
        ls=jax.tree_util.tree_leaves(a); lv=[q[0] for q in jax.tree_util.tree_leaves(b)]
        ex=sum(bool(jnp.array_equal(u,v)) for u,v in zip(ls,lv))
        w=max(float(jnp.abs(u.astype(jnp.float32)-v.astype(jnp.float32)).max()) for u,v in zip(ls,lv))
        return ex, len(ls), w
    ge, gt, gw = cmp(g1,gv); pe, pt, pw = cmp(p1,pv)
    print(f"  {label:<34} grads {ge}/{gt} ({gw:.2e})   params {pe}/{pt} ({pw:.2e})")

probe("dense+dense, float32", compute_dtype=jnp.float32, dtype_name="float32")
probe("dense+dense, BFLOAT16", compute_dtype=jnp.bfloat16, dtype_name="bfloat16")
probe("dense attn + TILED loss, bf16", compute_dtype=jnp.bfloat16, dtype_name="bfloat16",
      loss_backend="tiled", vocab_size=1024, semantic_vocab_size=1024, vocab_tile_size=128)
probe("dense attn + TILED loss, fp32", compute_dtype=jnp.float32, dtype_name="float32",
      loss_backend="tiled", vocab_size=1024, semantic_vocab_size=1024, vocab_tile_size=128)
