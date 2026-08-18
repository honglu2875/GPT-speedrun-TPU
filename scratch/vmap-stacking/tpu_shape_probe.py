"""Does vmap bit-exactness depend on shape, with no custom kernels at all?"""
import importlib.util, sys
from dataclasses import replace
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
ROOT = Path("/home/cubic27/GPT-speedrun-TPU")
spec = importlib.util.spec_from_file_location("ref", ROOT/"recipes/reference/train.py")
t = importlib.util.module_from_spec(spec); sys.modules["ref"]=t; spec.loader.exec_module(t)
parser = t.build_parser()
base = t.resolve_config(parser.parse_args(["--profile","smoke"]), "tpu", 256)
import contextlib
PREC = __import__("os").environ.get("PREC","default")
print("matmul precision:", PREC)
print(f"{'layers':>7}{'d_model':>9}{'seq':>6}{'batch':>7}   {'grads':>9} {'params':>9}  worst")
for layers, d, s, b in ((2,64,32,2),(2,64,64,4),(2,128,64,4),(2,128,256,8),
                        (2,256,256,8),(4,256,512,8)):
    cfg = replace(base, layers=layers, d_model=d, heads=2, seq_len=s, batch_size=b,
                  compute_dtype=jnp.float32, dtype_name="float32")
    pa, pb = t.init_params(cfg,1337), t.init_params(cfg,1338)
    o  = jax.tree_util.tree_map(jnp.asarray, t.init_optimizer(pa,cfg.steps))
    ob = jax.tree_util.tree_map(jnp.asarray, t.init_optimizer(pb,cfg.steps))
    rng = np.random.default_rng(0)
    tok = jnp.asarray(rng.integers(0,cfg.semantic_vocab_size,size=(b,cfg.seq_len+1)))
    x,y = tok[:,:-1], tok[:,1:]
    sp = jax.tree_util.tree_map(lambda u,v: jnp.stack([u,v]), pa, pb)
    so = jax.tree_util.tree_map(lambda u,v: jnp.stack([u,v]), o, ob)
    f = lambda p,oo,a,c: t._apply_training_update(p,oo,a,c,cfg,None)
    ctx = jax.default_matmul_precision(PREC) if PREC!='default' else contextlib.nullcontext()
    with ctx:
        p1,_,_,g1 = jax.jit(f)(pa,o,x,y)
        pv,_,_,gv = jax.jit(jax.vmap(f))(sp,so,jnp.stack([x,x]),jnp.stack([y,y]))
    def cmp(a,bb):
        ls=jax.tree_util.tree_leaves(a); lv=[q[0] for q in jax.tree_util.tree_leaves(bb)]
        return (sum(bool(jnp.array_equal(u,v)) for u,v in zip(ls,lv)), len(ls),
                max(float(jnp.abs(u-v).max()) for u,v in zip(ls,lv)))
    ge,gt,_ = cmp(g1,gv); pe,pt,pw = cmp(p1,pv)
    print(f"{layers:>7}{d:>9}{s:>6}{b:>7}   {ge:>4}/{gt:<4} {pe:>4}/{pt:<4}  {pw:.2e}")
