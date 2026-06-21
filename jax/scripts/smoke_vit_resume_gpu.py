"""GPU test for lossless suspend/resume (save_train_state / load_train_state).

Imports the generated ViT-Tiny/ImageNet trainer, runs a few Adam steps, saves the
full train state, reloads it into a FRESH (differently-initialised) template, and
checks two things on the ROCm/CUDA GPU:

  1. restore is exact — params, Adam (m, v, t), and the EMA shadow all come back
     bit-for-bit (max|Δ| == 0), not just the weights;
  2. continuation is bit-for-bit — one more train_step from the original state and
     from the resumed state produce identical params/opt_state/ema. If Adam moments
     or EMA history were lost on resume, this step would diverge.
"""
import importlib.util, os
import jax, jax.numpy as jnp
from jax import random

GEN = os.path.join(os.path.dirname(__file__), "..", ".lake", "build",
                   "generated_vit_tiny_imagenet.py")
spec = importlib.util.spec_from_file_location("vit_gen", GEN)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("devices:", jax.devices(), "| kind:", jax.devices()[0].device_kind)
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()[0]}"

def max_diff(a, b):
    return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))

B = 8
x = random.normal(random.PRNGKey(1), (B, 3 * 224 * 224), dtype=jnp.float32)
y = random.randint(random.PRNGKey(2), (B,), 0, 1000).astype(jnp.int32)
LR = jnp.float32(5e-4)

# initial state + 5 Adam steps
params = m.init_params(random.PRNGKey(0))
opt_state = (jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params), jnp.float32(0))
ema = params
step = 0
for _ in range(5):
    dk = jax.random.fold_in(random.PRNGKey(123), step)
    params, opt_state, _ = m.train_step(params, opt_state, x, y, LR, dk)
    ema = m.ema_update(ema, params)
    step += 1
print(f"ran {step} Adam steps; opt t = {float(opt_state[2])}")

# SAVE full state
path = "/tmp/vit_perf/resume_state.npz"
os.makedirs("/tmp/vit_perf", exist_ok=True)
m.save_train_state(path, (params, opt_state, ema), step)

# Branch A: continue the ORIGINAL state one step
dkA = jax.random.fold_in(random.PRNGKey(123), step)
pA, oA, _ = m.train_step(params, opt_state, x, y, LR, dkA)
emaA = m.ema_update(ema, pA)

# Branch B: reload into a FRESH, differently-initialised template, continue one step
t_params = m.init_params(random.PRNGKey(999))          # different init -> proves load overwrites
t_opt = (jax.tree.map(jnp.zeros_like, t_params), jax.tree.map(jnp.zeros_like, t_params), jnp.float32(0))
(rp, ro, rema), rstep = m.load_train_state(path, (t_params, t_opt, t_params))
assert rstep == step, f"step mismatch {rstep} != {step}"
assert max_diff(rp, params) == 0.0,    "params not restored exactly"
assert max_diff(ro, opt_state) == 0.0, "Adam (m,v,t) not restored exactly"
assert max_diff(rema, ema) == 0.0,     "EMA shadow not restored exactly"
print(f"[ok] restore exact: params/opt(m,v,t)/ema all max|Δ|=0, step={rstep}")

dkB = jax.random.fold_in(random.PRNGKey(123), rstep)
pB, oB, _ = m.train_step(rp, ro, x, y, LR, dkB)
emaB = m.ema_update(rema, pB)
dp, do, de = max_diff(pA, pB), max_diff(oA, oB), max_diff(emaA, emaB)
assert dp == 0.0 and do == 0.0 and de == 0.0, f"resumed step diverged: dp={dp} do={do} de={de}"
print(f"[ok] continuation bit-for-bit: params Δ={dp}, opt Δ={do}, ema Δ={de}")
print("\nLOSSLESS RESUME PASS")
