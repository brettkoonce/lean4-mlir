"""End-to-end GPU smoke for the LITERAL RSB-A2 ResNet-50 trainer (phase 5).

Exercises the full RSB-A2 train_step on a synthetic batch — every ingredient
landed across phases 1-4 wired together:
  * LAMB optimizer       -> opt_state = (m, v, t), trust-ratio step (phase 3)
  * BCE-with-logits      -> soft multi-hot [B,NC] targets (phase 4)
  * stochastic depth     -> drop_key threaded through the 16 bottleneck blocks (phase 1)
  * running BN           -> 53 buffers EMA-updated, train!=eval stats (phase 1)
  * model EMA            -> shadow weights diverge from live params
  * 25,557,032 params    -> exact torchvision ResNet-50

No tfds / no full run — just init/forward/train_step on a synthetic batch. The
real-data dataloader (mixup/cutmix + RandAugment + 3x repeated-aug) is validated
separately (warmup timing) and structurally in the codegen tests.

Run with the jax venv:
  /home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python3 \
      jax/scripts/smoke_r50_rsb_a2_gpu.py
"""
import importlib.util, os
import jax, jax.numpy as jnp
from jax import random

GEN = os.path.join(os.path.dirname(__file__), "..", ".lake", "build",
                   "generated_resnet50_imagenet.py")
spec = importlib.util.spec_from_file_location("r50_rsb", GEN)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("devices:", jax.devices(), "| kind:", jax.devices()[0].device_kind)
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()[0]}"

B, NC = 8, 1000
params = m.init_params(random.PRNGKey(0))
bn_state = m.init_bn_state()
opt_state = (jax.tree.map(jnp.zeros_like, params),    # LAMB m
             jax.tree.map(jnp.zeros_like, params),    # LAMB v
             jnp.float32(0))                          # LAMB t
ema_params = jax.tree.map(lambda p: p, params)        # EMA shadow starts at params
nparams = sum(int(p.size) for p in jax.tree.leaves(params))
print(f"params: {nparams:,} floats | BN layers: {len(bn_state)}")
assert nparams == 25_557_032, f"expected 25,557,032 params, got {nparams:,}"
assert len(bn_state) == 53, f"expected 53 BN layers, got {len(bn_state)}"

# Soft multi-hot targets (mixup-style: lam*onehot(a) + (1-lam)*onehot(b)) — the
# shape the BCE path consumes directly.
a = random.randint(random.PRNGKey(2), (B,), 0, NC)
b = random.randint(random.PRNGKey(3), (B,), 0, NC)
y = (0.7 * jax.nn.one_hot(a, NC) + 0.3 * jax.nn.one_hot(b, NC)).astype(jnp.float32)
x = random.normal(random.PRNGKey(1), (B, 3 * 224 * 224), dtype=jnp.float32)

losses = []
for step in range(6):
    dk = jax.random.fold_in(random.PRNGKey(99), step)   # per-step stochastic-depth key
    params, opt_state, bn_state, loss = m.train_step(params, opt_state, bn_state, x, y, jnp.float32(0.00125), dk)
    ema_params = m.ema_update(ema_params, params)
    losses.append(float(loss))
print("RSB-A2 train losses:", [f"{l:.4f}" for l in losses])
assert all(jnp.isfinite(jnp.array(l)) for l in losses), "non-finite loss"
assert losses[-1] < losses[0], f"loss not decreasing ({losses[0]:.4f} -> {losses[-1]:.4f})"
print(f"[ok] LAMB+BCE+SD trains on GPU: loss {losses[0]:.4f} -> {losses[-1]:.4f}")

# LAMB opt_state: t advanced, m/v finite
mm, vv, tt = opt_state
assert int(tt) == 6, f"LAMB step counter t={int(tt)} != 6"
assert all(bool(jnp.all(jnp.isfinite(z))) for z in jax.tree.leaves((mm, vv))), "non-finite m/v"
print(f"[ok] LAMB opt_state: t={int(tt)}, m/v finite")

# running BN buffers moved + finite
mv = max(float(jnp.max(jnp.abs(rm))) for rm, rv in bn_state)
vvb = max(float(jnp.max(jnp.abs(rv - 1.0))) for rm, rv in bn_state)
assert mv > 1e-5 and vvb > 1e-5, "running BN buffers did not update"
print(f"[ok] 53 BN buffers EMA-updated: max|mean|={mv:.3f}, max|var-1|={vvb:.3f}")

# EMA shadow diverged from live params (the shadow is the eval checkpoint)
d_ema = max(float(jnp.max(jnp.abs(e - p))) for e, p in zip(jax.tree.leaves(ema_params), jax.tree.leaves(params)))
assert d_ema > 1e-9, "EMA shadow == live params (ema_update is a no-op)"
print(f"[ok] EMA shadow diverged from live params (max Δ={d_ema:.2e})")

# eval (running stats, SD off via drop_key=None) finite + != train-mode
le, _ = m.forward(params, x, bn_state, False)
lt, _ = m.forward(params, x, bn_state, True)
assert bool(jnp.all(jnp.isfinite(le))) and float(jnp.max(jnp.abs(le - lt))) > 1e-3
print("[ok] eval(running, SD off) finite and != train(batch, SD on)")
print("\nR50 RSB-A2 END-TO-END VALIDATED (phase 5)")
