"""GPU smoke for the RSB-A3 ResNet-50 trainer (train @160 / test @224 split).

A3 is the cheap 100-epoch RSB tier (→78.1%). The new bit vs A2 is the
train/test RESOLUTION SPLIT: train images are 160×160, eval images 224×224, and
the generated `forward` infers the square side from the flat input length so the
same conv stack + GAP run at both sizes. This checks both paths work end-to-end:
  * train_step (LAMB+BCE) on a 160px flat batch → finite, decreasing loss
  * forward(training=True)  on 160px → logits [B,1000]  (GAP collapses 5×5)
  * forward(training=False) on 224px → logits [B,1000]  (GAP collapses 7×7)
  * 25,557,032 params, 53 BN, no EMA / no stochastic depth (A3)

A3 is the committed SHORT config, so:
Generate first:  LEAN_MLIR_SHORT=1 ./.lake/build/bin/resnet50-imagenet
Run:  /home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python3 jax/scripts/smoke_r50_a3_gpu.py
"""
import importlib.util, os
import jax, jax.numpy as jnp
from jax import random

GEN = os.path.join(os.path.dirname(__file__), "..", ".lake", "build",
                   "generated_resnet50_imagenet_short.py")
spec = importlib.util.spec_from_file_location("r50_a3", GEN)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("devices:", jax.devices(), "| kind:", jax.devices()[0].device_kind)
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()[0]}"

B, NC = 8, 1000
params = m.init_params(random.PRNGKey(0))
bn = m.init_bn_state()
opt = (jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params), jnp.float32(0))  # LAMB
nparams = sum(int(p.size) for p in jax.tree.leaves(params))
print(f"params: {nparams:,} | BN layers: {len(bn)}")
assert nparams == 25_557_032 and len(bn) == 53

# soft multi-hot targets (mixup-style) for the BCE path
a = random.randint(random.PRNGKey(2), (B,), 0, NC); b = random.randint(random.PRNGKey(3), (B,), 0, NC)
y = (0.7 * jax.nn.one_hot(a, NC) + 0.3 * jax.nn.one_hot(b, NC)).astype(jnp.float32)

# --- TRAIN path: 160px flat input ---
x160 = random.normal(random.PRNGKey(1), (B, 3 * 160 * 160), dtype=jnp.float32)
losses = []
for s in range(6):
    params, opt, bn, loss = m.train_step(params, opt, bn, x160, y, jnp.float32(0.002))
    losses.append(float(loss))
print("A3 train losses (160px):", [f"{l:.4f}" for l in losses])
assert all(jnp.isfinite(jnp.array(l)) for l in losses) and losses[-1] < losses[0]
assert int(opt[2]) == 6, "LAMB t counter"
print(f"[ok] train_step @160px (LAMB+BCE): {losses[0]:.4f} -> {losses[-1]:.4f}, t={int(opt[2])}")

# --- forward at BOTH resolutions resolves the right square + emits [B,1000] ---
lt, _ = m.forward(params, x160, bn, True)            # 160 -> 5x5 -> GAP
assert lt.shape == (B, NC) and bool(jnp.all(jnp.isfinite(lt))), f"train-fwd shape {lt.shape}"
x224 = random.normal(random.PRNGKey(4), (B, 3 * 224 * 224), dtype=jnp.float32)
le, _ = m.forward(params, x224, bn, False)           # 224 -> 7x7 -> GAP (eval res)
assert le.shape == (B, NC) and bool(jnp.all(jnp.isfinite(le))), f"eval-fwd shape {le.shape}"
print(f"[ok] forward infers resolution: 160px->{lt.shape}, 224px->{le.shape} (both finite)")

# --- host-pipeline batch ops at TRAIN res (the train loop runs these on the 160px
#     batch BEFORE train_step). augment_batch (numpy crop) and _cutmix hardcoded
#     224 until the trainRes fix, so they crashed/garbled the 160px batch — the
#     train_step-only checks above sail right past that. Exercise them here. ---
import numpy as np
xnp = np.asarray(x160)                                   # (B, 3*160*160)
xa = m.augment_batch(xnp, np.random.RandomState(0))      # numpy pad+crop @ train res
assert xa.shape == xnp.shape, f"augment_batch shape {xa.shape} != {xnp.shape} (trainRes ignored?)"
xc, yc = m._cutmix(x160, a, random.PRNGKey(7))           # reshapes to (B,3,res,res)
assert xc.shape == x160.shape and yc.shape == (B, NC), f"cutmix shapes {xc.shape}/{yc.shape}"
xm, ym = m._mixup(x160, a, random.PRNGKey(8))            # resolution-agnostic, sanity
assert xm.shape == x160.shape and ym.shape == (B, NC), f"mixup shapes {xm.shape}/{ym.shape}"
assert all(bool(jnp.all(jnp.isfinite(jnp.asarray(t)))) for t in (xa, xc, xm))
print(f"[ok] host pipeline @train-res 160: augment_batch{tuple(xa.shape)}, cutmix+mixup flat-preserved")

print("\nR50 RSB-A3 VALIDATED (160 train / 224 eval split)")
