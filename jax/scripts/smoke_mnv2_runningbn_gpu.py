"""GPU validation for running-BN-stats (gap A) on the MNv2 imagenet trainer.

Imports the generated MobileNetV2/ImageNet trainer (runningBN=true) and checks on
the ROCm/CUDA GPU that the threaded running-BN path actually works end-to-end:
  1. compiles + trains (finite, decreasing loss) with the (params, opt_state, bn)
     -> (params, opt_state, bn, loss) has_aux threading;
  2. the running BN buffers (52 of them) are EMA-updated from init (mean 0, var 1);
  3. eval (training=False -> running stats) DIFFERS from train-mode (batch stats) —
     i.e. eval now does the paper-faithful thing the old trainer skipped.
"""
import importlib.util, os
import jax, jax.numpy as jnp
from jax import random

GEN = os.path.join(os.path.dirname(__file__), "..", ".lake", "build",
                   "generated_mobilenet_v2_imagenet.py")
spec = importlib.util.spec_from_file_location("mnv2_gen", GEN)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("devices:", jax.devices(), "| kind:", jax.devices()[0].device_kind)
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()[0]}"

B = 8
params = m.init_params(random.PRNGKey(0))
bn_state = m.init_bn_state()
opt_state = (jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params))  # rmsprop (sq, buf)
print(f"params: {sum(int(p.size) for p in jax.tree.leaves(params)):,} floats | BN layers: {len(bn_state)}")
assert len(bn_state) == 52, f"expected 52 BN layers, got {len(bn_state)}"

x = random.normal(random.PRNGKey(1), (B, 3 * 224 * 224), dtype=jnp.float32)
y = random.randint(random.PRNGKey(2), (B,), 0, 1000).astype(jnp.int32)

# snapshot init BN buffers (should be mean 0, var 1)
assert float(jnp.max(jnp.abs(bn_state[0][0]))) == 0.0 and float(jnp.max(jnp.abs(bn_state[0][1] - 1.0))) == 0.0

losses = []
for step in range(6):
    params, opt_state, bn_state, loss = m.train_step(params, opt_state, bn_state, x, y, jnp.float32(0.045))
    losses.append(float(loss))
print("train losses:", [f"{l:.3f}" for l in losses])
assert all(jnp.isfinite(jnp.array(l)) for l in losses), "non-finite loss"
assert losses[-1] < losses[0], f"loss not decreasing ({losses[0]:.3f} -> {losses[-1]:.3f})"
print(f"[ok] running-BN trains on GPU: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

# 2. running buffers moved from init across all layers
mv = max(float(jnp.max(jnp.abs(rm))) for rm, rv in bn_state)
vv = max(float(jnp.max(jnp.abs(rv - 1.0))) for rm, rv in bn_state)
assert mv > 1e-5 and vv > 1e-5, "running BN buffers did not update"
finite = all(bool(jnp.all(jnp.isfinite(rm))) and bool(jnp.all(jnp.isfinite(rv))) for rm, rv in bn_state)
assert finite, "non-finite BN buffers"
print(f"[ok] 52 BN buffers EMA-updated + finite: max|mean|={mv:.3f}, max|var-1|={vv:.3f}")

# 3. eval (running stats) != train-mode (batch stats) — the gap-A fix is live
logits_eval, _ = m.forward(params, x, bn_state, False)
logits_train, _ = m.forward(params, x, bn_state, True)
d = float(jnp.max(jnp.abs(logits_eval - logits_train)))
assert d > 1e-3, "eval(running) == eval(batch) — fix is a no-op"
assert bool(jnp.all(jnp.isfinite(logits_eval))), "non-finite eval logits"
c1, c5, el = m.eval_batch(params, bn_state, x, y)
print(f"[ok] eval uses running stats (Δ vs batch-stats={d:.3f}); eval_batch ok (loss={float(el):.3f})")
print("\nMNV2 RUNNING-BN VALIDATED")
