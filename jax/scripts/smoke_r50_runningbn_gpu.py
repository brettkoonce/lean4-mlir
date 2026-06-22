"""GPU smoke test for the RSB-A2 ResNet-50 / ImageNet trainer (phase-1 skeleton).

Imports the generated ResNet-50/ImageNet trainer (runningBN=true) and checks on
the ROCm/CUDA GPU that the newly-threaded *bottleneck* running-BN path works
end-to-end:
  1. 53 BN layers (1 stem + 16 blocks·3 + 4 stage-first shortcuts) — the count
     the RSB-A2 plan calls for;
  2. compiles + trains (finite, decreasing loss) via the SGD+momentum train_step
     with the (params, velocity, bn) -> (params, velocity, bn, loss) has_aux
     threading through bottleneck_block / bottleneck_block_down;
  3. the running BN buffers are EMA-updated from init (mean 0, var 1) + finite;
  4. eval (training=False -> running stats) DIFFERS from train-mode (batch stats)
     — i.e. the gap-A fix is live for bottleneck blocks, not just basic blocks.

Run with the jax venv:
  /home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python3 \
      jax/scripts/smoke_r50_runningbn_gpu.py
"""
import importlib.util, os
import jax, jax.numpy as jnp
from jax import random

GEN = os.path.join(os.path.dirname(__file__), "..", ".lake", "build",
                   "generated_resnet50_imagenet.py")
spec = importlib.util.spec_from_file_location("r50_gen", GEN)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("devices:", jax.devices(), "| kind:", jax.devices()[0].device_kind)
assert jax.devices()[0].platform == "gpu", f"not on a GPU: {jax.devices()[0]}"

B = 8
params = m.init_params(random.PRNGKey(0))
bn_state = m.init_bn_state()
velocity = jax.tree.map(jnp.zeros_like, params)  # SGD+momentum opt_state
nparams = sum(int(p.size) for p in jax.tree.leaves(params))
print(f"params: {nparams:,} floats | BN layers: {len(bn_state)}")
assert nparams == 25_557_032, f"expected 25,557,032 params (torchvision R50), got {nparams:,}"
assert len(bn_state) == 53, f"expected 53 BN layers, got {len(bn_state)}"

x = random.normal(random.PRNGKey(1), (B, 3 * 224 * 224), dtype=jnp.float32)
y = random.randint(random.PRNGKey(2), (B,), 0, 1000).astype(jnp.int32)

# init BN buffers should be exactly mean 0, var 1
assert float(jnp.max(jnp.abs(bn_state[0][0]))) == 0.0 and float(jnp.max(jnp.abs(bn_state[0][1] - 1.0))) == 0.0

losses = []
for step in range(6):
    params, velocity, bn_state, loss = m.train_step(params, velocity, bn_state, x, y, jnp.float32(0.1))
    losses.append(float(loss))
print("train losses:", [f"{l:.3f}" for l in losses])
assert all(jnp.isfinite(jnp.array(l)) for l in losses), "non-finite loss"
assert losses[-1] < losses[0], f"loss not decreasing ({losses[0]:.3f} -> {losses[-1]:.3f})"
print(f"[ok] bottleneck running-BN trains on GPU: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

# 2. running buffers moved from init across all 53 layers + finite
mv = max(float(jnp.max(jnp.abs(rm))) for rm, rv in bn_state)
vv = max(float(jnp.max(jnp.abs(rv - 1.0))) for rm, rv in bn_state)
assert mv > 1e-5 and vv > 1e-5, "running BN buffers did not update"
finite = all(bool(jnp.all(jnp.isfinite(rm))) and bool(jnp.all(jnp.isfinite(rv))) for rm, rv in bn_state)
assert finite, "non-finite BN buffers"
print(f"[ok] 53 BN buffers EMA-updated + finite: max|mean|={mv:.3f}, max|var-1|={vv:.3f}")

# 3. eval (running stats) != train-mode (batch stats) — gap-A fix is live for bottleneck
logits_eval, _ = m.forward(params, x, bn_state, False)
logits_train, _ = m.forward(params, x, bn_state, True)
d = float(jnp.max(jnp.abs(logits_eval - logits_train)))
assert d > 1e-3, "eval(running) == eval(batch) — fix is a no-op"
assert bool(jnp.all(jnp.isfinite(logits_eval))), "non-finite eval logits"
c1, c5, el = m.eval_batch(params, bn_state, x, y)
print(f"[ok] eval uses running stats (Δ vs batch-stats={d:.3f}); eval_batch ok (loss={float(el):.3f})")
print("\nR50 BOTTLENECK RUNNING-BN VALIDATED (phase-1 skeleton)")
