"""GPU smoke test for the DeiT-Ti stochastic-depth codegen wiring.

Imports the generated ViT-Tiny/ImageNet trainer as a module (the training
loop is behind `if __name__ == '__main__'`, so import only defines the model
+ optimizer fns and the jax mesh) and exercises the *changed* path —
`transformer_block` with drop_key/keep_prob — on the ROCm GPU with a small
synthetic batch. No tfds, no full run.

Checks:
  1. params build + run forward on GPU (RocmDevice)
  2. eval path (drop_key=None) is deterministic and drop-free
  3. stochastic depth is actually active in train: different drop_key ->
     different logits; same drop_key -> identical logits (reproducible)
  4. a few real train_steps produce finite, decreasing loss
"""
import importlib.util, os, sys
import jax, jax.numpy as jnp
from jax import random

GEN = os.path.join(os.path.dirname(__file__), "..", ".lake", "build",
                   "generated_vit_tiny_imagenet.py")
spec = importlib.util.spec_from_file_location("vit_gen", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print("devices:", jax.devices())
assert "rocm" in str(jax.devices()[0]).lower(), "not on a ROCm device!"

B = 8
key = random.PRNGKey(0)
params = m.init_params(key)
nparams = sum(int(p.size) for p in jax.tree.leaves(params))
print(f"params built: {nparams:,} floats")

x = random.normal(random.PRNGKey(1), (B, 3 * 224 * 224), dtype=jnp.float32)
y = random.randint(random.PRNGKey(2), (B,), 0, 1000).astype(jnp.int32)

# 2. eval path: deterministic + drop-free
e1 = m.forward(params, x)            # drop_key=None
e2 = m.forward(params, x)
assert jnp.allclose(e1, e2), "eval path not deterministic"
print(f"[ok] eval forward deterministic; logits {e1.shape}, finite={bool(jnp.all(jnp.isfinite(e1)))}")

# 3. stochastic depth active in the train path
ka, kb = random.split(random.PRNGKey(7))
ta = m.forward(params, x, ka)
ta2 = m.forward(params, x, ka)       # same key -> identical
tb = m.forward(params, x, kb)        # different key -> different
assert jnp.allclose(ta, ta2), "drop path not reproducible under fixed key"
diff = float(jnp.max(jnp.abs(ta - tb)))
assert diff > 1e-4, f"drop path inactive (max|Δ|={diff:.2e})"
# and the train path must differ from the (drop-free) eval path
assert float(jnp.max(jnp.abs(ta - e1))) > 1e-4, "train path == eval path (no drop)"
print(f"[ok] stochastic depth active: same-key identical, diff-key max|Δ|={diff:.3f}")

# 4. a few real Adam train_steps
opt_m = jax.tree.map(jnp.zeros_like, params)
opt_v = jax.tree.map(jnp.zeros_like, params)
opt_state = (opt_m, opt_v, jnp.float32(0))
losses = []
for step in range(6):
    dk = jax.random.fold_in(random.PRNGKey(123), step)
    params, opt_state, loss = m.train_step(params, opt_state, x, y, jnp.float32(5e-4), dk)
    losses.append(float(loss))
print("train losses:", [f"{l:.4f}" for l in losses])
assert all(jnp.isfinite(jnp.array(l)) for l in losses), "non-finite loss"
assert losses[-1] < losses[0], f"loss not decreasing ({losses[0]:.4f} -> {losses[-1]:.4f})"
print(f"[ok] train_step on GPU: loss {losses[0]:.4f} -> {losses[-1]:.4f} (decreasing, finite)")
print("\nSMOKE PASS")
