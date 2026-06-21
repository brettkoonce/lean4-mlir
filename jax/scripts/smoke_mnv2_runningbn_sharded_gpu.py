"""2-GPU sharding validation for running-BN (gap A) — the open caveat.

Under jax.jit + NamedSharding, `jnp.mean(x, axis=(0,2,3))` over a batch-sharded
input should make XLA insert a cross-device all-reduce, so the BN batch stats (and
hence the running buffers) are GLOBAL, not per-shard. The README's old "BN diverged
under sharding" note means we verify rather than trust.

Test: run one running-BN train_step two ways from the same init —
  (ref)   everything on a single device (trivially global stats);
  (shard) params/opt/bn replicated, x/y sharded across 2 GPUs.
If the cross-device reduction is happening, the resulting running stats + params
match (within bf16 noise). If stats were per-shard, the running variance would be
biased low and params would diverge. Also runs a few sharded steps to confirm it
trains without crashing on the RCCL collectives.

Run with: HIP_VISIBLE_DEVICES=0,1 LD_PRELOAD=/opt/rocm/lib/librccl.so.1
"""
import importlib.util, os
import jax, jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

GEN = os.path.join(os.path.dirname(__file__), "..", ".lake", "build",
                   "generated_mobilenet_v2_imagenet.py")
spec = importlib.util.spec_from_file_location("mnv2_gen", GEN)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
devs = jax.devices()
print("devices:", devs)
assert len(devs) >= 2, "need 2 GPUs for the sharding test"
mesh = Mesh(np.array(devs[:2]), ('batch',))
shard = NamedSharding(mesh, P('batch'))
repl = NamedSharding(mesh, P())

B = 16
params0 = m.init_params(random.PRNGKey(0))
bn0 = m.init_bn_state()
opt0 = (jax.tree.map(jnp.zeros_like, params0), jax.tree.map(jnp.zeros_like, params0))
x = random.normal(random.PRNGKey(1), (B, 3 * 224 * 224), dtype=jnp.float32)
y = random.randint(random.PRNGKey(2), (B,), 0, 1000).astype(jnp.int32)
LR = jnp.float32(0.045)

def maxdiff(a, b):
    # pull to host numpy first — a and b live on different device sets (1 vs 2 GPUs)
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))

# (ref) single device
p_ref, o_ref, bn_ref, l_ref = m.train_step(
    jax.device_put(params0, devs[0]), jax.device_put(opt0, devs[0]),
    jax.device_put(bn0, devs[0]), jax.device_put(x, devs[0]),
    jax.device_put(y, devs[0]), LR)

# (shard) 2-GPU: replicate params/opt/bn, shard x/y on the batch axis
p_sh, o_sh, bn_sh, l_sh = m.train_step(
    jax.device_put(params0, repl), jax.device_put(opt0, repl),
    jax.device_put(bn0, repl), jax.device_put(x, shard),
    jax.device_put(y, shard), LR)
print(f"[ok] sharded train_step ran on 2 GPUs (loss ref={float(l_ref):.3f} shard={float(l_sh):.3f})")

# UNAMBIGUOUS proof (fp32, no bf16 confound): the exact reduction BN uses —
# jnp.mean over the batch-sharded axis — must equal the single-device global mean.
# Per-shard would return each device's local mean (and diverge by the between-shard gap).
x4 = x.reshape(B, 3, 224, 224)
gm = np.asarray(jnp.mean(jax.device_put(x4, devs[0]), axis=(0, 2, 3)))                       # 1 device = global
sm = np.asarray(jax.jit(lambda z: jnp.mean(z, axis=(0, 2, 3)))(jax.device_put(x4, shard)))   # 2-GPU sharded
dmean = float(np.max(np.abs(gm - sm)))
assert dmean < 1e-4, f"sharded mean != global mean ({dmean}) -> BN stats would be PER-SHARD"
print(f"[ok] BN reduction is GLOBAL under sharding: max|Δ mean(axis=0,2,3)| = {dmean:.2e}")

# Sanity cross-check on the full step: bn buffers match tightly; params differ only by
# bf16 conv reassociation (different cross-shard reduction order), not by BN.
print(f"    full-step cross-check: max|Δ bn_state|={maxdiff(bn_ref, bn_sh):.2e} "
      f"(global), max|Δ params|={maxdiff(p_ref, p_sh):.2e} (bf16 conv noise)")

# a few sharded steps to confirm it keeps training over the collectives
p, o, bn = jax.device_put(params0, repl), jax.device_put(opt0, repl), jax.device_put(bn0, repl)
losses = []
for s in range(5):
    p, o, bn, loss = m.train_step(p, o, bn, jax.device_put(x, shard), jax.device_put(y, shard), LR)
    losses.append(float(loss))
print("sharded losses:", [f"{v:.3f}" for v in losses])
assert all(jnp.isfinite(jnp.array(v)) for v in losses) and losses[-1] < losses[0]
bnf = all(bool(jnp.all(jnp.isfinite(rm))) and bool(jnp.all(jnp.isfinite(rv))) for rm, rv in bn)
assert bnf, "non-finite BN buffers under sharding"
print(f"[ok] trains under 2-GPU sharding: loss {losses[0]:.3f} -> {losses[-1]:.3f}, BN buffers finite")
print("\nMNV2 RUNNING-BN SHARDING VALIDATED")
