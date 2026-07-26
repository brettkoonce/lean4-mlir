"""Per-step memory + throughput probe for a generated phase-2 trainer.

Imports a generated_*.py as a module (its training loop is behind __main__),
builds params/opt-state/EMA exactly as the real loop does, shards them the same
way (params replicated, batch sharded over the device mesh), and runs real
train_step + ema_update calls on synthetic data.

Reports peak device bytes and steady-state ms/step. Excludes the tf.data input
pipeline, so ms/step is a *compute-only lower bound* on the real run's step time.

env: GEN=<path>  BATCH=<effective batch>  ACCUM=<grad accum, default 1>  STEPS=<n>
"""
import importlib.util, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax import random

GEN = os.environ["GEN"]
STEPS = int(os.environ.get("STEPS", "8"))
ACCUM = int(os.environ.get("ACCUM", "1"))

spec = importlib.util.spec_from_file_location("gen", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

nd = len(jax.devices())
micro = int(os.environ["BATCH"])
micro = (micro // nd) * nd or nd
B = micro * ACCUM          # effective batch fed to train_step (it re-splits internally)

print(f"gen={os.path.basename(GEN)} devices={nd} micro={micro} accum={ACCUM} effective={B}")

key = random.PRNGKey(0)
params = m.init_params(key)
nparams = sum(int(p.size) for p in jax.tree.leaves(params))
params = jax.device_put(params, m.replicated_sharding)
# `train_step` closes over WD_MASK, which the generated trainer only builds
# inside its `__main__` block — replicate that here.
if hasattr(m, "_wd_mask"):
    m.WD_MASK = m._wd_mask(params)
opt_state = (jax.tree.map(jnp.zeros_like, params),
             jax.tree.map(jnp.zeros_like, params), jnp.float32(0))
opt_state = jax.device_put(opt_state, m.replicated_sharding)
ema = jax.device_put(params, m.replicated_sharding)
print(f"params={nparams:,}  fp32 weights={nparams*4/2**20:.0f} MiB  "
      f"(x4 replicated state = {nparams*4*4/2**20:.0f} MiB/device)")

xh = np.asarray(random.normal(random.PRNGKey(1), (B, 3 * 224 * 224), dtype=jnp.float32))
yh = np.asarray(random.randint(random.PRNGKey(2), (B,), 0, 1000).astype(jnp.int32))

def stats():
    s = jax.devices()[0].memory_stats() or {}
    return s.get("peak_bytes_in_use", 0) / 2**30, s.get("bytes_limit", 0) / 2**30

# runningBN nets (ResNet/RSB) thread a BN buffer tuple through train_step and
# carry an EMA shadow of it; detect via the generated init_bn_state.
HAS_BN = hasattr(m, "init_bn_state")
if HAS_BN:
    bn = jax.device_put(m.init_bn_state(), m.replicated_sharding)
    ema_bn = bn
    print("running-BN net: threading bn_state through train_step")

ts = []
try:
    for step in range(STEPS):
        x = jax.device_put(xh, m.data_sharding)
        y = jax.device_put(yh, m.data_sharding)
        dk = jax.random.fold_in(random.PRNGKey(123), step)
        t0 = time.time()
        if HAS_BN:
            params, opt_state, bn, loss = m.train_step(
                params, opt_state, bn, x, y, jnp.float32(5e-4), dk)
            ema_bn = m.ema_update(ema_bn, bn)
        else:
            params, opt_state, loss = m.train_step(params, opt_state, x, y, jnp.float32(5e-4), dk)
        ema = m.ema_update(ema, params)
        jax.block_until_ready((params, loss))
        dt = (time.time() - t0) * 1e3
        if step >= 3:
            ts.append(dt)
        print(f"  step {step}: {dt:8.1f} ms  loss={float(loss):.4f}")
except Exception as e:
    peak, lim = stats()
    print(f"FAIL after {len(ts)} timed steps: {type(e).__name__}: {str(e)[:400]}")
    print(f"RESULT {os.path.basename(GEN)} micro={micro} accum={ACCUM} OOM peak={peak:.2f}GiB limit={lim:.2f}GiB")
    sys.exit(3)

peak, lim = stats()
med = float(np.median(ts)) if ts else float("nan")
spe = 1281167 // B
print(f"RESULT {os.path.basename(GEN)} micro={micro} accum={ACCUM} eff={B} "
      f"ms/step={med:.1f} peak={peak:.2f}GiB limit={lim:.2f}GiB "
      f"min/epoch={spe*med/1000/60:.1f} hr/80ep={spe*med/1000/3600*80:.1f} hr/300ep={spe*med/1000/3600*300:.1f}")
