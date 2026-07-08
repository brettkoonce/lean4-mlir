"""Time a probe module's jit'd train_step directly — the GPU number, nothing else.

The epoch loops in the teaching trainers are host-bound (numpy aug, 5.7 GB epoch
shuffles, per-epoch eval), so their epoch times measure Python. This loads a probe
module, builds synthetic resident batches, and times train_step with proper
block_until_ready, sweeping batch sizes to show utilization scaling:

  XLA_PYTHON_CLIENT_PREALLOCATE=false python jax/probe/step_timer.py \\
      jax/probe/probe_resnet50_imagenette_noaug.py --batches 192 512 1024

Works with any generated module exposing init_params(key) and
train_step(params, (m,v,t), x, y, lr) (the Adam-family imagenette trainers).
Feed the best img/s into estimate.py as the anchor.
"""
import argparse, importlib.util, time

import jax, jax.numpy as jnp
from jax import random


def load(path):
    spec = importlib.util.spec_from_file_location("probe_mod", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def time_batch(m, params, bs, img_flat, warmup, steps):
    opt = (jax.tree.map(jnp.zeros_like, params),
           jax.tree.map(jnp.zeros_like, params), jnp.float32(0))
    x = random.normal(random.PRNGKey(1), (bs, img_flat), dtype=jnp.float32)
    y = random.randint(random.PRNGKey(2), (bs,), 0, 10)
    lr = jnp.float32(1e-3)
    p = params
    for _ in range(warmup):                      # compile + autotune, unmeasured
        p, opt, loss = m.train_step(p, opt, x, y, lr)
    jax.block_until_ready(p)
    t0 = time.monotonic()
    for _ in range(steps):
        p, opt, loss = m.train_step(p, opt, x, y, lr)
    jax.block_until_ready(p)
    dt = time.monotonic() - t0
    ms = dt / steps * 1000
    stats = jax.local_devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use", 0) / 2**30
    print(f"  bs {bs:5d}: {ms:8.1f} ms/step  {bs / (dt / steps):9,.0f} img/s"
          f"   peak {peak:5.1f} GiB")
    return bs / (dt / steps)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("module", help="path to a generated probe .py")
    ap.add_argument("--batches", type=int, nargs="+", default=[192, 512, 1024])
    ap.add_argument("--img", type=int, default=224, help="square image side")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()

    d = jax.devices()[0]
    assert d.platform == "gpu", f"not on a GPU: {d}"
    print(f"━━━ step_timer ━━━ {a.module} on {d.device_kind}")
    m = load(a.module)
    params = m.init_params(random.PRNGKey(0))
    n = sum(int(p.size) for p in jax.tree.leaves(params))
    print(f"  params: {n:,} | synthetic data, jit train_step only "
          f"(warmup {a.warmup}, timed {a.steps})")
    best = 0.0
    for bs in a.batches:
        try:
            best = max(best, time_batch(m, params, bs, 3 * a.img * a.img,
                                        a.warmup, a.steps))
        except Exception as e:                    # OOM at big bs is a datapoint
            print(f"  bs {bs:5d}: FAILED ({type(e).__name__}: {str(e)[:80]})")
    if best:
        print(f"  anchor for estimate.py: {best:,.0f} img/s "
              f"(sec/epoch equiv: {49 * 192 / best:.1f})")


if __name__ == "__main__":
    main()
