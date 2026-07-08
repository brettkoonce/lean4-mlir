"""Time a probe module's jit'd train_step directly — the GPU number, nothing else.

The epoch loops in the teaching trainers are host-bound (numpy aug, 5.7 GB epoch
shuffles, per-epoch eval), so their epoch times measure Python. This times the jit
step on synthetic resident data with block_until_ready, one batch size per FRESH
subprocess (fragmentation from one size otherwise poisons the next — measured live
2026-07-08: bs512 asked for one 42 GiB chunk after bs192 shredded the arena):

  python jax/probe/step_timer.py jax/probe/probe_resnet50_imagenette_noaug.py \\
      --batches 192 512 1024 2048

Allocator env (PREALLOCATE off, 95% cap, cuda_malloc_async) is set internally.
OOM at a batch size prints FAILED and is itself a datapoint. Works with any
generated module exposing init_params(key) and train_step(params, (m,v,t), x, y, lr)
(the Adam-family imagenette trainers). Feed the best img/s into estimate.py.
"""
import argparse, importlib.util, os, subprocess, sys, time

ENV = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
       "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
       "TF_GPU_ALLOCATOR": "cuda_malloc_async"}


def run_one(args):
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    import jax, jax.numpy as jnp          # after env — allocator flags bind at init
    from jax import random

    spec = importlib.util.spec_from_file_location("probe_mod", args.module)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    bs = args.batches[0]
    params = m.init_params(random.PRNGKey(0))
    opt = (jax.tree.map(jnp.zeros_like, params),
           jax.tree.map(jnp.zeros_like, params), jnp.float32(0))
    x = random.normal(random.PRNGKey(1), (bs, 3 * args.img * args.img),
                      dtype=jnp.float32)
    y = random.randint(random.PRNGKey(2), (bs,), 0, 10)
    lr = jnp.float32(1e-3)
    p = params
    for _ in range(args.warmup):              # compile + autotune, unmeasured
        p, opt, loss = m.train_step(p, opt, x, y, lr)
    jax.block_until_ready(p)
    t0 = time.monotonic()
    for _ in range(args.steps):
        p, opt, loss = m.train_step(p, opt, x, y, lr)
    jax.block_until_ready(p)
    dt = time.monotonic() - t0
    ms = dt / args.steps * 1000
    stats = jax.local_devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use", 0) / 2**30
    print(f"  bs {bs:5d}: {ms:8.1f} ms/step  {bs / (dt / args.steps):9,.0f} img/s"
          f"   peak {peak:5.1f} GiB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("module", help="path to a generated probe .py")
    ap.add_argument("--batches", type=int, nargs="+", default=[192, 512, 1024])
    ap.add_argument("--img", type=int, default=224, help="square image side")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--_single", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._single:                              # child: exactly one batch size
        run_one(a)
        return

    print(f"━━━ step_timer ━━━ {a.module} (one fresh process per batch size)")
    for bs in a.batches:                       # parent: re-exec per size
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), a.module, "--_single",
             "--batches", str(bs), "--img", str(a.img),
             "--warmup", str(a.warmup), "--steps", str(a.steps)],
            env={**os.environ, **ENV})
        if r.returncode != 0:
            print(f"  bs {bs:5d}: FAILED (exit {r.returncode} — OOM is a datapoint)")
    print("  anchor for estimate.py: best img/s above "
          "(sec/epoch equiv = 9408 / img_s)")


if __name__ == "__main__":
    main()
