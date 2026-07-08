"""GPU-resident imagenette training profile — real data, real accuracy, zero host/step.

The teaching trainers' epoch loops are host-bound (numpy aug, 5.7 GB epoch shuffles):
on an A100 the GPU idles at ~0%. This keeps the WHOLE dataset on the GPU as uint8
(train 256px ≈ 1.9 GB, val 224px ≈ 0.6 GB) and does everything per-step on device:
gather by permutation, random 256→224 crop, hflip, normalize, train_step. The host
contributes a scalar step index. Use with the bf16 probe modules:

  python jax/probe/probe_train_gpu.py jax/probe/probe_resnet50_imagenette_bf16.py \\
      --epochs 5 --batch 512
  python jax/probe/probe_train_gpu.py jax/probe/probe_vit_tiny_imagenette_bf16.py \\
      --epochs 5 --batch 512

Prints per-epoch train time / img/s / top-1/top-5, then the img/s anchor for
estimate.py. Single-GPU by design (the profile question); OOM → lower --batch.
"""
import argparse, importlib.util, os, struct, time

for k, v in {"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
             "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}.items():
    os.environ.setdefault(k, v)

import numpy as np
import jax, jax.numpy as jnp
from jax import random

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def load_raw(path):
    """Raw uint8 records (1-byte label + 3*S*S image), NO fp32 blowup on host."""
    with open(path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    rec = len(data) // count
    side = int(np.sqrt((rec - 1) // 3))
    assert side * side * 3 + 1 == rec, (rec, side, path)
    data = data.reshape(count, rec)
    return (data[:, 1:].reshape(count, 3, side, side),
            data[:, 0].astype(np.int32), side)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("module", help="path to a generated probe .py (bf16 variants)")
    ap.add_argument("--data", default="data/imagenette")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    dev = jax.devices()[0]
    assert dev.platform == "gpu", f"not on a GPU: {dev}"
    spec = importlib.util.spec_from_file_location("probe_mod", a.module)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tr_u8, tr_y, tr_side = load_raw(os.path.join(a.data, "train.bin"))
    va_u8, va_y, va_side = load_raw(os.path.join(a.data, "val.bin"))
    DST, CROP = va_side, tr_side - va_side          # 224, 32
    tr_u8, tr_y = jnp.asarray(tr_u8), jnp.asarray(tr_y)
    va_u8, va_y = jnp.asarray(va_u8), jnp.asarray(va_y)
    mean, std = jnp.asarray(MEAN), jnp.asarray(STD)
    n, bs = tr_u8.shape[0], a.batch
    steps = n // bs
    print(f"━━━ probe_train_gpu ━━━ {os.path.basename(a.module)} on {dev.device_kind}")
    print(f"  resident: train {tr_u8.nbytes/2**30:.2f} GiB u8 @{tr_side}px, "
          f"val {va_u8.nbytes/2**30:.2f} GiB @{va_side}px | bs {bs}, {steps} steps/ep")

    def normalize(u8):                               # (B,3,D,D) u8 -> fp32 flat
        x = u8.astype(jnp.float32) / 255.0
        return ((x - mean) / std).reshape(u8.shape[0], -1)

    def crop_one(img, off, flip):                    # (3,S,S) -> (3,D,D)
        c = jax.lax.dynamic_slice(img, (0, off[0], off[1]), (3, DST, DST))
        return jax.lax.cond(flip, lambda t: t[:, :, ::-1], lambda t: t, c)

    @jax.jit
    def train_epoch_step(params, opt, perm, i, key, lr):
        idx = jax.lax.dynamic_slice_in_dim(perm, i * bs, bs)
        raw = jnp.take(tr_u8, idx, axis=0)
        ko, kf = random.split(random.fold_in(key, i))
        offs = random.randint(ko, (bs, 2), 0, CROP + 1)
        flips = random.bernoulli(kf, 0.5, (bs,))
        x = normalize(jax.vmap(crop_one)(raw, offs, flips))
        y = jnp.take(tr_y, idx, axis=0)
        return m.train_step(params, opt, x, y, lr)

    @jax.jit
    def eval_step(params, i):
        s = i * bs
        x = normalize(jax.lax.dynamic_slice_in_dim(va_u8, s, bs))
        y = jax.lax.dynamic_slice_in_dim(va_y, s, bs)
        return m.eval_batch(params, x, y)

    params = m.init_params(random.PRNGKey(0))
    opt = (jax.tree.map(jnp.zeros_like, params),
           jax.tree.map(jnp.zeros_like, params), jnp.float32(0))
    print(f"  params: {sum(int(p.size) for p in jax.tree.leaves(params)):,}")

    va_steps = va_u8.shape[0] // bs
    best = 0.0
    for ep in range(a.epochs):
        key = random.PRNGKey(100 + ep)
        perm = random.permutation(random.fold_in(key, 0xFFFF), n)
        t0 = time.monotonic()
        for i in range(steps):
            params, opt, loss = train_epoch_step(params, opt, perm, i, key,
                                                 jnp.float32(a.lr))
        jax.block_until_ready(params)
        dt = time.monotonic() - t0
        c1 = c5 = 0
        te0 = time.monotonic()
        for i in range(va_steps):
            r1, r5, _ = eval_step(params, i)
            c1 += int(r1); c5 += int(r5)
        te = time.monotonic() - te0
        seen = va_steps * bs
        best = max(best, c1 / seen)
        tag = " (compile)" if ep == 0 else ""
        print(f"  [ep {ep+1}] train {dt:6.1f}s  {steps*bs/dt:8,.0f} img/s   "
              f"top1 {c1/seen:.4f}  top5 {c5/seen:.4f}  eval {te:.1f}s{tag}",
              flush=True)
    stats = dev.memory_stats() or {}
    print(f"  peak {stats.get('peak_bytes_in_use', 0)/2**30:.1f} GiB | best top1 {best:.4f}")
    print(f"  anchor for estimate.py: steady img/s above "
          f"(--r50-sec-epoch = 9408 / img_s equiv)")


if __name__ == "__main__":
    main()
