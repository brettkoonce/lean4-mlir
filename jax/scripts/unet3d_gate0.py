#!/usr/bin/env python3
"""Gate 0 for the 3D UNet (planning/brats_25d_3d.md §3): does a rank-5 UNet train
step RUN on one 4060 Ti, at what ms/step, and in how much memory?

planning/archive/unet3d.md established that every op a 3D UNet needs COMPILES
(IREE and XLA); it says in as many words that compiling is not running. This
measures the running: a depth-4 3D UNet in the shape of `ReferenceNets.unetBrats`
(base 32, bottleneck 512, two 3³ conv+BN+ReLU per stage, 2³ max-pool, trilinear
×2 + concat on the way up, 4 modalities in, 4 classes out) trained on a 4-channel
patch with per-voxel cross-entropy under plain SGD, forward + backward + update,
through XLA on the GPU — the same compiler the Lean path drives through
`ffi/libpjrt_ffi.so`.

Also the 2D twin (same design, 3×3 kernels) at the from-scratch trainer's
16 × 240² batch, from the same harness, so the per-voxel comparison is between
two numbers measured the same way. The Lean R34 UNet's own step is 252 ms at
16 × 224² (runs/2026-09-25-brats-r34-xla/README.md).

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=3 \\
        .venv/bin/python jax/scripts/unet3d_gate0.py

No BraTS data is read: the input is random, this is a throughput measurement.
The gate in the plan: per-voxel throughput within ~3–5× of the 2D conv's and
no OOM at 128³ × batch 2 in the 4060 Ti's memory.
"""
import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

if jax.devices()[0].platform != "gpu":
    raise SystemExit("no GPU device — refusing to time a CPU fallback")

DEV = jax.devices()[0]


# ----------------------------------------------------------------------------- model
def conv_dn(nd):
    """Dimension numbers for NC(D)HW convolution at spatial rank `nd`."""
    sp = "DHW"[-nd:]
    return (f"NC{sp}", f"OI{sp}", f"NC{sp}")


def init_conv(key, ic, oc, k, nd):
    """He init for a k^nd kernel; BN gamma/beta at 1/0."""
    fan_in = ic * k ** nd
    w = jax.random.normal(key, (oc, ic) + (k,) * nd, jnp.float32) * jnp.sqrt(2.0 / fan_in)
    return {"w": w, "g": jnp.ones((oc,), jnp.float32), "b": jnp.zeros((oc,), jnp.float32)}


def conv_bn_relu(p, x, nd):
    """3^nd conv, same padding, BN over (N, spatial) in train mode, ReLU."""
    k = p["w"].shape[-1]
    pad = [(k // 2, k // 2)] * nd
    y = jax.lax.conv_general_dilated(x, p["w"], (1,) * nd, pad, dimension_numbers=conv_dn(nd))
    axes = (0,) + tuple(range(2, 2 + nd))
    mu = y.mean(axes, keepdims=True)
    var = y.var(axes, keepdims=True)
    shape = (1, -1) + (1,) * nd
    y = (y - mu) * jax.lax.rsqrt(var + 1e-5) * p["g"].reshape(shape) + p["b"].reshape(shape)
    return jax.nn.relu(y)


def maxpool2(x, nd):
    win = (1, 1) + (2,) * nd
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, win, win, "VALID")


def upsample2(x, nd):
    """Trilinear (bilinear at nd=2) ×2, align_corners=False like the Lean `bilinearUpsample`."""
    shape = x.shape[:2] + tuple(2 * s for s in x.shape[2:])
    return jax.image.resize(x, shape, method="linear")


def init_unet(key, nd, in_ch=4, n_classes=4, base=32):
    chans = [base, 2 * base, 4 * base, 8 * base]
    keys = iter(jax.random.split(key, 64))
    params = {"down": [], "up": []}
    ic = in_ch
    for oc in chans:
        params["down"].append([init_conv(next(keys), ic, oc, 3, nd), init_conv(next(keys), oc, oc, 3, nd)])
        ic = oc
    bott = 16 * base
    params["bott"] = [init_conv(next(keys), ic, bott, 3, nd), init_conv(next(keys), bott, bott, 3, nd)]
    ic = bott
    for oc in reversed(chans):
        params["up"].append([init_conv(next(keys), ic + oc, oc, 3, nd), init_conv(next(keys), oc, oc, 3, nd)])
        ic = oc
    params["head"] = {"w": jax.random.normal(next(keys), (n_classes, ic) + (1,) * nd, jnp.float32) * jnp.sqrt(1.0 / ic),
                      "b": jnp.zeros((n_classes,), jnp.float32)}
    return params


def unet_forward(params, x, nd):
    skips = []
    for c1, c2 in params["down"]:
        x = conv_bn_relu(c2, conv_bn_relu(c1, x, nd), nd)
        skips.append(x)
        x = maxpool2(x, nd)
    c1, c2 = params["bott"]
    x = conv_bn_relu(c2, conv_bn_relu(c1, x, nd), nd)
    for (c1, c2), skip in zip(params["up"], reversed(skips)):
        x = jnp.concatenate([upsample2(x, nd), skip], axis=1)
        x = conv_bn_relu(c2, conv_bn_relu(c1, x, nd), nd)
    h = params["head"]
    y = jax.lax.conv_general_dilated(x, h["w"], (1,) * nd, [(0, 0)] * nd, dimension_numbers=conv_dn(nd))
    return y + h["b"].reshape((1, -1) + (1,) * nd)


def loss_fn(params, x, y, nd):
    logits = unet_forward(params, x, nd)
    logp = jax.nn.log_softmax(logits, axis=1)
    onehot = jax.nn.one_hot(y, logits.shape[1], axis=1, dtype=jnp.float32)
    return -(onehot * logp).sum(axis=1).mean()


def make_step(nd, lr=1e-3):
    @jax.jit
    def step(params, x, y):
        loss, g = jax.value_and_grad(loss_fn)(params, x, y, nd)
        params = jax.tree_util.tree_map(lambda p, gg: p - lr * gg, params, g)
        return params, loss
    return step


def n_params(params):
    return sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))


def peak_gib():
    s = DEV.memory_stats() or {}
    return s.get("peak_bytes_in_use", 0) / 2 ** 30, s.get("bytes_limit", 0) / 2 ** 30


# ----------------------------------------------------------------------------- bench
def bench(nd, batch, side, steps, label):
    key = jax.random.PRNGKey(0)
    params = init_unet(key, nd)
    shape = (batch, 4) + (side,) * nd
    x = jax.random.normal(key, shape, jnp.float32)
    y = jax.random.randint(key, (batch,) + (side,) * nd, 0, 4, jnp.int32)
    step = make_step(nd)
    vox = batch * side ** nd
    print(f"--- {label}: {nd}D UNet, {n_params(params):,} params, input {shape} = {vox / 1e6:.2f} Mvox/step",
          flush=True)
    t0 = time.time()
    try:
        params, loss = step(params, x, y)
        jax.block_until_ready(loss)
    except Exception as e:  # XLA raises on OOM at compile or first run
        msg = str(e).splitlines()[0][:160]
        print(f"    FAILED at first step: {msg}", flush=True)
        return None
    t_compile = time.time() - t0
    times = []
    for _ in range(steps):
        t0 = time.time()
        params, loss = step(params, x, y)
        jax.block_until_ready(loss)
        times.append(time.time() - t0)
    med = float(np.median(times)) * 1000
    peak, limit = peak_gib()
    print(f"    first step (incl. compile) {t_compile:.1f} s; then median {med:.0f} ms/step "
          f"over {steps} ({min(times) * 1000:.0f}–{max(times) * 1000:.0f}); "
          f"{vox / med / 1e3:.2f} Mvox/s; peak {peak:.2f} GiB of {limit:.2f}; loss {float(loss):.3f}",
          flush=True)
    return {"label": label, "nd": nd, "batch": batch, "side": side, "ms": med, "mvox_s": vox / med / 1e3,
            "peak_gib": peak, "params": n_params(params)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--skip-2d", action="store_true")
    ap.add_argument("--configs", default="128:2,128:1,96:2,64:4",
                    help="3D configs as side:batch, comma-separated")
    args = ap.parse_args()
    print("backend:", jax.default_backend(), DEV, "jax", jax.__version__, flush=True)
    rows = []
    if not args.skip_2d:
        rows.append(bench(2, 16, 240, args.steps, "2D control (unetBrats shape, 16×240²)"))
    for cfg in args.configs.split(","):
        side, batch = (int(v) for v in cfg.split(":"))
        rows.append(bench(3, batch, side, args.steps, f"3D {side}³ × B{batch}"))
    rows = [r for r in rows if r]
    if rows and rows[0]["nd"] == 2:
        ref = rows[0]["mvox_s"]
        print("\nper-voxel throughput against the 2D control:")
        for r in rows[1:]:
            print(f"  {r['label']:<16} {r['mvox_s']:.2f} Mvox/s = {ref / r['mvox_s']:.1f}× slower per voxel; "
                  f"{r['ms']:.0f} ms/step, peak {r['peak_gib']:.2f} GiB")


if __name__ == "__main__":
    main()
