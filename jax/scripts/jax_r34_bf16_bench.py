#!/usr/bin/env python3
"""fp32-vs-bf16 throughput benchmark: ResNet-34 (ImageNet-shaped,
1000-class) on a SINGLE 7900 XTX via JAX. Synthetic data only.

Why single-GPU: the bf16/fp32 speedup is a per-GPU hardware property
(gfx1100 WMMA bf16 units), so the ratio transfers to the 2-GPU run
without risking the known multi-GPU sharding hang. We validate by
cross-checking the fp32 projection against the measured 15 hr /
30-epoch 2-GPU baseline.

Mixed precision = fp32 master weights, bf16 conv/matmul compute, fp32
softmax. That's the production-shippable config. `forward` is
parameterized by compute dtype; fp32 path casts are no-ops.

Usage: python3 scripts/jax_r34_bf16_bench.py [per_device_batch]
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad


def he(key, shape):
    fan_in = int(np.prod(shape[1:]))
    return random.normal(key, shape) * jnp.sqrt(2.0 / fan_in)


def conv(x, w, stride=1, pad=1):
    return jax.lax.conv_general_dilated(
        x, w, window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))


def bn(x, g, b, eps=1e-5):
    # BN reductions in fp32 for numerical stability even in bf16 mode,
    # then back to x.dtype — this mirrors real mixed-precision recipes.
    xf = x.astype(jnp.float32)
    mu = jnp.mean(xf, axis=(0, 2, 3), keepdims=True)
    va = jnp.var(xf,  axis=(0, 2, 3), keepdims=True)
    out = g.reshape(1, -1, 1, 1) * (xf - mu) / jnp.sqrt(va + eps) + b.reshape(1, -1, 1, 1)
    return out.astype(x.dtype)


def init_basic_block(key, ic, oc, stride):
    k1, k2, k3 = random.split(key, 3)
    out = {
        "w1": he(k1, (oc, ic, 3, 3)),
        "g1": jnp.ones(oc), "b1": jnp.zeros(oc),
        "w2": he(k2, (oc, oc, 3, 3)),
        "g2": jnp.ones(oc), "b2": jnp.zeros(oc),
    }
    if stride != 1 or ic != oc:
        out["wp"] = he(k3, (oc, ic, 1, 1))
        out["gp"] = jnp.ones(oc); out["bp"] = jnp.zeros(oc)
    return out


def fwd_basic_block(p, x, stride):
    h = jax.nn.relu(bn(conv(x, p["w1"], stride=stride), p["g1"], p["b1"]))
    h = bn(conv(h, p["w2"], stride=1), p["g2"], p["b2"])
    if "wp" in p:
        skip = bn(jax.lax.conv_general_dilated(
                    x, p["wp"], window_strides=(stride, stride), padding="VALID",
                    dimension_numbers=("NCHW", "OIHW", "NCHW")),
                  p["gp"], p["bp"])
    else:
        skip = x
    return jax.nn.relu(h + skip)


def init_stage(key, ic, oc, n_blocks, stride):
    keys = random.split(key, n_blocks)
    blocks = []
    for i in range(n_blocks):
        s = stride if i == 0 else 1
        in_ch = ic if i == 0 else oc
        blocks.append(init_basic_block(keys[i], in_ch, oc, s))
    return blocks


def fwd_stage(blocks, x, stride):
    for i, p in enumerate(blocks):
        s = stride if i == 0 else 1
        x = fwd_basic_block(p, x, s)
    return x


def init_r34(key):
    keys = random.split(key, 8)
    return {
        "stem_w": he(keys[0], (64, 3, 7, 7)),
        "stem_g": jnp.ones(64), "stem_b": jnp.zeros(64),
        "s1": init_stage(keys[1], 64,  64,  3, 1),
        "s2": init_stage(keys[2], 64,  128, 4, 2),
        "s3": init_stage(keys[3], 128, 256, 6, 2),
        "s4": init_stage(keys[4], 256, 512, 3, 2),
        "fc_w": he(keys[5], (512, 1000)),
        "fc_b": jnp.zeros(1000),
    }


def forward(p, x, dt):
    # Cast master params + input to compute dtype. fp32 → no-op.
    p = jax.tree_util.tree_map(lambda a: a.astype(dt), p)
    x = x.astype(dt)
    h = jax.lax.conv_general_dilated(x, p["stem_w"],
        window_strides=(2, 2), padding=((3, 3), (3, 3)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))
    h = jax.nn.relu(bn(h, p["stem_g"], p["stem_b"]))
    # Maxpool in fp32: reduce_window(max) VJP needs a constant -inf init,
    # and pooling-in-fp32 is standard for mixed precision anyway.
    h = h.astype(jnp.float32)
    h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max,
        window_dimensions=(1, 1, 3, 3), window_strides=(1, 1, 2, 2),
        padding=((0, 0), (0, 0), (1, 1), (1, 1)))
    h = h.astype(dt)
    h = fwd_stage(p["s1"], h, stride=1)
    h = fwd_stage(p["s2"], h, stride=2)
    h = fwd_stage(p["s3"], h, stride=2)
    h = fwd_stage(p["s4"], h, stride=2)
    h = jnp.mean(h.astype(jnp.float32), axis=(2, 3))  # GAP in fp32
    logits = h @ p["fc_w"].astype(jnp.float32) + p["fc_b"]
    return logits  # fp32 logits → stable softmax


def make_loss(dt):
    def loss_fn(p, x, y):
        logits = forward(p, x, dt)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])
    return loss_fn


def bench(label, dt, params, x, y, n_steps=50):
    step = jit(value_and_grad(make_loss(dt)))
    t0 = time.time()
    loss, grads = step(params, x, y)
    params = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, grads)
    jax.block_until_ready(loss)
    t_compile = time.time() - t0
    print(f"  [{label}] compile + step0: {t_compile:5.1f}s  loss={float(loss):.4f}")
    times = []
    for i in range(n_steps):
        t0 = time.time()
        loss, grads = step(params, x, y)
        params = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, grads)
        jax.block_until_ready(loss)
        times.append(time.time() - t0)
    return float(np.median(times))


def main():
    per_dev = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    print(f"backend = {jax.default_backend()}  devices = {jax.devices()}")
    BS = per_dev
    print(f"batch   = {BS} (single device), 224×224×3, 1000-class, ~21.3M params\n")

    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    params = init_r34(k1)
    x = random.normal(k2, (BS, 3, 224, 224))
    y = random.randint(random.fold_in(k2, 1), (BS,), 0, 1000)

    res = {}
    for label, dt in (("fp32", jnp.float32), ("bf16", jnp.bfloat16)):
        res[label] = bench(label, dt, params, x, y)

    print("\n=== throughput (single GPU) ===")
    base = res["fp32"]
    for label in ("fp32", "bf16"):
        ms = res[label] * 1000
        ips = BS / res[label]
        print(f"  {label}: {ms:7.1f} ms/step   {ips:6.0f} img/s   speedup ×{base/res[label]:.2f}")
    speedup = base / res["bf16"]

    # ImageNet projection. Real 2-GPU fp32 baseline: 15 hr / 30 epochs.
    # steps/epoch at batch 256 = 1,281,167 / 256 = 5004.
    print("\n=== ImageNet projection (1.28M imgs, batch 256, 2 GPUs) ===")
    base_2gpu_30ep_hr = 15.0
    print(f"  fp32 (measured baseline): {base_2gpu_30ep_hr:5.1f} hr / 30 ep")
    print(f"  bf16 (= fp32 / {speedup:.2f}):   {base_2gpu_30ep_hr/speedup:5.1f} hr / 30 ep"
          f"   → saves {base_2gpu_30ep_hr*(1-1/speedup):.1f} hr")
    for ep in (30, 90):
        print(f"    {ep} ep: fp32 {base_2gpu_30ep_hr/30*ep:5.1f} hr  →  bf16 {base_2gpu_30ep_hr/30*ep/speedup:5.1f} hr")
    # Sanity: also project from raw single-GPU img/s × 2 devices.
    proj_fp32 = (1281167 / (res['fp32'] and (BS/res['fp32'])*2)) * 30 / 3600
    print(f"\n  [cross-check] single-GPU img/s ×2 → fp32 {proj_fp32:.1f} hr/30ep "
          f"(vs 15 hr measured — validates the model if close)")


if __name__ == "__main__":
    main()
