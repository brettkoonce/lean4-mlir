#!/usr/bin/env python3
"""Throughput benchmark: ResNet-34 (ImageNet-shaped, 1000-class) on
2× 7900 XTX via JAX multi-GPU sharding. Synthetic data only — point
is to nail down images/sec so we can put a real ETA on the full
90-epoch ImageNet training run.

Reports per-step time after warmup, derives images/sec, projects
per-epoch and full-training wall-clock at standard 90-epoch budget.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def he(key, shape):
    fan_in = int(np.prod(shape[1:]))
    return random.normal(key, shape) * jnp.sqrt(2.0 / fan_in)


def conv(x, w, stride=1, pad=1):
    return jax.lax.conv_general_dilated(
        x, w, window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))


def bn(x, g, b, eps=1e-5):
    mu = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    va = jnp.var(x,  axis=(0, 2, 3), keepdims=True)
    return g.reshape(1, -1, 1, 1) * (x - mu) / jnp.sqrt(va + eps) + b.reshape(1, -1, 1, 1)


def bn_pair(c):
    return jnp.ones(c), jnp.zeros(c)


def init_basic_block(key, ic, oc, stride):
    """Two 3×3 convs with optional 1×1 projection on the skip path."""
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
    """ResNet-34: [conv7s2] [pool] [3,4,6,3] basic blocks, 1000-class head."""
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


def forward(p, x):
    # 7×7 conv stride 2, then 3×3 maxpool stride 2.
    h = jax.lax.conv_general_dilated(x, p["stem_w"],
        window_strides=(2, 2), padding=((3, 3), (3, 3)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))
    h = jax.nn.relu(bn(h, p["stem_g"], p["stem_b"]))
    h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max,
        window_dimensions=(1, 1, 3, 3), window_strides=(1, 1, 2, 2),
        padding=((0, 0), (0, 0), (1, 1), (1, 1)))
    h = fwd_stage(p["s1"], h, stride=1)
    h = fwd_stage(p["s2"], h, stride=2)
    h = fwd_stage(p["s3"], h, stride=2)
    h = fwd_stage(p["s4"], h, stride=2)
    h = jnp.mean(h, axis=(2, 3))  # global avg pool
    return h @ p["fc_w"] + p["fc_b"]


def loss_fn(p, x, y):
    logits = forward(p, x)
    return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])


def main():
    devices = jax.devices()
    n = len(devices)
    print(f"backend = {jax.default_backend()}")
    print(f"devices = {n} ({devices})")

    mesh = Mesh(np.array(devices), axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))
    replicated    = NamedSharding(mesh, P())

    # ImageNet R34 standard: batch 256 (128 per device on 2 GPUs).
    PER_DEV = 128
    BS = PER_DEV * n
    print(f"\nbatch    = {BS} ({n} × {PER_DEV}) at 224×224×3, 1000-class")
    print(f"params   ≈ 21.3M")

    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    params = init_r34(k1)
    params = jax.device_put(params, replicated)

    x_host = random.normal(k2, (BS, 3, 224, 224))
    y_host = random.randint(random.fold_in(k2, 1), (BS,), 0, 1000)
    x = jax.device_put(x_host, data_sharding)
    y = jax.device_put(y_host, data_sharding)

    train_step = jit(value_and_grad(loss_fn))

    print(f"\ncompiling (first step is XLA cost)…")
    t0 = time.time()
    loss, grads = train_step(params, x, y)
    params = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, grads)
    jax.block_until_ready(loss)
    t_compile = time.time() - t0
    print(f"  compile + step 0: {t_compile:.1f}s   loss={float(loss):.4f}")

    print(f"\nrunning 50 steady-state steps for throughput measurement…")
    times = []
    for i in range(50):
        t0 = time.time()
        loss, grads = train_step(params, x, y)
        params = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, grads)
        jax.block_until_ready(loss)
        dt = time.time() - t0
        times.append(dt)
        if i < 3 or i % 10 == 0:
            print(f"  step {i:3d}: {dt*1000:7.1f} ms  loss={float(loss):.4f}")

    times = np.array(times)
    median_ms = np.median(times) * 1000
    images_per_sec = BS / np.median(times)
    print(f"\nthroughput:")
    print(f"  median step time : {median_ms:7.1f} ms")
    print(f"  images/sec       : {images_per_sec:7.0f}")
    print(f"\nfull-ImageNet projections (1,281,167 train images):")
    steps_per_epoch = 1281167 // BS
    sec_per_epoch = steps_per_epoch * np.median(times)
    print(f"  steps/epoch      : {steps_per_epoch}")
    print(f"  sec/epoch        : {sec_per_epoch:7.0f}  ({sec_per_epoch/60:.1f} min)")
    for ep in (30, 60, 90):
        total_h = (sec_per_epoch * ep) / 3600
        print(f"  {ep:2d} epochs total : {total_h:6.1f} hr")


if __name__ == "__main__":
    main()
