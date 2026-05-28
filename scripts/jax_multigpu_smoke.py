#!/usr/bin/env python3
"""Multi-GPU JAX sanity-check on this mars dual-7900-XTX box.

Validates that JAX 0.10.0 + jax-rocm7-plugin 0.9.1.post4 (the fix-version
per upstream-issues/2026-04-jax-rocm-multigpu-mesh-hang/README.md) can
actually drive both GPUs on a ResNet-shaped JIT-compiled train step that
exercises the same primitives the codegen relies on (conv, bn, residual,
GAP, dense, softmax-CE). Not a real training run — synthetic data, 50
steps. The output you care about:
  1. `n_devices = 2` line — JAX sees both GPUs
  2. step times stay roughly constant (no hang)
  3. `rocm-smi --showuse` shows non-zero on both GPU 0 and GPU 1 mid-run
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def conv(x, w, stride=1):
    # NHWC convention is JAX-default; convert to NCHW to mirror our Lean codegen.
    return jax.lax.conv_general_dilated(
        x, w,
        window_strides=(stride, stride),
        padding=((1, 1), (1, 1)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))


def bn_train(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    var = jnp.var(x, axis=(0, 2, 3), keepdims=True)
    norm = (x - mean) / jnp.sqrt(var + eps)
    return gamma.reshape(1, -1, 1, 1) * norm + beta.reshape(1, -1, 1, 1)


def init(key, shape, scale=1.0):
    fan_in = np.prod(shape[1:])
    return random.normal(key, shape) * jnp.sqrt(2.0 / fan_in) * scale


def model_init(key):
    keys = random.split(key, 8)
    return {
        # (oc, ic, kh, kw) — small ResNet-like stack ending in 256 features.
        "w1": init(keys[0], (64,  3,   3, 3)),
        "g1": jnp.ones(64),  "b1": jnp.zeros(64),
        "w2": init(keys[1], (128, 64,  3, 3)),  # stride 2
        "g2": jnp.ones(128), "b2": jnp.zeros(128),
        # residual stage on 128-ch
        "w3": init(keys[2], (128, 128, 3, 3)),
        "g3": jnp.ones(128), "b3": jnp.zeros(128),
        "w4": init(keys[3], (128, 128, 3, 3)),
        "g4": jnp.ones(128), "b4": jnp.zeros(128),
        # final stage 128→256 stride 2
        "w5": init(keys[4], (256, 128, 3, 3)),
        "g5": jnp.ones(256), "b5": jnp.zeros(256),
        # head
        "wfc": init(keys[5], (256, 10)),
        "bfc": jnp.zeros(10),
    }


def forward(params, x):
    h  = jax.nn.relu(bn_train(conv(x,  params["w1"]),         params["g1"], params["b1"]))
    h  = jax.nn.relu(bn_train(conv(h,  params["w2"], stride=2), params["g2"], params["b2"]))
    # Residual block (stride 1, identity skip works at 128→128).
    h2 = jax.nn.relu(bn_train(conv(h,  params["w3"]),         params["g3"], params["b3"]))
    h2 = bn_train(conv(h2, params["w4"]), params["g4"], params["b4"])
    h  = jax.nn.relu(h + h2)
    h  = jax.nn.relu(bn_train(conv(h,  params["w5"], stride=2), params["g5"], params["b5"]))
    # global avg pool → dense.
    h  = jnp.mean(h, axis=(2, 3))
    return h @ params["wfc"] + params["bfc"]


def loss_fn(params, x, y):
    logits = forward(params, x)
    log_softmax = jax.nn.log_softmax(logits)
    return -jnp.mean(log_softmax[jnp.arange(y.shape[0]), y])


def main():
    devices = jax.devices()
    n_devices = len(devices)
    print(f"backend  = {jax.default_backend()}")
    print(f"devices  = {devices}")
    print(f"n_devices = {n_devices}")
    if n_devices < 2:
        print("WARN: <2 devices — multi-GPU test inconclusive")

    mesh = Mesh(np.array(devices), axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))
    replicated    = NamedSharding(mesh, P())

    key = random.PRNGKey(42)
    k1, k2 = random.split(key)
    params = model_init(k1)
    params = jax.device_put(params, replicated)

    # Synthetic batch — divisible by n_devices.
    per_dev = 32
    bs = per_dev * n_devices
    print(f"batch    = {bs} ({n_devices} × {per_dev}) on shape (B, 3, 224, 224)")

    x_host = random.normal(k2, (bs, 3, 224, 224))
    y_host = random.randint(random.fold_in(k2, 1), (bs,), 0, 10)
    x = jax.device_put(x_host, data_sharding)
    y = jax.device_put(y_host, data_sharding)

    train_step = jit(value_and_grad(loss_fn))

    print(f"\ncompiling + running 50 train steps with synthetic data...")
    print(f"(rocm-smi --showuse should show both GPUs nonzero mid-run)\n")
    losses = []
    for step in range(50):
        t0 = time.time()
        loss, grads = train_step(params, x, y)
        # SGD update (toy)
        lr = 0.01
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        loss_val = float(loss)
        losses.append(loss_val)
        t1 = time.time()
        if step < 5 or step % 5 == 0:
            print(f"  step {step:3d}: loss={loss_val:.4f}  ({(t1-t0)*1000:.0f} ms)")

    print(f"\nfirst loss: {losses[0]:.4f}")
    print(f"last  loss: {losses[-1]:.4f}")
    print(f"trend     : {'DOWN' if losses[-1] < losses[0] else 'flat/up'}  ({100*(losses[0]-losses[-1])/losses[0]:.1f}% delta)")
    print(f"OK")


if __name__ == "__main__":
    main()
