#!/usr/bin/env python3
"""Probe whether JAX multi-GPU sharding actually distributes compute on
this dual-7900-XTX box, beyond just "no longer hangs."

Three checks:
  1. Inspect `x.sharding.device_set` — confirms the tensor is mapped to
     both devices logically.
  2. Inspect the compiled HLO via `train_step.lower(...).compile()` —
     looks for `cross-replica-sum` / AllReduce ops that should appear
     when work is sharded across devices.
  3. Run a long train_step loop in a thread while polling
     `rocm-smi --showuse --json` every 200 ms; report
     (min, mean, max) utilization per GPU.

If (1) says both devices, (2) shows AllReduce, but (3) shows GPU 1 at
0%, that's diagnostic — work isn't physically distributed even though
the logical sharding is.
"""
import json
import subprocess
import threading
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def main():
    devices = jax.devices()
    n = len(devices)
    print(f"backend = {jax.default_backend()}")
    print(f"devices = {devices}  (n={n})")
    if n < 2:
        print("need 2+ devices"); return

    mesh = Mesh(np.array(devices), axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))
    replicated    = NamedSharding(mesh, P())

    # Toy MLP: (B, 1024) → (B, 256) → (B, 10).
    key = random.PRNGKey(0)
    k1, k2, k3 = random.split(key, 3)
    W1 = random.normal(k1, (1024, 256)) * 0.01
    b1 = jnp.zeros(256)
    W2 = random.normal(k2, (256, 10)) * 0.01
    b2 = jnp.zeros(10)
    params = (W1, b1, W2, b2)
    params = jax.device_put(params, replicated)

    bs = n * 256  # 512 on 2 devices
    x = jax.device_put(random.normal(k3, (bs, 1024)), data_sharding)
    y = jax.device_put(random.randint(random.fold_in(k3, 1), (bs,), 0, 10), data_sharding)

    # --- Check 1: tensor placement ---
    print(f"\n[1] x.sharding = {x.sharding}")
    print(f"    x.sharding.device_set = {x.sharding.device_set}")
    print(f"    actual devices = {[shard.device for shard in x.addressable_shards]}")

    def loss_fn(params, x, y):
        W1, b1, W2, b2 = params
        h = jax.nn.relu(x @ W1 + b1)
        logits = h @ W2 + b2
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])

    train_step = jit(value_and_grad(loss_fn))

    # --- Check 2: compiled HLO ---
    lowered = train_step.lower(params, x, y)
    compiled = lowered.compile()
    hlo = compiled.as_text()
    n_all_reduce = hlo.count("all-reduce") + hlo.count("AllReduce") + hlo.count("cross-replica-sum")
    n_all_gather = hlo.count("all-gather") + hlo.count("AllGather")
    print(f"\n[2] HLO mentions:")
    print(f"    all-reduce ops: {n_all_reduce}")
    print(f"    all-gather ops: {n_all_gather}")
    print(f"    HLO total size: {len(hlo)} chars")

    # --- Check 3: rocm-smi polling during workload ---
    print(f"\n[3] running 200 train steps with rocm-smi polling...")
    stop = threading.Event()
    samples = []   # list of [gpu0_pct, gpu1_pct]

    def poll():
        while not stop.is_set():
            try:
                r = subprocess.run(["rocm-smi", "--showuse", "--json"],
                                   capture_output=True, text=True, timeout=2)
                if r.returncode == 0 and r.stdout.strip():
                    data = json.loads(r.stdout)
                    g = []
                    for k in sorted(data.keys()):
                        try:
                            g.append(float(data[k].get("GPU use (%)", 0)))
                        except (ValueError, TypeError):
                            g.append(0.0)
                    if g:
                        samples.append(g)
            except Exception:
                pass
            time.sleep(0.2)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    for step in range(200):
        loss, grads = train_step(params, x, y)
        params = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, grads)
        float(loss)  # force materialize

    stop.set()
    poller.join(timeout=2)

    if not samples:
        print("    no rocm-smi samples collected!"); return
    arr = np.array(samples)
    print(f"    samples = {len(arr)}")
    for i in range(arr.shape[1]):
        col = arr[:, i]
        print(f"    GPU[{i}]  min={col.min():.0f}%  mean={col.mean():.1f}%  max={col.max():.0f}%  nonzero={int((col > 0).sum())}/{len(col)}")
    print(f"\nverdict: {'DISTRIBUTED' if arr[:, 1:].max() > 30 else 'GPU0-ONLY'}")


if __name__ == "__main__":
    main()
