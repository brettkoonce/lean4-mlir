#!/usr/bin/env python3
"""Sweep a list of per-epoch ViT-Tiny EMA .bin checkpoints on ONE GPU, computing
the canonical full-50,000-image ImageNet val top-1 AND top-5.

Runs single-GPU (HIP_VISIBLE_DEVICES pins one card), so there is NO batch
sharding — which is exactly why top-5 comes out correct here, unlike the
in-training 2-GPU eval where the `y[:,None]` broadcast under a sharded reduce
silently returned 0. Imports the generated trainer once and reuses its
forward/preprocess; only the weights reload per checkpoint.

Usage: eval_curve_worker.py <out.csv> <ep1,ep2,...>
"""
import os, sys, importlib.util
import numpy as np
import jax, jax.numpy as jnp

GEN = ".lake/build/generated_vit_tiny_imagenet.py"
CKPT_BASE = "/home/skoonce/vit_tiny_imagenet_bf16"
OUT = sys.argv[1]
EPOCHS = [int(e) for e in sys.argv[2].split(",") if e]

spec = importlib.util.spec_from_file_location("gen", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(f"worker on {jax.devices()} — {len(EPOCHS)} checkpoints", flush=True)

@jax.jit
def counts(params, x, y):
    logits = m.forward(params, x)            # forward-only (drop_key=None)
    c1 = jnp.sum(jnp.argmax(logits, axis=-1) == y)
    # top-k-FREE top-5: jax.lax.top_k's indices are broken on ROCm/gfx1100
    # (garbage on device, HIP error on host). Rank the true class instead:
    # top-5 correct iff fewer than 5 classes have a strictly-greater logit.
    true_logit = jnp.take_along_axis(logits, y[:, None], axis=1)   # (B,1)
    c5 = jnp.sum(jnp.sum(logits > true_logit, axis=1) < 5)
    return c1, c5

with open(OUT, "w") as f:
    f.write("epoch,top1,top5\n")

for ep in EPOCHS:
    path = f"{CKPT_BASE}_e{ep}.bin"
    if not os.path.exists(path):
        print(f"e{ep}: MISSING {path}", flush=True)
        continue
    params = m.init_params_from_file(path)
    # 50000 % 250 == 0, so batch 250 with drop_remainder counts every image once.
    ds = m.build_imagenet_iter('validation', 250, training=False, augment=False)
    c1 = c5 = n = 0
    for x, y in ds:
        a, b = counts(params, jnp.asarray(x), jnp.asarray(y))
        c1 += int(a); c5 += int(b); n += int(y.shape[0])
    t1, t5 = 100.0 * c1 / n, 100.0 * c5 / n
    with open(OUT, "a") as f:
        f.write(f"{ep},{t1:.3f},{t5:.3f}\n")
    print(f"e{ep}: top1={t1:.2f} top5={t5:.2f} (n={n})", flush=True)
