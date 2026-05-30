#!/usr/bin/env python3
"""fp32-vs-bf16 throughput benchmark for ViT on ImageNet (1000-class,
224², patch16) — single GPU, synthetic data. Matmul-bound, so this is
where bf16 should pay off (see reference_bf16_gfx1100_conv_vs_gemm:
bf16 GEMM ~7.6× on gfx1100).

Mixed precision = fp32 master weights, bf16 linear/attention matmuls,
fp32 LayerNorm + softmax. Projects 80-epoch ImageNet wall-clock on
2 GPUs from the measured single-GPU step time.

Usage: python3 jax/scripts/jax_vit_bench.py [tiny|small] [per_device_batch]
"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, value_and_grad

CFG = {
    "tiny":  dict(D=192, heads=3, mlp=768,  depth=12),  # ViT-Ti/16 ~5.7M
    "small": dict(D=384, heads=6, mlp=1536, depth=12),  # ViT-S/16  ~22M
}
PATCH, IMG, NCLS, NPATCH = 16, 224, 1000, (224 // 16) ** 2  # 196


def ln(x, g, b, eps=1e-6):
    xf = x.astype(jnp.float32)
    mu = jnp.mean(xf, -1, keepdims=True)
    va = jnp.var(xf, -1, keepdims=True)
    return ((g * (xf - mu) / jnp.sqrt(va + eps) + b)).astype(x.dtype)


def dense(x, w, b, dt):
    return (x.astype(dt) @ w.astype(dt)).astype(jnp.float32) + b


def init(key, D, heads, mlp, depth):
    ks = random.split(key, depth * 4 + 4)
    g = lambda k, sh: random.normal(k, sh) * (sh[0] ** -0.5)
    p = {
        "patch_w": random.normal(ks[0], (3 * PATCH * PATCH, D)) * (3 * PATCH * PATCH) ** -0.5,
        "patch_b": jnp.zeros(D),
        "cls": jnp.zeros((1, 1, D)),
        "pos": random.normal(ks[1], (1, NPATCH + 1, D)) * 0.02,
        "head_w": g(ks[2], (D, NCLS)), "head_b": jnp.zeros(NCLS),
        "hn_g": jnp.ones(D), "hn_b": jnp.zeros(D),
        "blocks": [],
    }
    for i in range(depth):
        b = ks[4 + i * 4: 8 + i * 4]
        p["blocks"].append({
            "n1g": jnp.ones(D), "n1b": jnp.zeros(D),
            "qkv_w": g(b[0], (D, 3 * D)), "qkv_b": jnp.zeros(3 * D),
            "proj_w": g(b[1], (D, D)), "proj_b": jnp.zeros(D),
            "n2g": jnp.ones(D), "n2b": jnp.zeros(D),
            "fc1_w": g(b[2], (D, mlp)), "fc1_b": jnp.zeros(mlp),
            "fc2_w": g(b[3], (mlp, D)), "fc2_b": jnp.zeros(D),
        })
    return p


def attn(x, bp, heads, dt):
    B, T, D = x.shape
    qkv = dense(x, bp["qkv_w"], bp["qkv_b"], dt)          # B,T,3D
    q, k, v = jnp.split(qkv, 3, -1)
    hd = D // heads
    sh = lambda z: z.reshape(B, T, heads, hd).transpose(0, 2, 1, 3)
    q, k, v = sh(q), sh(k), sh(v)
    # attention matmuls in bf16, softmax in fp32
    scores = (q.astype(dt) @ k.astype(dt).transpose(0, 1, 3, 2)).astype(jnp.float32) * (hd ** -0.5)
    a = jax.nn.softmax(scores, -1).astype(dt)
    o = (a @ v.astype(dt)).astype(jnp.float32)            # B,h,T,hd
    o = o.transpose(0, 2, 1, 3).reshape(B, T, D)
    return dense(o, bp["proj_w"], bp["proj_b"], dt)


def forward(p, x, heads, dt):
    B = x.shape[0]
    nh = IMG // PATCH                                      # 14 patches per side
    # patchify by reshape (avoids MIOpen's broken im2col conv on gfx1100):
    # (B,3,224,224) → (B,196,3·16·16) → matmul → (B,196,D)
    h = (x.reshape(B, 3, nh, PATCH, nh, PATCH)
           .transpose(0, 2, 4, 1, 3, 5)
           .reshape(B, NPATCH, 3 * PATCH * PATCH))
    h = dense(h, p["patch_w"], p["patch_b"], dt)          # B,196,D
    D = h.shape[2]
    h = jnp.concatenate([jnp.broadcast_to(p["cls"], (B, 1, D)), h], 1) + p["pos"]
    for bp in p["blocks"]:
        h = h + attn(ln(h, bp["n1g"], bp["n1b"]), bp, heads, dt)
        m = dense(jax.nn.gelu(dense(ln(h, bp["n2g"], bp["n2b"]), bp["fc1_w"], bp["fc1_b"], dt)),
                  bp["fc2_w"], bp["fc2_b"], dt)
        h = h + m
    cls = ln(h, p["hn_g"], p["hn_b"])[:, 0]
    return cls @ p["head_w"].astype(jnp.float32) + p["head_b"]


def make_loss(heads, dt):
    def loss(p, x, y):
        logits = forward(p, x, heads, dt)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])
    return loss


def bench(label, dt, p, x, y, heads, n=40):
    step = jit(value_and_grad(make_loss(heads, dt)))
    t0 = time.time(); l, g = step(p, x, y); p2 = jax.tree_util.tree_map(lambda a, b: a - 0.001 * b, p, g)
    jax.block_until_ready(l); tc = time.time() - t0
    print(f"  [{label}] compile+step0: {tc:5.1f}s  loss={float(l):.3f}")
    ts = []
    for _ in range(n):
        t0 = time.time(); l, g = step(p, x, y); p2 = jax.tree_util.tree_map(lambda a, b: a - 0.001 * b, p, g)
        jax.block_until_ready(l); ts.append(time.time() - t0)
    return float(np.median(ts))


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    BS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    c = CFG[size]
    print(f"ViT-{size} (D={c['D']} heads={c['heads']} mlp={c['mlp']} depth={c['depth']}) "
          f"batch={BS} single-GPU  devices={jax.devices()}\n")
    k = random.PRNGKey(0); k1, k2 = random.split(k)
    p = init(k1, **c)
    x = random.normal(k2, (BS, 3, IMG, IMG)); y = random.randint(random.fold_in(k2, 1), (BS,), 0, NCLS)
    res = {}
    for label, dt in (("fp32", jnp.float32), ("bf16", jnp.bfloat16)):
        res[label] = bench(label, dt, p, x, y, c["heads"])

    print("\n=== throughput (single GPU) ===")
    base = res["fp32"]
    for label in ("fp32", "bf16"):
        print(f"  {label}: {res[label]*1000:7.1f} ms/step   {BS/res[label]:6.0f} img/s   ×{base/res[label]:.2f}")
    spd = base / res["bf16"]

    # 80-epoch ImageNet projection on 2 GPUs (data-parallel, total batch = 2×BS).
    total_bs = BS * 2
    steps_ep = 1281167 // total_bs
    print(f"\n=== 80-epoch ImageNet projection (2 GPUs, batch {total_bs}, {steps_ep} steps/ep) ===")
    for label in ("fp32", "bf16"):
        # per-step time on 2 GPUs ≈ single-GPU per-device time (data-parallel)
        ep_s = steps_ep * res[label]
        print(f"  {label}: {ep_s/60:5.1f} min/ep  →  80 ep = {ep_s*80/3600:5.1f} hr")
    print(f"\n  bf16 speedup ×{spd:.2f}  (vs R34 conv where bf16 was ×0.98 — ViT is matmul-bound)")


if __name__ == "__main__":
    main()
