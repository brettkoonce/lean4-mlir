#!/usr/bin/env python3
"""ImageNet training-time ETA for the phase-2 (Lean→JAX) path.

Same idea as `lake run benchmark` (the verified-IREE Imagenette estimator), but for
the full 1000-class ImageNet/JAX trainers — a hand-picked sample, one net per chapter.
Synthetic data only (so no ImageNet on disk needed), bf16 to match the real runs, and
multi-GPU **for free** via `jax.device_count()` + a data-parallel mesh.

For each net: build it, time a steady-state synthetic-batch window after the XLA
compile, derive images/sec, and project per-epoch + full-run wall-clock at that net's
standard schedule (ResNet-34 = 90 epochs, ViT-Tiny = 300, the DeiT recipe).

Reference (single-/multi-7900-XTX, bf16, from jax/runs/*/RESULTS.md):
  ResNet-34 ~139 ms/step (~10.2 min/ep, ~15h/90ep) · ViT-Tiny ~185 ms/step (~7.7 min/ep).

Run:  /path/to/jax-venv/bin/python scripts/jax_imagenet_bench.py
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

DT = jnp.bfloat16            # match the bf16 reference runs
N_TRAIN = 1_281_167          # ImageNet-1k train images
WARMUP, MEASURE = 3, 40      # steps


# ── primitives ──────────────────────────────────────────────────────────────
def he(key, shape):
    fan_in = int(np.prod(shape[1:]))
    return (random.normal(key, shape) * jnp.sqrt(2.0 / fan_in)).astype(DT)


def conv(x, w, stride=1, pad=1):
    return jax.lax.conv_general_dilated(
        x, w, (stride, stride), ((pad, pad), (pad, pad)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))


def bn(x, g, b, eps=1e-5):
    mu = jnp.mean(x, (0, 2, 3), keepdims=True)
    va = jnp.var(x, (0, 2, 3), keepdims=True)
    return g.reshape(1, -1, 1, 1) * (x - mu) * jax.lax.rsqrt(va + eps) + b.reshape(1, -1, 1, 1)


def ln(x, w, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True); va = x.var(-1, keepdims=True)
    return w * (x - mu) * jax.lax.rsqrt(va + eps) + b


# ── ResNet-34 (Chapter 6) ────────────────────────────────────────────────────
def init_block(key, ic, oc, stride):
    k1, k2, k3 = random.split(key, 3)
    p = {"w1": he(k1, (oc, ic, 3, 3)), "g1": jnp.ones(oc, DT), "b1": jnp.zeros(oc, DT),
         "w2": he(k2, (oc, oc, 3, 3)), "g2": jnp.ones(oc, DT), "b2": jnp.zeros(oc, DT)}
    if stride != 1 or ic != oc:
        p |= {"wp": he(k3, (oc, ic, 1, 1)), "gp": jnp.ones(oc, DT), "bp": jnp.zeros(oc, DT)}
    return p


def fwd_block(p, x, stride):
    h = jax.nn.relu(bn(conv(x, p["w1"], stride), p["g1"], p["b1"]))
    h = bn(conv(h, p["w2"]), p["g2"], p["b2"])
    skip = x if "wp" not in p else bn(conv(x, p["wp"], stride, 0), p["gp"], p["bp"])
    return jax.nn.relu(h + skip)


def init_r34(key):
    k = random.split(key, 8)
    def stage(key, ic, oc, n, st):
        ks = random.split(key, n)
        return [init_block(ks[i], ic if i == 0 else oc, oc, st if i == 0 else 1) for i in range(n)]
    return {"sw": he(k[0], (64, 3, 7, 7)), "sg": jnp.ones(64, DT), "sb": jnp.zeros(64, DT),
            "s1": stage(k[1], 64, 64, 3, 1), "s2": stage(k[2], 64, 128, 4, 2),
            "s3": stage(k[3], 128, 256, 6, 2), "s4": stage(k[4], 256, 512, 3, 2),
            "fw": he(k[5], (512, 1000)), "fb": jnp.zeros(1000, DT)}


def fwd_r34(p, x):
    h = jax.nn.relu(bn(conv(x, p["sw"], 2, 3), p["sg"], p["sb"]))
    h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 1, 3, 3), (1, 1, 2, 2),
                              ((0, 0), (0, 0), (1, 1), (1, 1)))
    for s, st in (("s1", 1), ("s2", 2), ("s3", 2), ("s4", 2)):
        for i, blk in enumerate(p[s]):
            h = fwd_block(blk, h, st if i == 0 else 1)
    return jnp.mean(h, (2, 3)) @ p["fw"] + p["fb"]


# ── ViT-Tiny (Chapter 10): patch16, dim192, depth12, heads3, mlp768 ───────────
VD, VL, VH, VM, VP = 192, 12, 3, 768, 16
VTOK = (224 // VP) ** 2 + 1   # 197


def init_vit(key):
    ks = random.split(key, 4 + VL)
    blocks = []
    for i in range(VL):
        b = random.split(ks[4 + i], 6)
        blocks.append({
            "qkv_w": he(b[0], (VD, 3 * VD)), "qkv_b": jnp.zeros(3 * VD, DT),
            "o_w": he(b[1], (VD, VD)), "o_b": jnp.zeros(VD, DT),
            "fc1_w": he(b[2], (VD, VM)), "fc1_b": jnp.zeros(VM, DT),
            "fc2_w": he(b[3], (VM, VD)), "fc2_b": jnp.zeros(VD, DT),
            "n1w": jnp.ones(VD, DT), "n1b": jnp.zeros(VD, DT),
            "n2w": jnp.ones(VD, DT), "n2b": jnp.zeros(VD, DT)})
    return {"pw": he(ks[0], (3 * VP * VP, VD)), "pb": jnp.zeros(VD, DT),
            "cls": (random.normal(ks[1], (1, 1, VD)) * 0.02).astype(DT),
            "pos": (random.normal(ks[2], (1, VTOK, VD)) * 0.02).astype(DT),
            "blocks": blocks,
            "nw": jnp.ones(VD, DT), "nb": jnp.zeros(VD, DT),
            "hw": he(ks[3], (VD, 1000)), "hb": jnp.zeros(1000, DT)}


def attn(p, x):
    B, T, _ = x.shape; hd = VD // VH
    qkv = (x @ p["qkv_w"] + p["qkv_b"]).reshape(B, T, 3, VH, hd).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    a = jax.nn.softmax((q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5), axis=-1)
    return (a @ v).transpose(0, 2, 1, 3).reshape(B, T, VD) @ p["o_w"] + p["o_b"]


def fwd_vit(p, x):
    # Conv-free patch embed: non-overlapping 16×16 patchify is a reshape + matmul,
    # so ViT is pure matmul/attention (no MIOpen conv — leaner on RDNA3 too).
    B = x.shape[0]; g = 224 // VP
    xp = x.reshape(B, 3, g, VP, g, VP).transpose(0, 2, 4, 1, 3, 5).reshape(B, g * g, 3 * VP * VP)
    h = xp @ p["pw"] + p["pb"]                                   # (B,196,192)
    h = jnp.concatenate([jnp.broadcast_to(p["cls"], (B, 1, VD)), h], 1) + p["pos"]
    for b in p["blocks"]:
        h = h + attn(b, ln(h, b["n1w"], b["n1b"]))
        g = jax.nn.gelu(ln(h, b["n2w"], b["n2b"]) @ b["fc1_w"] + b["fc1_b"])
        h = h + (g @ b["fc2_w"] + b["fc2_b"])
    return ln(h, p["nw"], p["nb"])[:, 0] @ p["hw"] + p["hb"]


def make_loss(fwd):
    def loss(p, x, y):
        logits = fwd(p, x).astype(jnp.float32)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])
    return loss


# name, chapter, init, fwd, per-device batch, epochs, params
NETS = [
    ("ResNet-34", 6,  init_r34, fwd_r34, 128, 90,  "21.3M"),
    ("ViT-Tiny",  10, init_vit, fwd_vit, 256, 300, "5.7M"),
]


def bench(name, ch, init, fwd, per_dev, epochs, params, mesh, n, ds, rep):
    bs = per_dev * n
    k1, k2 = random.split(random.PRNGKey(0))
    p = jax.device_put(init(k1), rep)
    x = jax.device_put(random.normal(k2, (bs, 3, 224, 224), DT), ds)
    y = jax.device_put(random.randint(random.fold_in(k2, 1), (bs,), 0, 1000), ds)
    step = jit(value_and_grad(make_loss(fwd)))

    t0 = time.time()
    loss, g = step(p, x, y)
    p = jax.tree_util.tree_map(lambda a, b: a - 0.01 * b, p, g)
    jax.block_until_ready(loss)
    compile_s = time.time() - t0

    times = []
    for i in range(WARMUP + MEASURE):
        t0 = time.time()
        loss, g = step(p, x, y)
        p = jax.tree_util.tree_map(lambda a, b: a - 0.01 * b, p, g)
        jax.block_until_ready(loss)
        if i >= WARMUP:
            times.append(time.time() - t0)
    med = float(np.median(times))
    ips = bs / med
    spe = (N_TRAIN / bs) * med
    return {"name": name, "ch": ch, "params": params, "bs": bs, "epochs": epochs,
            "ms": med * 1000, "ips": ips, "min_ep": spe / 60, "hr": spe * epochs / 3600,
            "compile": compile_s}


def main():
    devs = jax.devices()
    n = len(devs)
    print(f"━━━ ImageNet (phase-2 Lean→JAX) training-time estimate ━━━")
    print(f"  backend: {jax.default_backend()}   devices: {n}   dtype: bf16   (synthetic input)")
    mesh = Mesh(np.array(devs), ("batch",))
    ds, rep = NamedSharding(mesh, P("batch")), NamedSharding(mesh, P())
    rows = []
    for spec in NETS:
        print(f"\n  ▸ probing {spec[0]} ({spec[4]}×{n} batch, 224², 1000-class)…")
        r = bench(*spec, mesh, n, ds, rep)
        print(f"    {r['ms']:.0f} ms/step · {r['ips']:.0f} img/s · {r['min_ep']:.1f} min/epoch "
              f"(compile {r['compile']:.0f}s)")
        rows.append(r)
    print(f"\n  ESTIMATED full-ImageNet training on YOUR setup ({n}× {jax.default_backend()}):\n")
    print(f"  {'Ch':<3}{'Net':<12}{'params':<9}{'min/epoch':<11}{'epochs':<8}full run")
    print("  " + "-" * 52)
    for r in rows:
        print(f"  {r['ch']:<3}{r['name']:<12}{r['params']:<9}{r['min_ep']:<11.1f}{r['epochs']:<8}{r['hr']:.1f} h")
    print("\n  * synthetic-input throughput × ImageNet step count; first run adds the one-time")
    print("    XLA compile shown above. Multi-GPU is data-parallel via jax.device_count().")


if __name__ == "__main__":
    main()
