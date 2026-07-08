#!/usr/bin/env python3
"""`lake run benchmark` for the JAX path — probe this GPU, print run-time estimates.

Nets (--net):
  r50-a3   (default) the R50-A3 demo (`jax/MainResnet50Imagenet.lean`
           `resnet50ImagenetConfigRSBFaithful`): ResNet-50 @160px train res, LAMB
           lr 8e-3 at effective batch 2048 (micro-batch x grad-accum, Ghost-BN per
           micro-step), BCE-with-logits, wd 0.02 with the timm no_weight_decay
           skip-list (BN gamma/beta + biases).
  vit-tiny the ViT demo probe (`probe_vit_tiny_imagenette.py`, AdamW) at the DeiT
           batch (1024) @224, bf16 matmul on CUDA (the transformer lever). Prints
           DeiT-300ep ImageNet estimates for ViT-Ti plus FLOP-scaled ViT-S/16 and
           ViT-B/16 rows (the estimate.py scaling model).

Like `lake run benchmark` (LEAN_MLIR_BENCH_SYNTH), this times the TRAIN STEP only:
one resident batch of real pixels reused every step, eval + host-side aug (mixup/
cutmix/RandAugment) excluded — the demos' own warmup ETA checks cover input-bound
runs. Models are imported from the committed Lean-generated probe modules, so the
graphs are the demos', not rewrites; only the head differs (10-class Imagenette
probe vs 1000-class ImageNet, negligible for both nets).

Usage (from repo root; data/imagenette from ./download_imagenette.sh):

  python jax/probe/benchmark.py                      # R50-A3, all visible GPUs
  python jax/probe/benchmark.py --net vit-tiny
  CUDA_VISIBLE_DEVICES=0 python jax/probe/benchmark.py --eff-steps 5

Per-device micro-batch defaults per net (r50-a3: 128, the demo's 512 across 4
GPUs; vit-tiny: 256, DeiT's 1024 across 4); accum is derived to keep the net's
effective batch, so numbers are comparable across any device count.
"""
import argparse
import importlib.util
import os
import sys
import time

ENV = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
       "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
       "TF_GPU_ALLOCATOR": "cuda_malloc_async"}
for k, v in ENV.items():
    os.environ.setdefault(k, v)

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, random, value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

IMAGENET_EPOCH_IMGS = 1_281_167
IMAGENETTE_EPOCH_IMGS = 9_469
VIT_GF = {"Ti": 1.26, "S": 4.61, "B": 17.58}   # fwd GFLOPs @224/16 (estimate.py)

# ── r50-a3 constants (resnet50ImagenetConfigRSBFaithful) ─────────────────────
LR = 0.008                # RSB-A3 lr @ bs2048
WD = 0.02                 # weight decay, skip-list below
BETA1, BETA2, EPS = 0.9, 0.999, 1e-6   # timm Lamb defaults
TRAIN_RES = 160           # A3 trains @160, evals @224 (eval excluded here)
# Reference anchor: klawd/ares 4x 4060 Ti, A3 micro-step ~250 ms @160px bs512
# (= per-device micro 128, fp32 era), from project memory / mi300x_rental_program.md.
REF_MICRO_MS_PER_DEV128 = 250.0

NETS = {
    "r50-a3": dict(module="probe_resnet50_imagenette_noaug.py",
                   per_dev=128, eff_batch=2048, res=160),
    "vit-tiny": dict(module="probe_vit_tiny_imagenette.py",
                     per_dev=256, eff_batch=1024, res=224),
}


def load_probe_module(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    spec = importlib.util.spec_from_file_location("probe_mod", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def forward_res(m, params, x):
    """The R50 probe module's forward minus its baked 224 reshape: x is [B,3,R,R]."""
    x = m.conv_bn(x, params[0][0], params[0][1], params[0][2], stride=(2, 2), padding='SAME')
    x = jax.nn.relu(x)
    x = m.max_pool2d(x, 3, 2)
    x = m.bottleneck_block_down(params, x, 1, 1)
    x = m.bottleneck_block(params, x, 5)
    x = m.bottleneck_block(params, x, 8)
    x = m.bottleneck_block_down(params, x, 11, 2)
    x = m.bottleneck_block(params, x, 15)
    x = m.bottleneck_block(params, x, 18)
    x = m.bottleneck_block(params, x, 21)
    x = m.bottleneck_block_down(params, x, 24, 2)
    x = m.bottleneck_block(params, x, 28)
    x = m.bottleneck_block(params, x, 31)
    x = m.bottleneck_block(params, x, 34)
    x = m.bottleneck_block(params, x, 37)
    x = m.bottleneck_block(params, x, 40)
    x = m.bottleneck_block_down(params, x, 43, 2)
    x = m.bottleneck_block(params, x, 47)
    x = m.bottleneck_block(params, x, 50)
    x = m.global_avg_pool(x)
    x = m.mm(x, params[53][0].T) + params[53][1]
    return x


def bce_loss(m, params, x, y):
    """BCE-with-logits over one-hot targets (timm --bce-loss, the RSB loss)."""
    logits = forward_res(m, params, x)
    tgt = jax.nn.one_hot(y, logits.shape[-1])
    return jnp.mean(jnp.sum(
        jnp.maximum(logits, 0) - logits * tgt + jnp.log1p(jnp.exp(-jnp.abs(logits))),
        axis=-1))


def lamb_update(params, grads, mom, vel, t):
    """LAMB: Adam moments + per-tensor trust ratio; wd only on ndim>=2 leaves
    (== timm no_weight_decay skip-list: BN gamma/beta + biases are the 1-D ones)."""
    t = t + 1
    bc1 = 1 - BETA1 ** t
    bc2 = 1 - BETA2 ** t

    def leaf(p, g, mo, ve):
        mo = BETA1 * mo + (1 - BETA1) * g
        ve = BETA2 * ve + (1 - BETA2) * g * g
        upd = (mo / bc1) / (jnp.sqrt(ve / bc2) + EPS)
        if p.ndim >= 2:
            upd = upd + WD * p
        wn = jnp.linalg.norm(p)
        un = jnp.linalg.norm(upd)
        trust = jnp.where(wn > 0, jnp.where(un > 0, wn / un, 1.0), 1.0)
        return p - LR * trust * upd, mo, ve

    flat_p, tree = jax.tree.flatten(params)
    flat_g = jax.tree.leaves(grads)
    flat_m = jax.tree.leaves(mom)
    flat_v = jax.tree.leaves(vel)
    out = [leaf(p, g, mo, ve) for p, g, mo, ve in zip(flat_p, flat_g, flat_m, flat_v)]
    new_p = jax.tree.unflatten(tree, [o[0] for o in out])
    new_m = jax.tree.unflatten(tree, [o[1] for o in out])
    new_v = jax.tree.unflatten(tree, [o[2] for o in out])
    return new_p, new_m, new_v, t


def fmt_dur(sec):
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.0f}m"
    return f"{sec / 3600:.1f}h"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", choices=sorted(NETS), default="r50-a3")
    ap.add_argument("--data", default="data/imagenette", help="imagenette bin dir")
    ap.add_argument("--per-dev", type=int, default=None,
                    help="per-device micro-batch (default per net; see header)")
    ap.add_argument("--eff-steps", type=int, default=8,
                    help="effective (post-accum) steps to time, median reported")
    ap.add_argument("--conv-dtype", choices=["bf16", "fp32"], default=None,
                    help="compute dtype lever (r50: conv, vit: matmul); default "
                         "bf16 on cuda, fp32 elsewhere (matches the demos)")
    args = ap.parse_args()

    net = NETS[args.net]
    m = load_probe_module(net["module"])
    res = net["res"]
    eff_batch = net["eff_batch"]

    backend = jax.default_backend()
    devices = jax.devices()
    n = len(devices)
    is_cuda = "cuda" in str(devices[0]).lower() or os.path.exists("/proc/driver/nvidia")
    conv_dt = args.conv_dtype or ("bf16" if is_cuda else "fp32")
    if conv_dt == "bf16":
        if args.net == "r50-a3":
            m.CONV_DT = jnp.bfloat16   # the demo's bf16Conv := true (cuDNN tensor cores)
        else:
            m.DT = jnp.bfloat16        # the vit _bf16 probe variant's one diff

    per_dev = args.per_dev or net["per_dev"]
    micro = per_dev * n
    accum = max(1, eff_batch // micro)
    print("━━━ jax benchmark ━━━ JAX-path training throughput on your GPU")
    if args.net == "r50-a3":
        print(f"net: R50-A3 demo (rsb-faithful tier) — R50 @{res}px, LAMB lr {LR} "
              f"@ eff-bs{eff_batch}, BCE, wd {WD} (skip norm/bias), conv {conv_dt}")
    else:
        print(f"net: ViT-Tiny probe (AdamW, the committed vit demo step) — @{res}px, "
              f"DeiT batch {eff_batch}, matmul {conv_dt}")
    print(f"backend={backend}  devices={n} ({devices[0].device_kind})")
    print(f"micro-batch {micro} ({n} x {per_dev}), accum {accum} → effective {micro * accum}")
    if micro * accum != eff_batch:
        print(f"  NOTE: effective batch {micro * accum} != {eff_batch} (per-dev x devices "
              f"doesn't divide it); estimates still use measured img/s")

    # ── real pixels from the imagenette download, one resident micro-batch set ──
    train_bin = os.path.join(args.data, "train.bin")
    if not os.path.exists(train_bin):
        sys.exit(f"missing {train_bin} — run ./download_imagenette.sh first")
    images, labels = m.load_imagenette(train_bin)   # [N, 3*224*224] normalized, [N]
    need = micro * accum
    idx = np.arange(need) % len(images)
    x4 = images[idx].reshape(-1, 3, 224, 224)
    if res != 224:
        off = (224 - res) // 2
        x4 = x4[:, :, off:off + res, off:off + res]  # center-crop (A3 train res)
    y1 = labels[idx]
    print(f"data: {train_bin} ({len(images)} images @224 → crop {res}, "
          f"{need} resident for probe)")

    mesh = Mesh(np.array(devices), axis_names=("batch",))
    shard = NamedSharding(mesh, P("batch"))
    repl = NamedSharding(mesh, P())

    params = m.init_params(random.PRNGKey(0))
    params = jax.device_put(params, repl)

    if args.net == "r50-a3":
        # x stays 4D (our own resolution-parametric forward)
        xs = [jax.device_put(x4[i * micro:(i + 1) * micro], shard) for i in range(accum)]
        ys = [jax.device_put(y1[i * micro:(i + 1) * micro], shard) for i in range(accum)]
        mom = jax.tree.map(jnp.zeros_like, params)
        vel = jax.tree.map(jnp.zeros_like, params)
        state = (params, mom, vel, jnp.int32(0))
        grad_fn = jit(value_and_grad(lambda p, x, y: bce_loss(m, p, x, y)))
        apply_fn = jit(lamb_update)

        def eff_step(state):
            """One rsb-faithful step: accum micro grads (Ghost-BN per micro), one LAMB apply."""
            params, mom, vel, t_opt = state
            loss, acc = grad_fn(params, xs[0], ys[0])
            for i in range(1, accum):
                li, gi = grad_fn(params, xs[i], ys[i])
                acc = jax.tree.map(jnp.add, acc, gi)
                loss = loss + li
            acc = jax.tree.map(lambda g: g / accum, acc)
            params, mom, vel, t_opt = apply_fn(params, acc, mom, vel, t_opt)
            return (params, mom, vel, t_opt), loss / accum
    else:
        # the generated module's own jit train_step (flat input, baked 224)
        xf = x4.reshape(need, -1)
        xs = [jax.device_put(xf[i * micro:(i + 1) * micro], shard) for i in range(accum)]
        ys = [jax.device_put(y1[i * micro:(i + 1) * micro], shard) for i in range(accum)]
        opt = (jax.tree.map(jnp.zeros_like, params),
               jax.tree.map(jnp.zeros_like, params), jnp.float32(0))
        state = (params, opt)
        lr = jnp.float32(m.LR)

        def eff_step(state):
            params, opt = state
            loss = jnp.float32(0)
            for i in range(accum):   # accum==1 at the default DeiT batch
                params, opt, li = m.train_step(params, opt, xs[i], ys[i], lr)
                loss = loss + li
            return (params, opt), loss / accum

    print(f"\ncompiling (first effective step is XLA cost)…")
    t0 = time.time()
    state, loss = eff_step(state)
    jax.block_until_ready(loss)
    print(f"  compile + step 0: {time.time() - t0:.1f}s   loss={float(loss):.4f}")

    times = []
    for i in range(args.eff_steps):
        t0 = time.time()
        state, loss = eff_step(state)
        jax.block_until_ready(loss)
        times.append(time.time() - t0)
        print(f"  eff-step {i}: {times[-1] * 1000:7.1f} ms  loss={float(loss):.4f}")

    eff_ms = float(np.median(times)) * 1000
    micro_ms = eff_ms / accum
    img_s = micro * accum / (eff_ms / 1000)

    print(f"\nprobe: median eff-step {eff_ms:.0f} ms  (micro ~{micro_ms:.0f} ms)  "
          f"→ {img_s:,.0f} img/s")

    print(f"\nestimates (train step only — eval, host aug, checkpoints excluded):")
    if args.net == "r50-a3":
        if per_dev == 128:
            # reference micro is bs512 across 4 devs = per-dev 128, so per-dev-equal
            # throughputs compare as (ref img/s x n/4) vs measured
            ref_img_s = 512 / (REF_MICRO_MS_PER_DEV128 / 1000) * n / 4
            print(f"  reference 4x4060Ti A3 micro ~{REF_MICRO_MS_PER_DEV128:.0f} ms @bs512 "
                  f"(fp32 era) → this run is {img_s / ref_img_s:.2f}x per-device")
        rows = [("Imagenette A3 100ep (this probe's data)", IMAGENETTE_EPOCH_IMGS, 100, 1.0),
                ("ImageNet   A3 100ep (the demo run)      ", IMAGENET_EPOCH_IMGS, 100, 1.0)]
    else:
        rows = [("Imagenette ViT-Ti 80ep (the jax demo)     ", IMAGENETTE_EPOCH_IMGS, 80, 1.0),
                ("ImageNet ViT-Ti/16 DeiT 300ep @224        ", IMAGENET_EPOCH_IMGS, 300, 1.0),
                ("ImageNet ViT-S/16  DeiT 300ep (FLOP-scaled)", IMAGENET_EPOCH_IMGS, 300,
                 VIT_GF["Ti"] / VIT_GF["S"]),
                ("ImageNet ViT-B/16  DeiT 300ep (FLOP-scaled)", IMAGENET_EPOCH_IMGS, 300,
                 VIT_GF["Ti"] / VIT_GF["B"])]
    for name, epoch_imgs, epochs, scale in rows:
        sec_ep = epoch_imgs / (img_s * scale)
        print(f"  {name}: {fmt_dur(sec_ep)}/epoch → total ~{fmt_dur(sec_ep * epochs)}")


if __name__ == "__main__":
    main()
