#!/usr/bin/env python3
"""A 3D UNet on BraTS (MSD Task01) in JAX — the reference a Lean 3D UNet would be tied to, and
the direct answer to the question planning/brats_25d_3d.md asks: **does volumetric context
beat the slice model on this data**, measured per volume on the same 73 validation patients
the 2D and 2.5D Lean nets are scored on (`lake exe brats-eval`).

The net is `ReferenceNets.unetBrats` with every 3×3 made 3×3×3 and every 2×2 pool 2×2×2:
depth 4, base 32, bottleneck 512, two conv+BN+ReLU per stage, trilinear ×2 + concat on the way
up, 4 modalities in, 4 classes out — 23.5M parameters. It trains on random 128³ patches at
batch 2 (nnU-Net's shape for this data), a third of them centred on a tumour voxel (nnU-Net's
foreground oversampling — at 0.5% enhancing tumour a uniform patch is almost all background),
mirrored at random along each axis, on Dice+CE (`unet-brats-train`'s default arm) under Adam
with warmup and cosine decay. BatchNorm rather than nnU-Net's InstanceNorm, because BN is what
the Lean codegen has and this is meant to be ported.

Scoring is per volume, the literature's protocol and `brats-eval`'s: every slice of every
validation volume, one Dice per patient and region (WT / TC / ET, BraTS's convention for an
absent region), mean over patients. The forward runs on two 128-deep z-windows covering the
155 slices with the overlap's logits averaged — the training depth, not a different one.

Data: whole volumes from `preprocess_brats.py --train-full --val-full` (`train_full.bin` /
`val_full.bin` beside their `.idx`), u8 as on disk, dequantised on the device.

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u \\
        jax/scripts/unet3d_brats.py --steps 4000 --out runs/<dir>/unet3d

Throughput on one 4060 Ti: ~0.9 s/step at 128³ × B2 (jax/scripts/unet3d_gate0.py), so 4,000
steps is ~1 h. The per-volume CSV has `brats-eval`'s columns, the ET false-alarm counts included,
so `scripts/probes/brats_tail.py` puts the two side by side under one post-process.

With `--eval-every N` every N steps writes `<out>_ckpt.npz` (params, BN stats, both Adam moments,
the step) and `<out>_pervol_s<step>.csv`; `--resume <out>_ckpt.npz` continues from that step on
the same schedule, so a dead run loses at most N steps.
"""
import argparse
import json
import os
import queue
import struct
import sys
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

if jax.devices()[0].platform != "gpu" and os.environ.get("ALLOW_CPU") != "1":
    raise SystemExit("no GPU device — refusing to train on a CPU fallback (ALLOW_CPU=1 overrides)")

MODALITIES = 4
NUM_CLASSES = 4
DEQUANT = 5.0 / 127.0            # preprocess_brats.quantize_u8's inverse, as in lean_f32_load_brats
REGIONS = [("WT", (1, 2, 3)), ("TC", (2, 3)), ("ET", (3,))]


# ----------------------------------------------------------------------------- corpus
class Corpus:
    """A whole-volume export: `<stem>.bin` (u8 records, image then mask) and `<stem>.idx`."""

    def __init__(self, data_dir, stem, size):
        self.size = size
        self.rec = MODALITIES * size * size + size * size
        with open(os.path.join(data_dir, f"{stem}.bin"), "rb") as f:
            n = struct.unpack("<I", f.read(4))[0]
        self.mm = np.memmap(os.path.join(data_dir, f"{stem}.bin"), dtype=np.uint8, mode="r",
                            offset=4, shape=(n, self.rec))
        with open(os.path.join(data_dir, f"{stem}.idx"), "rb") as f:
            nv = struct.unpack("<I", f.read(4))[0]
            counts = struct.unpack(f"<{nv}I", f.read(4 * nv))
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.vols = list(zip(starts.tolist(), counts))
        assert sum(counts) == n, (sum(counts), n)
        names_path = os.path.join(data_dir, f"{stem}.json")
        self.names = [v["name"] for v in json.load(open(names_path))["volumes"]] \
            if os.path.exists(names_path) else [str(i) for i in range(nv)]

    def volume(self, v):
        """(image u8 [count, 4, S, S], mask u8 [count, S, S]) of volume v."""
        start, count = self.vols[v]
        rec = np.asarray(self.mm[start:start + count])
        S = self.size
        return (rec[:, :MODALITIES * S * S].reshape(count, MODALITIES, S, S),
                rec[:, MODALITIES * S * S:].reshape(count, S, S))

    def tumour_slices(self, v):
        """z indices of volume v's slices that hold a tumour voxel (read once, cached)."""
        if not hasattr(self, "_tz"):
            self._tz = {}
        if v not in self._tz:
            start, count = self.vols[v]
            S = self.size
            masks = np.asarray(self.mm[start:start + count, MODALITIES * S * S:])
            self._tz[v] = np.where(masks.max(axis=1) > 0)[0]
        return self._tz[v]


class PatchSampler:
    """Random 128³ (or `patch`) crops of whole volumes, a `fg` fraction centred on a tumour
    voxel, each mirrored at random along the three axes. Fills a queue from a thread so the
    step never waits on the page cache."""

    def __init__(self, corpus, patch, batch, fg, seed, depth=8):
        self.c, self.P, self.B, self.fg = corpus, patch, batch, fg
        self.rng = np.random.RandomState(seed)
        self.q = queue.Queue(maxsize=depth)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def one(self):
        c, P, S = self.c, self.P, self.c.size
        v = self.rng.randint(len(c.vols))
        start, count = c.vols[v]
        D = min(P, count)
        if self.rng.rand() < self.fg and len(c.tumour_slices(v)):
            zc = int(self.rng.choice(c.tumour_slices(v)))
            m = np.asarray(c.mm[start + zc, MODALITIES * S * S:]).reshape(S, S)
            ys, xs = np.where(m > 0)
            k = self.rng.randint(len(ys))
            z0 = int(np.clip(zc - D // 2, 0, count - D))
            y0 = int(np.clip(ys[k] - P // 2, 0, S - P))
            x0 = int(np.clip(xs[k] - P // 2, 0, S - P))
        else:
            z0 = self.rng.randint(count - D + 1)
            y0 = self.rng.randint(S - P + 1)
            x0 = self.rng.randint(S - P + 1)
        rec = np.asarray(c.mm[start + z0:start + z0 + D])
        img = rec[:, :MODALITIES * S * S].reshape(D, MODALITIES, S, S)[:, :, y0:y0 + P, x0:x0 + P]
        msk = rec[:, MODALITIES * S * S:].reshape(D, S, S)[:, y0:y0 + P, x0:x0 + P]
        img = np.transpose(img, (1, 0, 2, 3))            # (C, D, H, W)
        if D < P:                                          # a volume shallower than the patch
            pad = P - D
            img = np.pad(img, ((0, 0), (0, pad), (0, 0), (0, 0)))
            msk = np.pad(msk, ((0, pad), (0, 0), (0, 0)))
        for ax in range(3):                                # mirroring along every axis
            if self.rng.rand() < 0.5:
                img = np.flip(img, axis=ax + 1)
                msk = np.flip(msk, axis=ax)
        return np.ascontiguousarray(img), np.ascontiguousarray(msk)

    def _run(self):
        while True:
            xs, ys = zip(*(self.one() for _ in range(self.B)))
            self.q.put((np.stack(xs), np.stack(ys)))

    def get(self):
        return self.q.get()


# ----------------------------------------------------------------------------- model
DN = ("NCDHW", "OIDHW", "NCDHW")


def init_conv(key, ic, oc, k):
    fan_in = ic * k ** 3
    return {"w": jax.random.normal(key, (oc, ic, k, k, k), jnp.float32) * jnp.sqrt(2.0 / fan_in),
            "g": jnp.ones((oc,), jnp.float32), "b": jnp.zeros((oc,), jnp.float32)}


def init_bn_state(p):
    oc = p["g"].shape[0]
    return {"mean": jnp.zeros((oc,), jnp.float32), "var": jnp.ones((oc,), jnp.float32)}


def init_unet(key, base=32):
    chans = [base, 2 * base, 4 * base, 8 * base]
    keys = iter(jax.random.split(key, 64))
    P = {"down": [], "up": []}
    ic = MODALITIES
    for oc in chans:
        P["down"].append([init_conv(next(keys), ic, oc, 3), init_conv(next(keys), oc, oc, 3)])
        ic = oc
    bott = 16 * base
    P["bott"] = [init_conv(next(keys), ic, bott, 3), init_conv(next(keys), bott, bott, 3)]
    ic = bott
    for oc in reversed(chans):
        P["up"].append([init_conv(next(keys), ic + oc, oc, 3), init_conv(next(keys), oc, oc, 3)])
        ic = oc
    P["head"] = {"w": jax.random.normal(next(keys), (NUM_CLASSES, ic, 1, 1, 1), jnp.float32) * jnp.sqrt(1.0 / ic),
                 "b": jnp.zeros((NUM_CLASSES,), jnp.float32)}
    # The BN running stats mirror the conv structure; the head has none.
    S = {"down": [[init_bn_state(c) for c in pair] for pair in P["down"]],
         "bott": [init_bn_state(c) for c in P["bott"]],
         "up": [[init_bn_state(c) for c in pair] for pair in P["up"]],
         "head": {}}
    return P, S


def conv_bn_relu(p, s, x, train):
    """3³ conv (same), BN, ReLU. Train mode normalises with the batch and returns its stats;
    eval mode uses the running `s`."""
    y = jax.lax.conv_general_dilated(x, p["w"], (1, 1, 1), [(1, 1)] * 3, dimension_numbers=DN)
    shape = (1, -1, 1, 1, 1)
    if train:
        mu = y.mean((0, 2, 3, 4))
        var = y.var((0, 2, 3, 4))
        stats = {"mean": mu, "var": var}
    else:
        mu, var = s["mean"], s["var"]
        stats = s
    y = (y - mu.reshape(shape)) * jax.lax.rsqrt(var.reshape(shape) + 1e-5) * p["g"].reshape(shape) + p["b"].reshape(shape)
    return jax.nn.relu(y), stats


def maxpool2(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 1, 2, 2, 2), (1, 1, 2, 2, 2), "VALID")


def upsample2(x):
    return jax.image.resize(x, x.shape[:2] + tuple(2 * d for d in x.shape[2:]), method="linear")


def unet_forward(P, S, x, train):
    """Returns (logits [B, 4, D, H, W], BN stats in S's shape)."""
    out = {"down": [], "up": [], "head": {}}
    skips = []
    for (c1, c2), (s1, s2) in zip(P["down"], S["down"]):
        x, n1 = conv_bn_relu(c1, s1, x, train)
        x, n2 = conv_bn_relu(c2, s2, x, train)
        out["down"].append([n1, n2])
        skips.append(x)
        x = maxpool2(x)
    (c1, c2), (s1, s2) = P["bott"], S["bott"]
    x, n1 = conv_bn_relu(c1, s1, x, train)
    x, n2 = conv_bn_relu(c2, s2, x, train)
    out["bott"] = [n1, n2]
    for (c1, c2), (s1, s2), skip in zip(P["up"], S["up"], reversed(skips)):
        x = jnp.concatenate([upsample2(x), skip], axis=1)
        x, n1 = conv_bn_relu(c1, s1, x, train)
        x, n2 = conv_bn_relu(c2, s2, x, train)
        out["up"].append([n1, n2])
    h = P["head"]
    y = jax.lax.conv_general_dilated(x, h["w"], (1, 1, 1), [(0, 0)] * 3, dimension_numbers=DN)
    return y + h["b"].reshape(1, -1, 1, 1, 1), out


def dequant(x_u8):
    return (x_u8.astype(jnp.float32) - 128.0) * DEQUANT


def loss_fn(P, S, x_u8, y, kind):
    logits, stats = unet_forward(P, S, dequant(x_u8), True)
    logp = jax.nn.log_softmax(logits, axis=1)
    onehot = jax.nn.one_hot(y, NUM_CLASSES, axis=1, dtype=jnp.float32)
    ce = -(onehot * logp).sum(axis=1).mean()
    if kind == "ce":
        return ce, stats
    # Soft Dice over the three tumour classes, each over the whole batch.
    p = jnp.exp(logp)[:, 1:]
    t = onehot[:, 1:]
    inter = (p * t).sum((0, 2, 3, 4))
    den = p.sum((0, 2, 3, 4)) + t.sum((0, 2, 3, 4))
    dice = (2.0 * inter + 1.0) / (den + 1.0)
    return ce + (1.0 - dice.mean()), stats


def make_step(kind, lr_fn, wd, total):
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def step(P, S, m, v, t, x_u8, y):
        (loss, stats), g = jax.value_and_grad(loss_fn, has_aux=True)(P, S, x_u8, y, kind)
        lr = lr_fn(t, total)
        m = jax.tree_util.tree_map(lambda a, gg: b1 * a + (1 - b1) * gg, m, g)
        v = jax.tree_util.tree_map(lambda a, gg: b2 * a + (1 - b2) * gg * gg, v, g)
        tf = t.astype(jnp.float32)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** tf), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** tf), v)
        P = jax.tree_util.tree_map(lambda p, a, b: p - lr * (a / (jnp.sqrt(b) + eps) + wd * p), P, mh, vh)
        # Running BN stats: the batch's on the first step, then momentum 0.1 (the Lean loop's).
        mom = jnp.where(t == 1, 1.0, 0.1)
        S = jax.tree_util.tree_map(lambda r, bs: (1 - mom) * r + mom * bs, S, stats)
        return P, S, m, v, loss

    return step


def lr_schedule(peak, warmup):
    def f(t, total):
        tf = t.astype(jnp.float32)
        warm = peak * tf / warmup
        prog = jnp.clip((tf - warmup) / jnp.maximum(total - warmup, 1), 0.0, 1.0)
        cos = peak * 0.5 * (1.0 + jnp.cos(jnp.pi * prog))
        return jnp.where(tf < warmup, warm, cos)
    return f


# ----------------------------------------------------------------------------- eval
def dice_of(inter, gt, pr):
    if gt == 0:
        return 1.0 if pr == 0 else 0.0
    return 2.0 * inter / (gt + pr)


def region_counts(conf, cls):
    idx = np.array(cls)
    inter = conf[np.ix_(idx, idx)].sum()
    return int(inter), int(conf[idx, :].sum()), int(conf[:, idx].sum())


def evaluate(P, S, val, depth, out_csv=None, log=print, limit=None):
    """Per-volume Dice over every slice of every validation volume, on z-windows of `depth`.
    `limit` scores only the first volumes (a smoke test, not a result)."""
    @jax.jit
    def fwd(x_u8):
        logits, _ = unet_forward(P, S, dequant(x_u8), False)
        return logits

    conf_all = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    conf_tum = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    n_tum = 0
    per_vol = []
    t0 = time.time()
    n_vols = len(val.vols) if limit is None else min(limit, len(val.vols))
    for v in range(n_vols):
        img, msk = val.volume(v)                  # [count, 4, S, S], [count, S, S]
        count = img.shape[0]
        x = np.transpose(img, (1, 0, 2, 3))[None]  # [1, 4, count, S, S]
        D = min(depth, count)
        starts = sorted(set([0, count - D] + list(range(0, count - D + 1, D))))
        logit_sum = np.zeros((NUM_CLASSES, count) + img.shape[2:], np.float32)
        hits = np.zeros((count,), np.float32)
        for z0 in starts:
            lg = np.asarray(fwd(jnp.asarray(x[:, :, z0:z0 + D])))[0]
            logit_sum[:, z0:z0 + D] += lg
            hits[z0:z0 + D] += 1
        pred = np.argmax(logit_sum / hits[None, :, None, None], axis=0).astype(np.int64)
        conf = np.bincount((msk.astype(np.int64) * NUM_CLASSES + pred).ravel(),
                           minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        conf_all += conf
        for z in range(count):
            if msk[z].max() > 0:
                n_tum += 1
                conf_tum += np.bincount((msk[z].astype(np.int64) * NUM_CLASSES + pred[z]).ravel(),
                                        minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        row = {"volume": v, "slices": count}
        for name, cls in REGIONS:
            i, g, p = region_counts(conf, cls)
            row.update({f"{name}_inter": i, f"{name}_gt": g, f"{name}_pred": p, f"{name}_dice": dice_of(i, g, p)})
        # Slices with no ground-truth ET, and those of them with >= 1 / >= 10 predicted ET pixels.
        clear = (msk == 3).sum((1, 2)) == 0
        pr_et = (pred == 3).sum((1, 2))
        row.update({"ET_clear_slices": int(clear.sum()), "ET_fa1": int((clear & (pr_et >= 1)).sum()),
                    "ET_fa10": int((clear & (pr_et >= 10)).sum())})
        per_vol.append(row)
    log(f"  scored {n_vols} volumes in {time.time() - t0:.0f} s")

    def pooled(label, conf, n):
        parts = [f"{name} {dice_of(*region_counts(conf, cls)):.4f}" for name, cls in REGIONS]
        tp = np.diag(conf).astype(np.float64)
        uni = conf.sum(0) + conf.sum(1) - np.diag(conf)
        miou = np.mean(np.where(uni > 0, tp / np.maximum(uni, 1), 0.0))
        return f"  pooled Dice, {label} ({n} slices): {'  '.join(parts)}  mIoU {miou:.4f}"

    log(pooled("tumour-bearing slices", conf_tum, n_tum))
    log(pooled("every slice", conf_all, int(conf_all.sum() // (val.size * val.size))))
    log(f"  per-volume Dice over {n_vols} patients (mean ± sd, median; mean over volumes with the region; volumes without it):")
    summary = {}
    for name, _ in REGIONS:
        d = np.array([r[f"{name}_dice"] for r in per_vol])
        present = np.array([r[f"{name}_dice"] for r in per_vol if r[f"{name}_gt"] > 0])
        log(f"    {name}: {d.mean():.4f} ± {d.std(ddof=1):.4f}  median {np.median(d):.4f}   "
            f"present-only {present.mean():.4f} (n={len(present)})   absent in {len(d) - len(present)}")
        summary[name] = float(d.mean())
    k = max(1, n_vols // 10)
    log(f"  tail over {n_vols} patients (worst-10% mean · n<0.7 · n<0.5):")
    for name, _ in REGIONS:
        d = np.array([r[f"{name}_dice"] for r in per_vol])
        log(f"    {name}: {np.sort(d)[:k].mean():.4f} · {(d < 0.7).sum()} · {(d < 0.5).sum()}")
    clear = sum(r["ET_clear_slices"] for r in per_vol)
    fa1, fa10 = sum(r["ET_fa1"] for r in per_vol), sum(r["ET_fa10"] for r in per_vol)
    log(f"  ET false alarms over {clear} slices with no ET: >=1 px on {fa1} ({fa1 / max(clear, 1):.4f}), "
        f">=10 px on {fa10} ({fa10 / max(clear, 1):.4f})")
    if out_csv:
        cols = ["volume", "slices"] + [f"{n}_{k}" for n, _ in REGIONS for k in ("inter", "gt", "pred", "dice")] \
            + ["ET_clear_slices", "ET_fa1", "ET_fa10"]
        with open(out_csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in per_vol:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
        log(f"  wrote {out_csv}")
    return summary


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/brats224")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--fg", type=float, default=1.0 / 3.0, help="fraction of patches centred on a tumour voxel")
    ap.add_argument("--loss", choices=["dicece", "ce"], default="dicece")
    ap.add_argument("--eval-every", type=int, default=0, help="also score per volume every N steps (0: end only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/unet3d", help="prefix for the params .npz, the CSV and the log")
    ap.add_argument("--init", default=None, help="a saved .npz to evaluate (no training)")
    ap.add_argument("--resume", default=None, help="a <out>_ckpt.npz to continue training from")
    ap.add_argument("--val-limit", type=int, default=None, help="smoke test: score only the first N volumes")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logf = open(args.out + "_log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"backend {jax.default_backend()} {jax.devices()[0]} jax {jax.__version__}; args {vars(args)}")
    size = json.load(open(os.path.join(args.data, "val_full.json")))["size"]
    val = Corpus(args.data, "val_full", size)
    key = jax.random.PRNGKey(args.seed)
    P, S = init_unet(key)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(P))
    log(f"3D UNet: {n_params:,} params; patch {args.patch}³ × B{args.batch}; {len(val.vols)} val volumes")

    if args.init:
        z = np.load(args.init, allow_pickle=True)
        P = jax.tree_util.tree_map(lambda _, a: jnp.asarray(a), P, z["P"].item())
        S = jax.tree_util.tree_map(lambda _, a: jnp.asarray(a), S, z["S"].item())
        evaluate(P, S, val, args.patch, args.out + "_pervol.csv", log, args.val_limit)
        return

    train = Corpus(args.data, "train_full", size)
    log(f"train: {len(train.vols)} volumes, {train.mm.shape[0]} slices")
    step = make_step(args.loss, lr_schedule(args.lr, args.warmup), args.wd, args.steps)
    m = jax.tree_util.tree_map(jnp.zeros_like, P)
    v = jax.tree_util.tree_map(jnp.zeros_like, P)
    t0_step = 0
    if args.resume:
        z = np.load(args.resume, allow_pickle=True)
        load = lambda ref, key: jax.tree_util.tree_map(lambda _, a: jnp.asarray(a), ref, z[key].item())
        P, S, m, v = load(P, "P"), load(S, "S"), load(P, "m"), load(P, "v")
        t0_step = int(z["t"])
        log(f"  resumed {args.resume} at step {t0_step}")
    # A resumed run draws fresh patches rather than replaying the first run's.
    sampler = PatchSampler(train, args.patch, args.batch, args.fg, args.seed + 1000003 * t0_step)

    def save(path, t):
        tree = lambda x: np.asarray(jax.tree_util.tree_map(np.asarray, x), dtype=object)
        np.savez(path + ".tmp.npz", P=tree(P), S=tree(S), m=tree(m), v=tree(v), t=t)
        os.replace(path + ".tmp.npz", path)

    losses, times = [], []
    t_start = time.time()
    for t in range(t0_step + 1, args.steps + 1):
        x, y = sampler.get()
        t0 = time.time()
        P, S, m, v, loss = step(P, S, m, v, jnp.asarray(t, jnp.int32), jnp.asarray(x), jnp.asarray(y.astype(np.int32)))
        loss = float(loss)
        times.append(time.time() - t0)
        losses.append(loss)
        if t <= 3 or t % 50 == 0:
            log(f"  step {t}/{args.steps}: loss={np.mean(losses[-50:]):.4f} ({times[-1] * 1000:.0f} ms; "
                f"median {np.median(times[-50:]) * 1000:.0f} ms; queue {sampler.q.qsize()}; "
                f"{(time.time() - t_start) / 60:.1f} min)")
        if np.isnan(loss):
            log("  loss is NaN — stopping"); break
        if args.eval_every and t % args.eval_every == 0 and t < args.steps:
            save(args.out + "_ckpt.npz", t)
            log(f"--- eval at step {t} (checkpoint {args.out}_ckpt.npz)")
            evaluate(P, S, val, args.patch, f"{args.out}_pervol_s{t}.csv", log, args.val_limit)
    log(f"trained {args.steps} steps in {(time.time() - t_start) / 60:.1f} min; median step {np.median(times) * 1000:.0f} ms")
    np.savez(args.out + "_params.npz",
             P=np.asarray(jax.tree_util.tree_map(np.asarray, P), dtype=object),
             S=np.asarray(jax.tree_util.tree_map(np.asarray, S), dtype=object))
    log(f"  saved {args.out}_params.npz")
    log("--- final eval")
    evaluate(P, S, val, args.patch, args.out + "_pervol.csv", log, args.val_limit)


if __name__ == "__main__":
    main()
