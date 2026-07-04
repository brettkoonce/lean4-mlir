"""P6 — trained-weight gains for MobileNetV2 (BN ⇒ reconstruct frozen stats).

The mnv2 net trained to 86.8% (runs/mobilenetv2_verified_crop_gpu0.log). Its
resume checkpoint is 3×(nP + 1120) — the +1120 is block-1's t=1 EXPAND
(32×32 conv + 3×32 for b/γ/β) which the CHECKPOINT keeps but the render /
MobileNetV2Layout.specs OMIT (t=1 ⇒ expand is identity, skipped). So the fix is
to skip those 1120 floats after the stem; the remaining params map 1:1 onto the
render args (verified: every BN γ then reads ~1). The rest is the r34 recipe
(reconstruct frozen eval-BN stats via a true-BN pass, gate, inject).

Loads the trained verified MobileNetV2 checkpoint
(.lake/build/mobilenetv2_adam_ckpt.bin, first nP of [params, adam_m, adam_v] —
the trainable params, in the render's non-stat arg order), reconstructs the
frozen eval-BN running stats via a true-batch-norm pass over a real Imagenette
calib batch (as for r34 — the stats aren't in the params block), gates on
recovering the trained accuracy, then injects into `mnv2_probe_ops(weights=…)`.
"""
import os
import re
import numpy as np
import jax
import jax.numpy as jnp

import adjoint_chain_probe as P
from adjoint_chain_probe import run_iree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, ".lake/build/mobilenetv2_adam_ckpt.bin")
VAL = "/home/skoonce/lean/claude_max/lean4-jax/data/imagenette/val.bin"
EPS = 1e-5
mlir = open(os.path.join(ROOT, "verified_mlir/mobilenetv2_fwd_eval.mlir")).read()
hdr = mlir[:mlir.index("{", mlir.index("func.func"))]
sig = [(m.group(1), tuple(int(d) for d in m.group(2).split("x")[:-1]))
       for m in re.finditer(r"%(\w+): tensor<([^>]+)>", hdr)]

cfg = [(32, 32, 16, 1), (16, 96, 24, 2), (24, 144, 24, 1),
       (24, 144, 32, 2), (32, 192, 32, 1), (32, 192, 32, 1),
       (32, 192, 64, 2), (64, 384, 64, 1), (64, 384, 64, 1), (64, 384, 64, 1),
       (64, 384, 96, 1), (96, 576, 96, 1), (96, 576, 96, 1),
       (96, 576, 160, 2), (160, 960, 160, 1), (160, 960, 160, 1),
       (160, 960, 320, 1)]

# ── trained params → the NON-stat render args (stats reconstructed below) ───
param_args = [(n, sh) for n, sh in sig
              if n != "x" and not n.endswith("nmu") and not n.endswith("nvar")]
flat = np.fromfile(CKPT, dtype=np.float32)          # full [params, m, v]
vals, off = {}, 0
for i, (n, sh) in enumerate(param_args):
    if i == 4:                                      # after the 4 stem args:
        off += 1120                                 # skip block-1's t=1 expand
    k = int(np.prod(sh))                            # (in ckpt, omitted by render)
    vals[n] = flat[off:off + k].reshape(sh)
    off += k

# ── real Imagenette val (4-byte header, class-sorted, ImageNet-norm) ────────
REC = 1 + 3 * 224 * 224
NREC = 3925
allb = np.fromfile(VAL, dtype=np.uint8, offset=4).reshape(NREC, REC)
mean = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
std = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]


def load(n):
    recs = allb[np.arange(0, NREC, NREC // n)[:n]]
    y = recs[:, 0].astype(np.int64)
    x = ((recs[:, 1:].reshape(-1, 3, 224, 224).astype(np.float32) / 255.0)
         - mean) / std
    return x, y


GAMMA = lambda t: {"st": "sg", "h": "hg"}.get(t, t + "g")
BETA = lambda t: {"st": "sbt", "h": "hbt"}.get(t, t + "bt")


def jx(a):
    return jnp.asarray(np.asarray(a, np.float32))


def jc(a, Wn, s, p, groups=1):
    return jax.lax.conv_general_dilated(
        a, jx(vals[Wn]), (s, s), ((p, p), (p, p)),
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'), feature_group_count=groups)


def relu6(x):
    return jnp.minimum(jnp.maximum(x, 0.0), 6.0)


stats = {}


def bn(z, tag, record):
    if record:
        stats[tag] = (z.mean(axis=(0, 2, 3)), z.var(axis=(0, 2, 3)))
    mu, var = stats[tag]
    istd = 1.0 / jnp.sqrt(var + EPS)
    return (z - mu[None, :, None, None]) * (istd * jx(vals[GAMMA(tag)]))[
        None, :, None, None] + jx(vals[BETA(tag)])[None, :, None, None]


def forward(x, record):
    h = relu6(bn(jc(x, "sW", 2, 1) + jx(vals["sb"])[None, :, None, None],
                 "st", record))
    for i, (ic, mid, oc, s) in enumerate(cfg, start=1):
        p = f"b{i}"
        inp = h
        a = h
        if mid != ic:
            a = relu6(bn(jc(a, p + "eW", 1, 0)
                         + jx(vals[p + "eb"])[None, :, None, None],
                         p + "e", record))
        a = relu6(bn(jc(a, p + "dW", s, 1, groups=mid)
                     + jx(vals[p + "db"])[None, :, None, None], p + "d", record))
        a = bn(jc(a, p + "pW", 1, 0) + jx(vals[p + "pb"])[None, :, None, None],
               p + "p", record)
        h = a + inp if (ic == oc and s == 1) else a
    h = relu6(bn(jc(h, "hW", 1, 0) + jx(vals["hb"])[None, :, None, None],
                 "h", record))
    return h.mean(axis=(2, 3)) @ jx(vals["Wd"]) + jx(vals["bd"])


gpu = jax.devices()[0]
with jax.default_device(gpu):
    xc, _ = load(256)
    forward(jnp.asarray(xc), True)                # freeze running stats
    stats = {k: (jax.device_get(mu), jax.device_get(v))
             for k, (mu, v) in stats.items()}
    stats = {k: (jnp.asarray(mu), jnp.asarray(v)) for k, (mu, v) in stats.items()}
    xg, yg = load(64)
    la = jax.device_get(forward(jnp.asarray(xg), False))
acc = float((la.argmax(1) == yg).mean())
print(f"GATE: reconstructed-stats trained MobileNetV2 top-1 on 64 = {acc:.1%}")
if acc < 0.20:
    raise SystemExit("gate failed — operating point wrong, aborting")

# inject the reconstructed frozen stats (numpy) so mnv2_probe_ops's calib no-ops
for tag, (mu, var) in stats.items():
    vals[tag + "nmu"] = np.asarray(jax.device_get(mu), np.float32)
    vals[tag + "nvar"] = np.asarray(jax.device_get(var), np.float32)
xm, _ = load(32)
vals["x"] = xm.reshape(32, 150528).astype(np.float32)
P.mnv2_probe_ops(tree_reduce=True, weights=vals)
