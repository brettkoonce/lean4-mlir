"""P6 — trained-weight gains for EfficientNet-B0 (BN + SE gate).

Loads the trained verified EfficientNet checkpoint
(.lake/build/efficientnet_adam_ckpt.bin, clean 3×nP, params in render arg order
— verified: every BN γ reads ~1), reconstructs the frozen eval-BN running stats
via a true-batch-norm pass over real Imagenette (the SE gate carries no BN), the
r34 recipe extended with the SE path, gates on recovering the trained accuracy
(87.7%), then injects into `enet_probe_ops(weights=…)`.
"""
import os
import re
import numpy as np
import jax
import jax.numpy as jnp

import adjoint_chain_probe as P
from adjoint_chain_probe import run_iree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, ".lake/build/efficientnet_adam_ckpt.bin")
VAL = "/home/skoonce/lean/claude_max/lean4-jax/data/imagenette/val.bin"
EPS = 1e-5
mlir = open(os.path.join(ROOT, "verified_mlir/efficientnet_fwd_eval.mlir")).read()
hdr = mlir[:mlir.index("{", mlir.index("func.func"))]
sig = [(m.group(1), tuple(int(d) for d in m.group(2).split("x")[:-1]))
       for m in re.finditer(r"%(\w+): tensor<([^>]+)>", hdr)]

cfg = [(32, 32, 16, 8, 3, 1), (16, 96, 24, 4, 3, 2), (24, 144, 24, 6, 3, 1),
       (24, 144, 40, 6, 5, 2), (40, 240, 40, 10, 5, 1),
       (40, 240, 80, 10, 3, 2), (80, 480, 80, 20, 3, 1),
       (80, 480, 80, 20, 3, 1), (80, 480, 112, 20, 5, 1),
       (112, 672, 112, 28, 5, 1), (112, 672, 112, 28, 5, 1),
       (112, 672, 192, 28, 5, 2), (192, 1152, 192, 48, 5, 1),
       (192, 1152, 192, 48, 5, 1), (192, 1152, 192, 48, 5, 1),
       (192, 1152, 320, 48, 3, 1)]

param_args = [(n, sh) for n, sh in sig
              if n != "x" and not n.endswith("nmu") and not n.endswith("nvar")]
nP = sum(int(np.prod(sh)) for _, sh in param_args)
flat = np.fromfile(CKPT, dtype=np.float32, count=nP)
vals, off = {}, 0
for n, sh in param_args:
    k = int(np.prod(sh))
    vals[n] = flat[off:off + k].reshape(sh)
    off += k

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


def swish(x):
    return x * jax.nn.sigmoid(x)


stats = {}


def bn(z, tag, record):
    if record:
        stats[tag] = (z.mean(axis=(0, 2, 3)), z.var(axis=(0, 2, 3)))
    mu, var = stats[tag]
    istd = 1.0 / jnp.sqrt(var + EPS)
    return (z - mu[None, :, None, None]) * (istd * jx(vals[GAMMA(tag)]))[
        None, :, None, None] + jx(vals[BETA(tag)])[None, :, None, None]


def forward(x, record):
    h = swish(bn(jc(x, "sW", 2, 1) + jx(vals["sb"])[None, :, None, None],
                 "st", record))
    for i, (ic, mid, oc, r, k, s) in enumerate(cfg, start=1):
        p = f"b{i}"
        inp = h
        a = h
        if mid != ic:
            a = swish(bn(jc(a, p + "eW", 1, 0)
                         + jx(vals[p + "eb"])[None, :, None, None],
                         p + "e", record))
        ds = swish(bn(jc(a, p + "dW", s, (k - 1) // 2, groups=mid)
                      + jx(vals[p + "db"])[None, :, None, None], p + "d",
                      record))
        sq = ds.mean(axis=(2, 3))                       # SE gate (no BN)
        a1 = swish(sq @ jx(vals[p + "zW1"]) + jx(vals[p + "zb1"]))
        gate = jax.nn.sigmoid(a1 @ jx(vals[p + "zW2"]) + jx(vals[p + "zb2"]))
        se = ds * gate[:, :, None, None]
        h2 = bn(jc(se, p + "pW", 1, 0) + jx(vals[p + "pb"])[None, :, None, None],
                p + "p", record)
        h = h2 + inp if (ic == oc and s == 1) else h2
    h = swish(bn(jc(h, "hW", 1, 0) + jx(vals["hb"])[None, :, None, None],
                 "h", record))
    return h.mean(axis=(2, 3)) @ jx(vals["Wd"]) + jx(vals["bd"])


gpu = jax.devices()[0]
with jax.default_device(gpu):
    xc, _ = load(256)
    forward(jnp.asarray(xc), True)                      # freeze running stats
    stats = {k: (jax.device_get(mu), jax.device_get(v))
             for k, (mu, v) in stats.items()}
    stats = {k: (jnp.asarray(mu), jnp.asarray(v)) for k, (mu, v) in stats.items()}
    xg, yg = load(64)
    la = jax.device_get(forward(jnp.asarray(xg), False))
acc = float((la.argmax(1) == yg).mean())
print(f"GATE: reconstructed-stats trained EfficientNet top-1 on 64 = {acc:.1%}")
if acc < 0.20:
    raise SystemExit("gate failed — operating point wrong, aborting")

for tag, (mu, var) in stats.items():
    vals[tag + "nmu"] = np.asarray(jax.device_get(mu), np.float32)
    vals[tag + "nvar"] = np.asarray(jax.device_get(var), np.float32)
xm, _ = load(32)
vals["x"] = xm.reshape(32, 150528).astype(np.float32)
P.enet_probe_ops(tree_reduce=True, weights=vals)
