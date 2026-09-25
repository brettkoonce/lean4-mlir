#!/usr/bin/env python3
"""C6 gate: the emitted PIL-bicubic geometric ops against PIL itself (planning/imagenet_parity.md C6).

timm runs ShearX/Y, TranslateX/Y and Rotate through PIL with BICUBIC resampling, and TF has no
bicubic projective warp, so `jax/Jax/Codegen.lean`'s `bicubicGeometryPy` writes PIL's affine
sampler out in the TF graph (a = -1, border clamp, truncation). This replays timm's own PIL calls —
`img.transform(size, AFFINE, data, BICUBIC, fillcolor)` and `img.rotate(deg, BICUBIC, fillcolor)` —
on one synthetic 224² image at three magnitudes per op, and compares.

Pass: worst mean |Δ| ≤ 0.2 grey levels and max |Δ| ≤ 1 (float accumulation order).
Control: the same ops through the bilinear block the emitter used before C6 (and still emits for
recipes without `augBicubic`) must come out at worst mean > 1, or the gate could not see a
bilinear regression.

Usage:
    .venv/bin/python scripts/aug_bicubic_pil_check.py [jax/generated/<an augBicubic trainer>.py]
"""
import os, sys
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "jax", "generated", "generated_efficientnet_b0_imagenet_full.py")
src = open(path).read().split("\n")
a = next(i for i, l in enumerate(src) if l.startswith("_AA_MAX"))
b = next(i for i, l in enumerate(src) if l.startswith("def _imagenet_decode_random_crop_flip"))
c = next((i for i in range(a, b) if src[i].startswith("# ── C6 (planning/imagenet_parity.md)")), None)
if c is None:
    sys.exit(f"⛔ {path} carries no bicubic block — not an augBicubic recipe")
full, bilinear = "\n".join(src[a:b]), "\n".join(src[a:c])
ns = {}
for label, block in (("bicubic", full), ("bilinear", bilinear)):
    g = {"tf": tf, "os": os, "np": np}
    exec(block, g)
    ns[label] = g

rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:224, 0:224]
base = np.stack([128 + 100 * np.sin(xx / 9.0), 128 + 100 * np.cos(yy / 13.0), (xx * yy) % 256], -1)
img = np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)
img[60:120, 80:160] = [250, 10, 40]
pil = Image.fromarray(img)
kw = dict(resample=Image.BICUBIC, fillcolor=(128, 128, 128))
W, H = pil.size
cases = []
for f in (0.3, -0.17, 0.05):
    cases.append((f"ShearX {f}", "_aa_shear_x", f, pil.transform(pil.size, Image.AFFINE, (1, f, 0, 0, 1, 0), **kw)))
    cases.append((f"ShearY {f}", "_aa_shear_y", f, pil.transform(pil.size, Image.AFFINE, (1, 0, 0, f, 1, 0), **kw)))
for p in (0.45, -0.2, 0.013):
    # ours maps output x to source x - pct·W; timm's signed pct makes the direction a coin flip
    cases.append((f"TranslateX {p}", "_aa_translate_x", p, pil.transform(pil.size, Image.AFFINE, (1, 0, -p * W, 0, 1, 0), **kw)))
    cases.append((f"TranslateY {p}", "_aa_translate_y", p, pil.transform(pil.size, Image.AFFINE, (1, 0, 0, 0, 1, -p * H), **kw)))
for d in (30.0, -12.5, 3.0):
    cases.append((f"Rotate {d}", "_aa_rotate", d, pil.rotate(d, **kw)))

worst = {"bicubic": 0.0, "bilinear": 0.0}
worst_max = 0
print(f"{'op':16s} {'bicubic mean':>13s} {'max':>4s}   {'bilinear mean':>13s}")
for name, fn, arg, ref in cases:
    ref = np.asarray(ref).astype(np.int32)
    row = {}
    for label in ("bicubic", "bilinear"):
        d = np.abs(ns[label][fn](tf.constant(img), arg).numpy().astype(np.int32) - ref)
        row[label] = d
        worst[label] = max(worst[label], d.mean())
    worst_max = max(worst_max, int(row["bicubic"].max()))
    print(f"{name:16s} {row['bicubic'].mean():13.3f} {row['bicubic'].max():4d}   {row['bilinear'].mean():13.3f}")

ok = worst["bicubic"] <= 0.2 and worst_max <= 1
ctl = worst["bilinear"] > 1.0
print(f"bicubic : worst mean {worst['bicubic']:.3f}, max {worst_max}  {'✅' if ok else '⛔'}")
print(f"CONTROL bilinear: worst mean {worst['bilinear']:.3f}  {'✅ red, as it must be' if ctl else '⛔ GREEN — the gate is blind'}")
if not (ok and ctl):
    sys.exit("⛔ C6 bicubic geometry is not PIL's")
print("✅ the emitted geometric ops are PIL BICUBIC, as timm runs them")
