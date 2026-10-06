#!/usr/bin/env python3
"""The emitted train-crop samplers against their references (planning/gelu_erf_and_torchvision_crop.md §2.4).

Part 1, `cropTorchvision`: `_torchvision_rrc`, exec'd out of a generated shim, against torchvision's
own `RandomResizedCrop.get_params` (run in .venv-timm, where torch lives), N draws at each of four
image shapes: square, 3:4 portrait, 16:9 landscape and a 2:5 portrait that reaches the fallback.
Compared: the Kolmogorov distance of the crop's scale (h·w / H·W) and log aspect (log w/h), the
fallback rate, and the mean offset as a fraction of its range.

Pass: KS ≤ 0.01 on both, fallback rates within 0.005, offset means within 0.01.
Controls, which must go red: TF's `sample_distorted_bounding_box` as the pipeline emitted it before
(the 10% floor, uniform aspect, whole-image fallback), and the emitted sampler with the aspect drawn
uniformly instead of log-uniformly.

Part 2, `cropFallbackCenter` (B0): the trainer's `_imagenet_decode_random_crop_flip`, eager, on a
1600×120 JPEG where TF's sampler mostly returns the whole image. Every window must be a sampled
crop or EfficientNet's centre window (side int(0.875·120) = 105, offsets (H − s + 1) // 2); none may
be the whole image, and the centre window must occur.

Part 3, `trainResize`: TF's antialiased bilinear and bicubic resize against PIL's on four crop sizes
down to 224² (what torchvision's and timm's resized_crop run). Pass: mean |Δ| ≤ 0.3 grey levels;
bilinear max |Δ| ≤ 1 (bicubic's max is PIL's clamping of its overshoot, as C6 measured).

Usage:
    .venv/bin/python scripts/gates/crop_sampler_gate.py [--n 200000]
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(ROOT, "jax", "generated")
TIMM_PY = os.path.join(ROOT, ".venv-timm", "bin", "python")
SHAPES = [(375, 375), (500, 375), (360, 640), (500, 200)]   # (H, W)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200_000)
ap.add_argument("--shim", default=os.path.join(GEN, "generated_vit_tiny_imagenet_shim.py"))
ap.add_argument("--b0", default=os.path.join(GEN, "generated_efficientnet_b0_imagenet.py"))
args = ap.parse_args()


def block(path, start, stop_prefix):
    src = open(path).read().split("\n")
    a = next((i for i, l in enumerate(src) if l.startswith(start)), None)
    if a is None:
        sys.exit(f"⛔ {path} has no `{start}` — not a recipe with this flag")
    b = next(i for i in range(a + 1, len(src)) if src[i].startswith(stop_prefix))
    return "\n".join(src[a:b])


def load(code, name):
    g = {"tf": tf, "np": np, "os": os}
    exec(code, g)
    return g[name]


rrc_src = block(args.shim, "def _torchvision_rrc", "def _imagenet_decode_random_crop_flip")
ours = load(rrc_src, "_torchvision_rrc")
uniform_ratio = load(rrc_src.replace(
    "tf.exp(tf.random.uniform([10], np.log(3. / 4), np.log(4. / 3)))",
    "tf.random.uniform([10], 3. / 4, 4. / 3)"), "_torchvision_rrc")
assert uniform_ratio is not ours


def tf_sampler(shape):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        shape, bounding_boxes=bbox, min_object_covered=0.1, aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0), max_attempts=10, use_image_if_no_bounding_boxes=True)
    return tf.stack([begin[0], begin[1], size[0], size[1]])


def draw(fn, H, W, n):
    shape = tf.constant([H, W, 3], tf.int32)
    ds = tf.data.Dataset.range(n).map(lambda _: fn(shape), num_parallel_calls=tf.data.AUTOTUNE)
    return np.concatenate([b.numpy() for b in ds.batch(20_000)]).astype(np.int64)


def torchvision_draws(n):
    code = f"""
import sys, numpy as np, torch
from torchvision.transforms import RandomResizedCrop
torch.manual_seed(0)
out = {{}}
for H, W in {SHAPES!r}:
    img = torch.empty(3, H, W)
    out[f"{{H}}x{{W}}"] = np.array([RandomResizedCrop.get_params(img, [0.08, 1.0], [3. / 4, 4. / 3])
                                   for _ in range({n})], dtype=np.int64)
np.savez(sys.argv[1], **out)
"""
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "tv.npz")
        subprocess.run([TIMM_PY, "-c", code, f], check=True)
        z = np.load(f)
        return {k: z[k] for k in z.files}


def fallback_window(H, W):   # torchvision's, for counting how often a sampler took it
    r = W / H
    if r < 3 / 4:
        w, h = W, int(round(W / (3 / 4)))
    elif r > 4 / 3:
        h, w = H, int(round(H * (4 / 3)))
    else:
        w, h = W, H
    return np.array([(H - h) // 2, (W - w) // 2, h, w])


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b)).max()


def stats(win, H, W):
    i, j, h, w = win.T
    fb = (win == fallback_window(H, W)).all(1)
    oi = np.where(h < H, i / np.maximum(H - h, 1), 0.5)
    oj = np.where(w < W, j / np.maximum(W - w, 1), 0.5)
    return dict(scale=h * w / (H * W), logr=np.log(w / h), fb=fb.mean(), oi=oi.mean(), oj=oj.mean())


def compare(s, t):
    return dict(ks_scale=ks(s["scale"], t["scale"]), ks_logr=ks(s["logr"], t["logr"]),
                dfb=abs(s["fb"] - t["fb"]), doff=max(abs(s["oi"] - t["oi"]), abs(s["oj"] - t["oj"])))


def passes(c):
    return c["ks_scale"] <= 0.01 and c["ks_logr"] <= 0.01 and c["dfb"] <= 0.005 and c["doff"] <= 0.01


print(f"torchvision reference: {args.n} draws per shape (.venv-timm)")
tv = torchvision_draws(args.n)
ok, red = True, {"tf_sampler": False, "uniform_ratio": False}
print(f"{'shape':>9s} {'sampler':>14s} {'KS scale':>9s} {'KS logr':>8s} {'fallback':>9s} {'tv fb':>7s} {'Δoffset':>8s}")
for H, W in SHAPES:
    t = stats(tv[f"{H}x{W}"], H, W)
    for label, fn in (("emitted", ours), ("tf_sampler", tf_sampler), ("uniform_ratio", uniform_ratio)):
        s = stats(draw(fn, H, W, args.n), H, W)
        c = compare(s, t)
        good = passes(c)
        if label == "emitted":
            ok &= good
        elif not good:
            red[label] = True
        mark = ("✅" if good else "⛔") if label == "emitted" else ("red" if not good else "green")
        print(f"{H:4d}x{W:<4d} {label:>14s} {c['ks_scale']:9.4f} {c['ks_logr']:8.4f} {s['fb']:9.4f} "
              f"{t['fb']:7.4f} {c['doff']:8.4f}  {mark}")

# ── Part 2: B0's EfficientNet fallback ──
b0_src = block(args.b0, "def _imagenet_decode_random_crop_flip", "def _imagenet_decode_center_crop")
if "_cs = " not in b0_src:
    sys.exit(f"⛔ {args.b0}: no centre-crop fallback in the train crop")
g = {"tf": tf, "np": np, "os": os, "_IMG_SIZE": 224, "_CROP_PADDING": 32, "_AUG_SEED": None,
     "_autoaugment": lambda x: x}
exec(b0_src, g)
H, W = 120, 1600
jpeg = tf.io.encode_jpeg(tf.zeros([H, W, 3], tf.uint8))
seen, real = [], tf.io.decode_and_crop_jpeg
tf.io.decode_and_crop_jpeg = lambda b, w, channels: (seen.append(w.numpy()), real(b, w, channels=channels))[1]
try:
    for _ in range(300):
        g["_imagenet_decode_random_crop_flip"](jpeg)
finally:
    tf.io.decode_and_crop_jpeg = real
seen = np.array(seen)
s = int(0.875 * min(H, W))
centre = np.array([(H - s + 1) // 2, (W - s + 1) // 2, s, s])
n_centre = int((seen == centre).all(1).sum())
n_whole = int(((seen[:, 2] == H) & (seen[:, 3] == W)).sum())
b0_ok = n_centre > 0 and n_whole == 0
print(f"B0 fallback on {H}x{W}: {n_centre}/{len(seen)} centre windows {centre.tolist()}, "
      f"{n_whole} whole-image  {'✅' if b0_ok else '⛔'}")

# ── Part 3: the resize kernels against PIL ──
from PIL import Image
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:480, 0:640]
base = np.stack([128 + 100 * np.sin(xx / 9.0), 128 + 100 * np.cos(yy / 13.0), (xx * yy) % 256], -1)
img = np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)
rs_ok = True
for h, w in ((480, 640), (300, 300), (150, 200), (80, 80)):
    for name, pil in (("bilinear", Image.BILINEAR), ("bicubic", Image.BICUBIC)):
        ref = np.asarray(Image.fromarray(img[:h, :w]).resize((224, 224), pil)).astype(np.float64)
        out = np.clip(np.round(tf.image.resize([img[:h, :w]], [224, 224], method=name, antialias=True)[0].numpy()), 0, 255)
        d = np.abs(out - ref)
        good = d.mean() <= 0.3 and (name == "bicubic" or d.max() <= 1)
        rs_ok &= good
        print(f"resize {h}x{w} -> 224 {name:8s} mean |Δ| {d.mean():.3f} max {d.max():.0f}  {'✅' if good else '⛔'}")

print(f"emitted sampler: {'✅ matches torchvision' if ok else '⛔ differs from torchvision'}")
for k, v in red.items():
    print(f"CONTROL {k}: {'✅ red, as it must be' if v else '⛔ GREEN on every shape — the gate is blind'}")
sys.exit(0 if ok and b0_ok and rs_ok and all(red.values()) else 1)
