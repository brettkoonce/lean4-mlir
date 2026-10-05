import os, sys, time, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp_proto as P
src = open(sys.argv[1]).read().split("\n")
a = next(i for i, l in enumerate(src) if l.startswith("_AA_MAX"))
b = next(i for i, l in enumerate(src) if l.startswith("def _imagenet_decode_random_crop_flip"))
g = {"tf": tf, "os": os, "np": np}; exec("\n".join(src[a:b]), g)
rng = np.random.default_rng(0)
imgs = [tf.constant(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(40)]
def rotvec(deg):
    # the emitted _aa_rotate's matrix, reproduced by calling it through a spy
    return None
def bench(fn, reps=3):
    f = tf.function(fn); f(imgs[0]); f(imgs[1])
    t = time.perf_counter()
    for _ in range(reps):
        for x in imgs: r = f(x)
    r.numpy(); return (time.perf_counter() - t) / (reps * len(imgs)) * 1e3
# capture the rotate vec the emitted code builds, by swapping _aa_transform for a recorder
cap = {}
orig_T = g["_aa_transform"]
def rec(img, vec): cap["vec"] = vec; return orig_T(img, vec)
g["_aa_transform"] = rec
ang = 23.0
ref_rot = g["_aa_rotate"](imgs[0], ang).numpy(); vec = cap["vec"]; g["_aa_transform"] = orig_T
cases = [
  ("rotate  emitted",       lambda x: orig_T(x, vec)),
  ("rotate  u8 taps",       lambda x: P.transform_u8taps(x, vec)),
  ("rotate  one gather",    lambda x: P.transform_onegather(x, vec)),
  ("rotate  TF bilinear",   lambda x: tf.raw_ops.ImageProjectiveTransformV3(images=tf.expand_dims(x, 0),
        transforms=tf.constant([[0.9, 0.1, 0.0, -0.1, 0.9, 0.0, 0.0, 0.0]]), output_shape=[224, 224],
        interpolation="BILINEAR", fill_mode="CONSTANT", fill_value=128.0)[0]),
  ("shearX  emitted",       lambda x: g["_aa_warp1d"](x, 0.2, 0.1, 1)),
  ("shearX  u8",            lambda x: P.warp1d_u8(x, 0.2, 0.1, 1)),
  ("shearY  emitted",       lambda x: g["_aa_warp1d"](x, 0.2, 0.1, 0)),
  ("shearY  u8",            lambda x: P.warp1d_u8(x, 0.2, 0.1, 0)),
]
for name, fn in cases:
    print(f"{name:22s} {bench(fn):6.2f} ms/img")
# exactness vs the emitted versions over all 40 images, several angles / shears
worst = {}
for x in imgs[:10]:
    for d in (30.0, -12.5, 3.0, 23.0):
        cap.clear(); g["_aa_transform"] = rec; e = g["_aa_rotate"](x, d).numpy(); v = cap["vec"]; g["_aa_transform"] = orig_T
        for nm, fn in (("u8 taps", P.transform_u8taps), ("one gather", P.transform_onegather)):
            dd = np.abs(fn(x, v).numpy().astype(int) - e.astype(int))
            worst[nm] = max(worst.get(nm, (0, 0)), (dd.max(), int((dd > 0).sum())))
    for s in (0.3, -0.17, 0.05):
        for ax in (0, 1):
            dd = np.abs(P.warp1d_u8(x, s, 0.5 * s, ax).numpy().astype(int) - g["_aa_warp1d"](x, s, 0.5 * s, ax).numpy().astype(int))
            worst["warp1d u8"] = max(worst.get("warp1d u8", (0, 0)), (dd.max(), int((dd > 0).sum())))
for k, (m, n) in worst.items():
    print(f"exactness vs emitted: {k:11s} max |Δ| {m}, worst-case differing pixels {n}")
