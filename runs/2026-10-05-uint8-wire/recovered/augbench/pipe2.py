# tf.data throughput of a generated trainer's train iterator with stages knocked out by flag.
# flags (comma list): nodecode, nora, u8 (no normalize/erase/transpose: uint8 HWC out), noerase, nondet
import os, sys, time, resource
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TFDS_DATA_DIR"] = "/home/skoonce/tensorflow_datasets"
import numpy as np, tensorflow as tf
path, flags, B, NB = sys.argv[1], set(filter(None, sys.argv[2].split(","))), int(sys.argv[3]), int(sys.argv[4])
src = open(path).read(); src = src[:src.index("def prefetch_to_device")]
src = src.replace("import jax", "jax = None  # ").replace("from jax", "# from jax")
def sub(a, b):
    global src
    assert a in src, a
    src = src.replace(a, b)
PP0 = "        img = (_imagenet_decode_random_crop_flip(b)"
if "nodecode" in flags:
    sub(PP0, "        img = (tf.cast(tf.zeros([_IMG_SIZE, _IMG_SIZE, 3], tf.uint8) + tf.cast(tf.strings.length(b) * 0, tf.uint8), tf.float32)) if True else (_imagenet_decode_random_crop_flip(b)")
if "nora" in flags:
    import re
    m = re.search(r"\n    img = _randaugment\(img, [^\n]*\n", src); assert m
    src = src.replace(m.group(0), "\n")
if "noerase" in flags:
    sub("            img = _random_erase(img)", "            pass")
if "u8" in flags:
    old = src[src.index("        img = tf.cast(img, tf.float32)              # 0..255 HWC"):src.index("        return img, ex['label']")]
    src = src.replace(old, "        img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)\n")
if "pilrot" in flags:
    sub("_RA_INC = True", """from PIL import Image as _PILImage
def _pil_rot(x, d):
    return np.asarray(_PILImage.fromarray(x).rotate(float(d), resample=_PILImage.BICUBIC, fillcolor=(128, 128, 128)))
def _aa_rotate_pil(img, deg):
    out = tf.numpy_function(_pil_rot, [img, tf.cast(deg, tf.float32)], tf.uint8, stateful=False)
    out.set_shape(img.shape); return out
_AA_OPS['Rotate'] = (_aa_rotate_pil, _aa_rot, True)
_RA_INC = True""")
if "norot" in flags:
    sub("_RA_INC = True", "_AA_OPS['Rotate'] = (lambda img, deg: img, _aa_rot, True)\n_RA_INC = True")
if "nondet" in flags:
    sub("    ds = ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE)",
        "    ds = ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)")
# AutoGraph re-reads each function's source from its FILE by line number, so the edited
# source must live in a file of its own or the edits are silently replaced by the original.
import hashlib
mod = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mod_" + hashlib.md5(src.encode()).hexdigest()[:10] + ".py")
open(mod, "w").write(src)
g = {"__name__": "pipe"}; exec(compile(src, mod, "exec"), g)
it = iter(g["build_imagenet_iter"]("train", B, True, True))
for _ in range(8): next(it)
c0 = resource.getrusage(resource.RUSAGE_SELF); t = time.perf_counter(); st = []
for i in range(NB):
    next(it); st.append(time.perf_counter())
c1 = resource.getrusage(resource.RUSAGE_SELF); wall = st[-1] - t
cpu = (c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime)
dt = np.diff([t] + st)
print(f"{os.path.basename(path)[10:-3]:28s} {','.join(sorted(flags)) or 'as-shipped':22s} {NB*B/wall:6.0f} img/s  "
      f"median {np.median(dt)*1e3:4.0f} ms/batch  {cpu/wall:4.1f} cores")
