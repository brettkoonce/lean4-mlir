# End-to-end tf.data throughput of the JAX trainer's train iterator (CPU only, no device copies).
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TFDS_DATA_DIR"] = "/home/skoonce/tensorflow_datasets"
import numpy as np, tensorflow as tf
src = open(sys.argv[1]).read(); src = src[:src.index("def prefetch_to_device")]
src = src.replace("import jax", "jax = None  # ").replace("from jax", "# from jax")
mode = sys.argv[2]; B = int(sys.argv[3]); NB = int(sys.argv[4])
if mode == "nondet":
    src = src.replace("    ds = ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE)",
                      "    ds = ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)")
if mode == "u8":
    old = src[src.index("        img = tf.cast(img, tf.float32)              # 0..255 HWC"):src.index("        return img, ex['label']")]
    src = src.replace(old, "        img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)  # uint8 HWC wire\n")
PP0 = "        img = (_imagenet_decode_random_crop_flip(b)"
assert PP0 in src and "    img = _randaugment(img, 2," in src
if mode == "src":      # no decode at all: the source + shuffle + repeat + flat_map + batch ceiling
    src = src.replace(PP0, "        img = (tf.zeros([_IMG_SIZE, _IMG_SIZE, 3], tf.uint8) + tf.cast(tf.strings.length(b) * 0, tf.uint8)) if True else (_imagenet_decode_random_crop_flip(b)")
if mode == "decode":   # decode + crop + resize + flip, no RandAugment
    src = src.replace("    img = _randaugment(img, 2,", "    img = img if True else _randaugment(img, 2,")
print("mode", mode, "decode present in _pp:", "if True else (_imagenet_decode" not in src, "RA present:", "img if True else _randaugment" not in src)
g = {"__name__": "pipe"}; exec(compile(src, sys.argv[1], "exec"), g)
it = iter(g["build_imagenet_iter"]("train", B, True, True))
for _ in range(8): next(it)
t = time.perf_counter(); stamps = []
for i in range(NB):
    next(it); stamps.append(time.perf_counter())
dt = np.diff([t] + stamps)
import resource; ru = resource.getrusage(resource.RUSAGE_SELF)
print(f"cpu {(ru.ru_utime + ru.ru_stime):.0f} s total")
print(f"{mode:7s} B={B}: {NB*B/(stamps[-1]-t):7.0f} img/s mean | median batch {np.median(dt)*1e3:6.0f} ms "
      f"({B/np.median(dt):6.0f} img/s) | max {dt.max()*1e3:6.0f} ms")
