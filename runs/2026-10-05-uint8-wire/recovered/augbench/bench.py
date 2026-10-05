# Per-stage CPU cost of the ViT-S JAX trainer's augmentation, single-threaded, on real ImageNet JPEGs.
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, tensorflow as tf, tensorflow_datasets as tfds
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
src = open(sys.argv[1]).read()
src = src[:src.index("def prefetch_to_device")]
src = src.replace("import jax", "jax = None  # ").replace("from jax", "# from jax")
g = {"__name__": "augbench"}
exec(compile(src, sys.argv[1], "exec"), g)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ds = tfds.load("imagenet2012", split="train", decoders={"image": tfds.decode.SkipDecoding()},
               data_dir="/home/skoonce/tensorflow_datasets")
raw = [ex["image"] for ex in ds.take(N)]
def timeit(fn, xs, reps=1):
    fn(xs[0]); fn(xs[1])
    t = time.perf_counter()
    for _ in range(reps):
        for x in xs: r = fn(x)
    _ = np.asarray(r)
    return (time.perf_counter() - t) / (len(xs) * reps) * 1e3
S = g["_IMG_SIZE"]
@tf.function
def dcr(b):
    shape = tf.io.extract_jpeg_shape(b)
    bb, sz, _ = tf.image.sample_distorted_bounding_box(shape, tf.constant([0.,0.,1.,1.], shape=[1,1,4]),
        min_object_covered=0.1, aspect_ratio_range=(3/4, 4/3), area_range=(0.08, 1.0), max_attempts=10,
        use_image_if_no_bounding_boxes=True)
    oy, ox, _ = tf.unstack(bb); th, tw, _ = tf.unstack(sz)
    img = tf.io.decode_and_crop_jpeg(b, tf.stack([oy, ox, th, tw]), channels=3)
    return tf.image.resize([img], [S, S], method="bicubic", antialias=True)[0]
print(f"decode+crop+resize      {timeit(dcr, raw):7.2f} ms/img")
imgs = [tf.cast(tf.clip_by_value(dcr(b), 0, 255), tf.uint8) for b in raw[:100]]
res = {}
for nm in g["_RA_OPS"]:
    fn, magf, _ = g["_AA_OPS"][nm]
    f = tf.function(lambda x, nm=nm: g["_aa_apply_op"](x, nm, 9.0))
    res[nm] = timeit(f, imgs)
for nm, v in sorted(res.items(), key=lambda kv: -kv[1]):
    print(f"  RA {nm:14s}      {v:7.2f} ms/img")
ra = tf.function(lambda x: g["_randaugment"](tf.cast(x, tf.float32), 2, 9.0, 0.5))
print(f"randaugment(2, m9) full {timeit(ra, imgs, 3):7.2f} ms/img (expected {sum(res.values())/15*2*0.5:.2f} from the op table)")
fimgs = [tf.cast(x, tf.float32) for x in imgs]
er = tf.function(lambda x: g["_random_erase"]((x - g["_MEAN_RGB"]) / g["_STD_RGB"]))
print(f"normalize + erase(p.25) {timeit(er, fimgs):7.2f} ms/img")
pp = tf.function(lambda b: g["build_imagenet_iter"].__code__ and None)
M, SD = g["_MEAN_RGB"], g["_STD_RGB"]
nrm = tf.function(lambda x: (tf.cast(x, tf.float32) - M) / SD)
tr = tf.function(lambda x: tf.reshape(tf.transpose((tf.cast(x, tf.float32) - M) / SD, [2, 0, 1]), [-1]))
er1 = tf.function(lambda x: g["_random_erase"](x))
print(f"normalize only          {timeit(nrm, imgs):7.2f} ms/img")
print(f"normalize+transpose     {timeit(tr, imgs):7.2f} ms/img")
print(f"erase(p.25) only        {timeit(er1, fimgs):7.2f} ms/img")
