#!/usr/bin/env python3
"""Per-image scoring of the ViT-Ti JAX e300 checkpoint: {EMA, live} x {bf16 matmuls (the trainer's
own eval), f32 matmuls (the verified path's eval precision)}. Writes one byte per val image (1 =
top-1 correct, tfds `validation` file order) plus the labels, for a McNemar pairing against the
verified run's bitmaps. Writes only under $OUT (RESULTS.md: bitmaps/jax_e300_*)."""
import os, sys, importlib.util
import numpy as np
import jax, jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

GEN  = os.environ["GEN"]
CKPT = os.environ["CKPT"]
OUT  = os.environ["OUT"]
BATCH = 250

spec = importlib.util.spec_from_file_location("gen", GEN)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(f"backend={jax.default_backend()} devices={len(jax.devices())} gen={GEN}")

_p = m.init_params(jax.random.PRNGKey(0))
_opt = (jax.tree.map(jnp.zeros_like, _p), jax.tree.map(jnp.zeros_like, _p), jnp.float32(0))
n_file = len([k for k in np.load(CKPT).files if k.startswith("l")])
n_tmpl = len(jax.tree.leaves((_p, _opt, _p)))
assert n_file == n_tmpl, f"layout mismatch: file {n_file} arrays, template {n_tmpl}"
(live, _o, ema), step = m.load_train_state(CKPT, (_p, _opt, _p))
print(f"loaded {CKPT} step={step} ({n_file} arrays ✓)")

ds = tfds.load('imagenet2012', split='validation', decoders={'image': tfds.decode.SkipDecoding()},
               data_dir=os.environ.get('TFDS_DATA_DIR'))
def _pp(ex):
    img = tf.cast(m._imagenet_decode_center_crop(ex['image']), tf.float32)
    img = tf.transpose((img - m._MEAN_RGB) / m._STD_RGB, [2, 0, 1])
    return tf.reshape(img, [3 * m._IMG_SIZE * m._IMG_SIZE]), ex['label']
ds = ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
batches = [(x, y) for x, y in tfds.as_numpy(ds)]
labels = np.concatenate([y for _, y in batches]).astype("<i4")
labels.tofile(f"{OUT}_labels.bin")
print(f"val: {labels.size} images, crop_pct {m._CROP_PCT:.4f}, {m._IMG_SIZE}px")

for dt_name, dt in (("bf16", jnp.bfloat16), ("f32", jnp.float32)):
    m.DT = dt          # `mm` reads the global at trace time
    @jax.jit
    def score(pr, x, y):
        lg = m.forward(pr, x)
        t = jnp.take_along_axis(lg, y[:, None], axis=1)
        return jnp.argmax(lg, -1) == y, jnp.sum(lg > t, axis=1) < 5
    for arm, pr in (("ema", ema), ("live", live)):
        b1, b5 = [], []
        for x, y in batches:
            a, b = score(pr, jnp.asarray(x), jnp.asarray(y))
            b1.append(np.asarray(a)); b5.append(np.asarray(b))
        b1 = np.concatenate(b1).astype(np.uint8); b5 = np.concatenate(b5)
        b1.tofile(f"{OUT}_{arm}_{dt_name}.bin")
        print(f"{arm:4s} {dt_name:4s}  top-1 {int(b1.sum())}/{b1.size} = {100*b1.mean():.3f}%   "
              f"top-5 {int(b5.sum())}/{b5.size} = {100*b5.mean():.3f}%", flush=True)
