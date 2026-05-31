#!/usr/bin/env python3
"""Canonical full-50,000-image ImageNet val for the trained R34 bf16 model.

The in-training eval uses tfds .batch(drop_remainder=True), which drops the
last partial batch (50000 % batch_size images) — so its top-1 is over 49,920,
not 50,000. This script evaluates ALL 50,000 with drop_remainder=False,
counting every image exactly once (no padding games: we just sum correct
predictions over real images), to get the official number.

Reuses the generated trainer's own forward/eval/param-load/preprocess code by
importing it as a module, so numerics match the run exactly.
"""
import os, sys, importlib.util
import numpy as np
import jax, jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

GEN = ".lake/build/generated_resnet34_imagenet.py"
CKPT = os.environ.get("CKPT", "/home/skoonce/r34_imagenet_bf16.bin")
BATCH = int(os.environ.get("BATCH", "250"))   # 50000 % 250 == 0, but we don't rely on it

# Import the generated module (defines forward, eval_batch, init_params_from_file,
# the preprocess helpers, DT/CONV_DT, etc.). It guards train under __main__.
spec = importlib.util.spec_from_file_location("genr34", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print(f"backend={jax.default_backend()} devices={len(jax.devices())}")
print(f"loading {CKPT}")
params = m.init_params_from_file(CKPT)
params = jax.device_put(params, m.replicated_sharding)

# Build a full-val iterator WITHOUT drop_remainder, reusing the module's
# center-crop + normalize + CHW-flatten preprocessing exactly.
ds = tfds.load('imagenet2012', split='validation',
               decoders={'image': tfds.decode.SkipDecoding()},
               data_dir=os.environ.get('TFDS_DATA_DIR'))
def _pp(ex):
    img = m._imagenet_decode_center_crop(ex['image'])
    img = tf.cast(img, tf.float32)
    img = (img - m._MEAN_RGB) / m._STD_RGB
    img = tf.transpose(img, [2, 0, 1])
    img = tf.reshape(img, [3 * m._IMG_SIZE * m._IMG_SIZE])
    return img, ex['label']
ds = ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH, drop_remainder=False)        # <-- keep ALL 50k
ds = ds.prefetch(tf.data.AUTOTUNE)

c1 = c5 = total = 0
for x, y in tfds.as_numpy(ds):
    x = jax.device_put(jnp.asarray(x))
    y = jax.device_put(jnp.asarray(y))
    logits = m.forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    _, top5 = jax.lax.top_k(logits, 5)
    c1 += int(jnp.sum(preds == y))
    c5 += int(jnp.sum(jnp.any(top5 == y[:, None], axis=-1)))
    total += int(y.shape[0])

print(f"\n=== FULL VAL ({total} images) ===")
print(f"top-1: {c1}/{total} = {c1/total:.4f}")
print(f"top-5: {c5}/{total} = {c5/total:.4f}")
