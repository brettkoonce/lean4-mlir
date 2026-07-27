#!/usr/bin/env python3
"""A/B the ImageNet validation preprocessing against timm's protocol.

Our emitted `_imagenet_decode_center_crop` is the Google/TPU "Inception" eval:
crop a centred square of side `crop_pct * min(h,w)` straight out of the JPEG,
then `tf.image.resize(..., BICUBIC)` it to 224. timm instead resizes the WHOLE
image so its shorter side is `img_size / crop_pct` (aspect preserved, PIL
bicubic, which is antialiased) and only then centre-crops `img_size`.

The field of view is identical -- both end up with a centred square of side
crop_pct*min(h,w) -- so the difference is purely resampling:

  tf-current   crop -> tf.image.resize BICUBIC, antialias=False   (what we ship)
  tf-aa        crop -> tf.image.resize BICUBIC, antialias=True    (one-line fix)
  timm         resize shorter side (antialiased) -> centre crop   (timm's order)

Reuses the generated trainer's forward/param-load so numerics match the run.
No drop_remainder: all 50,000 images counted exactly once.

env: GEN=<generated trainer>  CKPT=<weights .bin>  MODE=tf-current|tf-aa|timm
     BATCH=250  LIMIT=<optional cap on images, for a quick look>
"""
import os, sys, importlib.util
import numpy as np
import jax, jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

GEN = os.environ["GEN"]
CKPT = os.environ["CKPT"]
MODE = os.environ.get("MODE", "tf-current")
BATCH = int(os.environ.get("BATCH", "250"))
LIMIT = int(os.environ.get("LIMIT", "0"))

spec = importlib.util.spec_from_file_location("gen", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

S = m._IMG_SIZE
CROP_PCT = S / (S + m._CROP_PADDING)          # 224/256 = 0.875, timm's default


def pp_tf(image_bytes, antialias):
    """Crop-then-resize (our current path). antialias toggles the only knob."""
    shape = tf.io.extract_jpeg_shape(image_bytes)
    h, w = shape[0], shape[1]
    side = tf.cast(CROP_PCT * tf.cast(tf.minimum(h, w), tf.float32), tf.int32)
    oy, ox = (h - side) // 2, (w - side) // 2
    img = tf.io.decode_and_crop_jpeg(image_bytes, tf.stack([oy, ox, side, side]), channels=3)
    return tf.image.resize([img], [S, S], method=tf.image.ResizeMethod.BICUBIC,
                           antialias=antialias)[0]


def pp_timm(image_bytes):
    """timm: resize shorter side to S/crop_pct (aspect kept, antialiased), then centre crop."""
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    target = tf.cast(tf.round(S / CROP_PCT), tf.int32)          # 256
    scale = tf.cast(target, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32)
    nh = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    nw = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, [nh, nw], method=tf.image.ResizeMethod.BICUBIC,
                          antialias=True)                        # PIL-like
    oy, ox = (nh - S) // 2, (nw - S) // 2
    return tf.image.crop_to_bounding_box(img, oy, ox, S, S)


params = m.init_params_from_file(CKPT)
params = jax.device_put(params, m.replicated_sharding)

ds = tfds.load("imagenet2012", split="validation",
               decoders={"image": tfds.decode.SkipDecoding()},
               data_dir=os.environ.get("TFDS_DATA_DIR"))

def _pp(ex):
    b = ex["image"]
    if MODE == "tf-current":
        img = pp_tf(b, antialias=False)
    elif MODE == "tf-aa":
        img = pp_tf(b, antialias=True)
    elif MODE == "timm":
        img = pp_timm(b)
    else:
        raise SystemExit(f"unknown MODE {MODE}")
    img = tf.cast(img, tf.float32)
    img = (img - m._MEAN_RGB) / m._STD_RGB
    img = tf.transpose(img, [2, 0, 1])
    return tf.reshape(img, [3 * S * S]), ex["label"]

ds = ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE)
if LIMIT:
    ds = ds.take(LIMIT)
ds = ds.batch(BATCH, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

c1 = c5 = total = 0
for x, y in tfds.as_numpy(ds):
    logits = m.forward(params, jax.device_put(jnp.asarray(x)))
    y = jnp.asarray(y)
    c1 += int(jnp.sum(jnp.argmax(logits, -1) == y))
    _, top5 = jax.lax.top_k(logits, 5)
    c5 += int(jnp.sum(jnp.any(top5 == y[:, None], axis=-1)))
    total += int(y.shape[0])

print(f"PREPROC {MODE}|{c1}|{c5}|{total}|{100.0*c1/total:.3f}|{100.0*c5/total:.3f}")
