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

GEN = os.environ.get("GEN", ".lake/build/generated_mobilenet_v2_imagenet.py")
CKPT = os.environ.get("CKPT", "/home/skoonce/mnv2_imagenet_bf16.bin")
BATCH = int(os.environ.get("BATCH", "250"))   # 50000 % 250 == 0, but we don't rely on it
# Running-BN eval (gap A) needs the trained BN mean/var, which live in the
# companion `.state.npz`, NOT in the params-only `.bin`. Defaults to the sibling
# state file; set STATE= explicitly to override, or STATE=none to eval with
# freshly-initialised BN stats (wrong — only for debugging).
STATE = os.environ.get("STATE", CKPT[:-4] + ".state.npz" if CKPT.endswith(".bin") else "")

# Import the generated module (defines forward, eval_batch, init_params_from_file,
# the preprocess helpers, DT/CONV_DT, etc.). It guards train under __main__.
spec = importlib.util.spec_from_file_location("genr34", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print(f"backend={jax.default_backend()} devices={len(jax.devices())}")
print(f"loading {CKPT}")
params = m.init_params_from_file(CKPT)

# BN running stats. `forward`/`eval_batch` gained `bn`/`training` args when
# running-BN landed (gap A), so this script must supply them or it TypeErrors.
bn = m.init_bn_state()
if STATE and STATE != "none" and os.path.exists(STATE):
    print(f"loading BN running stats from {STATE}")
    opt_state = (jax.tree.map(jnp.ones_like, params), jax.tree.map(jnp.zeros_like, params))
    (params, _opt, bn), _step = m.load_train_state(STATE, (params, opt_state, bn))
    print(f"  state step={_step}")
else:
    print(f"WARNING: no state file at {STATE!r} — evaluating with FRESH BN stats "
          f"(this will be badly wrong for a running-BN net)")

params = jax.device_put(params, m.replicated_sharding)
bn = jax.device_put(bn, m.replicated_sharding)

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
    # Use the trainer's own eval_batch so numerics match the run exactly —
    # including its top-5 (it ranks the true class rather than calling
    # jax.lax.top_k, whose indices are broken on ROCm/gfx1100).
    b1, b5, _loss = m.eval_batch(params, bn, x, y)
    c1 += int(b1)
    c5 += int(b5)
    total += int(y.shape[0])

print(f"\n=== FULL VAL ({total} images) ===")
print(f"top-1: {c1}/{total} = {c1/total:.4f}")
print(f"top-5: {c5}/{total} = {c5/total:.4f}")
