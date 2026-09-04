#!/usr/bin/env python3
"""Canonical full-50,000-image ImageNet val for the trained R34 bf16 model.

R34 is a BatchNorm net evaluated on running statistics (gap A), so this loads
the FULL-state checkpoint (<base>.state.npz = params + SGD velocity + bn
buffers + step), NOT the params-only <base>.bin. The .bin carries no running
mean/var, and evaluating it alone silently runs the net against fresh
(mean 0, var 1) buffers — it does not fail, it just scores a different network.

R34 trains without EMA (its state tuple has no shadow), so the reported weights
are the live params — unlike B0 / ConvNeXt / ViT, whose .bin IS the EMA.

The in-training eval uses .batch(drop_remainder=True) (drops the last partial
batch); this counts all 50,000 with drop_remainder=False.

Reuses the generated trainer's own forward/preprocess/init/load_train_state by
importing it as a module, so numerics match the run exactly.

  CKPT=/home/skoonce/r34_imagenet_bf16.state.npz \
    /home/skoonce/.venv-cuda/bin/python3 jax/scripts/eval_r34_full50k.py
"""
import os, sys, importlib.util
import numpy as np
import jax, jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

GEN = ".lake/build/generated_resnet34_imagenet.py"
CKPT = os.environ.get("CKPT", "/home/skoonce/r34_imagenet_bf16.state.npz")
BATCH = int(os.environ.get("BATCH", "250"))   # 50000 % 250 == 0, but we don't rely on it

# Import the generated module (defines forward, eval_batch, init_params_from_file,
# the preprocess helpers, DT/CONV_DT, etc.). It guards train under __main__.
spec = importlib.util.spec_from_file_location("genr34", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print(f"backend={jax.default_backend()} devices={len(jax.devices())}")

assert CKPT.endswith(".state.npz"), (
    f"running-BN eval needs the full-state .state.npz (it carries the BN "
    f"buffers); the .bin has params only. Got {CKPT}")
print(f"loading full train state {CKPT}")
# Template matching the tuple the trainer saves, in order:
#   (params, velocity, bn_state)      -- SGD+momentum, no EMA shadow
# Only the tree STRUCTURE matters; load_train_state overwrites every leaf.
_p = m.init_params(jax.random.PRNGKey(0))
state, step = m.load_train_state(CKPT, (_p, jax.tree.map(jnp.zeros_like, _p), m.init_bn_state()))
params, _velocity, bn_state = state
print(f"loaded step={step}; eval uses the running-BN buffers from the checkpoint")
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

@jax.jit
def _score(pr, bn, x, y):
    logits, _ = m.forward(pr, x, bn, False)      # running-BN eval, drop-path off
    top5 = jax.lax.top_k(logits, 5)[1]
    return (jnp.sum(jnp.argmax(logits, axis=-1) == y),
            jnp.sum(jnp.any(top5 == y[:, None], axis=-1)))

c1 = c5 = total = 0
for x, y in tfds.as_numpy(ds):
    a, b = _score(params, bn_state, jnp.asarray(x), jnp.asarray(y))
    c1 += int(a); c5 += int(b); total += int(y.shape[0])

print(f"\n=== FULL VAL ({total} images) ===")
print(f"top-1: {c1}/{total} = {c1/total:.4f}")
print(f"top-5: {c5}/{total} = {c5/total:.4f}")
