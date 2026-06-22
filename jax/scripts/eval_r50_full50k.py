#!/usr/bin/env python3
"""Canonical full-50,000-image ImageNet val for the trained RSB-A2 ResNet-50.

Differs from eval_r34_full50k.py in two ways the RSB-A2 recipe forces:
  1. running BN (gap A): the eval forward needs the trained running mean/var, so
     we load the FULL-state checkpoint (<base>.state.npz = params + LAMB m/v/t +
     EMA shadow + bn buffers + step), NOT the params-only <base>.bin. The .bin
     lacks BN stats and would eval against fresh (mean 0, var 1) buffers.
  2. EMA: the eval uses the EMA shadow weights (what RSB-A2 reports), not the
     live params.

The in-training eval uses .batch(drop_remainder=True) (drops the last partial
batch); this counts all 50,000 with drop_remainder=False.

Reuses the generated trainer's forward / preprocess / init / load_train_state by
importing it as a module, so numerics match the run exactly.

  CKPT=/home/skoonce/r50_rsb_a2_imagenet.state.npz \
    /home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python3 \
    jax/scripts/eval_r50_full50k.py
"""
import os, importlib.util
import numpy as np
import jax, jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

GEN  = ".lake/build/generated_resnet50_imagenet.py"
CKPT = os.environ.get("CKPT", "/home/skoonce/r50_rsb_a2_imagenet.state.npz")
BATCH = int(os.environ.get("BATCH", "200"))   # 50000 % 200 == 0 (not relied on)

spec = importlib.util.spec_from_file_location("genr50", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(f"backend={jax.default_backend()} devices={len(jax.devices())}")

assert CKPT.endswith(".state.npz"), \
    f"runningBN eval needs the full-state .state.npz (has BN buffers); got {CKPT}"
print(f"loading full train state {CKPT}")
# Rebuild a template matching the saved state tuple, in the exact order the
# trainer saves it: (params, opt_state, ema_params, bn_state).  LAMB opt_state =
# (m, v, t); EMA on; runningBN on  -> see MainResnet50Imagenet.lean / Codegen.
_p   = m.init_params(jax.random.PRNGKey(0))
_opt = (jax.tree.map(jnp.zeros_like, _p), jax.tree.map(jnp.zeros_like, _p), jnp.float32(0))
template = (_p, _opt, jax.tree.map(lambda a: a, _p), m.init_bn_state())
state, step = m.load_train_state(CKPT, template)
params, _opt_state, ema_params, bn_state = state
print(f"loaded step={step}; eval uses EMA shadow + running-BN buffers")
ema_params = jax.device_put(ema_params, m.replicated_sharding)

# Full-val iterator WITHOUT drop_remainder, reusing the module's center-crop +
# normalize + CHW-flatten preprocessing exactly.
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
ds = ds.batch(BATCH, drop_remainder=False)        # keep ALL 50k
ds = ds.prefetch(tf.data.AUTOTUNE)

c1 = c5 = total = 0
for x, y in tfds.as_numpy(ds):
    x = jax.device_put(jnp.asarray(x))
    y = jax.device_put(jnp.asarray(y))
    logits, _ = m.forward(ema_params, x, bn_state, False)   # running-BN eval, SD off
    preds = jnp.argmax(logits, axis=-1)
    _, top5 = jax.lax.top_k(logits, 5)
    c1 += int(jnp.sum(preds == y))
    c5 += int(jnp.sum(jnp.any(top5 == y[:, None], axis=-1)))
    total += int(y.shape[0])

print(f"\n=== FULL VAL ({total} images) ===")
print(f"top-1: {c1}/{total} = {c1/total:.4f}")
print(f"top-5: {c5}/{total} = {c5/total:.4f}")
print("(RSB-A2 reference: 79.8% top-1)")
