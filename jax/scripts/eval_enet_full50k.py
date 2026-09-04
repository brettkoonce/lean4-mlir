#!/usr/bin/env python3
"""Canonical full-50,000-image ImageNet val for the trained EfficientNet-B0.

Differs from the params-only pattern in two ways B0's recipe forces, both of
which are silent if you get them wrong — the script still runs and still prints
a plausible number:

  1. running BN (gap A): the eval forward needs the trained running mean/var, so
     this loads the FULL-state checkpoint (<base>.state.npz = params + RMSProp
     sq/buf + EMA weights + EMA BN + BN buffers + step), NOT the params-only
     <base>.bin. The .bin holds EMA *parameters* and no BN statistics at all;
     evaluating it alone runs the net against fresh (mean 0, var 1) buffers.
  2. EMA on BOTH: B0 reports the EMA shadow, and the shadow covers the BN
     buffers as well as the weights. Eval therefore pairs `ema_params` with
     `ema_bn`. Pairing averaged weights with the *live* BN statistics is the
     mismatch that sent validation loss to 1e8 before `ema_bn` existed, so the
     pairing here is the one the trainer's own in-loop eval uses.

The in-training eval uses .batch(drop_remainder=True) (drops the last partial
batch); this counts all 50,000 with drop_remainder=False.

Reuses the generated trainer's forward / preprocess / init / load_train_state by
importing it as a module, so numerics match the run exactly.

  CKPT=/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet.state.npz \
    /home/skoonce/.venv-cuda/bin/python3 jax/scripts/eval_enet_full50k.py
"""
import os, importlib.util
import numpy as np
import jax, jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

# The artifact that TRAINED the checkpoint. `_full` and the 80-epoch variant
# differ only in the baked EPOCHS constant, but point this at the wrong net and
# the checkpoint is being run in a graph it was never trained in — see
# planning/imagenet_rerun_sweep.md C6.
GEN   = os.environ.get("GEN", ".lake/build/generated_efficientnet_b0_imagenet_full.py")
CKPT  = os.environ.get("CKPT", "/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet.state.npz")
BATCH = int(os.environ.get("BATCH", "200"))   # 50000 % 200 == 0 (not relied on)

spec = importlib.util.spec_from_file_location("genenet", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(f"backend={jax.default_backend()} devices={len(jax.devices())}")

assert CKPT.endswith(".state.npz"), (
    f"running-BN + EMA eval needs the full-state .state.npz (it carries the BN "
    f"buffers and the EMA shadows); the .bin has neither. Got {CKPT}")
print(f"loading full train state {CKPT}")
# Template matching the tuple the trainer saves, in order:
#   (params, opt_state, ema_params, ema_bn, bn_state)
# RMSProp opt_state = (sq, buf); TF form inits sq to 1.0, buf to 0. Only the
# tree STRUCTURE matters here — load_train_state overwrites every leaf.
_p    = m.init_params(jax.random.PRNGKey(0))
_opt  = (jax.tree.map(jnp.ones_like, _p), jax.tree.map(jnp.zeros_like, _p))
_bn   = m.init_bn_state()
state, step = m.load_train_state(CKPT, (_p, _opt, _p, _bn, _bn))
params, _opt_state, ema_params, ema_bn, bn_state = state
print(f"loaded step={step}; eval uses the EMA weights paired with the EMA BN buffers")
ema_params = jax.device_put(ema_params, m.replicated_sharding)
params     = jax.device_put(params,     m.replicated_sharding)

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

@jax.jit
def _score(pr, bn, x, y):
    logits, _ = m.forward(pr, x, bn, False)       # running-BN eval, drop-path off
    top5 = jax.lax.top_k(logits, 5)[1]
    return (jnp.sum(jnp.argmax(logits, axis=-1) == y),
            jnp.sum(jnp.any(top5 == y[:, None], axis=-1)))

for tag, pr, bs in (("EMA (reported)", ema_params, ema_bn),
                    ("raw weights   ", params,     bn_state)):
    c1 = c5 = total = 0
    for x, y in tfds.as_numpy(ds):
        a, b = _score(pr, bs, jnp.asarray(x), jnp.asarray(y))
        c1 += int(a); c5 += int(b); total += int(y.shape[0])
    print(f"{tag}  top-1 {c1}/{total} = {c1/total:.4f}   top-5 {c5}/{total} = {c5/total:.4f}")

print("(EfficientNet-B0 reference: 77.1% top-1 / 93.3% top-5)")
