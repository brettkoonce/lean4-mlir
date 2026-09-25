#!/usr/bin/env python3
"""Score BOTH arms of a ConvNeXt run — EMA and raw weights — over all 50,000.

Sibling of eval_convnext_full50k.py, which loads the `<base>.bin` (that file IS
`ema_params`) and is correct as written for the reported number. This one opens
the full-state `<base>.state.npz` instead, because the RAW weights exist only
there: the trainer saves `(params, opt_state, ema_params)` and writes only the
EMA to `.bin`. On EfficientNet-B0 the EMA was worth +0.82 points, which is the
kind of thing worth knowing for a net whose recipe turns EMA on.

ConvNeXt is a LayerNorm net, so unlike the B0/R34 scripts there are NO BN
buffers to pair with the weights — `forward(params, x, drop_key=None)` takes
params alone, and an arm is just its parameter tree.

The in-training eval already covers all 50,000 here (the generated pipeline
batches validation with `drop_remainder=training`, i.e. False), so this is a
confirmation of the EMA number and a first measurement of the raw one.

  GEN=.lake/build/generated_convnext_tiny_imagenet_full.py \
  CKPT=/home/skoonce/convnext_t300_3060/convnext_tiny_imagenet.state.npz \
    /home/skoonce/.venv-cuda/bin/python3 jax/scripts/eval_convnext_arms_full50k.py

⚠ Point GEN at the artifact that TRAINED the checkpoint.
"""
import os, importlib.util
import jax, jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

GEN   = os.environ.get("GEN", ".lake/build/generated_convnext_tiny_imagenet_full.py")
CKPT  = os.environ.get("CKPT", "/home/skoonce/convnext_t300_3060/convnext_tiny_imagenet.state.npz")
BATCH = int(os.environ.get("BATCH", "250"))

assert CKPT.endswith(".state.npz"), (
    f"the raw arm lives only in the full-state .state.npz; {CKPT} looks like the "
    f"EMA-only .bin — use eval_convnext_full50k.py for that")

spec = importlib.util.spec_from_file_location("gen", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(f"backend={jax.default_backend()} devices={len(jax.devices())}")

# Only the tree STRUCTURE matters — load_train_state overwrites every leaf.
# AdamW opt_state is (m, v, count); see the trainer's opt_state init.
_p   = m.init_params(jax.random.PRNGKey(0))
_opt = (jax.tree.map(jnp.zeros_like, _p), jax.tree.map(jnp.zeros_like, _p), jnp.float32(0))
state, step = m.load_train_state(CKPT, (_p, _opt, _p))
params, _opt_state, ema_params = state
print(f"loaded {CKPT}  step={step}")

params     = jax.device_put(params,     m.replicated_sharding)
ema_params = jax.device_put(ema_params, m.replicated_sharding)

# Full-val iterator WITHOUT drop_remainder, reusing the module's
# center-crop + normalize + CHW-flatten preprocessing exactly — identical to
# eval_convnext_full50k.py, so the two scripts' EMA numbers are comparable.
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
ds = ds.batch(BATCH, drop_remainder=False)      # keep ALL 50k
ds = ds.prefetch(tf.data.AUTOTUNE)

# Per-image, not summed on device, so `DUMP_CORRECT` can write the paired bitmap. The totals are
# the same integers either way.
@jax.jit
def _score(pr, x, y):
    logits = m.forward(pr, x)                  # forward-only: drop_key=None
    _, top5 = jax.lax.top_k(logits, 5)
    return (jnp.argmax(logits, axis=-1) == y,
            jnp.any(top5 == y[:, None], axis=-1))

# ▶ `DUMP_CORRECT=<prefix>` writes `<prefix>_{ema,raw}.bin`: one byte per val image, 1 = top-1
# correct, in tfds `validation` FILE order. That is `LEAN_MLIR_DUMP_CORRECT`'s format, so
# `scripts/demos/mcnemar.py` pairs a reference arm against a verified checkpoint directly. The order
# matches the verified side's single-producer val drain: neither shuffles, and tf.data's map keeps
# order. It also writes `<prefix>_labels.bin` (int32 LE, same order), so an alignment claim can be
# CHECKED, not assumed. Unset ⇒ nothing is written and the output is unchanged.
DUMP = os.environ.get("DUMP_CORRECT")
import numpy as np
for tag, key, pr in (("EMA (reported)", "ema", ema_params), ("raw weights   ", "raw", params)):
    c1 = c5 = total = 0
    bits, labels = [], []
    for x, y in tfds.as_numpy(ds):
        a, b = _score(pr, jnp.asarray(x), jnp.asarray(y))
        a = np.asarray(a); b = np.asarray(b)
        c1 += int(a.sum()); c5 += int(b.sum()); total += int(y.shape[0])
        if DUMP:
            bits.append(a.astype(np.uint8)); labels.append(np.asarray(y, dtype=np.int32))
    print(f"{tag}  top-1 {c1}/{total} = {c1/total:.4f}   top-5 {c5}/{total} = {c5/total:.4f}")
    if DUMP:
        np.concatenate(bits).tofile(f"{DUMP}_{key}.bin")
        np.concatenate(labels).astype("<i4").tofile(f"{DUMP}_labels.bin")
        print(f"    per-image top-1 bitmap -> {DUMP}_{key}.bin ({total} bytes)")
