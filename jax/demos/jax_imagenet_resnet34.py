#!/usr/bin/env python3
"""Phase-2 ResNet-34 on full ImageNet (1000-class) — multi-GPU JAX demo.

Validates that our existing JAX codegen patterns scale from 10-class
Imagenette to 1000-class ImageNet across all 6 GPUs on ares. Standalone:
does not generate Lean MLIR or run phase-3; once it works, the next step
is to wire `.imagenet` through jax/Jax/Codegen.lean so the trainer is
emitted from a NetSpec like every other model.

Architecture, optimizer, init, and conv_bn are taken verbatim from the
patterns in jax/Jax/Codegen.lean's emit_helpers. Differences vs the
codegen-emitted trainers:
  - 1000-class head (vs Imagenette's 10)
  - tfds streaming (vs in-RAM np.frombuffer of train.bin/val.bin)
  - eval uses *batch* BN stats (codegen-emitted variants don't have
    running stats either; phase-3 does)

Usage:
  TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets \\
  python3 jax/demos/jax_imagenet_resnet34.py --epochs 3 --batch 192
"""

import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# tf import has to happen before any jax allocation if you want to keep
# tf off the GPU; we explicitly disable tf's GPU use below.
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds


# ─────────────────────────────────────────────────────────────────────
# tfds input pipeline (mostly cribbed from flax/examples/imagenet)
# ─────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32).reshape(1, 3, 1, 1)
STD_RGB  = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32).reshape(1, 3, 1, 1)


def _decode_and_random_crop_then_flip(image_bytes):
  shape = tf.io.extract_jpeg_shape(image_bytes)
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      shape, bounding_boxes=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3./4, 4./3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      use_image_if_no_bounding_boxes=True)
  oy, ox, _ = tf.unstack(bbox_begin)
  th, tw, _ = tf.unstack(bbox_size)
  window = tf.stack([oy, ox, th, tw])
  img = tf.io.decode_and_crop_jpeg(image_bytes, window, channels=3)
  img = tf.image.resize([img], [IMAGE_SIZE, IMAGE_SIZE],
                        method=tf.image.ResizeMethod.BICUBIC)[0]
  img = tf.image.random_flip_left_right(img)
  return img


def _decode_and_center_crop(image_bytes):
  shape = tf.io.extract_jpeg_shape(image_bytes)
  h, w = shape[0], shape[1]
  padded = tf.cast(
      ((IMAGE_SIZE / (IMAGE_SIZE + CROP_PADDING)) *
       tf.cast(tf.minimum(h, w), tf.float32)), tf.int32)
  oy = (h - padded) // 2
  ox = (w - padded) // 2
  window = tf.stack([oy, ox, padded, padded])
  img = tf.io.decode_and_crop_jpeg(image_bytes, window, channels=3)
  img = tf.image.resize([img], [IMAGE_SIZE, IMAGE_SIZE],
                        method=tf.image.ResizeMethod.BICUBIC)[0]
  return img


def build_dataset(split, batch_size, training):
  ds = tfds.load('imagenet2012', split=split,
                 decoders={'image': tfds.decode.SkipDecoding()},
                 data_dir=os.environ.get('TFDS_DATA_DIR'))

  def _preprocess(example):
    img_bytes = example['image']
    if training:
      img = _decode_and_random_crop_then_flip(img_bytes)
    else:
      img = _decode_and_center_crop(img_bytes)
    img = tf.cast(img, tf.float32)               # 0-255
    img = tf.transpose(img, [2, 0, 1])           # HWC → CHW
    return img, example['label']

  if training:
    ds = ds.shuffle(8192, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat()
  ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return tfds.as_numpy(ds)


# ─────────────────────────────────────────────────────────────────────
# ResNet-34 model (cribbed from jax/Jax/Codegen.lean's residualBlock emit)
# ─────────────────────────────────────────────────────────────────────

def conv2d(x, w, b, padding='SAME', stride=(1,1)):
  x = jax.lax.conv_general_dilated(x, w, stride, padding,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
  return x + b.reshape(1, -1, 1, 1)


def conv_bn(x, w, gamma, beta, stride=(1,1), padding='SAME'):
  x = jax.lax.conv_general_dilated(x, w, stride, padding,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
  mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
  var  = jnp.var (x, axis=(0, 2, 3), keepdims=True)
  x = (x - mean) / jnp.sqrt(var + 1e-5)
  return x * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)


def basic_block(p, x, stride=(1,1)):
  """Two 3x3 convBn + skip. Stride==2 path downsamples + 1x1 projects skip."""
  (w1, g1, b1) = p[0]; (w2, g2, b2) = p[1]
  residual = x
  out = jax.nn.relu(conv_bn(x, w1, g1, b1, stride=stride))
  out = conv_bn(out, w2, g2, b2)
  if stride != (1,1) or len(p) > 2:
    (wp, gp, bp) = p[2]
    residual = conv_bn(residual, wp, gp, bp, stride=stride)
  return jax.nn.relu(out + residual)


def he_init(k, shape):
  # Conv weight (oc, ic, kh, kw): fan_in = ic*kh*kw
  fan_in = int(np.prod(shape[1:]))
  std = float(np.sqrt(2.0 / fan_in))
  return random.normal(k, shape, dtype=jnp.float32) * std


def init_resnet34(key):
  ps = []
  k, key = random.split(key); ps.append(make_convbn(k,  64,    3, 7))
  # block configs: (ic, oc, n_blocks, first_stride)
  for ic, oc, n, fs in [(64, 64, 3, 1), (64, 128, 4, 2),
                        (128, 256, 6, 2), (256, 512, 3, 2)]:
    for i in range(n):
      in_c = ic if i == 0 else oc
      stride = fs if i == 0 else 1
      blk = []
      k, key = random.split(key); blk.append(make_convbn(k, oc, in_c, 3))
      k, key = random.split(key); blk.append(make_convbn(k, oc, oc,   3))
      if stride != 1 or in_c != oc:
        k, key = random.split(key); blk.append(make_convbn(k, oc, in_c, 1))
      ps.append(blk)
  # Dense 512 → 1000
  k, key = random.split(key)
  w = he_init(k, (1000, 512))
  ps.append((w, jnp.zeros(1000)))
  return ps


def make_convbn(k, oc, ic, ks):
  return (he_init(k, (oc, ic, ks, ks)), jnp.ones(oc), jnp.zeros(oc))


def forward(params, x):
  # x: (B, 3, 224, 224), uint8-as-float32 in [0, 255]
  x = x / 255.0
  x = (x - MEAN_RGB) / STD_RGB
  # Stem: conv-bn 3→64, 7x7 stride 2 → maxPool 3x3 stride 2
  (w, g, b) = params[0]
  x = jax.nn.relu(conv_bn(x, w, g, b, stride=(2,2)))
  x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max,
                            (1, 1, 3, 3), (1, 1, 2, 2), 'SAME')
  # 4 residual stages
  idx = 1
  for ic, oc, n, fs in [(64, 64, 3, 1), (64, 128, 4, 2),
                        (128, 256, 6, 2), (256, 512, 3, 2)]:
    for i in range(n):
      stride = (fs, fs) if i == 0 else (1, 1)
      x = basic_block(params[idx], x, stride=stride)
      idx += 1
  # GAP + dense
  x = jnp.mean(x, axis=(2, 3))               # (B, 512)
  (wd, bd) = params[idx]
  return x @ wd.T + bd                       # (B, 1000)


def loss_fn(params, x, y, label_smoothing=0.1):
  logits = forward(params, x)
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  oh = jax.nn.one_hot(y, 1000)
  smooth = (1 - label_smoothing) * oh + label_smoothing / 1000
  return -jnp.mean(jnp.sum(smooth * log_probs, axis=-1))


def accuracy(params, x, y):
  logits = forward(params, x)
  return jnp.mean(jnp.argmax(logits, axis=-1) == y)


# ─────────────────────────────────────────────────────────────────────
# Optimizer (Adam with cosine + warmup, mirrors phase-2/3 codegen)
# ─────────────────────────────────────────────────────────────────────

def adam_step(params, grads, opt_state, lr, wd=1e-4, b1=0.9, b2=0.999, eps=1e-8):
  m, v, step = opt_state
  step = step + 1
  m = jax.tree.map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
  v = jax.tree.map(lambda vi, gi: b2 * vi + (1 - b2) * (gi * gi), v, grads)
  m_hat = jax.tree.map(lambda mi: mi / (1 - b1 ** step), m)
  v_hat = jax.tree.map(lambda vi: vi / (1 - b2 ** step), v)
  new_params = jax.tree.map(
      lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + wd * p),
      params, m_hat, v_hat)
  return new_params, (m, v, step)


def lr_at(step, base_lr, warmup_steps, total_steps):
  warm = base_lr * step / jnp.maximum(warmup_steps, 1)
  prog = (step - warmup_steps) / jnp.maximum(total_steps - warmup_steps, 1)
  cos  = base_lr * 0.5 * (1 + jnp.cos(jnp.pi * jnp.clip(prog, 0.0, 1.0)))
  return jnp.where(step < warmup_steps, warm, cos)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--epochs', type=int, default=3)
  ap.add_argument('--batch',  type=int, default=192)  # 32 per device on 6 GPUs
  ap.add_argument('--lr',     type=float, default=0.1)
  ap.add_argument('--warmup-epochs', type=int, default=1)
  ap.add_argument('--eval-batch', type=int, default=192)
  ap.add_argument('--log-every', type=int, default=100)
  args = ap.parse_args()

  devices = jax.devices()
  n = len(devices)
  print(f'Devices: {devices}')
  mesh = Mesh(np.array(devices), axis_names=('batch',))
  data_shard = NamedSharding(mesh, P('batch'))
  repl_shard = NamedSharding(mesh, P())

  assert args.batch % n == 0, f'batch {args.batch} must divide n_devices {n}'

  print(f'Building tfds pipelines (batch={args.batch}) ...')
  train_iter = iter(build_dataset('train', args.batch, training=True))
  val_iter_factory = lambda: build_dataset('validation', args.eval_batch, training=False)

  steps_per_epoch = 1281167 // args.batch
  total_steps = steps_per_epoch * args.epochs
  warmup_steps = steps_per_epoch * args.warmup_epochs

  print(f'steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  '
        f'warmup_steps={warmup_steps}')

  key = random.PRNGKey(0)
  params = init_resnet34(key)
  params = jax.device_put(params, repl_shard)
  opt_state = (jax.tree.map(jnp.zeros_like, params),
               jax.tree.map(jnp.zeros_like, params),
               jnp.float32(0))
  opt_state = jax.device_put(opt_state, repl_shard)

  @jit
  def train_step(params, opt_state, x, y, lr):
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    params, opt_state = adam_step(params, grads, opt_state, lr)
    return params, opt_state, loss

  @jit
  def eval_step(params, x, y):
    return accuracy(params, x, y)

  t0 = time.time()
  step = 0
  for epoch in range(args.epochs):
    epoch_loss_sum = 0.0
    epoch_steps = 0
    ep_t0 = time.time()
    for _ in range(steps_per_epoch):
      x, y = next(train_iter)
      x = jax.device_put(x, data_shard)
      y = jax.device_put(y, data_shard)
      cur_lr = float(lr_at(step, args.lr, warmup_steps, total_steps))
      params, opt_state, loss = train_step(params, opt_state, x, y, cur_lr)
      epoch_loss_sum += float(loss)
      epoch_steps += 1
      step += 1
      if step % args.log_every == 0:
        print(f'  step {step}/{total_steps}  loss={float(loss):.4f}  '
              f'lr={cur_lr:.5f}  ({(time.time()-t0)/step*1000:.0f}ms/step avg)')

    print(f'Epoch {epoch+1}/{args.epochs}: '
          f'avg loss={epoch_loss_sum/max(1,epoch_steps):.4f}  '
          f'wall={time.time()-ep_t0:.1f}s')

    # Validation pass (50K / eval_batch ≈ 260 steps)
    print(f'  Running validation ...')
    correct = 0
    total = 0
    ev_t0 = time.time()
    for x, y in val_iter_factory():
      x = jax.device_put(x, data_shard)
      y = jax.device_put(y, data_shard)
      acc_b = float(eval_step(params, x, y))
      correct += acc_b * y.shape[0]
      total += y.shape[0]
    print(f'  val top-1: {correct:.0f}/{total} = {correct/total:.4f}  '
          f'({time.time()-ev_t0:.1f}s)')

  print(f'Total wall: {time.time()-t0:.1f}s')


if __name__ == '__main__':
  main()
