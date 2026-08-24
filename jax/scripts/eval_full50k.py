#!/usr/bin/env python3
"""eval_full50k.py — score ANY ImageNet checkpoint over the full 50,000 val images.

The net-agnostic peer of the six `eval_<net>_full50k.py` scripts. Those were copies of one
another (three still carry ResNet-34's docstring), so each emitter change had to be applied
six times and in practice was applied to one: `eval_mnv2_full50k.py` learned the `bn`/
`training` args when running-BN landed and `eval_enet_full50k.py` did not, and none of them
learned the `take` argument that `eval_batch` grew with the partial-batch fix (`ccca380`).
This is the one writer.

    GEN=.lake/build/generated_<net>.py CKPT=<path>.state.npz ../.venv/bin/python \
        scripts/eval_full50k.py

## Why this scores 50,000 and the training log does not

In-training eval batched the val split with `drop_remainder=True` and drained a hardcoded
195 batches, so 195x256 = 49,920 images were scored and 80 were silently dropped — on every
ImageNet number this repo has quoted. `ccca380` fixed the trainer; this script never had the
bug, because it batches with `drop_remainder=False` and `eval_batch`'s `take` masks the short
final batch. Both now agree with timm's `validate.py` denominator.

## ⚠ The state layout is READ from the module, not guessed

`save_train_state` flattens a tuple to `l0..l{n-1}` and `load_train_state` rebuilds it from a
template, so a template with the wrong SHAPE silently mis-assigns arrays — the parameters of
one slot land in another and the result is a plausible-looking percentage off garbage. The
tuple differs per net (`(params, opt_state, bn_state)` for R50/MNv2, and
`(params, opt_state, ema_params, ema_bn, bn_state)` for EfficientNet-B0), and it has changed
over time, which is how `eval_r50_full50k.py` came to carry an EMA slot that RSB-A3 — which
sets `useEMA := false` — never wrote.

So the layout is parsed out of the generated module's own `save_train_state` call, the
optimizer's leaf count is DERIVED from the file rather than assumed, and the total is checked
against the array count in the `.npz`. A mismatch means the checkpoint predates the module;
this refuses rather than falling back, because the fallback scores garbage.
"""
import os, re, sys, importlib.util
import numpy as np
import jax, jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

GEN    = os.environ["GEN"]
CKPT   = os.environ["CKPT"]
# The BN running stats live in the `.state.npz`, never in the params-only `.bin`. Default to
# the sibling state file so passing either path does the right thing.
STATE  = os.environ.get("STATE") or (CKPT[:-4] + ".state.npz" if CKPT.endswith(".bin") else CKPT)
BATCH  = int(os.environ.get("BATCH", "250"))
REGION = os.environ.get("REGION", "auto")     # auto | live | ema
LABEL  = os.environ.get("LABEL", os.path.basename(CKPT))

spec = importlib.util.spec_from_file_location("gen", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print(f"=== {LABEL} ===")
print(f"backend={jax.default_backend()} devices={len(jax.devices())}")
# The batch is sharded over the device mesh, so it has to divide evenly. Refusing here beats
# the shape error 200 lines later, and beats silently dropping to one device.
if BATCH % len(jax.devices()):
    sys.exit(f"BATCH={BATCH} is not divisible by {len(jax.devices())} devices — "
             f"pick a multiple, or pin devices with CUDA_VISIBLE_DEVICES")
print(f"gen={GEN}")
print(f"state={STATE}")
# `_CROP_PCT` was introduced by the timm-protocol change (`f8cd3a9`); a module without it is a
# pre-change one, which is exactly the case the equality gate wants to run.
_crop = getattr(m, "_CROP_PCT", None)
print(f"eval protocol: {m._IMG_SIZE}px, "
      + (f"crop_pct={_crop:.6f} (timm resize-then-crop, antialiased)" if _crop is not None
         else "PRE-f8cd3a9 module — crop-then-resize, aliased"))

assert STATE.endswith(".state.npz"), f"running-BN eval needs the full-state .state.npz; got {STATE}"

# ── the state layout, read out of the module that wrote it ────────────────────────────────
src = open(GEN).read()
mo = re.search(r"save_train_state\(f'\{_ckpt_base\}\.state\.npz',\s*\((.*?)\),", src)
if not mo:
    sys.exit(f"could not find the save_train_state call in {GEN} — cannot determine the layout")
names = [s.strip() for s in mo.group(1).split(",")]
print(f"state tuple (from the module): ({', '.join(names)})")

_p  = m.init_params(jax.random.PRNGKey(0))
_bn = m.init_bn_state()
P, B = len(jax.tree.leaves(_p)), len(jax.tree.leaves(_bn))
N = len([k for k in np.load(STATE).files if k.startswith("l")])

PARAM_SLOTS = {"params", "ema_params"}
BN_SLOTS    = {"bn_state", "ema_bn"}
n_p = sum(1 for k in names if k in PARAM_SLOTS)
n_b = sum(1 for k in names if k in BN_SLOTS)
if n_p + n_b + 1 != len(names):
    sys.exit(f"unrecognised slot in {names} (known: {PARAM_SLOTS | BN_SLOTS} + opt_state)")

# The optimizer is the only slot whose leaf count is not fixed by the net, so DERIVE it from
# what is actually in the file. LAMB/Adam carry a scalar step counter alongside their moments,
# which is why this is `k*P` or `k*P + 1` rather than a multiple of P.
opt_leaves = N - n_p * P - n_b * B
_z = lambda: jax.tree.map(jnp.zeros_like, _p)
if   opt_leaves % P == 0:            opt = tuple(_z() for _ in range(opt_leaves // P))
elif (opt_leaves - 1) % P == 0:      opt = tuple(_z() for _ in range((opt_leaves - 1) // P)) + (jnp.float32(0),)
else:
    sys.exit(f"cannot factor the optimizer state: {N} arrays in the file, params={P} bn={B}, "
             f"layout {names} leaves {opt_leaves} for opt_state, which is neither k*{P} nor k*{P}+1.\n"
             f"⚠ the checkpoint was written by a different module version than {GEN}")
print(f"  params={P} bn={B} opt={opt_leaves} leaves -> {N} total (file has {N}) ✓")

slot = {"params": _p, "ema_params": _p, "bn_state": _bn, "ema_bn": _bn, "opt_state": opt}
template = tuple(slot[k] for k in names)
assert len(jax.tree.leaves(template)) == N, "template/file leaf-count mismatch"

state, step = m.load_train_state(STATE, template)
got = dict(zip(names, state))
print(f"loaded step={step}")

# `auto` mirrors what the trainer's own eval does: a net with an EMA shadow is scored on the
# shadow (and on the EMA-lagged BN buffers that pair with it), a net without one on the live
# weights. `live` forces the un-averaged weights, which is the useful comparison when an EMA
# run is suspected of the warmup bug.
use_ema = "ema_params" in got and REGION != "live"
if REGION == "ema" and "ema_params" not in got:
    sys.exit(f"REGION=ema but this checkpoint has no EMA slot: {names}")
params = got["ema_params"] if use_ema else got["params"]
bn     = got.get("ema_bn", got["bn_state"]) if use_ema else got["bn_state"]
print(f"scoring the {'EMA shadow' if use_ema else 'live weights'} + "
      f"{'EMA-lagged' if use_ema and 'ema_bn' in got else 'running'} BN buffers")

params = jax.device_put(params, m.replicated_sharding)
bn     = jax.device_put(bn, m.replicated_sharding)

# ── the val split, through the module's OWN preprocessing ─────────────────────────────────
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
ds = ds.batch(BATCH, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

# The graph is compiled for one batch width, so the short final batch is PADDED up to it and
# `take` masks the padding out of the counts — the same thing the fixed trainer drain does.
# The batch is sharded exactly as the trainer's own drain shards it, so `eval_batch` runs on
# every device rather than compiling for one and replicating the work.
#
# ⚠ `take` post-dates these checkpoints, and scoring one through a STALE module is the whole
# point of the equality gate below — so tolerate a 4-argument `eval_batch` rather than
# TypeError on it. Without `take` there is no way to mask padding, so a batch width that
# leaves a tail would silently score zeros: refuse instead.
import inspect
_HAS_TAKE = "take" in inspect.signature(m.eval_batch).parameters
if not _HAS_TAKE and 50000 % BATCH:
    sys.exit(f"{GEN} predates `take`, so the tail cannot be masked, and BATCH={BATCH} leaves "
             f"{50000 % BATCH} images over. Pick a BATCH dividing 50000 (e.g. 250 on 5 devices).")

c1 = c5 = total = 0
for x, y in tfds.as_numpy(ds):
    n = int(y.shape[0])
    if n < BATCH:
        x = np.concatenate([x, np.zeros((BATCH - n,) + x.shape[1:], x.dtype)])
        y = np.concatenate([y, np.zeros((BATCH - n,), y.dtype)])
    xs = jax.device_put(jnp.asarray(x), m.data_sharding)
    ys = jax.device_put(jnp.asarray(y), m.data_sharding)
    b1, b5, _loss = m.eval_batch(params, bn, xs, ys, n) if _HAS_TAKE \
               else m.eval_batch(params, bn, xs, ys)
    c1 += int(b1); c5 += int(b5); total += n

print(f"\n=== FULL VAL ({total} images) ===")
print(f"top-1: {c1}/{total} = {100.0 * c1 / total:.2f}%")
print(f"top-5: {c5}/{total} = {100.0 * c5 / total:.2f}%")
