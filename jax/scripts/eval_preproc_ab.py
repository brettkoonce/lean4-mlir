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

⚠⚠ **THE CROP RATIO IS READ OUT OF THE EMITTED FUNCTION, NOT RECOMPUTED (fixed 2026-08-14).**
This used to be `CROP_PCT = S / (S + m._CROP_PADDING)`, which is correct only for the nets whose
config leaves `testCropRatio = 0`. `Jax/Codegen.lean` emits TWO branches for the eval crop, and a
config that SETS `testCropRatio` takes the other one and never reads `_CROP_PADDING` at all --
which is RSB-A3's case: `MainResnet50Imagenet.lean:102` sets `0.95`, the generated trainer crops at
`0.950000`, and this script scored it at **0.875**. That is an 8% narrower field of view than the
run's own eval, i.e. a different measurement rather than an error bar, and on FixRes-style
train@160/eval@224 recipes object scale is exactly the axis the recipe is exploiting.
▶ ConvNeXt-T and ViT-Ti -- the two nets `recipe_fidelity_diffs.md` C1 measured -- take the
`_CROP_PADDING` branch, so C1's ~0.2 pt stands unaffected. R50 is the one it was wrong for, and
R50 is the net the §5.8 `[TODO, resize/eval reconciliation]` is about.

env: GEN=<generated trainer>  CKPT=<weights .bin>  MODE=tf-current|tf-aa|timm
     BATCH=250  LIMIT=<optional cap on images, for a quick look>
"""
import os, sys, re, inspect, importlib.util
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


def _crop_pct_of(mod):
    """The eval crop ratio THIS trainer uses, recovered by reading its own emitted function.

    Two emitted branches (`Jax/Codegen.lean`, `_imagenet_decode_center_crop`):
      testCropRatio > 0  ->  `padded = tf.cast((0.950000 * tf.cast(tf.minimum(h, w), ...`
      otherwise          ->  `padded = tf.cast(((_IMG_SIZE / (_IMG_SIZE + _CROP_PADDING)) * ...`
    Recovered by reading rather than by fitting, and the fallback is asserted to be the branch that
    is actually present -- so a third branch appearing later fails here instead of scoring quietly.
    """
    # ⭐ Current trainers emit `_CROP_PCT` as ONE module constant, whichever way it was configured,
    # so there is nothing to parse. The two branches below are for trainers generated before that.
    if hasattr(mod, "_CROP_PCT"):
        return float(mod._CROP_PCT), "_CROP_PCT (emitted constant)"
    src = inspect.getsource(mod._imagenet_decode_center_crop)
    lit = re.search(r"padded\s*=\s*tf\.cast\(\(\s*([0-9]*\.[0-9]+)\s*\*", src)
    if lit:
        return float(lit.group(1)), "testCropRatio (legacy inline literal)"
    if "_CROP_PADDING" in src:
        return S / (S + mod._CROP_PADDING), "_IMG_SIZE/(_IMG_SIZE+_CROP_PADDING) (legacy)"
    raise SystemExit(
        "could not recover the eval crop ratio from _imagenet_decode_center_crop -- it matches "
        "neither emitted branch. Read the function and extend `_crop_pct_of`; do NOT let this "
        "fall back to 0.875, which is what made this script silently wrong on RSB-A3.")


CROP_PCT, CROP_SRC = _crop_pct_of(m)
print(f"PREPROC-CONFIG img={S} crop_pct={CROP_PCT:.6f}  ({CROP_SRC})", file=sys.stderr)


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
    """timm: resize shorter side to S/crop_pct (aspect kept, antialiased), then centre crop.

    ⭐ **`antialias=True` IS PIL here, and that is now measured rather than assumed
    (2026-08-14).** This arm claims to be timm's protocol while calling TF, so the claim rests
    entirely on TF's antialiased bicubic agreeing with PIL's. Diffed across the downscale range
    ImageNet val actually spans (0.7x to 11.7x onto a 256 shorter side), it holds FLAT at mean
    |Δ| ≈ 0.30 of 255, max ≈ 1 -- the two are the same resampler to within a quantisation step.
    ⚠ The control is the `antialias=False` arm, which does NOT hold flat: it tracks PIL at 0.31
    when there is no downscale and diverges with the ratio (1.05 at 2.9x, 2.54 at 3.9x, 7.24 at
    5.9x, 13.73 at 11.7x). So the aliasing penalty this script measures is concentrated on the
    LARGE images in the val set, and a single scalar for it is an average over a skewed
    distribution rather than a per-image constant."""
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


# ── BATCH-NORM NETS ────────────────────────────────────────────────────────────────────────────
# ⚠⚠ **THIS SCRIPT COULD NOT RUN ON A BN NET AT ALL UNTIL 2026-08-14, AND THAT IS WHY ITS ONLY
# RECORDED MEASUREMENT (`recipe_fidelity_diffs.md` C1) IS ViT-Ti AND ConvNeXt-T.** The emitter
# gives a LayerNorm net `forward(params, x, drop_key=None)` and a BN net
# `forward(params, x, bn, training, drop_key=None)`; the call below passed two arguments, so R50,
# MobileNetV2, EfficientNet and MNv4 raised `TypeError` rather than scoring. ▶ It follows that
# blueprint §5.8's "port these weights into a standard PIL-resize eval and they fall to ~74.4%"
# was NOT produced by this script — nothing in this repo can produce an R50 number this way yet.
#
# ⭐ The running statistics are recoverable, unlike on the verified side. The JAX trainer's
# `.state.npz` is `jax.tree.leaves((params, opt_state, bn_state))`, so the BN buffers ARE
# checkpointed here — where the verified path's `.bin` is exactly `[θ|m|v]` and drops them
# (`next_session_verified_trainer_code.md` §2b). Same gap, opposite outcome: this side can
# re-score a finished run today; that side cannot until the format changes.
BN_STATE = None
if "bn" in inspect.signature(m.forward).parameters:
    npz = re.sub(r"\.bin$", "", CKPT) + ".state.npz"
    if not os.path.exists(npz):
        raise SystemExit(
            f"{GEN} is a BatchNorm net -- its forward takes the running statistics -- and they are\n"
            f"  NOT in {CKPT}, which holds parameters only. They live in the sibling train state:\n"
            f"    {npz}   (missing)\n"
            "  Scoring without them would normalise by whatever `init_bn_state()` returns, which is\n"
            "  not a slightly-off number. Point CKPT at a checkpoint whose .state.npz survives.")
    # bn_state is the LAST element of the saved tuple, so it is the last N leaves. Taken by
    # position because `load_train_state` needs an opt_state template this script has no way to
    # build -- and asserted against the template's own leaf count, so a change to the saved
    # tuple's shape fails here instead of silently reading optimizer moments as variances.
    tmpl = m.init_bn_state()
    leaves = jax.tree.leaves(tmpl)
    d = np.load(npz)
    total_leaves = sum(1 for k in d.files if k.startswith("l"))
    if total_leaves < len(leaves):
        raise SystemExit(f"{npz} has {total_leaves} leaves, fewer than bn_state's {len(leaves)}")
    off = total_leaves - len(leaves)
    got = [jnp.asarray(d[f"l{off + i}"]) for i in range(len(leaves))]
    for a, b in zip(got, leaves):
        if a.shape != jnp.asarray(b).shape:
            raise SystemExit(
                f"bn_state leaf shape mismatch reading {npz}: {a.shape} vs template {jnp.asarray(b).shape}.\n"
                "  The saved tuple is not (params, opt_state, bn_state) any more -- fix the slice.")
    BN_STATE = jax.device_put(jax.tree.unflatten(jax.tree.structure(tmpl), got),
                              m.replicated_sharding)
    print(f"PREPROC-CONFIG bn_state: {len(leaves)} arrays from {npz}", file=sys.stderr)


def _forward(x):
    """`training=False` — running statistics, never the eval batch's own (transductive).

    ⚠ A BN forward returns `(logits, bn_out)`, an LN one returns `logits`. Unwrapped the way the
    trainer's own `eval_batch` does (`logits, _ = forward(params, x, bn, False)`) rather than by
    testing the value, so this reads the same as the code it has to agree with."""
    if BN_STATE is None:
        return m.forward(params, x)
    logits, _ = m.forward(params, x, BN_STATE, False)
    return logits

ds = tfds.load("imagenet2012", split="validation",
               decoders={"image": tfds.decode.SkipDecoding()},
               data_dir=os.environ.get("TFDS_DATA_DIR"))

def _pp(ex):
    b = ex["image"]
    if MODE == "shipped":
        # ⭐ **THE TRAINER'S OWN FUNCTION, not a re-implementation of it.** Every other arm here
        # describes an alternative; this one has to BE the thing under test, or the script drifts
        # from the emitter and starts A/B-ing its own memory of it. That is not hypothetical: the
        # `tf-current` arm below was written as "what we ship" and stopped being so the moment the
        # emitter moved to timm's protocol, while its name went on claiming otherwise.
        # ▶ With the emitter now on timm's protocol, `shipped` and `timm` must agree EXACTLY. That
        # equality is the check that the conversion landed.
        img = m._imagenet_decode_center_crop(b)
    elif MODE == "tf-current":
        img = pp_tf(b, antialias=False)     # the LEGACY protocol (crop→resize, aliased)
    elif MODE == "tf-aa":
        img = pp_tf(b, antialias=True)      # legacy order, antialiased
    elif MODE == "timm":
        img = pp_timm(b)
    else:
        raise SystemExit(f"unknown MODE {MODE} — one of shipped | tf-current | tf-aa | timm")
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
    logits = _forward(jax.device_put(jnp.asarray(x)))
    y = jnp.asarray(y)
    c1 += int(jnp.sum(jnp.argmax(logits, -1) == y))
    _, top5 = jax.lax.top_k(logits, 5)
    c5 += int(jnp.sum(jnp.any(top5 == y[:, None], axis=-1)))
    total += int(y.shape[0])

print(f"PREPROC {MODE}|{c1}|{c5}|{total}|{100.0*c1/total:.3f}|{100.0*c5/total:.3f}")
