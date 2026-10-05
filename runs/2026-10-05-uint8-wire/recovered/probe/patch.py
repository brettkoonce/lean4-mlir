# Perf-only variants of the ViT-S JAX trainer: uint8 wire (+ optional PIL Rotate). NOT recipe-exact:
# erasing is done in pixel space (noise*std+mean, rounded to uint8).
import sys
src_path, out_path, arms = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
s = open(src_path).read()
def sub(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
old_tail = """        img = tf.cast(img, tf.float32)              # 0..255 HWC
        img = (img - _MEAN_RGB) / _STD_RGB          # normalize, still HWC
        if training and augment:
            img = _random_erase(img)
        img = tf.transpose(img, [2, 0, 1])          # HWC -> CHW
        img = tf.reshape(img, [3 * _IMG_SIZE * _IMG_SIZE])  # flat to match forward()
"""
if "u8" in arms:
    sub(old_tail, """        if training and augment:
            img = _random_erase_px(tf.cast(img, tf.float32))
            img = tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)   # uint8 HWC wire
        else:
""" + "".join("    " + l + "\n" for l in old_tail.rstrip("\n").split("\n") if "_random_erase" not in l and "if training and augment" not in l))
    a = s.index("def _random_erase(img):"); b = s.index("def build_imagenet_iter")
    er = s[a:b].replace("def _random_erase(img):", "def _random_erase_px(img):")
    assert er.count("tf.random.normal(tf.shape(img))") == 1
    er = er.replace("tf.random.normal(tf.shape(img))", "(tf.random.normal(tf.shape(img)) * _STD_RGB + _MEAN_RGB)")
    s = s[:b] + er + s[b:]
    sub("            x, y = next(train_iter)\n", "            x, y = next(train_iter)\n            x = _prep_u8(x)\n")
    sub("@jit\ndef eval_batch(params, x, y, take):", """_U8_MEAN = jnp.array([0.485 * 255, 0.456 * 255, 0.406 * 255], jnp.float32)
_U8_STD = jnp.array([0.229 * 255, 0.224 * 255, 0.225 * 255], jnp.float32)
@jit
def _prep_u8(x):
    x = (x.astype(jnp.float32) - _U8_MEAN) / _U8_STD
    return jnp.transpose(x, (0, 3, 1, 2)).reshape(x.shape[0], -1)

@jit
def eval_batch(params, x, y, take):""")
if "pilrot" in arms:
    sub("_RA_INC = True", """from PIL import Image as _PILImage
def _pil_rot(x, d):
    return np.asarray(_PILImage.fromarray(x).rotate(float(d), resample=_PILImage.BICUBIC, fillcolor=(128, 128, 128)))
def _aa_rotate_pil(img, deg):
    out = tf.numpy_function(_pil_rot, [img, tf.cast(deg, tf.float32)], tf.uint8, stateful=False)
    out.set_shape(img.shape); return out
_AA_OPS['Rotate'] = (_aa_rotate_pil, _aa_rot, True)
_RA_INC = True""")
open(out_path, "w").write(s)
print("wrote", out_path, sorted(arms))
