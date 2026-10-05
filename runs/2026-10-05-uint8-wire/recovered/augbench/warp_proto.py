# Prototype bicubic warps (PIL a=-1), compared with the emitted ones for exactness and speed.
import tensorflow as tf

def _pil_cub(t):
    a = -1.0
    def near(d): return ((a + 2.0) * d - (a + 3.0)) * d * d + 1.0
    def far(d):  return ((a * d - 5.0 * a) * d + 8.0 * a) * d - 4.0 * a
    return tf.stack([far(1.0 + t), near(t), near(1.0 - t), far(2.0 - t)], -1)

def _coords(img, vec):
    H = tf.shape(img)[0]; W = tf.shape(img)[1]
    Hf = tf.cast(H, tf.float32); Wf = tf.cast(W, tf.float32)
    v = [tf.cast(t, tf.float32) for t in vec]
    ys, xs = tf.meshgrid(tf.range(Hf), tf.range(Wf), indexing='ij')
    den = v[6] * xs + v[7] * ys + 1.0
    u = (v[0] * xs + v[1] * ys + v[2]) / den
    w = (v[3] * xs + v[4] * ys + v[5]) / den
    inside = (u >= -0.5) & (u < Wf - 0.5) & (w >= -0.5) & (w < Hf - 0.5)
    return H, W, u, w, inside

def transform_u8taps(img, vec):
    """Same 16-tap loop and summation order as the emitted one, gathering uint8 (4x fewer bytes)."""
    H, W, u, w, inside = _coords(img, vec)
    x0 = tf.floor(u); y0 = tf.floor(w)
    wx = _pil_cub(u - x0); wy = _pil_cub(w - y0)
    offs = tf.constant([-1.0, 0.0, 1.0, 2.0])
    xi = tf.clip_by_value(tf.cast(x0[..., None] + offs, tf.int32), 0, W - 1)
    yi = tf.clip_by_value(tf.cast(y0[..., None] + offs, tf.int32), 0, H - 1)
    flat = tf.reshape(img, [-1, 3])
    out = tf.zeros([H, W, 3], tf.float32)
    for i in range(4):
        row = tf.zeros([H, W, 3], tf.float32)
        for j in range(4):
            row += wx[:, :, j:j+1] * tf.cast(tf.gather(flat, yi[:, :, i] * W + xi[:, :, j]), tf.float32)
        out += wy[:, :, i:i+1] * row
    out = tf.where(inside[..., None], out, 128.0)
    return tf.cast(tf.clip_by_value(tf.floor(out), 0.0, 255.0), tf.uint8)

def transform_onegather(img, vec):
    """One uint8 gather of all 16 taps, then two small contractions."""
    H, W, u, w, inside = _coords(img, vec)
    x0 = tf.floor(u); y0 = tf.floor(w)
    wx = _pil_cub(u - x0); wy = _pil_cub(w - y0)
    offs = tf.constant([-1, 0, 1, 2])
    xi = tf.clip_by_value(tf.cast(x0, tf.int32)[..., None] + offs, 0, W - 1)
    yi = tf.clip_by_value(tf.cast(y0, tf.int32)[..., None] + offs, 0, H - 1)
    idx = yi[:, :, :, None] * W + xi[:, :, None, :]                      # [H,W,4,4]
    g = tf.cast(tf.gather(tf.reshape(img, [-1, 3]), idx), tf.float32)    # [H,W,4,4,3]
    row = tf.einsum('hwj,hwijc->hwic', wx, g)
    out = tf.einsum('hwi,hwic->hwc', wy, row)
    out = tf.where(inside[..., None], out, 128.0)
    return tf.cast(tf.clip_by_value(tf.floor(out), 0.0, 255.0), tf.uint8)

def warp1d_u8(img, a, c, axis):
    """The emitted 1-D warp, gathering uint8."""
    H = tf.shape(img)[0]; W = tf.shape(img)[1]
    Hf = tf.cast(H, tf.float32); Wf = tf.cast(W, tf.float32)
    a = tf.cast(a, tf.float32); c = tf.cast(c, tf.float32)
    ys, xs = tf.meshgrid(tf.range(Hf), tf.range(Wf), indexing='ij')
    if axis == 1: u = xs + a * ys + c; n = W; nf = Wf
    else:         u = ys + a * xs + c; n = H; nf = Hf
    inside = (u >= -0.5) & (u < nf - 0.5)
    u0 = tf.floor(u); t = u - u0
    a_ = -1.0
    def near(d): return ((a_ + 2.0) * d - (a_ + 3.0)) * d * d + 1.0
    def far(d):  return ((a_ * d - 5.0 * a_) * d + 8.0 * a_) * d - 4.0 * a_
    wts = [far(1.0 + t), near(t), near(1.0 - t), far(2.0 - t)]
    flat = tf.reshape(img, [-1, 3])
    iy = tf.cast(ys, tf.int32); ix = tf.cast(xs, tf.int32)
    out = tf.zeros([H, W, 3], tf.float32)
    for k in range(4):
        tap = tf.clip_by_value(tf.cast(u0, tf.int32) + (k - 1), 0, n - 1)
        idx = iy * W + tap if axis == 1 else tap * W + ix
        out += wts[k][:, :, None] * tf.cast(tf.gather(flat, idx), tf.float32)
    out = tf.where(inside[..., None], out, 128.0)
    return tf.cast(tf.clip_by_value(tf.floor(out), 0.0, 255.0), tf.uint8)
