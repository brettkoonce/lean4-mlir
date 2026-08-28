"""Correctness gate for the box-aware FPN affine (`F32.fpnAffine`).

Run it after ANY change to `fpn_affine_one`, `fpn_decode_boxes` or
`fpn_encode_boxes` in `ffi/f32_helpers.c`:

    scripts/../.venv/bin/python3 scripts/check_fpn_affine.py

Two rules this file exists to enforce, both learned the hard way in this repo:

1. **It compiles `ffi/f32_helpers.c` itself and calls into that.** It does not
   re-implement the transform and it does not test a copy. `deploy/export_onnx.py`
   shipped a different model for a day because its "validation" compared against a
   re-derivation that had drifted from the spec; a gate that tests a copy tests the
   copy.

2. **The reference is `preprocess_visdrone.encode_targets_fpn` itself**, imported
   from the repo — the same function that wrote the targets on disk. An affine is
   not shape-invariant the way the hflip is: scaling changes `max(w,h)`, which
   changes both the FPN level a box lands on and which anchor wins the wh-IoU. So
   the only honest check is that the C target equals a FULL RE-ENCODE of the
   transformed boxes by the real encoder.

Three properties, in increasing strength:
  1. identity   -- scale 1, translate 0 must return the target unchanged, which
                   proves the decode is a true inverse of the encode on REAL data.
  2. re-encode  -- for random (scale, translate), the C target must equal the
                   Python encoder run on the same transformed boxes.
  3. pixels     -- the warp must invert under the inverse transform, away from
                   the borders where the mean fill enters.
"""
import ctypes, sys, numpy as np
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from preprocess_visdrone import encode_targets_fpn, FPN_GRIDS, FPN_T_LO, FPN_T_HI

ANCHORS = [
    [(0.006935, 0.014941), (0.015750, 0.028005), (0.033728, 0.035028)],
    [(0.023961, 0.070528), (0.055662, 0.068706), (0.093187, 0.094324)],
    [(0.060280, 0.168604), (0.107559, 0.204684), (0.181239, 0.149031)],
]
PX, NTOT = 448, 185220
GEO = np.array([[g, 3] for g in FPN_GRIDS], dtype=np.int32).ravel()
ANC = np.array([a for lvl in ANCHORS for a in lvl], dtype=np.float32).ravel()

def _build():
    """Compile the SHIPPING ffi/f32_helpers.c into a shared object and dlopen it.

    Undefined Lean runtime symbols are fine: they belong to the LEAN_EXPORT
    wrappers, RTLD_LAZY never resolves them, and `fpn_affine_one` is plain C.
    """
    import subprocess, tempfile, os
    inc = subprocess.run(['lean', '--print-prefix'], capture_output=True,
                         text=True, check=True).stdout.strip() + '/include'
    so = os.path.join(tempfile.gettempdir(), 'fpn_affine_gate.so')
    subprocess.run(['gcc', '-O2', '-fPIC', '-shared', '-I', inc,
                    REPO + '/ffi/f32_helpers.c', '-o', so, '-lm'], check=True)
    return so


# RTLD_LAZY: the Lean runtime symbols the LEAN_EXPORT wrappers reference are
# never called from here, so they must not be resolved at load time.
lib = ctypes.CDLL(_build(), mode=os.RTLD_LAZY)
lib.fpn_affine_one.restype = None
lib.fpn_affine_one.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]

BXBUF = (ctypes.c_char * (12348 * 20))()


def c_affine(img, tgt, s, tx, ty, wh_thr=1.0, area_thr=0.1):
    img = np.ascontiguousarray(img, dtype=np.float32).copy()
    tgt = np.ascontiguousarray(tgt, dtype=np.float32).copy()
    scratch = np.zeros_like(img)
    lib.fpn_affine_one(
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tgt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(BXBUF, ctypes.c_void_p),
        3, PX, PX,
        GEO.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 3,
        ANC.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        s, tx, ty, FPN_T_LO, FPN_T_HI, wh_thr, area_thr)
    return img, tgt


def decode_py(tgt):
    """Python inverse of encode_targets_fpn, for the reference path."""
    out, off = [], 0
    for si, g in enumerate(FPN_GRIDS):
        A, gg = len(ANCHORS[si]), g * g
        blk = tgt[off:off + A * 15 * gg].reshape(A, 15, g, g)
        for a in range(A):
            ci, cj = np.nonzero(blk[a, 4] > 0.5)
            for i, j in zip(ci, cj):
                out.append((int(np.argmax(blk[a, 5:15, i, j])),
                            (blk[a, 0, i, j] + j) / g, (blk[a, 1, i, j] + i) / g,
                            float(blk[a, 2, i, j]), float(blk[a, 3, i, j])))
        off += A * 15 * gg
    return out


def reference(tgt, s, tx, ty, wh_thr=1.0, area_thr=0.1):
    """Transform the decoded boxes, then hand them to the REAL encoder."""
    boxes = []
    for cid, cx, cy, w, h in decode_py(tgt):
        ncx, ncy = (cx - 0.5) * s + 0.5 + tx, (cy - 0.5) * s + 0.5 + ty
        nw, nh = w * s, h * s
        x1, x2 = max(ncx - nw / 2, 0.0), min(ncx + nw / 2, 1.0)
        y1, y2 = max(ncy - nh / 2, 0.0), min(ncy + nh / 2, 1.0)
        cw, ch = x2 - x1, y2 - y1
        if cw <= 0 or ch <= 0: continue
        if cw * PX < wh_thr or ch * PX < wh_thr: continue
        if (cw * ch) / (nw * nh + 1e-12) < area_thr: continue
        # encode_targets_fpn takes PIXEL corners plus the image size
        boxes.append((cid, x1 * PX, y1 * PX, x2 * PX, y2 * PX))
    tg, _, _ = encode_targets_fpn(PX, PX, boxes, [np.array(a) for a in ANCHORS], PX)
    return np.concatenate([t.ravel() for t in tg])


REC = 3 * PX * PX + NTOT * 4
def load(i):
    with open(REPO + '/data/visdrone_fpn/val.bin', 'rb') as f:
        f.seek(4 + i * REC)
        img = np.frombuffer(f.read(3 * PX * PX), dtype=np.uint8)
        tgt = np.frombuffer(f.read(NTOT * 4), dtype=np.float32)
    mean = np.array([.485, .456, .406], np.float32).reshape(3, 1, 1)
    std = np.array([.229, .224, .225], np.float32).reshape(3, 1, 1)
    x = ((img.reshape(3, PX, PX).astype(np.float32) / 255.0) - mean) / std
    return x.ravel().copy(), tgt.copy()


FAILED = []

print('1. IDENTITY (s=1, t=0) — decode must be a true inverse of encode')
bad = 0
for i in range(12):
    img, tgt = load(i)
    _, t2 = c_affine(img, tgt, 1.0, 0.0, 0.0)
    d = np.abs(t2 - tgt).max()
    npos0, npos1 = int((tgt > 0.5).sum()), int((t2 > 0.5).sum())
    if d > 1e-6 or npos0 != npos1:
        bad += 1
        print(f'   ⛔ record {i}: max|d| {d:.3e}, positives {npos0} -> {npos1}')
print(f'   {"⛔ " + str(bad) + " FAILED" if bad else "✅ 12/12 byte-identical"}')
if bad: FAILED.append('identity')

print('\n2. RE-ENCODE — C target vs preprocess_visdrone.encode_targets_fpn')
rng = np.random.default_rng(0)
worst, worstd = 0.0, None
for i in range(8):
    img, tgt = load(i)
    for _ in range(6):
        s = float(rng.uniform(0.7, 1.4))
        tx, ty = float(rng.uniform(-.15, .15)), float(rng.uniform(-.15, .15))
        _, got = c_affine(img, tgt, s, tx, ty)
        ref = reference(tgt, s, tx, ty)
        d = float(np.abs(got - ref).max())
        if d > worst: worst, worstd = d, (i, s, tx, ty)
        nga, nra = int((got[4::15].size and (got > 0.5).sum())), int((ref > 0.5).sum())
print(f'   worst max|d| over 48 (record, scale, translate) draws: {worst:.3e}')
print(f'   at {worstd}')
print(f'   {"✅ exact" if worst == 0.0 else "⛔ MISMATCH"}')
if worst != 0.0: FAILED.append('re-encode')

print('\n3. PIXELS — warp then inverse-warp must return the interior')
img, tgt = load(3)
s, tx, ty = 1.25, 0.07, -0.04
w1, _ = c_affine(img, tgt, s, tx, ty)
w2, _ = c_affine(w1, tgt, 1.0 / s, -tx / s, -ty / s)
a = img.reshape(3, PX, PX)[:, 80:368, 80:368]
b = w2.reshape(3, PX, PX)[:, 80:368, 80:368]
print(f'   interior max|d| {np.abs(a - b).max():.3f}  mean {np.abs(a - b).mean():.4f}'
      f'  (two bilinear resamples, so a blur residual is expected)')
print(f'   correlation {np.corrcoef(a.ravel(), b.ravel())[0,1]:.6f}')

if FAILED:
    raise SystemExit(f"\n⛔ FAILED: {', '.join(FAILED)} — do not train on this "
                     f"augmenter. A wrong re-encode is silent: the loss still "
                     f"descends, it just descends toward the wrong targets.")
print('\n✅ all gates pass')
