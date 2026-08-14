#!/usr/bin/env python3
"""geo_aug_pil_diff.py — our TF geometric RandAugment ops vs PIL, which is what timm CALLS.

  scripts/geo_aug_pil_diff.py          # needs a venv with both tensorflow and PIL

⚠⚠ **WHY THIS EXISTS AND WHAT IT CORRECTED (2026-08-14).** The standing worry was that our
RandAugment geometry — implemented in TensorFlow (`tf.raw_ops.ImageProjectiveTransformV3`) — used
a different angle/matrix convention from timm's, which calls PIL (`Image.rotate`,
`Image.transform(AFFINE)`). timm's geometric ops are thin PIL wrappers, so diffing against PIL IS
diffing against timm for these ops. Measured: **it does not.** Shear, translate and rotate all
agree to mean |Δ| ≤ 0.5 of 255 on a smooth image.

⚠⚠⚠ **THE METHOD IS THE POINT, AND THE FIRST ATTEMPT GOT THE OPPOSITE ANSWER.** Run on RANDOM
NOISE, the same comparison reports 85–93% of pixels differing with mean |Δ| ≈ 10 — which reads as
a broken transform and is not one. On uncorrelated noise a HALF-PIXEL resampling difference makes
every pixel disagree by an arbitrary amount, so noise cannot separate "wrong geometry" from "same
geometry, different resampler". A smooth image with one hard edge can: a wrong transform shows
large INTERIOR error, a resampling difference shows error only where the gradient is steep.
▶ Never validate a resampler on noise.

⚠ What this does NOT cover, and it is the open question the angle worry should have been about:
the geometric INTERPOLATION CHOICE. We always use BILINEAR. timm's RandAugment takes its resample
mode from `hparams['interpolation']`, which without an explicit `-i` flag in the arg string is a
per-call RANDOM choice over (BILINEAR, BICUBIC). Settling that needs timm's source, not PIL's.
⚠ Also uncovered: the ORDER and PROBABILITY of ops, the magnitude mappings (`_RA_INC`), and the
photometric ops. This file is about geometry alone.
"""
import os, numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
import tensorflow as tf
from PIL import Image
H=W=64; FILL=128
yy,xx = np.mgrid[0:H,0:W].astype(np.float64)
# smooth, low-frequency, with a hard diagonal edge so a wrong TRANSFORM is visible as structure
sm = (127 + 60*np.sin(2*np.pi*xx/W) * np.cos(2*np.pi*yy/H))
sm += 40*(xx+yy > W)                       # a hard edge, to catch a shifted/mirrored map
img = np.clip(np.stack([sm, sm*0.8+20, sm*0.6+40], -1), 0, 255).astype(np.uint8)

def _t(im, vec):
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(tf.cast(im,tf.float32),0),
        transforms=tf.reshape(tf.cast(vec,tf.float32),[1,8]),
        output_shape=tf.stack([tf.shape(im)[0],tf.shape(im)[1]]),
        fill_value=float(FILL), interpolation='BILINEAR', fill_mode='CONSTANT')
    return tf.cast(tf.clip_by_value(out[0],0.,255.),tf.uint8).numpy()

ours = {
 "ShearX":    lambda l: _t(img,[1.,l,0.,0.,1.,0.,0.,0.]),
 "ShearY":    lambda l: _t(img,[1.,0.,0.,l,1.,0.,0.,0.]),
 "TranslateX":lambda p: _t(img,[1.,0.,-p*float(W),0.,1.,0.,0.,0.]),
}
def ours_rot(deg):
    th=deg*np.pi/180.; cs,sn=np.cos(th),np.sin(th); Hf,Wf=float(H),float(W)
    xo=((Wf-1.)-(cs*(Wf-1.)-sn*(Hf-1.)))/2.; yo=((Hf-1.)-(sn*(Wf-1.)+cs*(Hf-1.)))/2.
    return _t(img,[cs,-sn,xo,sn,cs,yo,0.,0.])
P=Image.fromarray(img); F=(FILL,)*3
pil = {
 "ShearX":    lambda l: np.array(P.transform((W,H),Image.AFFINE,(1,l,0,0,1,0),resample=Image.BILINEAR,fillcolor=F)),
 "ShearY":    lambda l: np.array(P.transform((W,H),Image.AFFINE,(1,0,0,l,1,0),resample=Image.BILINEAR,fillcolor=F)),
 "TranslateX":lambda p: np.array(P.transform((W,H),Image.AFFINE,(1,0,-p*W,0,1,0),resample=Image.BILINEAR,fillcolor=F)),
}
def pil_rot(deg): return np.array(P.rotate(deg,resample=Image.BILINEAR,fillcolor=F))

def cmp(name,a,b):
    a=a.astype(np.int32);b=b.astype(np.int32);d=np.abs(a-b)
    # interior only: exclude a 2px frame, where fill/edge handling legitimately differs
    di=d[2:-2,2:-2]
    print(f"  {name:26} whole: max {d.max():3d} mean {d.mean():6.3f} | INTERIOR: max {di.max():3d} mean {di.mean():6.3f}")

print("── ours (TF) vs timm/PIL, on a SMOOTH image ──")
for l in (0.3,-0.15):
    cmp(f"ShearX {l}", ours["ShearX"](l), pil["ShearX"](l))
    cmp(f"ShearY {l}", ours["ShearY"](l), pil["ShearY"](l))
for p in (0.2,-0.1):
    cmp(f"TranslateX {p}", ours["TranslateX"](p), pil["TranslateX"](p))
for d in (30.,-21.,7.5):
    cmp(f"Rotate {d}", ours_rot(d), pil_rot(d))
print("\n  (a genuinely different TRANSFORM shows large INTERIOR error on a smooth image;")
print("   a resampling/fill difference shows near-zero interior and error only at the frame)")
