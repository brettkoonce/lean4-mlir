#!/usr/bin/env python3
"""Do the CNN descent rungs' pool hypotheses hold on MNIST at the trained Chapter-3 weights?

The rungs in LeanMlir/Proofs/Training/SgdDescent/Cnn.lean take the pool condition
`MaxPool2MarginQUpTo delta T` on the conv2 PRE-activation z2: every 2x2 window is dead (all
cells <= 0), or the cell dominating it is more than 2*delta above every cell that is not its
twin. Twins are the cells the theorem may let tie:

  conv2 rungs (`cnn_conv2_sgd_descends`, `cnn_conv2_bias_sgd_descends`): `ConvPatchEq` --
      the two cells read identical zero-padded 3x3x32 patches of x1 = relu(conv1 x0), so
      they are equal for every conv2 kernel and bias.
  conv1 rungs (`cnn_conv1_sgd_descends`, `cnn_conv1_bias_sgd_descends`): `ConvPatchEq2` --
      for every offset of the conv2 window both reads fall in the padding, or both land on
      cells with identical zero-padded 3x3 patches of x0, so the cells are equal for every
      conv1 kernel and bias too.

Since delta is the step radius times a weight bound, a small enough learning rate makes it as
small as wanted: the condition is satisfiable at an image exactly when every live window's
dominating cell is STRICTLY above its non-twins. The rungs also need every conv pre-activation
and both head pre-activations nonzero (the relu margins), so those are counted too. For each
satisfying image the script reports the largest admissible delta (half the smallest non-twin
gap) and the smallest |pre-activation|, which bound the learning rate a concrete instance can
take.

Three conditions are counted per image:

  old       `MaxPool2MarginQ` on relu(z2) as the rungs took it before: every window's
            post-ReLU maximum attained at one cell. A dead window or a twin tie fails it.
  no-twin   dead, or a unique pre-activation maximum: what the binary32 rungs
            (`cnn_*_float_sgd_descends`) still need. Their pool backward routes the cotangent
            to EVERY cell attaining the maximum (the rendered compare-and-select), so at a tie
            the float gradient is not the loss gradient.
  twin      the rungs' `MaxPool2MarginQUpTo`, per rung family (conv2 / conv1 twins), together
            with the relu margins' nonzero clauses.

Arithmetic is float64 standing in for the reals; twin cells compute bit-identical values, since
they run the same operations on the same numbers. Weights: the trained `cnnVerified` parameter
dump (`.lake/build/cnn_verified_params.bin`, 3489130 float32 in `toSpecs` order, plus one
trailing float). CPU only.

  python3 scripts/probes/mnist_pool_twin_probe.py --n 10000
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
from _mnist_io import mnist  # noqa: E402

SHAPES = [(32, 1, 3, 3), (32,), (32, 32, 3, 3), (32,), (6272, 512), (512,), (512, 512), (512,),
          (512, 10), (10,)]


def load_params(path):
    a = np.frombuffer(open(path, "rb").read(), dtype="<f4").astype(np.float64)
    need = sum(int(np.prod(s)) for s in SHAPES)
    assert a.size >= need, f"{path}: {a.size} floats, need {need}"
    out, i = [], 0
    for s in SHAPES:
        n = int(np.prod(s))
        out.append(a[i:i + n].reshape(s))
        i += n
    return out


def patches(x):
    """[N,C,H,W] -> [N,H,W,C,3,3], zero-padded (the `convPad` reads at each output cell)."""
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    return np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3)).transpose(0, 2, 3, 1, 4, 5)


def conv(x, W, b):
    return np.einsum("nhwckl,ockl->nohw", patches(x), W, optimize=True) + b[None, :, None, None]


def window_cells(a):
    """[N,H,W,...] at H = 2h -> [N,h,w,4,...], cells in (a,b) order (0,0),(0,1),(1,0),(1,1)."""
    n, H, W = a.shape[:3]
    r = a.reshape((n, H // 2, 2, W // 2, 2) + a.shape[3:])
    r = np.moveaxis(r, 2, 3)
    return r.reshape((n, H // 2, W // 2, 4) + a.shape[3:])


def pair_eq(cells):
    """[N,h,w,4,F] -> [N,h,w,4,4] exact equality of the feature vectors."""
    return np.all(cells[:, :, :, :, None, :] == cells[:, :, :, None, :, :], axis=-1)


def conv1_signature(x0):
    """Per conv2 cell of a 28x28 map: for each of the 9 conv2 offsets, an in-range flag and the
    zero-padded 3x3 patch of x0 there (zeros when out of range). Equal signatures = ConvPatchEq2."""
    n, _, H, W = x0.shape
    P0 = patches(x0).reshape(n, H, W, 9)
    P0p = np.pad(P0, ((0, 0), (1, 1), (1, 1), (0, 0)))
    inr = np.pad(np.ones((H, W)), 1)
    sig = []
    for a in range(3):
        for b in range(3):
            sig.append(np.broadcast_to(inr[a:a + H, b:b + W][None, :, :, None], (n, H, W, 1)))
            sig.append(P0p[:, a:a + H, b:b + W, :])
    return np.concatenate(sig, axis=-1)


def window_check(z2w, twin):
    """z2w [N,h,w,4,C] pre-activations, twin [N,h,w,4,4] position twins.

    Returns per image: ok_twin, ok_notwin, ok_old, the smallest non-twin gap below a dominating
    cell over live windows, and window counts (dead, positive ties, positive ties all twins)."""
    v = np.moveaxis(z2w, -1, 3)                    # [N,h,w,C,4]
    tw = twin[:, :, :, None, :, :]                 # [N,h,w,1,4,4]
    dead = np.all(v <= 0, axis=-1)                 # [N,h,w,C]
    mx = v.max(axis=-1, keepdims=True)
    dom = v == mx                                  # [N,h,w,C,4]
    eye = np.eye(4, dtype=bool)
    # a non-twin cell l != k tying a dominating cell k
    tie = dom[..., :, None] & dom[..., None, :] & ~eye
    bad_twin = np.any(tie & ~tw, axis=(-1, -2)) & ~dead
    bad_notwin = (dom.sum(-1) > 1) & ~dead
    r = np.maximum(v, 0)
    bad_old = (r == r.max(-1, keepdims=True)).sum(-1) > 1
    # smallest gap mx - v[l] over dominating k and non-twin l (the margin 2*delta must stay below)
    gapm = np.where(dom[..., :, None] & ~tw & ~eye, (mx[..., None] - v[..., None, :]), np.inf)
    gap = np.where(dead, np.inf, gapm.min(axis=(-1, -2)))
    nimg = v.shape[0]
    flat = lambda a: a.reshape(nimg, -1)
    postie = (dom.sum(-1) > 1) & ~dead
    postie_twin = postie & ~bad_twin
    return (~flat(bad_twin).any(1), ~flat(bad_notwin).any(1), ~flat(bad_old).any(1),
            flat(gap).min(1), flat(dead).sum(1), flat(postie).sum(1), flat(postie_twin).sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(ROOT, ".lake/build/cnn_verified_params.bin"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args()
    if not os.path.exists(a.params):
        sys.exit(f"missing {a.params}: the trained cnnVerified parameter dump (MainMnistCnnVerified "
                 "with LEAN_MLIR_DUMP_PARAMS) is not on this box")
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = load_params(a.params)
    X, y = mnist("test")
    n = min(a.n, X.shape[0])
    tot = dict(images=0, correct=0, old=0, notwin=0, twin2=0, twin1=0, rung2=0, rung1=0,
               windows=0, dead=0, postie=0, postie_twin2=0, postie_twin1=0)
    delta2, delta1, minz = [], [], []
    for s in range(0, n, a.batch):
        x0 = X[s:s + a.batch, None].astype(np.float64) / 255.0
        nb = x0.shape[0]
        z1 = conv(x0, W1, b1)
        x1 = np.maximum(z1, 0)
        z2 = conv(x1, W2, b2)
        pool = np.maximum(z2, 0).reshape(nb, 32, 14, 2, 14, 2).max(axis=(3, 5)).reshape(nb, -1)
        z3 = pool @ W3 + b3
        z4 = np.maximum(z3, 0) @ W4 + b4
        logits = np.maximum(z4, 0) @ W5 + b5
        z2w = window_cells(np.moveaxis(z2, 1, -1))                     # [N,14,14,4,32]
        tw2 = pair_eq(window_cells(patches(x1).reshape(nb, 28, 28, -1)))
        tw1 = pair_eq(window_cells(conv1_signature(x0)))
        ok2, okn, oko, gap2, dead, postie, pt2 = window_check(z2w, tw2)
        ok1, _, _, gap1, _, _, pt1 = window_check(z2w, tw1)
        nz1 = np.all(z1.reshape(nb, -1) != 0, axis=1)
        nz2 = np.all(z2.reshape(nb, -1) != 0, axis=1)
        nz34 = np.all(z3 != 0, axis=1) & np.all(z4 != 0, axis=1)
        r2 = ok2 & nz2 & nz34
        r1 = ok1 & nz1 & nz2 & nz34
        tot["images"] += nb
        tot["correct"] += int((logits.argmax(1) == y[s:s + nb]).sum())
        tot["old"] += int(oko.sum())
        tot["notwin"] += int(okn.sum())
        tot["twin2"] += int(ok2.sum())
        tot["twin1"] += int(ok1.sum())
        tot["rung2"] += int(r2.sum())
        tot["rung1"] += int(r1.sum())
        tot["windows"] += nb * 14 * 14 * 32
        tot["dead"] += int(dead.sum())
        tot["postie"] += int(postie.sum())
        tot["postie_twin2"] += int(pt2.sum())
        tot["postie_twin1"] += int(pt1.sum())
        delta2 += list(gap2[r2] / 2)
        delta1 += list(gap1[r1] / 2)
        minz += list(np.minimum(np.abs(z2.reshape(nb, -1)).min(1),
                                np.minimum(np.abs(z3).min(1), np.abs(z4).min(1)))[r2])
    t = tot
    print(f"MNIST test images: {t['images']}  (trained cnnVerified, test acc on these "
          f"{100 * t['correct'] / t['images']:.2f}%)")
    print(f"windows (per channel): {t['windows']}  dead {t['dead']} "
          f"({100 * t['dead'] / t['windows']:.2f}%)  positive ties {t['postie']} "
          f"({100 * t['postie'] / t['windows']:.2f}%), of them twins: conv2 {t['postie_twin2']}, "
          f"conv1 {t['postie_twin1']}")
    print(f"images satisfying:  old MaxPool2MarginQ {t['old']}   no-twin (binary32 rungs) "
          f"{t['notwin']}")
    print(f"  MaxPool2MarginQUpTo, conv2 twins {t['twin2']} (+ relu clauses: {t['rung2']}); "
          f"conv1 twins {t['twin1']} (+ relu clauses: {t['rung1']})")
    if delta2:
        d = np.array(delta2)
        print(f"  conv2 rungs: admissible delta median {np.median(d):.3e}, min {d.min():.3e}; "
              f"smallest |pre-activation| median {np.median(minz):.3e}, min {min(minz):.3e}")
    if delta1:
        d = np.array(delta1)
        print(f"  conv1 rungs: admissible delta median {np.median(d):.3e}, min {d.min():.3e}")


if __name__ == "__main__":
    main()
