#!/usr/bin/env python3
"""FPN neck (top-down merge) forward + gradient, central-FD verified — brick #3.

The multi-scale neck's new piece is the DAG topology, not new primitives: it
composes 1x1 lateral convs, the already-verified bilinear upsample, and adds.
This nails the merge math + its backward in numpy (matching the codegen's
bilinearWeights1D exactly) before it is emitted as StableHLO, so the hand-derived
DAG VJP has a ground truth. Topology (retinanet/FPN top-down):

    P5 = conv1x1(C5, W5)                       # [B,256,g5,g5]
    P4 = conv1x1(C4, W4) + upsample2(P5)       # [B,256,g4=2g5,g4]
    P3 = conv1x1(C3, W3) + upsample2(P4)       # [B,256,g3=2g4,g3]

Backward routes each Pn gradient to its lateral (conv) AND up to the coarser
level (upsample^T), accumulating: dP4_total = dP4 + up^T(dP3), etc.

Run: python3 scripts/fpn_neck_check.py
"""
import numpy as np

np.random.seed(0)


def bilinear_weights_1d(in_len, scale):
    """Match LeanMlir bilinearWeights1D (align_corners=False). -> [out_len, in_len]."""
    out_len = in_len * scale
    den = 2 * scale
    W = np.zeros((out_len, in_len))
    for i in range(out_len):
        num = 2 * i + 1 - scale
        y0 = num // den if num >= 0 else -((-num + den - 1) // den)
        wy = (num - y0 * den) / den
        cl = lambda x: 0 if x < 0 else (in_len - 1 if x >= in_len else x)
        W[i, cl(y0)] += 1.0 - wy
        W[i, cl(y0 + 1)] += wy
    return W


def upsample2(x):
    """Separable bilinear x2 on [B,C,H,W] via Wy·x·Wx^T (matches emitBilinearUpsample)."""
    B, C, H, Wd = x.shape
    Wy = bilinear_weights_1d(H, 2)      # [2H, H]
    Wx = bilinear_weights_1d(Wd, 2)     # [2W, W]
    # y = Wy · x along H, then · Wx^T along W
    y = np.einsum('oh,bchw->bcow', Wy, x)
    y = np.einsum('bcow,vw->bcov', y, Wx)
    return y


def upsample2_T(g):
    """VJP of upsample2: transpose matmuls."""
    B, C, oH, oW = g.shape
    H, Wd = oH // 2, oW // 2
    Wy = bilinear_weights_1d(H, 2); Wx = bilinear_weights_1d(Wd, 2)
    d = np.einsum('bcov,vw->bcow', g, Wx)      # undo Wx
    d = np.einsum('oh,bcow->bchw', Wy, d)      # undo Wy
    return d


def conv1x1(x, W):
    """x [B,ic,H,W], W [oc,ic] -> [B,oc,H,W]."""
    return np.einsum('oi,bihw->bohw', W, x)


def fpn_forward(C3, C4, C5, W3, W4, W5):
    P5 = conv1x1(C5, W5)
    P4 = conv1x1(C4, W4) + upsample2(P5)
    P3 = conv1x1(C3, W3) + upsample2(P4)
    return P3, P4, P5


def fpn_grad(C3, C4, C5, W3, W4, W5, dP3, dP4, dP5):
    """VJP: given dP3,dP4,dP5 -> dC3,dC4,dC5 (and dW*, not FD-checked here)."""
    # P3 = conv(C3) + up(P4):  dP4 += up^T(dP3); dC3 via conv^T(dP3)
    dP4_tot = dP4 + upsample2_T(dP3)
    # P4 = conv(C4) + up(P5):  dP5 += up^T(dP4_tot); dC4 via conv^T(dP4_tot)
    dP5_tot = dP5 + upsample2_T(dP4_tot)
    dC3 = np.einsum('oi,bohw->bihw', W3, dP3)
    dC4 = np.einsum('oi,bohw->bihw', W4, dP4_tot)
    dC5 = np.einsum('oi,bohw->bihw', W5, dP5_tot)
    dW3 = np.einsum('bohw,bihw->oi', dP3, C3)
    dW4 = np.einsum('bohw,bihw->oi', dP4_tot, C4)
    dW5 = np.einsum('bohw,bihw->oi', dP5_tot, C5)
    return dC3, dC4, dC5, dW3, dW4, dW5


def main():
    B, oc = 2, 8
    c3, c4, c5 = 6, 10, 12
    g5 = 3                      # g4=6, g3=12
    C3 = np.random.randn(B, c3, 4 * g5, 4 * g5)
    C4 = np.random.randn(B, c4, 2 * g5, 2 * g5)
    C5 = np.random.randn(B, c5, g5, g5)
    W3 = np.random.randn(oc, c3) * 0.3
    W4 = np.random.randn(oc, c4) * 0.3
    W5 = np.random.randn(oc, c5) * 0.3
    # random cotangents on the 3 outputs
    dP3 = np.random.randn(B, oc, 4 * g5, 4 * g5)
    dP4 = np.random.randn(B, oc, 2 * g5, 2 * g5)
    dP5 = np.random.randn(B, oc, g5, g5)

    def loss(C3, C4, C5):
        P3, P4, P5 = fpn_forward(C3, C4, C5, W3, W4, W5)
        return (P3 * dP3).sum() + (P4 * dP4).sum() + (P5 * dP5).sum()

    dC3, dC4, dC5, *_ = fpn_grad(C3, C4, C5, W3, W4, W5, dP3, dP4, dP5)

    h = 1e-6
    err = 0.0
    for name, arr, ana in [("C3", C3, dC3), ("C4", C4, dC4), ("C5", C5, dC5)]:
        fd = np.zeros_like(arr)
        for idx in np.ndindex(*arr.shape):
            ap = arr.copy(); ap[idx] += h
            am = arr.copy(); am[idx] -= h
            if name == "C3":
                fd[idx] = (loss(ap, C4, C5) - loss(am, C4, C5)) / (2 * h)
            elif name == "C4":
                fd[idx] = (loss(C3, ap, C5) - loss(C3, am, C5)) / (2 * h)
            else:
                fd[idx] = (loss(C3, C4, ap) - loss(C3, C4, am)) / (2 * h)
        e = np.abs(ana - fd).max()
        err = max(err, e)
        print(f"  d/d{name}: max|ana-FD| = {e:.2e}")
    ok = err < 1e-6
    print(f"FPN neck gradient check: max err {err:.2e} -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
