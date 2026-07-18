#!/usr/bin/env python3
"""Compile + numerically validate the Lean-emitted anchor-YOLO-loss block
(brick #2, WS-C). Mirrors the DIoU probe's rigor for the full A-anchor loss
(per-anchor DIoU box + focal-BCE objectness + masked softmax-CE class).

  1. lake build anchor-loss-probe
  2. .lake/build/bin/anchor-loss-probe B gH gW A <out.mlir>
  3. this script: iree-compile (CPU) + run, then
     (a) emitted forward   vs an independent numpy forward,
     (b) emitted d_pred     vs a numpy analytic gradient, and
     (c) that numpy gradient vs f64 finite differences — on the box + class
         channels only. Objectness uses a DETACHED focal weight (the grad is
         α·w·(p−t) with w treated constant), so it is intentionally NOT the FD of
         the forward; it is checked via (b) against the same detached formula.

Anchors/γ/λ match anchorLossProbeModule: anchor i = (0.02+0.03i, 0.03+0.04i),
γ=2, λ_box=5, λ_noobj=0.5, NC=10.

Run:  <jax-venv>/bin/python scripts/anchor_loss_probe_check.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import iree.runtime as rt  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from diou_probe_check import np_diou_forward, np_diou_grad, sigmoid  # noqa: E402

PROBE = ".lake/build/bin/anchor-loss-probe"
IREE_COMPILE = ".venv/bin/iree-compile"
NC, P = 10, 15
GAMMA, LBOX, LNOOBJ = 2.0, 5.0, 0.5


def anchors_for(A):
    return [(0.02 + 0.03 * i, 0.03 + 0.04 * i) for i in range(A)]


def cls_weight_map(ct, clsw):
    """Per-cell class weight w_{c(cell)} from the one-hot target (T1b).

    Depends only on the target, so it is exactly constant w.r.t. the logits --
    which is why the weighted gradient below is still FD-checkable."""
    if clsw is None:
        return 1.0
    w = np.asarray(clsw, dtype=np.float64).reshape(1, -1, *([1] * (ct.ndim - 2)))
    return ((ct > 0.5) * w).sum(1, keepdims=True)


def np_forward(pred, tgt, mask, gH, gW, anchors, clsw=None):
    B = pred.shape[0]
    total = 0.0
    for a, (aw, ah) in enumerate(anchors):
        base = a * P
        bp, bt = pred[:, base:base + 4], tgt[:, base:base + 4]
        op = pred[:, base + 4:base + 5]
        cp, ct = pred[:, base + 5:base + 15], tgt[:, base + 5:base + 15]
        ma, m4 = mask[:, a], mask[:, a:a + 1]
        boxloss = LBOX * np_diou_forward(bp, bt, ma, gH, gW, aw, ah)
        z, t = op, m4
        p = sigmoid(z)
        pt = t * p + (1 - t) * (1 - p)
        w = np.maximum(1 - pt, 1e-12) ** GAMMA
        alpha = t + (1 - t) * LNOOBJ
        bce = np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z)))
        objloss = float(np.sum(alpha * w * bce))
        cshift = cp - cp.max(1, keepdims=True)
        lsm = cshift - np.log(np.exp(cshift).sum(1, keepdims=True))
        clsloss = float(np.sum(-(ct * lsm) * m4 * cls_weight_map(ct, clsw)))
        total += boxloss + objloss + clsloss
    return total / B


def np_grad(pred, tgt, mask, gH, gW, anchors, clsw=None):
    B = pred.shape[0]
    grad = np.zeros_like(pred)
    for a, (aw, ah) in enumerate(anchors):
        base = a * P
        bp, bt = pred[:, base:base + 4], tgt[:, base:base + 4]
        op = pred[:, base + 4:base + 5]
        cp, ct = pred[:, base + 5:base + 15], tgt[:, base + 5:base + 15]
        ma, m4 = mask[:, a], mask[:, a:a + 1]
        grad[:, base:base + 4] = LBOX / B * np_diou_grad(bp, bt, ma, gH, gW, aw, ah)
        z, t = op, m4
        p = sigmoid(z)
        pt = t * p + (1 - t) * (1 - p)
        w = np.maximum(1 - pt, 1e-12) ** GAMMA
        alpha = t + (1 - t) * LNOOBJ
        grad[:, base + 4:base + 5] = alpha * w * (p - t) / B
        cshift = cp - cp.max(1, keepdims=True)
        ex = np.exp(cshift); sm = ex / ex.sum(1, keepdims=True)
        grad[:, base + 5:base + 15] = (sm - ct) * m4 * cls_weight_map(ct, clsw) / B
    return grad


def make_runner(B, gH, gW, A):
    td = tempfile.mkdtemp()
    mlir = os.path.join(td, "anchor_loss_gen.mlir")
    r = subprocess.run([PROBE, str(B), str(gH), str(gW), str(A), mlir],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr); sys.exit("probe emit failed")
    vmfb = os.path.join(td, "al.vmfb")
    r = subprocess.run([IREE_COMPILE, mlir, "--iree-hal-target-backends=llvm-cpu",
                        "-o", vmfb], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[:3000]); sys.exit("iree-compile failed")
    ctx = rt.SystemContext(config=rt.Config("local-task"))
    with open(vmfb, "rb") as f:
        ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, f.read()))
    fn = ctx.modules.anchor_loss_probe["main"]

    def run(pred, tgt):
        out = fn(pred.astype(np.float32), tgt.astype(np.float32))
        return np.asarray(out[0]).item(), np.asarray(out[1]).astype(np.float64)
    return run


def make_data(B, gH, gW, A, seed):
    rng = np.random.RandomState(seed)
    pred = rng.randn(B, A * P, gH, gW) * 0.5
    tgt = np.zeros_like(pred)
    mask = (rng.rand(B, A, gH, gW) < 0.3).astype(np.float64)
    for a in range(A):
        base = a * P
        tgt[:, base + 0] = rng.rand(B, gH, gW)
        tgt[:, base + 1] = rng.rand(B, gH, gW)
        tgt[:, base + 2] = rng.uniform(0.01, 0.08, (B, gH, gW))
        tgt[:, base + 3] = rng.uniform(0.02, 0.10, (B, gH, gW))
        tgt[:, base + 4] = mask[:, a]          # obj target = per-anchor mask
        ci = rng.randint(0, NC, (B, gH, gW))
        for c in range(NC):
            tgt[:, base + 5 + c] = (ci == c).astype(float)
    return pred, tgt, mask


def check(B=2, gH=5, gW=5, A=3, seed=0):
    anchors = anchors_for(A)
    pred, tgt, mask = make_data(B, gH, gW, A, seed)
    run = make_runner(B, gH, gW, A)
    loss, dpred = run(pred, tgt)

    ref = np_forward(pred, tgt, mask, gH, gW, anchors)
    frel = abs(loss - ref) / max(abs(ref), 1e-9)
    npg = np_grad(pred, tgt, mask, gH, gW, anchors)
    gmax = np.abs(dpred - npg).max()

    # (c) oracle: numpy grad vs f64 FD on box+class channels (skip obj ch base+4)
    obj_ch = [a * P + 4 for a in range(A)]
    hf = 1e-6
    fd_err = 0.0
    for idx in np.ndindex(*pred.shape):
        if idx[1] in obj_ch:
            continue
        pp = pred.copy(); pp[idx] += hf
        pm = pred.copy(); pm[idx] -= hf
        fdv = (np_forward(pp, tgt, mask, gH, gW, anchors)
               - np_forward(pm, tgt, mask, gH, gW, anchors)) / (2 * hf)
        fd_err = max(fd_err, abs(npg[idx] - fdv))

    ok = frel < 1e-4 and gmax < 1e-3 and fd_err < 1e-5
    print(f"anchor-loss probe  B={B} gH={gH} gW={gW} A={A} seed={seed}")
    print(f"  emitted forward  vs numpy         : rel={frel:.2e}  {'PASS' if frel<1e-4 else 'FAIL'}")
    print(f"  emitted backward vs numpy analytic: max={gmax:.2e}  {'PASS' if gmax<1e-3 else 'FAIL'}")
    print(f"  numpy grad vs f64 FD (box+cls)    : max={fd_err:.2e}  {'PASS' if fd_err<1e-5 else 'FAIL'}")
    return ok


def main():
    ok = True
    for seed in (0, 1):
        ok &= check(B=2, gH=5, gW=5, A=3, seed=seed)
    ok &= check(B=1, gH=4, gW=6, A=6, seed=2)   # A=6 (the real config)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
