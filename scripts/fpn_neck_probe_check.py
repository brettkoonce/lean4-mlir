#!/usr/bin/env python3
"""Compile + numerically validate the Lean-emitted FPN-neck forward + backward
(detection-infra brick #3, planning/yolo_fpn.md bite 2).

  1. lake build fpn-neck-probe
  2. .lake/build/bin/fpn-neck-probe B oc c3 c4 c5 g5 <out.mlir>
  3. this script: iree-compile (CPU) + iree.runtime run, then compare
       - emitted forward P3/P4/P5  vs numpy fpn_forward
       - emitted backward dC/dW     vs the f64-FD-verified fpn_grad oracle

The neck's new piece is the DAG topology (1x1 lateral convs + the already-verified
bilinear upsample + adds), not a new primitive; this pins the merge + its
hand-derived VJP before wiring into the multi-scale train step. numpy math mirrors
scripts/fpn_neck_check.py exactly.

Run:  <jax-venv>/bin/python scripts/fpn_neck_probe_check.py
Needs iree.compiler + iree.runtime. CPU only.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import iree.runtime as rt  # noqa: E402

PROBE = ".lake/build/bin/fpn-neck-probe"
IREE_COMPILE = ".venv/bin/iree-compile"


# ── numpy oracle (mirrors scripts/fpn_neck_check.py + emitBilinearUpsample) ──
def bilinear_weights_1d(in_len, scale):
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
    B, C, H, Wd = x.shape
    Wy = bilinear_weights_1d(H, 2)
    Wx = bilinear_weights_1d(Wd, 2)
    y = np.einsum('oh,bchw->bcow', Wy, x)
    y = np.einsum('bcow,vw->bcov', y, Wx)
    return y


def upsample2_T(g):
    B, C, oH, oW = g.shape
    H, Wd = oH // 2, oW // 2
    Wy = bilinear_weights_1d(H, 2); Wx = bilinear_weights_1d(Wd, 2)
    d = np.einsum('bcov,vw->bcow', g, Wx)
    d = np.einsum('oh,bcow->bchw', Wy, d)
    return d


def conv1x1(x, W):
    return np.einsum('oi,bihw->bohw', W, x)


def fpn_forward(C3, C4, C5, W3, W4, W5):
    P5 = conv1x1(C5, W5)
    P4 = conv1x1(C4, W4) + upsample2(P5)
    P3 = conv1x1(C3, W3) + upsample2(P4)
    return P3, P4, P5


def fpn_grad(C3, C4, C5, W3, W4, W5, dP3, dP4, dP5):
    dP4_tot = dP4 + upsample2_T(dP3)
    dP5_tot = dP5 + upsample2_T(dP4_tot)
    dC3 = np.einsum('oi,bohw->bihw', W3, dP3)
    dC4 = np.einsum('oi,bohw->bihw', W4, dP4_tot)
    dC5 = np.einsum('oi,bohw->bihw', W5, dP5_tot)
    dW3 = np.einsum('bohw,bihw->oi', dP3, C3)
    dW4 = np.einsum('bohw,bihw->oi', dP4_tot, C4)
    dW5 = np.einsum('bohw,bihw->oi', dP5_tot, C5)
    return dC3, dC4, dC5, dW3, dW4, dW5


def make_runner(B, oc, c3, c4, c5, g5):
    """Emit + compile once; return f(C3,C4,C5,W3,W4,W5,dP3,dP4,dP5) -> outputs."""
    td = tempfile.mkdtemp()
    mlir = os.path.join(td, "fpn_neck_gen.mlir")
    r = subprocess.run([PROBE, str(B), str(oc), str(c3), str(c4), str(c5),
                        str(g5), mlir], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr); sys.exit("probe emit failed")
    vmfb = os.path.join(td, "fpn.vmfb")
    r = subprocess.run([IREE_COMPILE, mlir, "--iree-hal-target-backends=llvm-cpu",
                        "-o", vmfb], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[:3000]); sys.exit("iree-compile failed")
    cfg = rt.Config("local-task")
    ctx = rt.SystemContext(config=cfg)
    with open(vmfb, "rb") as f:
        vm = rt.VmModule.copy_buffer(ctx.instance, f.read())
    ctx.add_vm_module(vm)
    fn = ctx.modules.fpn_neck_probe["main"]

    def run(*arrs):
        out = fn(*[a.astype(np.float32) for a in arrs])
        return [np.asarray(o).astype(np.float64) for o in out]
    return run


def fd_grad(loss, arr, h=1e-6):
    fd = np.zeros_like(arr)
    for idx in np.ndindex(*arr.shape):
        ap = arr.copy(); ap[idx] += h
        am = arr.copy(); am[idx] -= h
        fd[idx] = (loss(ap) - loss(am)) / (2 * h)
    return fd


def check(B=2, oc=8, c3=6, c4=10, c5=12, g5=3, seed=0):
    rng = np.random.RandomState(seed)
    g4, g3 = 2 * g5, 4 * g5
    C3 = rng.randn(B, c3, g3, g3)
    C4 = rng.randn(B, c4, g4, g4)
    C5 = rng.randn(B, c5, g5, g5)
    W3 = rng.randn(oc, c3) * 0.3
    W4 = rng.randn(oc, c4) * 0.3
    W5 = rng.randn(oc, c5) * 0.3
    dP3 = rng.randn(B, oc, g3, g3)
    dP4 = rng.randn(B, oc, g4, g4)
    dP5 = rng.randn(B, oc, g5, g5)

    # (0) FD-verify the numpy oracle itself in f64 (the honest gate). loss is the
    #     random-cotangent inner product; its C/W grads are exactly fpn_grad.
    dC3, dC4, dC5, dW3, dW4, dW5 = fpn_grad(C3, C4, C5, W3, W4, W5, dP3, dP4, dP5)

    def L_from(C3_, C4_, C5_, W3_, W4_, W5_):
        P3, P4, P5 = fpn_forward(C3_, C4_, C5_, W3_, W4_, W5_)
        return (P3 * dP3).sum() + (P4 * dP4).sum() + (P5 * dP5).sum()

    oracle = 0.0
    for name, arr, ana, wrap in [
        ("C3", C3, dC3, lambda a: L_from(a, C4, C5, W3, W4, W5)),
        ("C4", C4, dC4, lambda a: L_from(C3, a, C5, W3, W4, W5)),
        ("C5", C5, dC5, lambda a: L_from(C3, C4, a, W3, W4, W5)),
        ("W3", W3, dW3, lambda a: L_from(C3, C4, C5, a, W4, W5)),
        ("W4", W4, dW4, lambda a: L_from(C3, C4, C5, W3, a, W5)),
        ("W5", W5, dW5, lambda a: L_from(C3, C4, C5, W3, W4, a)),
    ]:
        oracle = max(oracle, np.abs(ana - fd_grad(wrap, arr)).max())

    # (1) emitted module vs oracle
    run = make_runner(B, oc, c3, c4, c5, g5)
    eP3, eP4, eP5, eC3, eC4, eC5, eW3, eW4, eW5 = run(
        C3, C4, C5, W3, W4, W5, dP3, dP4, dP5)
    rP3, rP4, rP5 = fpn_forward(C3, C4, C5, W3, W4, W5)

    def rel(a, b):
        return np.abs(a - b).max() / max(np.abs(b).max(), 1e-9)

    fwd_err = max(rel(eP3, rP3), rel(eP4, rP4), rel(eP5, rP5))
    dc_err = max(np.abs(eC3 - dC3).max(), np.abs(eC4 - dC4).max(),
                 np.abs(eC5 - dC5).max())
    dw_err = max(np.abs(eW3 - dW3).max(), np.abs(eW4 - dW4).max(),
                 np.abs(eW5 - dW5).max())

    ok = oracle < 1e-5 and fwd_err < 1e-4 and dc_err < 1e-3 and dw_err < 1e-3
    print(f"FPN neck probe  B={B} oc={oc} c=({c3},{c4},{c5}) g5={g5} seed={seed}")
    print(f"  oracle (numpy grad) vs f64 FD : max_err={oracle:.2e}  "
          f"{'PASS' if oracle < 1e-5 else 'FAIL'}")
    print(f"  emitted forward  vs numpy     : rel={fwd_err:.2e}  "
          f"{'PASS' if fwd_err < 1e-4 else 'FAIL'}")
    print(f"  emitted dC       vs oracle    : max_err={dc_err:.2e}  "
          f"{'PASS' if dc_err < 1e-3 else 'FAIL'}")
    print(f"  emitted dW       vs oracle    : max_err={dw_err:.2e}  "
          f"{'PASS' if dw_err < 1e-3 else 'FAIL'}")
    return ok


def main():
    ok = True
    for seed in (0, 1, 2):
        ok &= check(seed=seed)
    # asymmetric channels + a different coarsest grid
    ok &= check(B=1, oc=4, c3=3, c4=5, c5=7, g5=2, seed=5)
    ok &= check(B=2, oc=16, c3=8, c4=8, c5=8, g5=4, seed=9)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
