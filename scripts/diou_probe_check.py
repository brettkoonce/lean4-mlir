#!/usr/bin/env python3
"""Compile + numerically validate the Lean-emitted DIoU box-loss FORWARD block
(brick #1, planning/yolo_drone.md WS-D — chunk 2a).

  1. lake build diou-loss-probe
  2. .lake/build/bin/diou-loss-probe B gH gW <out.mlir>
  3. this script: iree-compile (CPU) + iree.runtime run, then compare the emitted
     `loss` to an independent numpy forward that mirrors emitDiouForward exactly.

Forward-only for now: this confirms the box construction (sigmoid on xy, exp on
wh), the IoU / center-distance / enclosing-box arithmetic, and the masked sum are
emitted correctly, before the backward VJP probe. numpy reference math is the
same as scripts/diou_grad_check.py.

Run:  <jax-venv>/bin/python scripts/diou_probe_check.py
Needs iree.compiler + iree.runtime. CPU only.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import iree.runtime as rt  # noqa: E402

PROBE = ".lake/build/bin/diou-loss-probe"
IREE_COMPILE = ".venv/bin/iree-compile"
EPS = 1e-9


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def np_diou_forward(pred, tgt, mask, gH, gW, anchorW=1.0, anchorH=1.0):
    """Mirror emitDiouForward: loss = Σ_cells mask·(1 - DIoU). pred/tgt [B,4,gH,gW]."""
    B = pred.shape[0]
    tx, ty, tw, th = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    gx, gy, gw, gh = tgt[:, 0:1], tgt[:, 1:2], tgt[:, 2:3], tgt[:, 3:4]
    ci = np.broadcast_to(np.arange(gH, dtype=np.float64).reshape(1, 1, gH, 1), (B, 1, gH, gW))
    cj = np.broadcast_to(np.arange(gW, dtype=np.float64).reshape(1, 1, 1, gW), (B, 1, gH, gW))
    cx = (cj + sigmoid(tx)) / gW
    cy = (ci + sigmoid(ty)) / gH
    w, h = anchorW * np.exp(tw), anchorH * np.exp(th)
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    tcx = (cj + gx) / gW
    tcy = (ci + gy) / gH
    X0, X1, Y0, Y1 = tcx - gw / 2, tcx + gw / 2, tcy - gh / 2, tcy + gh / 2
    iw = np.maximum(0.0, np.minimum(x1, X1) - np.maximum(x0, X0))
    ih = np.maximum(0.0, np.minimum(y1, Y1) - np.maximum(y0, Y0))
    inter = iw * ih
    union = np.maximum(w * h + gw * gh - inter, EPS)
    iou = inter / union
    rho2 = (cx - tcx) ** 2 + (cy - tcy) ** 2
    cw = np.maximum(x1, X1) - np.minimum(x0, X0)
    ch = np.maximum(y1, Y1) - np.minimum(y0, Y0)
    c2 = np.maximum(cw * cw + ch * ch, EPS)
    diou = iou - rho2 / c2
    cell = (1.0 - diou) * mask[:, None, :, :]
    return float(cell.sum())


def np_diou_grad(pred, tgt, mask, gH, gW, anchorW=1.0, anchorH=1.0):
    """Grid form of the FD-verified diou_loss_grad (sum + mask). -> [B,4,gH,gW]."""
    B = pred.shape[0]
    tx, ty, tw, th = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    gx, gy, gw, gh = tgt[:, 0:1], tgt[:, 1:2], tgt[:, 2:3], tgt[:, 3:4]
    ci = np.broadcast_to(np.arange(gH, dtype=np.float64).reshape(1, 1, gH, 1), (B, 1, gH, gW))
    cj = np.broadcast_to(np.arange(gW, dtype=np.float64).reshape(1, 1, 1, gW), (B, 1, gH, gW))
    sx, sy = sigmoid(tx), sigmoid(ty)
    cx, cy = (cj + sx) / gW, (ci + sy) / gH
    w, h = anchorW * np.exp(tw), anchorH * np.exp(th)
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    tcx, tcy = (cj + gx) / gW, (ci + gy) / gH
    X0, X1, Y0, Y1 = tcx - gw / 2, tcx + gw / 2, tcy - gh / 2, tcy + gh / 2
    iw = np.maximum(0.0, np.minimum(x1, X1) - np.maximum(x0, X0))
    ih = np.maximum(0.0, np.minimum(y1, Y1) - np.maximum(y0, Y0))
    inter = iw * ih
    U = np.maximum(w * h + gw * gh - inter, EPS)
    iwp, ihp = (iw > 0).astype(float), (ih > 0).astype(float)
    a0, a1 = (x0 > X0).astype(float), (x1 < X1).astype(float)
    b0, b1 = (y0 > Y0).astype(float), (y1 < Y1).astype(float)
    dI_dx0, dI_dx1 = ih * (-a0 * iwp), ih * (a1 * iwp)
    dI_dy0, dI_dy1 = iw * (-b0 * ihp), iw * (b1 * ihp)
    dI_dcx, dI_dcy = dI_dx0 + dI_dx1, dI_dy0 + dI_dy1
    dI_dw, dI_dh = 0.5 * (dI_dx1 - dI_dx0), 0.5 * (dI_dy1 - dI_dy0)

    def d_iou(dI, dAp):
        return (dI * U - inter * (dAp - dI)) / (U * U)
    dIoU_dcx, dIoU_dcy = d_iou(dI_dcx, 0.0), d_iou(dI_dcy, 0.0)
    dIoU_dw, dIoU_dh = d_iou(dI_dw, h), d_iou(dI_dh, w)
    rho2 = (cx - tcx) ** 2 + (cy - tcy) ** 2
    cw = np.maximum(x1, X1) - np.minimum(x0, X0)
    ch = np.maximum(y1, Y1) - np.minimum(y0, Y0)
    c2 = np.maximum(cw * cw + ch * ch, EPS)
    e1, e0 = (x1 > X1).astype(float), (x0 < X0).astype(float)
    f1, f0 = (y1 > Y1).astype(float), (y0 < Y0).astype(float)
    dcw_dcx, dcw_dw = (-e0 + e1), 0.5 * (e1 + e0)
    dch_dcy, dch_dh = (-f0 + f1), 0.5 * (f1 + f0)
    dc2_dcx, dc2_dcy = 2 * cw * dcw_dcx, 2 * ch * dch_dcy
    dc2_dw, dc2_dh = 2 * cw * dcw_dw, 2 * ch * dch_dh

    def d_pen(drho2, dc2):
        return (drho2 * c2 - rho2 * dc2) / (c2 * c2)
    dpen_dcx = d_pen(2 * (cx - tcx), dc2_dcx)
    dpen_dcy = d_pen(2 * (cy - tcy), dc2_dcy)
    dpen_dw, dpen_dh = d_pen(0.0, dc2_dw), d_pen(0.0, dc2_dh)
    dL_dcx, dL_dcy = -dIoU_dcx + dpen_dcx, -dIoU_dcy + dpen_dcy
    dL_dw, dL_dh = -dIoU_dw + dpen_dw, -dIoU_dh + dpen_dh
    m = mask[:, None, :, :]
    return np.concatenate([
        dL_dcx * sx * (1 - sx) / gW * m,
        dL_dcy * sy * (1 - sy) / gH * m,
        dL_dw * w * m,
        dL_dh * h * m], axis=1)


def make_runner(B, gH, gW, anchorW=1.0, anchorH=1.0):
    """Emit + compile once; return a f(pred,tgt,mask) -> (loss, d_pred) callable."""
    td = tempfile.mkdtemp()
    mlir = os.path.join(td, "diou_loss_gen.mlir")
    r = subprocess.run([PROBE, str(B), str(gH), str(gW),
                        f"aw={round(anchorW*1000)}", f"ah={round(anchorH*1000)}", mlir],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr); sys.exit("probe emit failed")
    vmfb = os.path.join(td, "diou.vmfb")
    r = subprocess.run([IREE_COMPILE, mlir, "--iree-hal-target-backends=llvm-cpu",
                        "-o", vmfb], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[:3000]); sys.exit("iree-compile failed")
    cfg = rt.Config("local-task")
    ctx = rt.SystemContext(config=cfg)
    with open(vmfb, "rb") as f:
        vm = rt.VmModule.copy_buffer(ctx.instance, f.read())
    ctx.add_vm_module(vm)
    fn = ctx.modules.diou_probe["main"]

    def run(pred, tgt, mask):
        out = fn(pred.astype(np.float32), tgt.astype(np.float32), mask.astype(np.float32))
        return np.asarray(out[0]).item(), np.asarray(out[1]).astype(np.float64)
    return run


def check(B=2, gH=7, gW=7, seed=0, anchorW=1.0, anchorH=1.0):
    rng = np.random.RandomState(seed)
    pred = rng.randn(B, 4, gH, gW).astype(np.float64)
    pred[:, 2:] -= 3.0                       # start w/h small (exp(-3) ~ 0.05)
    tgt = np.zeros((B, 4, gH, gW))
    tgt[:, 0] = rng.rand(B, gH, gW)          # x,y cell offset in [0,1]
    tgt[:, 1] = rng.rand(B, gH, gW)
    tgt[:, 2] = rng.uniform(0.01, 0.08, (B, gH, gW))   # w_rel
    tgt[:, 3] = rng.uniform(0.02, 0.10, (B, gH, gW))   # h_rel
    mask = (rng.rand(B, gH, gW) < 0.5).astype(np.float64)

    # (0) FD-verify the numpy analytic oracle itself, in f64 (the honest gate:
    #     "finite differences don't care what we believe the derivative is").
    #     f64 lets h=1e-6 without roundoff; the emitted f32 vmfb cannot (its FD
    #     noise floor ~ ε·|loss|/2h ~ 1e-2, which is why we check the emitter
    #     against this verified oracle, not against a raw f32 FD).
    npg = np_diou_grad(pred, tgt, mask, gH, gW, anchorW, anchorH)
    hf = 1e-6
    fd = np.zeros_like(pred)
    for idx in np.ndindex(*pred.shape):
        pp = pred.copy(); pp[idx] += hf
        pm = pred.copy(); pm[idx] -= hf
        fd[idx] = (np_diou_forward(pp, tgt, mask, gH, gW, anchorW, anchorH)
                   - np_diou_forward(pm, tgt, mask, gH, gW, anchorW, anchorH)) / (2 * hf)
    oracle_err = np.abs(npg - fd).max()

    run = make_runner(B, gH, gW, anchorW, anchorH)
    loss, dpred = run(pred, tgt, mask)
    ref = np_diou_forward(pred, tgt, mask, gH, gW, anchorW, anchorH)
    frel = abs(loss - ref) / max(abs(ref), 1e-9)
    gmax = np.abs(dpred - npg).max()          # emitted vs verified oracle

    ok = frel < 1e-4 and oracle_err < 1e-5 and gmax < 1e-3
    print(f"DIoU probe  B={B} gH={gH} gW={gW} seed={seed} anchor=({anchorW},{anchorH})")
    print(f"  oracle (numpy grad) vs f64 FD : max_err={oracle_err:.2e}  "
          f"{'PASS' if oracle_err < 1e-5 else 'FAIL'}")
    print(f"  emitted forward  vs numpy     : rel={frel:.2e}  "
          f"{'PASS' if frel < 1e-4 else 'FAIL'}")
    print(f"  emitted backward vs oracle    : max_err={gmax:.2e}  "
          f"{'PASS' if gmax < 1e-3 else 'FAIL'}")
    return ok


def main():
    ok = True
    for seed in (0, 1, 2):
        ok &= check(B=2, gH=7, gW=7, seed=seed)
    ok &= check(B=1, gH=5, gW=6, seed=7)
    # anchor-prior path (brick #2): w = anchorW*exp(tw)
    ok &= check(B=2, gH=7, gW=7, seed=0, anchorW=0.05, anchorH=0.08)
    ok &= check(B=2, gH=7, gW=7, seed=3, anchorW=0.13, anchorH=0.15)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
