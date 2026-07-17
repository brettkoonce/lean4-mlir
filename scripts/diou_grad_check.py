#!/usr/bin/env python3
"""Standalone DIoU box-loss forward + analytic gradient, central-FD verified.

Brick #1 of the detection-infra build (planning/yolo_drone.md WS-D). The single
biggest weakness of the current YOLOv1 head is its box regression: √-MSE on
(x,y,w,h) with an ε-floor that has no positive constraint (w/h can go negative
and recover only slowly) and no notion of overlap. This is the reference for the
replacement — an IoU-family loss with a positive box parameterization — verified
here in numpy BEFORE it is emitted as StableHLO, so the hand-derived VJP that
goes into the codegen has a numeric ground truth (mirrors scripts/
seg_loss_probe_check.py: "finite differences don't care what we believe the
derivative is, which is the point").

Box parameterization (positive by construction), per foreground cell (i,j):
    cx = (j + sigmoid(tx)) / gW      cy = (i + sigmoid(ty)) / gH
    w  = exp(tw)                     h  = exp(th)          # always > 0
so the head's raw outputs (tx,ty,tw,th) are unconstrained, and the box is always
valid. Loss = mean over foreground cells of (1 - DIoU), where

    DIoU = IoU - rho2 / c2
    IoU  = inter / union
    rho2 = (cx-CX)^2 + (cy-CY)^2                      # center distance^2
    c2   = enclosing_w^2 + enclosing_h^2              # smallest enclosing box diag^2

The analytic gradient d(loss)/d(tx,ty,tw,th) is hand-derived below and checked
against central finite differences of the forward. Run:
    python3 scripts/diou_grad_check.py
"""
import numpy as np

np.random.seed(0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_boxes(t, cj, ci, gW, gH):
    """raw params t=[N,4] (tx,ty,tw,th) -> center-size + xyxy, all [N]."""
    sx, sy = sigmoid(t[:, 0]), sigmoid(t[:, 1])
    cx = (cj + sx) / gW
    cy = (ci + sy) / gH
    w = np.exp(t[:, 2])
    h = np.exp(t[:, 3])
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return cx, cy, w, h, x0, y0, x1, y1, sx, sy


def diou_loss_forward(t, gt, cj, ci, gW, gH):
    """Mean (1 - DIoU) over N cells. gt=[N,4] target (cx,cy,w,h) normalized."""
    cx, cy, w, h, x0, y0, x1, y1, _, _ = build_boxes(t, cj, ci, gW, gH)
    CX, CY, W, H = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    X0, Y0, X1, Y1 = CX - W / 2, CY - H / 2, CX + W / 2, CY + H / 2
    iw = np.maximum(0.0, np.minimum(x1, X1) - np.maximum(x0, X0))
    ih = np.maximum(0.0, np.minimum(y1, Y1) - np.maximum(y0, Y0))
    inter = iw * ih
    union = w * h + W * H - inter
    iou = inter / np.maximum(union, 1e-12)
    rho2 = (cx - CX) ** 2 + (cy - CY) ** 2
    cw = np.maximum(x1, X1) - np.minimum(x0, X0)
    ch = np.maximum(y1, Y1) - np.minimum(y0, Y0)
    c2 = cw * cw + ch * ch
    diou = iou - rho2 / np.maximum(c2, 1e-12)
    return np.mean(1.0 - diou)


def diou_loss_grad(t, gt, cj, ci, gW, gH):
    """Analytic d(mean(1-DIoU))/d t, shape [N,4]."""
    N = t.shape[0]
    cx, cy, w, h, x0, y0, x1, y1, sx, sy = build_boxes(t, cj, ci, gW, gH)
    CX, CY, W, H = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    X0, Y0, X1, Y1 = CX - W / 2, CY - H / 2, CX + W / 2, CY + H / 2

    iw = np.maximum(0.0, np.minimum(x1, X1) - np.maximum(x0, X0))
    ih = np.maximum(0.0, np.minimum(y1, Y1) - np.maximum(y0, Y0))
    inter = iw * ih
    union = w * h + W * H - inter
    U = np.maximum(union, 1e-12)

    # indicators for the piecewise min/max (sub-gradients)
    iw_pos = (iw > 0).astype(float)
    ih_pos = (ih > 0).astype(float)
    a0 = (x0 > X0).astype(float)          # max(x0,X0) picks x0
    a1 = (x1 < X1).astype(float)          # min(x1,X1) picks x1
    b0 = (y0 > Y0).astype(float)
    b1 = (y1 < Y1).astype(float)

    # d inter / d{x0,x1,y0,y1}
    diw_dx0 = -a0 * iw_pos
    diw_dx1 = a1 * iw_pos
    dih_dy0 = -b0 * ih_pos
    dih_dy1 = b1 * ih_pos
    dI_dx0 = ih * diw_dx0
    dI_dx1 = ih * diw_dx1
    dI_dy0 = iw * dih_dy0
    dI_dy1 = iw * dih_dy1

    # corners -> center-size: x0=cx-w/2, x1=cx+w/2
    dI_dcx = dI_dx0 + dI_dx1
    dI_dcy = dI_dy0 + dI_dy1
    dI_dw = -0.5 * dI_dx0 + 0.5 * dI_dx1
    dI_dh = -0.5 * dI_dy0 + 0.5 * dI_dy1
    # area_p = w*h
    dAp_dw, dAp_dh = h, w

    # dIoU = (dI*U - I*dU)/U^2 ; dU = dAp - dI
    def d_iou(dI, dAp):
        dU = dAp - dI
        return (dI * U - inter * dU) / (U * U)

    dIoU_dcx = d_iou(dI_dcx, 0.0)
    dIoU_dcy = d_iou(dI_dcy, 0.0)
    dIoU_dw = d_iou(dI_dw, dAp_dw)
    dIoU_dh = d_iou(dI_dh, dAp_dh)

    # rho2/c2 term
    rho2 = (cx - CX) ** 2 + (cy - CY) ** 2
    cw = np.maximum(x1, X1) - np.minimum(x0, X0)
    ch = np.maximum(y1, Y1) - np.minimum(y0, Y0)
    c2 = np.maximum(cw * cw + ch * ch, 1e-12)
    drho2_dcx = 2.0 * (cx - CX)
    drho2_dcy = 2.0 * (cy - CY)
    # enclosing: cw = max(x1,X1) - min(x0,X0)
    e1 = (x1 > X1).astype(float)          # max picks x1
    e0 = (x0 < X0).astype(float)          # min picks x0
    f1 = (y1 > Y1).astype(float)
    f0 = (y0 < Y0).astype(float)
    dcw_dx1, dcw_dx0 = e1, -e0
    dch_dy1, dch_dy0 = f1, -f0
    dcw_dcx = dcw_dx0 + dcw_dx1
    dcw_dw = -0.5 * dcw_dx0 + 0.5 * dcw_dx1
    dch_dcy = dch_dy0 + dch_dy1
    dch_dh = -0.5 * dch_dy0 + 0.5 * dch_dy1
    dc2_dcx = 2.0 * cw * dcw_dcx
    dc2_dcy = 2.0 * ch * dch_dcy
    dc2_dw = 2.0 * cw * dcw_dw
    dc2_dh = 2.0 * ch * dch_dh

    # d(rho2/c2) = (drho2*c2 - rho2*dc2)/c2^2
    def d_pen(drho2, dc2):
        return (drho2 * c2 - rho2 * dc2) / (c2 * c2)

    dpen_dcx = d_pen(drho2_dcx, dc2_dcx)
    dpen_dcy = d_pen(drho2_dcy, dc2_dcy)
    dpen_dw = d_pen(0.0, dc2_dw)
    dpen_dh = d_pen(0.0, dc2_dh)

    # loss = 1 - IoU + rho2/c2 ; d loss/dθ = -dIoU + dpen
    dL_dcx = -dIoU_dcx + dpen_dcx
    dL_dcy = -dIoU_dcy + dpen_dcy
    dL_dw = -dIoU_dw + dpen_dw
    dL_dh = -dIoU_dh + dpen_dh

    # chain to raw params: cx=(j+sigmoid(tx))/gW -> dcx/dtx = sx(1-sx)/gW ; w=exp(tw) -> dw/dtw = w
    dcx_dtx = sx * (1 - sx) / gW
    dcy_dty = sy * (1 - sy) / gH
    g = np.zeros_like(t)
    g[:, 0] = dL_dcx * dcx_dtx
    g[:, 1] = dL_dcy * dcy_dty
    g[:, 2] = dL_dw * w
    g[:, 3] = dL_dh * h
    return g / N          # mean over cells


def main():
    N, gW, gH = 200, 14, 14
    # random foreground cells + raw params; targets = small VisDrone-like boxes
    cj = np.random.randint(0, gW, N).astype(float)
    ci = np.random.randint(0, gH, N).astype(float)
    t = np.random.randn(N, 4) * 0.8
    t[:, 2:] -= 3.0                       # start w/h small (exp(-3)~0.05), realistic
    gt = np.zeros((N, 4))
    gt[:, 0] = (cj + np.random.rand(N)) / gW
    gt[:, 1] = (ci + np.random.rand(N)) / gH
    gt[:, 2] = np.random.uniform(0.01, 0.08, N)
    gt[:, 3] = np.random.uniform(0.02, 0.10, N)

    ana = diou_loss_grad(t, gt, cj, ci, gW, gH)

    # PER-ELEMENT central FD (not a column sum — a per-cell error can cancel in
    # the sum and pass a weaker check). Each cell's box is independent, so
    # perturbing t[i,k] only moves that cell's (1-DIoU) term.
    h = 1e-6
    fd = np.zeros_like(t)
    for i in range(N):
        for k in range(4):
            tp = t.copy(); tp[i, k] += h
            tm = t.copy(); tm[i, k] -= h
            fd[i, k] = (diou_loss_forward(tp, gt, cj, ci, gW, gH)
                        - diou_loss_forward(tm, gt, cj, ci, gW, gH)) / (2 * h)
    err = np.abs(ana - fd)
    print("DIoU box-loss gradient check (analytic vs per-element central FD)")
    print(f"  cells N={N}, grid {gH}x{gW}")
    for k, nm in enumerate(["tx", "ty", "tw", "th"]):
        print(f"  d loss / d {nm}:  max_abs_err={err[:, k].max():.2e}")
    ok = err.max() < 1e-7
    print(f"  max abs err = {err.max():.2e}  ->  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
