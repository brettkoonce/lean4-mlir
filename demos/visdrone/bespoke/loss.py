"""PyTorch twin of `emitMultiScaleYoloLoss` / `emitAnchorYoloLoss` / `emitDiouForward`.

Exact transcription of LeanMlir/MlirCodegen.lean:5130-5596. Verified against the
golden numpy mirrors in scripts/anchor_loss_probe_check.py and
scripts/fpn_loss_breakdown.py.

Per-anchor slot layout (base = a*15):
    +0 tx  +1 ty  +2 tw  +3 th   +4 obj   +5..+15 class logits
The target's objectness channel IS the assignment mask -- there is no separate
mask tensor.

    loss = Σ_scales (1/B) Σ_a [ 5.0·Σ mask·(1-DIoU) + Σ α·w_foc·BCE + Σ mask·w_cls·CE ]

Everything is SUMMED over cells, anchors and scales; the only division is by
batch size. Nothing is normalized by grid area or by the number of positives --
that is why the FPN arm runs at lr 4e-4 rather than the single-scale 7e-4.

TWO DELIBERATE NON-AUTOGRAD CHOICES, both matching the emitted backward:
  * the focal modulating factor w_foc is DETACHED (MlirCodegen.lean:5456-5459
    emits grad = α·w_foc·(p-t)/B, i.e. w_foc held constant).
  * the emitted backward for tw/th has no cap indicator, so above the exp cap it
    keeps a nonzero gradient where clamp_max would zero it. Reproduced here with
    a straight-through clamp so the twin's gradient matches Lean's, not torch's.
"""
import torch
import torch.nn.functional as F

P = 15
NC = 10
LAMBDA_BOX = 5.0
EXP_CAP = 8.0
EPS_UNION = 1e-9
EPS_FOCAL = 1e-12

# T1b sqrt-inverse-frequency class weights (fpnClsWeights), Σ f_c·w_c = 1
CLS_WEIGHTS = [0.8058, 1.4377, 2.1196, 0.5579, 1.3407, 1.7916,
               2.9778, 3.7281, 2.6187, 1.2694]


def _cap_straight_through(t, cap):
    """min(t, cap) in the forward, identity in the backward.

    The emitted backward computes dw/dtw = w with no cap indicator, so a plain
    torch.clamp_max would diverge from Lean exactly where the cap is active.
    """
    return t + (t.clamp(max=cap) - t).detach()


def diou_loss(pred, tgt, mask, aw, ah):
    """pred/tgt: [B,4,g,g] = (tx,ty,tw,th) / (gx,gy,gw,gh). mask: [B,1,g,g].

    Returns the SUM of mask·(1-DIoU). Mirrors emitDiouForward (MlirCodegen:5130).
    """
    B, _, gh, gw = pred.shape
    dev, dt = pred.device, pred.dtype
    cj = torch.arange(gw, device=dev, dtype=dt).view(1, 1, 1, gw)
    ci = torch.arange(gh, device=dev, dtype=dt).view(1, 1, gh, 1)
    inv_w, inv_h = 1.0 / gw, 1.0 / gh

    tx, ty, tw, th = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    gx, gy, gwid, ghgt = tgt[:, 0:1], tgt[:, 1:2], tgt[:, 2:3], tgt[:, 3:4]

    # predicted box, image-normalized
    cx = (cj + torch.sigmoid(tx)) * inv_w
    cy = (ci + torch.sigmoid(ty)) * inv_h
    w = aw * torch.exp(_cap_straight_through(tw, EXP_CAP))
    h = ah * torch.exp(_cap_straight_through(th, EXP_CAP))
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2

    # target box: centre is cell-relative, but w/h are RAW image-normalized
    tcx = (cj + gx) * inv_w
    tcy = (ci + gy) * inv_h
    X0, X1 = tcx - gwid / 2, tcx + gwid / 2
    Y0, Y1 = tcy - ghgt / 2, tcy + ghgt / 2

    iw = (torch.min(x1, X1) - torch.max(x0, X0)).clamp(min=0)
    ih = (torch.min(y1, Y1) - torch.max(y0, Y0)).clamp(min=0)
    inter = iw * ih
    union = (w * h + gwid * ghgt - inter).clamp(min=EPS_UNION)
    iou = inter / union

    rho2 = (cx - tcx) ** 2 + (cy - tcy) ** 2
    cw = torch.max(x1, X1) - torch.min(x0, X0)
    ch = torch.max(y1, Y1) - torch.min(y0, Y0)
    c2 = (cw ** 2 + ch ** 2).clamp(min=EPS_UNION)

    return ((1.0 - (iou - rho2 / c2)) * mask).sum()


def anchor_yolo_loss(pred, tgt, anchors, gamma=2.0, cls_weights=None,
                     breakdown=None):
    """pred/tgt: [B, A*15, g, g]. Returns the SUM (caller divides by B)."""
    total = pred.new_zeros(())
    for a, (aw, ah) in enumerate(anchors):
        base = a * P
        p, t = pred[:, base:base + P], tgt[:, base:base + P]
        mask = t[:, 4:5]                                     # obj channel IS the mask

        box = LAMBDA_BOX * diou_loss(p[:, 0:4], t[:, 0:4], mask, aw, ah)

        # objectness: alpha-weighted focal BCE over ALL cells, detached modulator
        z, tobj = p[:, 4:5], t[:, 4:5]
        prob = torch.sigmoid(z)
        pt = tobj * prob + (1 - tobj) * (1 - prob)
        w_foc = ((1 - pt).clamp(min=EPS_FOCAL) ** gamma).detach()
        alpha = tobj + (1 - tobj) * 0.5
        bce = z.clamp(min=0) - z * tobj + torch.log1p(torch.exp(-z.abs()))
        obj = (alpha * w_foc * bce).sum()

        # class: softmax CE masked to positives, weighted by the GT class weight
        lsm = F.log_softmax(p[:, 5:5 + NC], dim=1)
        onehot = t[:, 5:5 + NC]
        per_cell = (onehot * lsm).sum(dim=1, keepdim=True) * mask
        if cls_weights is not None:
            w = torch.as_tensor(cls_weights, device=pred.device, dtype=pred.dtype)
            w_cell = (onehot * w.view(1, NC, 1, 1)).sum(dim=1, keepdim=True)
            per_cell = per_cell * w_cell
        cls = -per_cell.sum()

        total = total + box + obj + cls
        if breakdown is not None:
            with torch.no_grad():
                breakdown["box"] += box.item()
                breakdown["obj"] += obj.item()
                breakdown["cls"] += cls.item()
                breakdown["obj_pos"] += (alpha * w_foc * bce * tobj).sum().item()
                breakdown["obj_neg"] += (alpha * w_foc * bce * (1 - tobj)).sum().item()
                breakdown["npos"] += mask.sum().item()
    return total


def fpn_loss(logits, targets, scales, gamma=2.0, cls_weights=CLS_WEIGHTS,
             breakdown=None):
    """logits/targets: [B, NTOT] laid out [P3|P4|P5], each C-order [A*15, g, g]."""
    B = logits.shape[0]
    off = 0
    total = logits.new_zeros(())
    for g, anchors in scales:
        n = len(anchors) * P * g * g
        p = logits[:, off:off + n].view(B, len(anchors) * P, g, g)
        t = targets[:, off:off + n].view(B, len(anchors) * P, g, g)
        total = total + anchor_yolo_loss(p, t, anchors, gamma, cls_weights,
                                         breakdown)
        off += n
    assert off == logits.shape[1], f"target layout mismatch: {off} vs {logits.shape[1]}"
    # NOTE: breakdown accumulates RAW SUMS across calls. The caller divides by
    # (num_batches * B) to get a per-image figure. Dividing by B here would
    # re-divide the running total on every batch and silently collapse it.
    return total / B
