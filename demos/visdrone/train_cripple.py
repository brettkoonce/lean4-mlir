#!/usr/bin/env python3
"""Bite 0 of planning/yolo_scoring.md: CRIPPLE THE DETECTOR THAT WORKS.

The thread's four training arms each tried to fix the broken detector and got no
signal until 4.5 h later. This goes the other way: start from a measured working
control and add ONE piece of the Lean detector's scoring design at a time. The
rung where the number craters names the cause, and each rung is ~25 min.

Control = runs/scratch_squash_12ep_448 (YOLOv8s, random init, no aug, 448 squash,
12 ep) -> mAP@0.5 = 0.1399.

The Lean scoring design differs from YOLOv8's on two separable axes, and each is
a one-line patch to the loss -- no architecture surgery, so nothing else moves:

  --cripple nobg
      The class loss is masked to foreground anchors. Background anchors get NO
      classification gradient, so nothing ever teaches the class score to be
      small on background. This is the Lean class head's defect exactly: its
      softmax CE is multiplied by `%{pa}_cls_maskb` (a broadcast of the target's
      own objectness channel) in MlirCodegen, on both loss and gradient.
      The fg mask is derived from the target itself -- TAL writes nonzero
      target_scores only on assigned anchors -- so no state has to be threaded
      through. The normalizer `target_scores_sum` is already foreground-only, so
      masking the loss does not change it and the comparison stays clean.

  --cripple hardtarget
      TAL's quality-weighted soft target is binarized to a hard 1.0, so every
      assigned anchor is trained toward "definitely an object" regardless of how
      good its box is. This is the Lean objectness target exactly (constant 1.0).
      Patched at the ASSIGNER, not in the loss, so the loss AND its normalizer
      AND the box loss's per-anchor weighting all see the hard target
      consistently -- binarizing inside the BCE alone would silently rescale the
      cls gain and confound the result.

PREDICTION, recorded before running (planning/yolo_scoring.md):
    nobg       craters to < 0.03   <-- the load-bearing ingredient
    hardtarget costs something, but far less

If `nobg` does NOT crater by >=5x, the diagnosis in that doc is wrong: stop and
re-measure rather than proceeding to the port.

    HIP_VISIBLE_DEVICES=0 ./.venv/bin/python3 -u train_cripple.py --cripple nobg
"""
import argparse
from pathlib import Path

import torch
from ultralytics.utils import loss as ul_loss
from ultralytics.utils import tal as ul_tal


def patch_nobg():
    """Mask the classification loss to foreground anchors."""
    orig_init = ul_loss.v8DetectionLoss.__init__

    def init(self, model, tal_topk=10, tal_topk2=None):
        orig_init(self, model, tal_topk, tal_topk2)
        base = self.bce

        def masked_bce(pred, target):
            raw = base(pred, target)
            # TAL writes nonzero target_scores ONLY on assigned anchors, so the
            # target doubles as the foreground mask -- no plumbing required.
            fg = (target.sum(-1, keepdim=True) > 0).to(raw.dtype)
            return raw * fg

        self.bce = masked_bce

    ul_loss.v8DetectionLoss.__init__ = init
    print("CRIPPLE: class loss masked to foreground (no background training)")


def patch_hardtarget():
    """Binarize TAL's quality-weighted target to a constant 1.0."""
    orig_fwd = ul_tal.TaskAlignedAssigner.forward

    def fwd(self, *a, **k):
        tl, tb, ts, fg, tgi = orig_fwd(self, *a, **k)
        return tl, tb, (ts > 0).to(ts.dtype), fg, tgi

    ul_tal.TaskAlignedAssigner.forward = fwd
    print("CRIPPLE: target scores binarized to 1.0 (no quality awareness)")


def patch_staticassign():
    """Replace TAL with the Lean encoder's STATIC assignment.

    Lean assigns each GT to exactly ONE slot: pick the FPN level by object size
    (max(w,h) px vs thresholds 24/64), then the cell containing the GT centre,
    and let a later GT overwrite an earlier one on a collision. TAL instead
    picks a dynamic top-k of anchors per GT, ranked by an alignment metric that
    mixes predicted class score with predicted-box IoU.

    Reproduced here exactly, including the last-write-wins collision rule, so
    the only thing that changes versus the control is WHICH anchors are
    positive. YOLOv8 is anchor-free (one slot per cell), so the Lean design's 3
    anchors per cell are not modelled -- that is rung L5, not this one.

    The grid layout is derived at RUNTIME, not hardcoded: training runs at
    imgsz (A=4116 at 448) but ultralytics' validator letterbox-pads to the next
    stride multiple, so the same assigner is called with A=4725 (480px) during
    val. Hardcoding 448 crashes at the first validation pass, after a full epoch
    of training has already been spent.
    """
    T_LO, T_HI = 24.0, 64.0
    _cache = {}

    def layout(A, anc_points):
        """(grids, strides, offsets) for a 3-level 1:2:4 pyramid totalling A."""
        if A not in _cache:
            # A = g^2 + (g/2)^2 + (g/4)^2 = g^2 * 21/16
            g0 = int(round((A * 16 / 21) ** 0.5))
            grids = (g0, g0 // 2, g0 // 4)
            assert sum(g * g for g in grids) == A, f"cannot factor A={A} into a 1:2:4 pyramid"
            offs = (0, grids[0] ** 2, grids[0] ** 2 + grids[1] ** 2)
            # P3 is row-major in j, so consecutive anchors differ by one stride
            s0 = float(anc_points[1, 0] - anc_points[0, 0])
            _cache[A] = (grids, (s0, s0 * 2, s0 * 4), offs)
            print(f"  static assigner layout: A={A} grids={grids} strides={_cache[A][1]}")
        return _cache[A]

    def fwd(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        B, A, NC = pd_scores.shape
        GRIDS, STRIDES, OFFSETS = layout(A, anc_points)
        dev = pd_scores.device
        tgt_lab = torch.full((B, A), NC, dtype=torch.long, device=dev)
        tgt_box = torch.zeros((B, A, 4), dtype=gt_bboxes.dtype, device=dev)
        tgt_scr = torch.zeros((B, A, NC), dtype=pd_scores.dtype, device=dev)
        fg = torch.zeros((B, A), dtype=torch.bool, device=dev)
        tgt_idx = torch.zeros((B, A), dtype=torch.long, device=dev)

        for b in range(B):
            valid = mask_gt[b, :, 0].nonzero().flatten()
            for gi in valid.tolist():
                x0, y0, x1, y1 = gt_bboxes[b, gi].tolist()
                w, h = x1 - x0, y1 - y0
                if w <= 0 or h <= 0:
                    continue
                m = max(w, h)
                lvl = 0 if m < T_LO else (1 if m < T_HI else 2)
                g, s = GRIDS[lvl], STRIDES[lvl]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                j = min(int(cx / s), g - 1)
                i = min(int(cy / s), g - 1)
                a = OFFSETS[lvl] + i * g + j
                # last write wins, exactly as encode_targets_fpn does
                tgt_scr[b, a] = 0.0
                tgt_lab[b, a] = gt_labels[b, gi, 0].long()
                tgt_box[b, a] = gt_bboxes[b, gi]
                tgt_scr[b, a, gt_labels[b, gi, 0].long()] = 1.0
                fg[b, a] = True
                tgt_idx[b, a] = gi
        return tgt_lab, tgt_box, tgt_scr, fg, tgt_idx

    ul_tal.TaskAlignedAssigner.forward = fwd
    print("CRIPPLE: static best-cell assignment (1 slot/GT, last-write-wins)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cripple",
                    choices=["none", "nobg", "hardtarget", "both", "staticassign",
                             "staticassign_nobg"],
                    required=True)
    ap.add_argument("--model", default="yolov8s.yaml")
    ap.add_argument("--data", default="data/visdrone_squash.yaml")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.cripple in ("nobg", "both", "staticassign_nobg"):
        patch_nobg()
    if args.cripple in ("hardtarget", "both"):
        patch_hardtarget()
    if args.cripple in ("staticassign", "staticassign_nobg"):
        patch_staticassign()

    # import AFTER patching so the trainer picks up the patched classes
    from ultralytics import YOLO

    data = Path(args.data).resolve()
    if not data.exists():
        raise SystemExit(f"missing {data} -- run make_squash_dataset.py first")

    YOLO(args.model).train(
        data=str(data),
        project=str(Path("runs").resolve()),
        name=args.name or f"cripple_{args.cripple}_12ep_448",
        exist_ok=True,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=8,
        amp=False,
        val=True,
        plots=True,
        seed=0,
        # identical to the control arm
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
        scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
        erasing=0.0, auto_augment=None, close_mosaic=0,
    )


if __name__ == "__main__":
    main()
