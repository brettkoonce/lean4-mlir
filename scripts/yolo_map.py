#!/usr/bin/env python3
"""mAP@0.5 scorer for the YOLOv1 Pets detector (Workstream A, planning/yolo_demo_v2.md).

Reads a whole-val-set logits dump (from `yolov1-pets-infer 0 <data> <out>`) plus
the matching detection-record `val.bin`, and reports per-class AP@0.5 + mAP over
the entire set — the honest metric that replaces v1's hand-counted "64/64".

Decode fix (v2 Workstream A item 4): rank detections by sigmoid(conf logit)
(the objectness the focal-BCE head actually shaped), class from argmax of the
class slots. The old obj×class product was dog-biased in ranking; dropped here.

Record layout (157,728 bytes/record after a 4-byte <I count header), matching
preprocess_pets_det.py / preprocess_pets_mosaic.py:
    [0        , 150528)  image      uint8  3x224x224
    [150528   , 156408)  target     f32    30x7x7   (unused here)
    [156408   , 156604)  mask       f32    7x7      (unused here)
    [156604   , 156608)  nb         int32           (# GT boxes, <=56)
    [156608   , 157728)  boxes      56x20 bytes: int32 cid + 4x f32 (xmin,ymin,
                                    xmax,ymax), all normalized to [0,1]

Usage:
    python3 scripts/yolo_map.py <logits.bin> <val.bin> [--iou 0.5] [--conf-thresh 0.001]
"""
import argparse
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required (.venv/bin/pip install numpy)", file=sys.stderr)
    sys.exit(1)

GRID = 7
PER_CELL = 30            # 2 boxes * 5 + 20 classes
FLAT = PER_CELL * GRID * GRID  # 1470

# Record geometry.
IMG_BYTES    = 3 * 224 * 224          # 150528 uint8
TARGET_BYTES = PER_CELL * GRID * GRID * 4  # 5880
MASK_BYTES   = GRID * GRID * 4        # 196
NB_OFFSET    = IMG_BYTES + TARGET_BYTES + MASK_BYTES     # 156604
BOXES_OFFSET = NB_OFFSET + 4                             # 156608
MAX_BBOXES   = 56
BOX_STRIDE   = 20                     # int32 cid + 4x f32
RECORD_SIZE  = BOXES_OFFSET + MAX_BBOXES * BOX_STRIDE    # 157728
assert RECORD_SIZE == 157728, RECORD_SIZE

# Only cat/dog appear in Pets (mapped onto VOC ids so labels line up).
CLASS_NAMES = {7: "cat", 11: "dog"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iou(a, b):
    """IoU of two [xmin, ymin, xmax, ymax] boxes."""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def decode(pred_flat, conf_thresh, nms_iou):
    """Decode one [1470] prediction → list of (cid, conf, [xmin,ymin,xmax,ymax]).

    conf = sigmoid(objectness logit); class = argmax(class slots). Per-class
    greedy NMS at nms_iou. conf_thresh is kept low so the PR curve is complete
    (AP integrates over all operating points)."""
    pred = pred_flat.reshape(PER_CELL, GRID, GRID)
    dets = []
    for i in range(GRID):
        for j in range(GRID):
            conf = float(sigmoid(pred[4, i, j]))
            if conf < conf_thresh:
                continue
            cid = int(pred[10:30, i, j].argmax())
            x_cell = pred[0, i, j]; y_cell = pred[1, i, j]
            w_rel = max(float(pred[2, i, j]), 0.0)
            h_rel = max(float(pred[3, i, j]), 0.0)
            cx = (j + x_cell) / GRID
            cy = (i + y_cell) / GRID
            dets.append((cid, conf,
                         [float(cx - w_rel / 2), float(cy - h_rel / 2),
                          float(cx + w_rel / 2), float(cy + h_rel / 2)]))
    # Per-class greedy NMS.
    kept = []
    for c in set(d[0] for d in dets):
        cd = sorted((d for d in dets if d[0] == c), key=lambda d: -d[1])
        while cd:
            top = cd.pop(0)
            kept.append(top)
            cd = [d for d in cd if iou(top[2], d[2]) < nms_iou]
    return kept


def read_gt(val_path):
    """Return (n_records, list-per-record of [(cid, box)])."""
    raw = np.fromfile(val_path, dtype=np.uint8)
    n = int(np.frombuffer(raw[:4], dtype="<u4")[0])
    gts = []
    for r in range(n):
        base = 4 + r * RECORD_SIZE
        nb = int(np.frombuffer(raw[base + NB_OFFSET: base + NB_OFFSET + 4], dtype="<i4")[0])
        boxes = []
        blk = base + BOXES_OFFSET
        for k in range(min(nb, MAX_BBOXES)):
            off = blk + k * BOX_STRIDE
            cid = int(np.frombuffer(raw[off: off + 4], dtype="<i4")[0])
            xyxy = np.frombuffer(raw[off + 4: off + 20], dtype="<f4").astype(float).tolist()
            boxes.append((cid, xyxy))
        gts.append(boxes)
    return n, gts


def average_precision(confs, tps, n_gt):
    """All-point (VOC2010+) AP from parallel confidence / TP-flag arrays."""
    if n_gt == 0:
        return float("nan")
    if not confs:
        return 0.0
    order = np.argsort(-np.asarray(confs))
    tp = np.asarray(tps, dtype=np.float64)[order]
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # Monotone-decreasing precision envelope, integrate over recall.
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logits", help="logits.bin dump ([N,1470] f32) from yolov1-pets-infer")
    ap.add_argument("val_bin", help="matching detection-record val.bin")
    ap.add_argument("--iou", type=float, default=0.5, help="TP IoU threshold (default 0.5)")
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--conf-thresh", type=float, default=0.001,
                    help="min sigmoid(conf) to consider a detection (low = full PR curve)")
    args = ap.parse_args()

    logits = np.fromfile(args.logits, dtype=np.float32)
    n_pred = logits.size // FLAT
    logits = logits[: n_pred * FLAT].reshape(n_pred, FLAT)

    n_gt_rec, gts = read_gt(args.val_bin)
    n = min(n_pred, n_gt_rec)
    if n_pred != n_gt_rec:
        print(f"WARN: {n_pred} logit rows vs {n_gt_rec} GT records; scoring first {n}",
              file=sys.stderr)

    # Per-class accumulators over the whole set.
    classes = sorted(CLASS_NAMES)
    confs = {c: [] for c in classes}
    tps   = {c: [] for c in classes}
    n_gt  = {c: 0 for c in classes}

    for r in range(n):
        gt_boxes = gts[r]
        for c in classes:
            n_gt[c] += sum(1 for (cid, _) in gt_boxes if cid == c)
        matched = [False] * len(gt_boxes)
        dets = decode(logits[r], args.conf_thresh, args.nms_iou)
        dets.sort(key=lambda d: -d[1])         # high confidence first
        for (cid, conf, box) in dets:
            if cid not in confs:               # non cat/dog (shouldn't happen)
                continue
            best_iou, best_k = 0.0, -1
            for k, (gcid, gbox) in enumerate(gt_boxes):
                if gcid != cid or matched[k]:
                    continue
                i = iou(box, gbox)
                if i > best_iou:
                    best_iou, best_k = i, k
            is_tp = best_iou >= args.iou and best_k >= 0
            if is_tp:
                matched[best_k] = True
            confs[cid].append(conf)
            tps[cid].append(1.0 if is_tp else 0.0)

    print(f"scored {n} records from {Path(args.val_bin).name}  (IoU={args.iou})")
    aps = []
    for c in classes:
        apc = average_precision(confs[c], tps[c], n_gt[c])
        aps.append(apc)
        ndet = len(confs[c])
        print(f"  {CLASS_NAMES[c]:>3}: AP@{args.iou:.2f} = {apc:.4f}   "
              f"(GT={n_gt[c]}, dets={ndet})")
    valid = [a for a in aps if a == a]         # drop NaN (no-GT) classes
    mean = sum(valid) / len(valid) if valid else float("nan")
    print(f"  mAP@{args.iou:.2f} = {mean:.4f}")


if __name__ == "__main__":
    main()
