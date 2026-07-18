#!/usr/bin/env python3
"""mAP@0.5 scorer for the VisDrone single-grid YOLOv1 baseline (WS-A, planning/yolo_drone.md).

Reads a whole-val-set logits dump (from `yolov1-pets-infer 0 data/visdrone <out>`)
plus the matching VisDrone detection-record `val.bin` (produced by
preprocess_visdrone.py — the SAME 157,728-byte record layout as the Pets path,
so the Lean loader/codegen are unchanged) and reports:

  * a CLASS-AGNOSTIC localization AP@0.5 — "did the detector box *anything* in
    the right place, regardless of label", the honest floor for the collapse story;
  * per-class AP@0.5 + mAP over the 10 VisDrone classes.

This is a copy of scripts/yolo_map.py with two changes: the 10-class VisDrone
name map (ids 0..9, matching preprocess_visdrone.py) and the extra class-agnostic
row. The decode is identical: rank by sigmoid(conf logit), class from argmax of
the class slots, per-class greedy NMS.

The single 7x7 grid emits at most ONE box per cell (<=49 detections/image after
NMS) against VisDrone's ~70 GT boxes/image — recall is structurally capped well
below 1.0. That cap is the WS-A point, quantified here.

Usage:
    python3 scripts/yolo_map_visdrone.py <logits.bin> data/visdrone/val.bin [--iou 0.5]
"""
import argparse
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required (.venv/bin/pip install numpy)", file=sys.stderr)
    sys.exit(1)

PER_CELL   = 30          # 2 boxes * 5 + 20 class slots
MAX_BBOXES = 56
BOX_STRIDE = 20

# Geometry defaults = 224/7×7 (WS-A). set_geometry() overrides for the 448/14 rung.
GRID = 7
FLAT = TARGET_BYTES = MASK_BYTES = NB_OFFSET = BOXES_OFFSET = RECORD_SIZE = IMG_BYTES = 0


def set_geometry(size, grid):
    global GRID, FLAT, IMG_BYTES, TARGET_BYTES, MASK_BYTES, NB_OFFSET, BOXES_OFFSET, RECORD_SIZE
    GRID = grid
    FLAT = PER_CELL * GRID * GRID
    IMG_BYTES = 3 * size * size
    TARGET_BYTES = PER_CELL * GRID * GRID * 4
    MASK_BYTES = GRID * GRID * 4
    NB_OFFSET = IMG_BYTES + TARGET_BYTES + MASK_BYTES
    BOXES_OFFSET = NB_OFFSET + 4
    RECORD_SIZE = BOXES_OFFSET + MAX_BBOXES * BOX_STRIDE


set_geometry(224, 7)

# VisDrone kept classes, ids 0..9 (preprocess_visdrone.py remaps file 1..10 -> 0..9).
CLASS_NAMES = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
               5: "truck", 6: "tricycle", 7: "awning-tri", 8: "bus", 9: "motor"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iou(a, b):
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def decode(pred_flat, conf_thresh, nms_iou, box_param="raw"):
    """box_param: 'raw' = the √-MSE head (cx=(j+x)/G, w=w directly);
    'diou' = the positive DIoU parameterization (cx=(j+σ(x))/G, w=exp(w))."""
    pred = pred_flat.reshape(PER_CELL, GRID, GRID)
    dets = []
    for i in range(GRID):
        for j in range(GRID):
            conf = float(sigmoid(pred[4, i, j]))
            if conf < conf_thresh:
                continue
            cid = int(pred[10:30, i, j].argmax())
            x_cell = pred[0, i, j]; y_cell = pred[1, i, j]
            if box_param == "diou":
                cx = (j + float(sigmoid(x_cell))) / GRID
                cy = (i + float(sigmoid(y_cell))) / GRID
                w_rel = float(np.exp(pred[2, i, j]))
                h_rel = float(np.exp(pred[3, i, j]))
            else:
                w_rel = max(float(pred[2, i, j]), 0.0)
                h_rel = max(float(pred[3, i, j]), 0.0)
                cx = (j + x_cell) / GRID
                cy = (i + y_cell) / GRID
            dets.append((cid, conf,
                         [float(cx - w_rel / 2), float(cy - h_rel / 2),
                          float(cx + w_rel / 2), float(cy + h_rel / 2)]))
    kept = []
    for c in set(d[0] for d in dets):
        cd = sorted((d for d in dets if d[0] == c), key=lambda d: -d[1])
        while cd:
            top = cd.pop(0)
            kept.append(top)
            cd = [d for d in cd if iou(top[2], d[2]) < nms_iou]
    return kept


def load_anchors_file(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            w, h = ln.split()
            rows.append((float(w), float(h)))
    return rows


def decode_anchor(pred_flat, anchors, grid, conf_thresh, nms_iou):
    """Decode the A-anchor head [A*15, grid, grid]: per anchor a, box_a =
    anchor_a·exp(tw/th), cx=(j+σ(tx))/grid, obj=σ, class=argmax(softmax)."""
    A = len(anchors); Pn = 15
    pred = pred_flat.reshape(A * Pn, grid, grid)
    dets = []
    for a, (aw, ah) in enumerate(anchors):
        base = a * Pn
        for i in range(grid):
            for j in range(grid):
                obj = float(sigmoid(pred[base + 4, i, j]))
                if obj < conf_thresh:
                    continue
                cl = pred[base + 5:base + 15, i, j]
                cid = int(cl.argmax())
                e = np.exp(cl - cl.max()); clsp = float(e[cid] / e.sum())
                cx = (j + float(sigmoid(pred[base + 0, i, j]))) / grid
                cy = (i + float(sigmoid(pred[base + 1, i, j]))) / grid
                w = aw * float(np.exp(pred[base + 2, i, j]))
                h = ah * float(np.exp(pred[base + 3, i, j]))
                dets.append((cid, obj * clsp,
                             [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    kept = []
    for c in set(d[0] for d in dets):
        cd = sorted((d for d in dets if d[0] == c), key=lambda d: -d[1])
        while cd:
            top = cd.pop(0); kept.append(top)
            cd = [d for d in cd if iou(top[2], d[2]) < nms_iou]
    return kept


FPN_GRIDS = (56, 28, 14)   # P3 / P4 / P5 at 448px input (matches process_split_fpn)


def _nms_per_class(dets, nms_iou):
    kept = []
    for c in set(d[0] for d in dets):
        cd = sorted((d for d in dets if d[0] == c), key=lambda d: -d[1])
        while cd:
            top = cd.pop(0); kept.append(top)
            cd = [d for d in cd if iou(top[2], d[2]) < nms_iou]
    return kept


def decode_anchor_raw(pred_block, anchors, grid, conf_thresh):
    """One scale's anchor decode → raw (cid, conf, xyxy) dets, NO NMS (so the FPN
    caller can merge across scales before a single NMS). Vectorized over all
    A·grid² cells (the scalar per-cell loop is O(millions) at diffuse early-model
    density). `tw,th` are capped at 8 to match the training-time exp cap."""
    A = len(anchors); Pn = 15
    pred = pred_block.reshape(A, Pn, grid, grid).astype(np.float64)
    anch = np.asarray(anchors, dtype=np.float64)                 # [A,2]
    obj = 1.0 / (1.0 + np.exp(-np.clip(pred[:, 4], -60, 60)))     # [A,g,g]
    keep = obj >= conf_thresh
    if not keep.any():
        return []
    cls = pred[:, 5:15]                                           # [A,10,g,g]
    cid = cls.argmax(axis=1)                                      # [A,g,g]
    e = np.exp(cls - cls.max(axis=1, keepdims=True))
    clsp = e.max(axis=1) / e.sum(axis=1)                          # prob of argmax
    conf = obj * clsp
    jj = np.arange(grid).reshape(1, 1, grid)                      # col (W) index
    ii = np.arange(grid).reshape(1, grid, 1)                      # row (H) index
    sx = 1.0 / (1.0 + np.exp(-np.clip(pred[:, 0], -60, 60)))
    sy = 1.0 / (1.0 + np.exp(-np.clip(pred[:, 1], -60, 60)))
    cx = (jj + sx) / grid
    cy = (ii + sy) / grid
    w = anch[:, 0].reshape(A, 1, 1) * np.exp(np.minimum(pred[:, 2], 8.0))
    h = anch[:, 1].reshape(A, 1, 1) * np.exp(np.minimum(pred[:, 3], 8.0))
    x0 = (cx - w / 2)[keep]; y0 = (cy - h / 2)[keep]
    x1 = (cx + w / 2)[keep]; y1 = (cy + h / 2)[keep]
    boxes = np.stack([x0, y0, x1, y1], axis=1).tolist()
    return list(zip(cid[keep].tolist(), conf[keep].tolist(), boxes))


def decode_fpn(pred_flat, scales, conf_thresh, nms_iou, topk=1000):
    """Decode the flat FPN head [Ntot] = [P3|P4|P5]: per-scale anchor decode, then
    MERGE all scales' dets and run one class-wise NMS (planning/yolo_fpn.md bite 8).
    `scales` = [(grid, anchors), ...] in P3,P4,P5 order. Keeps only the top-`topk`
    dets by confidence before NMS — bounds the O(n²) NMS at diffuse early-model
    density (VisDrone has ~70 GT/img, so 1000 is ample headroom)."""
    dets = []
    off = 0
    for (grid, anchors) in scales:
        length = len(anchors) * 15 * grid * grid
        dets += decode_anchor_raw(pred_flat[off:off + length], anchors, grid, conf_thresh)
        off += length
    if len(dets) > topk:
        dets = sorted(dets, key=lambda d: -d[1])[:topk]
    return _nms_per_class(dets, nms_iou)


def read_gt(val_path):
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
    ap.add_argument("val_bin", help="data/visdrone/val.bin")
    ap.add_argument("--iou", type=float, default=0.5, help="TP IoU threshold (default 0.5)")
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--conf-thresh", type=float, default=0.001)
    ap.add_argument("--grid", type=int, default=7, help="detection grid side (7 for 224, 14 for 448)")
    ap.add_argument("--size", type=int, default=None, help="input px (default grid*32)")
    ap.add_argument("--box-param", choices=["raw", "diou"], default="raw",
                    help="'raw' for √-MSE head, 'diou' for the exp/sigmoid DIoU head")
    ap.add_argument("--anchors", default=None,
                    help="anchor priors file → decode the A*15 anchor head (brick #2). "
                         "GT is still read from val_bin with the single-box geometry.")
    ap.add_argument("--fpn", default=None,
                    help="dir with anchors_fpn_{p3,p4,p5}.txt → decode the 3-scale FPN "
                         "head (brick #3): per-scale decode, merge, one NMS. Use --grid 14. "
                         "GT is still read from val_bin with the single-box geometry.")
    args = ap.parse_args()
    set_geometry(args.size if args.size else args.grid * 32, args.grid)

    logits_raw = np.fromfile(args.logits, dtype=np.float32)
    if args.fpn:
        scales = [(g, load_anchors_file(str(Path(args.fpn) / f"anchors_fpn_{p}.txt")))
                  for g, p in zip(FPN_GRIDS, ("p3", "p4", "p5"))]
        pred_w = sum(len(a) * 15 * g * g for (g, a) in scales)
        n_pred = logits_raw.size // pred_w
        logits = logits_raw[: n_pred * pred_w].reshape(n_pred, pred_w)
        decode_fn = lambda row: decode_fpn(row, scales, args.conf_thresh, args.nms_iou)
    elif args.anchors:
        anchor_list = load_anchors_file(args.anchors)
        pred_w = len(anchor_list) * 15 * GRID * GRID
        n_pred = logits_raw.size // pred_w
        logits = logits_raw[: n_pred * pred_w].reshape(n_pred, pred_w)
        decode_fn = lambda row: decode_anchor(row, anchor_list, GRID, args.conf_thresh, args.nms_iou)
    else:
        n_pred = logits_raw.size // FLAT
        logits = logits_raw[: n_pred * FLAT].reshape(n_pred, FLAT)
        decode_fn = lambda row: decode(row, args.conf_thresh, args.nms_iou, args.box_param)

    n_gt_rec, gts = read_gt(args.val_bin)
    n = min(n_pred, n_gt_rec)
    if n_pred != n_gt_rec:
        print(f"WARN: {n_pred} logit rows vs {n_gt_rec} GT records; scoring first {n}",
              file=sys.stderr)

    classes = sorted(CLASS_NAMES)
    confs = {c: [] for c in classes}
    tps   = {c: [] for c in classes}
    n_gt  = {c: 0 for c in classes}
    # Class-agnostic accumulators (label ignored; localization only).
    ca_confs, ca_tps, ca_ngt = [], [], 0

    for r in range(n):
        gt_boxes = gts[r]
        for c in classes:
            n_gt[c] += sum(1 for (cid, _) in gt_boxes if cid == c)
        ca_ngt += len(gt_boxes)

        dets = decode_fn(logits[r])
        dets.sort(key=lambda d: -d[1])

        # Per-class matching.
        matched = [False] * len(gt_boxes)
        for (cid, conf, box) in dets:
            if cid not in confs:
                continue
            best_iou, best_k = 0.0, -1
            for k, (gcid, gbox) in enumerate(gt_boxes):
                if gcid != cid or matched[k]:
                    continue
                iv = iou(box, gbox)
                if iv > best_iou:
                    best_iou, best_k = iv, k
            is_tp = best_iou >= args.iou and best_k >= 0
            if is_tp:
                matched[best_k] = True
            confs[cid].append(conf)
            tps[cid].append(1.0 if is_tp else 0.0)

        # Class-agnostic matching (any label counts, greedy by conf).
        ca_matched = [False] * len(gt_boxes)
        for (cid, conf, box) in dets:
            best_iou, best_k = 0.0, -1
            for k, (gcid, gbox) in enumerate(gt_boxes):
                if ca_matched[k]:
                    continue
                iv = iou(box, gbox)
                if iv > best_iou:
                    best_iou, best_k = iv, k
            is_tp = best_iou >= args.iou and best_k >= 0
            if is_tp:
                ca_matched[best_k] = True
            ca_confs.append(conf)
            ca_tps.append(1.0 if is_tp else 0.0)

    print(f"scored {n} records from {Path(args.val_bin).name}  (IoU={args.iou})")
    ca_ap = average_precision(ca_confs, ca_tps, ca_ngt)
    ca_recall = (sum(ca_tps) / ca_ngt) if ca_ngt else float("nan")
    print(f"  class-agnostic localization AP@{args.iou:.2f} = {ca_ap:.4f}   "
          f"(GT boxes={ca_ngt}, dets={len(ca_confs)}, TP={int(sum(ca_tps))}, "
          f"recall={ca_recall:.4f})")
    aps = []
    for c in classes:
        apc = average_precision(confs[c], tps[c], n_gt[c])
        aps.append(apc)
        print(f"  {CLASS_NAMES[c]:>11}: AP@{args.iou:.2f} = {apc:.4f}   "
              f"(GT={n_gt[c]}, dets={len(confs[c])})")
    valid = [a for a in aps if a == a]
    mean = sum(valid) / len(valid) if valid else float("nan")
    print(f"  mAP@{args.iou:.2f} = {mean:.4f}   (mean over {len(valid)} classes)")


if __name__ == "__main__":
    main()
