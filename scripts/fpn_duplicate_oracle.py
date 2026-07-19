#!/usr/bin/env python3
"""bite 0c of planning/yolo_assignment.md -- duplicate accounting.

How much of the pinned mAP is actually lost to near-miss DUPLICATES rather than
to assignment? Re-score the existing logits with progressively stronger oracles
and read off the ceiling each one buys:

  baseline      : the real scorer -- per-class IoU-NMS at --nms-iou
  centre-NMS r  : suppress any detection whose CENTRE is within r*(1/56) of a
                  higher-scoring one. At 2-5px boxes IoU-NMS cannot see these
                  duplicates at all (measured: 0.41% of ring boxes reach IoU 0.5),
                  so this is the duplicate handling IoU-NMS *wishes* it did.
  oracle-1/GT   : an upper bound no real detector can reach -- keep, for each GT,
                  ONLY the single highest-confidence detection that matches it,
                  and drop every other detection in the image. This is perfect
                  duplicate suppression AND perfect FP rejection combined.

If mAP stays pinned under centre-NMS, duplicates are not what is costing the mAP
and center sampling cannot buy it back by making them mergeable.

Usage: fpn_duplicate_oracle.py <logits.bin> <visdrone448/val.bin> [--fpn data/visdrone]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.yolo_map_visdrone import (  # noqa: E402
    CLASS_NAMES, FPN_GRIDS, average_precision, decode_anchor_raw, iou,
    load_anchors_file, read_gt, set_geometry, _nms_per_class)


def centre_nms(dets, r):
    """Suppress a detection if a higher-scoring detection of the SAME class has a
    centre within r (relative units). Distance-based, not IoU-based."""
    kept = []
    for c in set(d[0] for d in dets):
        cd = sorted((d for d in dets if d[0] == c), key=lambda d: -d[1])
        while cd:
            top = cd.pop(0); kept.append(top)
            tx = (top[2][0] + top[2][2]) / 2; ty = (top[2][1] + top[2][3]) / 2
            cd = [d for d in cd
                  if abs((d[2][0] + d[2][2]) / 2 - tx) > r
                  or abs((d[2][1] + d[2][3]) / 2 - ty) > r]
    return kept


def score(all_dets, gts, n, iou_thr, oracle_1_per_gt=False):
    classes = sorted(CLASS_NAMES)
    confs = {c: [] for c in classes}; tps = {c: [] for c in classes}
    n_gt = {c: 0 for c in classes}
    ca_confs, ca_tps, ca_ngt = [], [], 0
    for r in range(n):
        gt_boxes = gts[r]
        for c in classes:
            n_gt[c] += sum(1 for (cid, _) in gt_boxes if cid == c)
        ca_ngt += len(gt_boxes)
        dets = sorted(all_dets[r], key=lambda d: -d[1])
        if oracle_1_per_gt:
            best = {}
            for (cid, conf, box) in dets:
                for k, (gcid, gbox) in enumerate(gt_boxes):
                    if gcid == cid and iou(box, gbox) >= iou_thr:
                        if k not in best or conf > best[k][1]:
                            best[k] = (cid, conf, box)
                        break
            dets = sorted(best.values(), key=lambda d: -d[1])
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
            is_tp = best_iou >= iou_thr and best_k >= 0
            if is_tp:
                matched[best_k] = True
            confs[cid].append(conf); tps[cid].append(1.0 if is_tp else 0.0)
        ca_matched = [False] * len(gt_boxes)
        for (cid, conf, box) in dets:
            best_iou, best_k = 0.0, -1
            for k, (gcid, gbox) in enumerate(gt_boxes):
                if ca_matched[k]:
                    continue
                iv = iou(box, gbox)
                if iv > best_iou:
                    best_iou, best_k = iv, k
            is_tp = best_iou >= iou_thr and best_k >= 0
            if is_tp:
                ca_matched[best_k] = True
            ca_confs.append(conf); ca_tps.append(1.0 if is_tp else 0.0)
    aps = [average_precision(confs[c], tps[c], n_gt[c]) for c in classes]
    valid = [a for a in aps if a == a]
    return (average_precision(ca_confs, ca_tps, ca_ngt),
            sum(ca_tps) / max(ca_ngt, 1),
            sum(valid) / len(valid) if valid else float("nan"),
            len(ca_confs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits"); ap.add_argument("val_bin")
    ap.add_argument("--fpn", default="data/visdrone")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--conf-thresh", type=float, default=0.001)
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()
    set_geometry(args.grid * 32, args.grid)

    scales = [(g, load_anchors_file(str(Path(args.fpn) / f"anchors_fpn_{p}.txt")))
              for g, p in zip(FPN_GRIDS, ("p3", "p4", "p5"))]
    pred_w = sum(len(a) * 15 * g * g for (g, a) in scales)
    lg = np.fromfile(args.logits, dtype=np.float32)
    npred = lg.size // pred_w
    lg = lg[:npred * pred_w].reshape(npred, pred_w)
    n_gt_rec, gts = read_gt(args.val_bin)
    n = min(npred, n_gt_rec)
    print(f"records: {n}\n")

    raw_dets = []
    for r in range(n):
        dets, off = [], 0
        for (g, anch) in scales:
            ln = len(anch) * 15 * g * g
            dets += decode_anchor_raw(lg[r, off:off + ln], anch, g, args.conf_thresh)
            off += ln
        if len(dets) > args.topk:
            dets = sorted(dets, key=lambda d: -d[1])[:args.topk]
        raw_dets.append(dets)
    print(f"raw dets/img (pre-NMS, top-{args.topk} capped): "
          f"{np.mean([len(d) for d in raw_dets]):.1f}\n")

    print("=" * 78)
    print("0c  DUPLICATE ACCOUNTING -- what does perfect duplicate handling buy?")
    print("=" * 78)
    print(f"  {'variant':<34} {'dets/img':>9} {'recall':>8} {'ca-AP':>8} {'mAP':>9}")
    print("  " + "-" * 74)

    variants = [("baseline: IoU-NMS @%.2f" % args.nms_iou,
                 lambda d: _nms_per_class(d, args.nms_iou), False)]
    for rr in (1, 2, 3):
        variants.append((f"centre-NMS r={rr} cell(s) @P3",
                         (lambda rv: (lambda d: centre_nms(d, rv / 56.0)))(rr), False))
    variants.append(("ORACLE: 1 det per GT, rest dropped",
                     lambda d: _nms_per_class(d, args.nms_iou), True))

    for label, fn, oracle in variants:
        allsets = [fn(raw_dets[r]) for r in range(n)]
        ca, rec, mp, nd = score(allsets, gts, n, args.iou, oracle_1_per_gt=oracle)
        print(f"  {label:<34} {nd/n:>9.1f} {rec:>8.4f} {ca:>8.4f} {mp:>9.4f}")
    print()


if __name__ == "__main__":
    main()
