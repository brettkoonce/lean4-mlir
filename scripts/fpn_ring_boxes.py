#!/usr/bin/env python3
"""bites 0b + 0c of planning/yolo_assignment.md, on the existing e12 logits.

0b -- THE NMS QUESTION. Center sampling only helps if the boxes emitted by the
newly-positive ring cells MERGE with the centre cell's box under NMS. At 2-5px
VisDrone box sizes that is doubtful, and there is a hard geometric reason: the
decode is cx=(j+sigma(tx))/g, so a cell's predicted centre is confined STRICTLY
INSIDE ITSELF. A ring cell one step away is >=1 cell (8px at P3) from the GT
centre, while the GT box is 2-5px wide -- the two boxes cannot even touch, so
their IoU is 0 and NMS can never merge them.

Reported: IoU(ring box, centre box) and IoU(ring box, GT) distributions, plus
the fraction of ring cells that could EVER reach IoU>=0.5 with the GT (a pure
geometry bound: best case over centre-in-cell and free w/h).

0c -- DUPLICATE ACCOUNTING. How much of the mAP loss is duplicate-driven? Re-score
with a spatial-cluster oracle that keeps only the highest-objectness detection per
GT neighbourhood (suppressing by CENTRE DISTANCE, which catches the near-duplicates
that IoU-NMS misses at tiny box sizes). The gap to the real mAP upper-bounds what
perfect duplicate handling alone would buy, independent of assignment.

Usage: fpn_ring_boxes.py <logits.bin> <fpn_val.bin> [--anchors data/visdrone]
                         [--gt data/visdrone448/val.bin]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

P = 15
GRIDS = (56, 28, 14)


def load_anchors_file(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            w, h = ln.split()
            rows.append((float(w), float(h)))
    return rows


def iou_cwh(b1, b2):
    """b*: [...,4] as (cx,cy,w,h) -> elementwise IoU."""
    ix = (np.minimum(b1[..., 0] + b1[..., 2] / 2, b2[..., 0] + b2[..., 2] / 2)
          - np.maximum(b1[..., 0] - b1[..., 2] / 2, b2[..., 0] - b2[..., 2] / 2))
    iy = (np.minimum(b1[..., 1] + b1[..., 3] / 2, b2[..., 1] + b2[..., 3] / 2)
          - np.maximum(b1[..., 1] - b1[..., 3] / 2, b2[..., 1] - b2[..., 3] / 2))
    ix = np.clip(ix, 0, None); iy = np.clip(iy, 0, None)
    inter = ix * iy
    ua = b1[..., 2] * b1[..., 3] + b2[..., 2] * b2[..., 3] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-12), 0.0)


def best_reachable_iou(gt, ci, cj, g, span):
    """Best IoU with gt=(cx,cy,w,h) obtainable from cell (ci,cj): centre clamped
    into the (span-widened) cell, w/h free."""
    cx, cy, w, h = gt[..., 0], gt[..., 1], gt[..., 2], gt[..., 3]
    px = np.clip(cx, (cj - span) / g, (cj + 1 + span) / g)
    py = np.clip(cy, (ci - span) / g, (ci + 1 + span) / g)
    best = np.zeros_like(cx)
    for m in (1.0, 1.25, 1.5, 2.0, 3.0):
        cand = iou_cwh(np.stack([px, py, w * m, h * m], -1),
                       np.stack([cx, cy, w, h], -1))
        best = np.maximum(best, cand)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits"); ap.add_argument("valbin")
    ap.add_argument("--anchors", default="data/visdrone")
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    args = ap.parse_args()

    anchors = [np.array(load_anchors_file(f"{args.anchors}/anchors_fpn_{p}.txt"))
               for p in ("p3", "p4", "p5")]
    A = len(anchors[0])
    ntot = sum(A * P * g * g for g in GRIDS)
    rec_img = 3 * args.imgsz * args.imgsz
    rec = rec_img + ntot * 4

    logits = np.fromfile(args.logits, dtype=np.float32)
    nl = logits.size // ntot
    logits = logits[:nl * ntot].reshape(nl, ntot)
    raw = np.fromfile(args.valbin, dtype=np.uint8)
    nrec = (raw.size - 4) // rec                       # skip the 4-byte header
    tgt = raw[4:4 + nrec * rec].reshape(nrec, rec)[:, rec_img:] \
        .copy().view(np.float32).reshape(nrec, ntot)
    n = min(nl, nrec)
    print(f"records: {n}  (header-corrected target read)\n")

    ring_iou_ctr, ring_iou_gt, reach0, reach05, dists = [], [], [], [], []
    off = 0
    for si, g in enumerate(GRIDS):
        blk = A * P * g * g
        L = logits[:n, off:off + blk].reshape(n, A, P, g, g)
        T = tgt[:n, off:off + blk].reshape(n, A, P, g, g)
        off += blk
        for a in range(A):
            aw, ah = anchors[si][a]
            obj_t = T[:, a, 4] > 0.5                    # [n,g,g]
            idx = np.argwhere(obj_t)                    # [K,3] (rec, ci, cj)
            if idx.size == 0:
                continue
            r_, ci_, cj_ = idx[:, 0], idx[:, 1], idx[:, 2]
            # GT box recovered from the encoded target
            t0 = T[r_, a, 0, ci_, cj_]; t1 = T[r_, a, 1, ci_, cj_]
            gt = np.stack([(cj_ + t0) / g, (ci_ + t1) / g,
                           T[r_, a, 2, ci_, cj_], T[r_, a, 3, ci_, cj_]], -1)

            def decode(ii, jj):
                sx = 1.0 / (1.0 + np.exp(-np.clip(L[r_, a, 0, ii, jj], -60, 60)))
                sy = 1.0 / (1.0 + np.exp(-np.clip(L[r_, a, 1, ii, jj], -60, 60)))
                w = aw * np.exp(np.minimum(L[r_, a, 2, ii, jj], 8.0))
                h = ah * np.exp(np.minimum(L[r_, a, 3, ii, jj], 8.0))
                return np.stack([(jj + sx) / g, (ii + sy) / g, w, h], -1)

            ctr_box = decode(ci_, cj_)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ii, jj = ci_ + dy, cj_ + dx
                    ok = (ii >= 0) & (ii < g) & (jj >= 0) & (jj < g)
                    if not ok.any():
                        continue
                    ii = np.clip(ii, 0, g - 1); jj = np.clip(jj, 0, g - 1)
                    rb = decode(ii, jj)
                    ring_iou_ctr.append(iou_cwh(rb, ctr_box)[ok])
                    ring_iou_gt.append(iou_cwh(rb, gt)[ok])
                    reach0.append(best_reachable_iou(gt, ii, jj, g, 0.0)[ok])
                    reach05.append(best_reachable_iou(gt, ii, jj, g, 0.5)[ok])
            dists.append(np.full(idx.shape[0], 1.0 / g))

    cat = lambda v: np.concatenate(v)
    ric, rig, r0, r05 = cat(ring_iou_ctr), cat(ring_iou_gt), cat(reach0), cat(reach05)

    print("=" * 74)
    print("0b  RING-CELL BOXES: do they merge under NMS?")
    print("=" * 74)
    print(f"  ring cells examined: {ric.size}\n")
    print("  MEASURED on the e12 logits:")
    for nm, v in (("IoU(ring box, centre box)", ric), ("IoU(ring box, GT)", rig)):
        print(f"    {nm:26s} mean={v.mean():.4f} median={np.median(v):.4f} "
              f"p90={np.percentile(v,90):.4f} max={v.max():.4f}")
        print(f"    {'':26s} frac >= nms_iou({args.nms_iou}) = "
              f"{100.0*(v>=args.nms_iou).mean():.2f}%")
    print("\n  GEOMETRY BOUND (best box any ring cell could EVER emit vs the GT):")
    print(f"    span 0.0 (today cx=(j+sigma)/g)   : frac reaching IoU>=0.5 = "
          f"{100.0*(r0>=0.5).mean():6.2f}%   mean best IoU={r0.mean():.4f}")
    print(f"    span 0.5 (YOLOv5 2*sigma-0.5)     : frac reaching IoU>=0.5 = "
          f"{100.0*(r05>=0.5).mean():6.2f}%   mean best IoU={r05.mean():.4f}")


if __name__ == "__main__":
    main()
