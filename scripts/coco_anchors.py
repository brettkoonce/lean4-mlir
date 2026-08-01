#!/usr/bin/env python3
"""k-means anchor priors over MS-COCO box sizes, per FPN scale.

The COCO twin of scripts/visdrone_anchors.py + visdrone_fpn_coverage.py's
--save-anchors. Reuses that file's IoU metric and k-means verbatim (imported,
not copied) so the two datasets' priors are computed by identical code; only
the box source differs.

Why COCO needs its own priors: VisDrone's are drone-altitude tiny (P3 median
~3x7 px at 448) while COCO objects are photographic and often fill the frame.
Handing the VisDrone priors to a COCO build would put nearly every GT box on
the largest anchor of P5, which this script's report makes visible.

Boxes are routed to a scale by the SAME rule the preprocessor uses
(preprocess_coco.fpn_scale_of: max(w,h)*input_px against 24/64 px), then
k-means runs within each scale — so the priors match the assignment they will
be used under.

Usage:
  python3 scripts/coco_anchors.py <coco_dir> [--save DIR] [--num A]
                                  [--thresh LO HI] [--val-only]

Without --save it reports; with --save DIR it writes
anchors_fpn_{p3,p4,p5}.txt in the format preprocess_coco.py --fpn reads.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.visdrone_anchors import wh_iou, kmeans_anchors  # noqa: E402
import preprocess_coco as pc                                  # noqa: E402

SCALE_NAMES = ("P3", "P4", "P5")


def collect_wh_by_scale(coco_dir, splits, thresh, input_px=448):
    """All kept GT (w_rel, h_rel) partitioned by FPN scale, using the
    preprocessor's own filter and its own scale rule."""
    pc.FPN_T_LO, pc.FPN_T_HI = thresh
    buckets = [[], [], []]
    for split in splits:
        j = Path(coco_dir) / "annotations" / f"instances_{split}2017.json"
        per_image, _names, _n = pc.load_split_annotations(str(j), "all")
        for (_fn, iw, ih, boxes) in per_image:
            for (_cid, x0, y0, x1, y1) in boxes:
                wr, hr = (x1 - x0) / iw, (y1 - y0) / ih
                buckets[pc.fpn_scale_of(wr, hr, input_px)].append((wr, hr))
    return [np.array(b, dtype=np.float64) for b in buckets]


def save_anchors(anchors, out_dir, thresh, input_px=448):
    for s, nm in enumerate(SCALE_NAMES):
        g = pc.FPN_GRIDS[s]
        p = Path(out_dir) / f"anchors_fpn_{nm.lower()}.txt"
        with open(p, "w") as f:
            f.write(f"# COCO FPN {nm} anchors (grid {g}, stride {input_px//g}), "
                    f"size-assigned k-means (thresh {thresh[0]:.0f}/"
                    f"{thresh[1]:.0f}px), w_rel h_rel\n")
            for a in anchors[s]:
                f.write(f"{a[0]:.6f} {a[1]:.6f}\n")
        print(f"  wrote {p}")


def main():
    args = sys.argv[1:]
    save = None
    if "--save" in args:
        i = args.index("--save"); save = args[i + 1]; del args[i:i + 2]
    A = 3
    if "--num" in args:
        i = args.index("--num"); A = int(args[i + 1]); del args[i:i + 2]
    thresh = (pc.FPN_T_LO, pc.FPN_T_HI)
    if "--thresh" in args:
        i = args.index("--thresh")
        thresh = (float(args[i + 1]), float(args[i + 2])); del args[i:i + 3]
    splits = ("train", "val")
    if "--val-only" in args:
        splits = ("val",); args.remove("--val-only")
    coco_dir = args[0] if args else "data/coco"

    print(f"collecting GT box sizes from {coco_dir} (splits {splits}) ...")
    buckets = collect_wh_by_scale(coco_dir, splits, thresh)
    out = []
    for s, nm in enumerate(SCALE_NAMES):
        wh = buckets[s]
        if len(wh) < A:
            print(f"ERROR: scale {nm} has only {len(wh)} boxes, need >= {A} "
                  f"for k-means. Widen --thresh or use --num 1.", file=sys.stderr)
            sys.exit(1)
        anchors = kmeans_anchors(wh, A)
        out.append(anchors)
        best = wh_iou(wh, anchors).max(axis=1)
        px = wh * 448
        print(f"\n{nm}: {len(wh)} boxes ({100.0*len(wh)/sum(len(b) for b in buckets):.1f}%)"
              f" | median {np.median(px[:,0]):.1f}x{np.median(px[:,1]):.1f} px@448"
              f" | mean best-IoU={best.mean():.3f} recall@0.5={np.mean(best>0.5):.3f}")
        for a in anchors:
            print(f"    ({a[0]:.4f}, {a[1]:.4f})   ({a[0]*448:6.1f}, {a[1]*448:6.1f}) px")
    if save:
        print()
        save_anchors(out, save, thresh)


if __name__ == "__main__":
    main()
