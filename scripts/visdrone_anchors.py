#!/usr/bin/env python3
"""k-means anchor priors over VisDrone box sizes (detection-infra brick #2, WS-C).

Anchors are the "learned scales" an anchor-based detector predicts against: each
cell emits A boxes, box_a = anchor_a · decode(pred), so the network regresses a
small residual off a prior instead of a raw size. The priors are the k-means
centroids of the ground-truth (w, h) distribution, clustered with IoU distance
(YOLOv5's metric: d = 1 - IoU, so big and small boxes are weighted by shape
overlap, not Euclidean px). Host-side, computed once.

Reports, for A in {3,5,6,9}: the anchor (w_rel, h_rel) priors (normalized to the
model input, i.e. fraction of image side) and the mean best-anchor IoU + recall
@0.5 over all GT boxes — the "how well do these priors cover the data" score that
tells us how many anchors this dataset actually needs.

Usage: python3 scripts/visdrone_anchors.py [visdrone_dir=data/visdrone] [A...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess_visdrone import parse_visdrone_txt  # reuse the exact box filter


def collect_wh(visdrone_dir):
    """All kept GT (w_rel, h_rel), normalized to [0,1] by image size."""
    from PIL import Image
    train = Path(visdrone_dir) / "VisDrone2019-DET-train"
    anns = train / "annotations"
    imgs = train / "images"
    wh = []
    for txt in sorted(anns.glob("*.txt")):
        img = imgs / f"{txt.stem}.jpg"
        if not img.exists():
            continue
        iw, ih = Image.open(img).size
        for (_cid, xmin, ymin, xmax, ymax) in parse_visdrone_txt(txt):
            wh.append(((xmax - xmin) / iw, (ymax - ymin) / ih))
    return np.array(wh, dtype=np.float64)


def wh_iou(wh, anchors):
    """IoU between boxes wh [N,2] and anchors [A,2], aligned at origin -> [N,A]."""
    w = np.minimum(wh[:, None, 0], anchors[None, :, 0])
    h = np.minimum(wh[:, None, 1], anchors[None, :, 1])
    inter = w * h
    area_wh = (wh[:, 0] * wh[:, 1])[:, None]
    area_a = (anchors[:, 0] * anchors[:, 1])[None, :]
    return inter / (area_wh + area_a - inter + 1e-12)


def kmeans_anchors(wh, A, iters=100, seed=0):
    rng = np.random.RandomState(seed)
    # init from A random distinct GT boxes
    centroids = wh[rng.choice(len(wh), A, replace=False)].copy()
    for _ in range(iters):
        d = 1.0 - wh_iou(wh, centroids)      # [N,A]
        assign = d.argmin(axis=1)
        new = np.array([wh[assign == a].mean(axis=0) if (assign == a).any()
                        else centroids[a] for a in range(A)])
        if np.allclose(new, centroids, atol=1e-9):
            centroids = new
            break
        centroids = new
    # sort by area
    centroids = centroids[np.argsort(centroids[:, 0] * centroids[:, 1])]
    return centroids


def main():
    args = sys.argv[1:]
    # --save PATH --num A : write the A anchors (one "w_rel h_rel" per line) for
    # the preprocessor + codegen to consume, then exit.
    save = None
    if "--save" in args:
        i = args.index("--save"); save = args[i + 1]; del args[i:i + 2]
    save_num = 6
    if "--num" in args:
        i = args.index("--num"); save_num = int(args[i + 1]); del args[i:i + 2]
    visdrone_dir = next((a for a in args if not a.isdigit()), "data/visdrone")
    As = [int(a) for a in args if a.isdigit()] or [3, 5, 6, 9]
    print(f"collecting GT box sizes from {visdrone_dir} ...")
    wh = collect_wh(visdrone_dir)
    if save:
        anchors = kmeans_anchors(wh, save_num)
        with open(save, "w") as f:
            f.write(f"# VisDrone k-means anchors, A={save_num}, w_rel h_rel (fraction of image side)\n")
            for a in anchors:
                f.write(f"{a[0]:.6f} {a[1]:.6f}\n")
        best = wh_iou(wh, anchors).max(axis=1)
        print(f"wrote {save}: {save_num} anchors, recall@0.5={np.mean(best>0.5):.3f}")
        return
    px = wh * 448  # report in 448-input pixels too
    print(f"  {len(wh)} boxes | w_rel median {np.median(wh[:,0]):.4f} "
          f"({np.median(px[:,0]):.1f}px)  h_rel median {np.median(wh[:,1]):.4f} "
          f"({np.median(px[:,1]):.1f}px)")
    for A in As:
        anchors = kmeans_anchors(wh, A)
        iou = wh_iou(wh, anchors)
        best = iou.max(axis=1)
        print(f"\nA={A} anchors (w_rel,h_rel  |  px@448):  "
              f"mean best-IoU={best.mean():.3f}  recall@0.5={np.mean(best>0.5):.3f}")
        for a in anchors:
            print(f"    ({a[0]:.4f}, {a[1]:.4f})   ({a[0]*448:5.1f}, {a[1]*448:5.1f}) px")


if __name__ == "__main__":
    main()
