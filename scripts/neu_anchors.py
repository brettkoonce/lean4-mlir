#!/usr/bin/env python3
"""k-means anchor priors over NEU-DET box sizes, per FPN scale.

The NEU-DET twin of scripts/coco_anchors.py: the IoU metric and k-means come
from scripts/visdrone_anchors.py (imported, not copied), the box source and
the train split come from preprocess_neu_det.py, so the priors are fitted on
exactly the images the detector trains on and by exactly the code that fitted
VisDrone's and COCO's.

Why NEU needs its own priors: VisDrone's largest P5 prior is 0.18 × 0.15 of
the frame, fitted to drone-altitude objects a few pixels across; a NEU defect
often spans half the crop. Under VisDrone's table every NEU box would sit on
P5 against anchors several times too small, and the box loss would start from
a residual `exp` cannot reach.

Boxes are routed to a scale by the SAME rule the preprocessor uses
(preprocess_visdrone.fpn_scale_of: max(w,h)·448 against 24 / 64 px), then
k-means runs within each scale. NEU has essentially no P3 boxes (nothing is
under 11 source px), so P3 gets a fixed fallback table spanning its size band:
the codegen needs 3 anchors on every level, a level with no positives never
trains its box residual, and the values only matter for decoding what the head
emits there. The report prints the recall@0.5 of the fitted table, which is
the coverage ceiling Gate 0 of planning/neu_det_fpn_demo.md asks for (≥ 0.9).

Usage:
  python3 scripts/neu_anchors.py [neu_dir=data/neu_det] [--save DIR] [--num A]
                                 [--seed N]
"""
import sys
from pathlib import Path

import numpy as np

# Both the repo root (the preprocessors) and scripts/ itself: `scripts.` as a
# package name collides with an unrelated `scripts` module in some venvs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visdrone_anchors import wh_iou, kmeans_anchors   # noqa: E402
import preprocess_visdrone as pv                     # noqa: E402
import preprocess_neu_det as pn                      # noqa: E402

SCALE_NAMES = ("P3", "P4", "P5")
INPUT_PX = 448
# Fallback priors for a scale with fewer boxes than anchors, in px @448: a
# square and the two aspects, at the middle of the band the scale owns.
FALLBACK_PX = {0: [(12, 12), (8, 20), (20, 8)],
               1: [(40, 40), (28, 56), (56, 28)],
               2: [(120, 120), (80, 200), (200, 80)]}


def collect_wh_by_scale(neu_dir, seed):
    per_image = pn.load_split(neu_dir, pn.split_stems(neu_dir, seed)["train"])
    buckets = [[], [], []]
    for (_stem, iw, ih, boxes) in per_image:
        for (_cid, x0, y0, x1, y1) in boxes:
            wr, hr = (x1 - x0) / iw, (y1 - y0) / ih
            buckets[pv.fpn_scale_of(wr, hr, INPUT_PX)].append((wr, hr))
    return len(per_image), [np.array(b, dtype=np.float64).reshape(-1, 2) for b in buckets]


def save_anchors(anchors, out_dir, fallback):
    for s, nm in enumerate(SCALE_NAMES):
        g = pv.FPN_GRIDS[s]
        p = Path(out_dir) / f"anchors_fpn_{nm.lower()}.txt"
        with open(p, "w") as f:
            how = ("FIXED fallback (no train boxes on this level)" if s in fallback
                   else f"size-assigned k-means (thresh {pv.FPN_T_LO:.0f}/{pv.FPN_T_HI:.0f}px)")
            f.write(f"# NEU-DET FPN {nm} anchors (grid {g}, stride {INPUT_PX // g}), "
                    f"{how}, w_rel h_rel — scripts/neu_anchors.py\n")
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
    seed = pn.SEED
    if "--seed" in args:
        i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
    neu_dir = args[0] if args else "data/neu_det"

    print(f"collecting train-split GT box sizes from {neu_dir} (split seed {seed}) ...")
    n_img, buckets = collect_wh_by_scale(neu_dir, seed)
    nb = sum(len(b) for b in buckets)
    print(f"  {n_img} train images, {nb} boxes ({nb / max(n_img, 1):.2f}/img)")
    out, fallback = [], set()
    covered = []          # best-IoU per box against ITS OWN scale's anchors
    for s, nm in enumerate(SCALE_NAMES):
        wh = buckets[s]
        g = pv.FPN_GRIDS[s]
        if len(wh) < A:
            anchors = np.array([(w / INPUT_PX, h / INPUT_PX) for (w, h) in FALLBACK_PX[s]][:A])
            fallback.add(s)
            print(f"\n{nm} (grid {g}, stride {INPUT_PX // g}): {len(wh)} boxes "
                  f"({100.0 * len(wh) / nb:.1f}%) — fewer than {A}, using the FIXED fallback table")
        else:
            anchors = kmeans_anchors(wh, A)
            best = wh_iou(wh, anchors).max(axis=1)
            covered.extend(best.tolist())
            px = wh * INPUT_PX
            print(f"\n{nm} (grid {g}, stride {INPUT_PX // g}): {len(wh)} boxes "
                  f"({100.0 * len(wh) / nb:.1f}%) | median {np.median(px[:, 0]):.0f}x"
                  f"{np.median(px[:, 1]):.0f} px@448 | mean best-IoU={best.mean():.3f} "
                  f"recall@0.5={np.mean(best > 0.5):.3f}")
        out.append(anchors)
        for a in anchors:
            print(f"    ({a[0]:.4f}, {a[1]:.4f})   ({a[0] * INPUT_PX:6.1f}, {a[1] * INPUT_PX:6.1f}) px")
    if len(wh := buckets[0]) and 0 in fallback:
        best = wh_iou(wh, out[0]).max(axis=1); covered.extend(best.tolist())
    covered = np.array(covered)
    print(f"\nALL {len(covered)} boxes vs their own level's {A} anchors: mean best-IoU="
          f"{covered.mean():.3f}  recall@0.5={np.mean(covered > 0.5):.3f}"
          f"   (Gate 0 wants >= 0.9; VisDrone's A=6 single grid was 0.76)")
    if save:
        print()
        save_anchors(out, save, fallback)


if __name__ == "__main__":
    main()
