#!/usr/bin/env python3
"""Multi-scale (FPN) target-encoding COVERAGE analysis — the load-bearing test of
the whole FPN build (planning/yolo_fpn.md bite 5, the thesis gate).

The anchor A=6 detector plateaus at ~5% recall because only ~61% of GT boxes get
a unique (cell, anchor) slot at a single 14x14 grid — the rest are lost to
collisions (VisDrone packs ~70 tiny objects/image). The FPN claim is that adding
finer scales (56x56 stride-8, 28x28 stride-16) cuts collisions hugely. This script
MEASURES that, cheaply, before any of the expensive Lean integration (bites 3/4/6/7)
or the 4 GB data re-encode (bite 8) is built. If coverage does not jump well above
61%, the neck is not worth wiring.

Encoding rule (per GT box, at 448px input):
  * scale by size: max(w,h)*448 < T_lo -> P3 (56); < T_hi -> P4 (28); else P5 (14)
  * cell by center at that scale's grid
  * best of that scale's 3 k-means anchors (wh-IoU)
Coverage = unique (scale, cell, anchor) slots / total GT boxes, summed over images.

Also reports:
  * the single-scale 14x14 A=6 baseline (reproduces the ~61%)
  * a "joint best across all 9 anchors" upper bound (assign to the globally best
    (scale, anchor) by IoU, cell by that scale) — collisions-only ceiling

Usage: python3 scripts/visdrone_fpn_coverage.py [visdrone_dir=data/visdrone]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess_visdrone import parse_visdrone_txt, load_anchors
from scripts.visdrone_anchors import wh_iou, kmeans_anchors

INPUT_PX = 448
GRIDS = {"P3": 56, "P4": 28, "P5": 14}       # strides 8 / 16 / 32
SCALE_NAMES = ["P3", "P4", "P5"]
# size thresholds in input px on max(w,h): [<T_lo -> P3] [<T_hi -> P4] [else P5]
T_LO, T_HI = 24.0, 64.0


def collect_per_image(visdrone_dir):
    """[(iw, ih, [(cid,xmin,ymin,xmax,ymax)...])] over the train split."""
    train = Path(visdrone_dir) / "VisDrone2019-DET-train"
    anns, imgs = train / "annotations", train / "images"
    out = []
    for txt in sorted(anns.glob("*.txt")):
        img = imgs / f"{txt.stem}.jpg"
        if not img.exists():
            continue
        boxes = parse_visdrone_txt(txt)
        if not boxes:
            continue
        iw, ih = Image.open(img).size      # header-only read, fast
        out.append((iw, ih, boxes))
    return out


def all_wh(per_image):
    wh = []
    for iw, ih, boxes in per_image:
        for (_c, x0, y0, x1, y1) in boxes:
            wh.append(((x1 - x0) / iw, (y1 - y0) / ih))
    return np.array(wh, dtype=np.float64)


def scale_of(w_rel, h_rel):
    m = max(w_rel, h_rel) * INPUT_PX
    return 0 if m < T_LO else (1 if m < T_HI else 2)


def per_scale_anchors(per_image, k=3):
    """k-means k anchors within each scale's size-assigned GT subset."""
    buckets = [[], [], []]
    for iw, ih, boxes in per_image:
        for (_c, x0, y0, x1, y1) in boxes:
            wr, hr = (x1 - x0) / iw, (y1 - y0) / ih
            buckets[scale_of(wr, hr)].append((wr, hr))
    anchors = []
    for s, b in enumerate(buckets):
        arr = np.array(b, dtype=np.float64)
        anchors.append(kmeans_anchors(arr, min(k, len(arr))))
    return anchors, [len(b) for b in buckets]


def best_a(w_rel, h_rel, anchors):
    return int(wh_iou(np.array([[w_rel, h_rel]]), anchors)[0].argmax())


def multiscale_coverage(per_image, anchors):
    total = slots = 0
    for iw, ih, boxes in per_image:
        filled = set()
        for (_c, x0, y0, x1, y1) in boxes:
            wr, hr = (x1 - x0) / iw, (y1 - y0) / ih
            cx, cy = (x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih
            s = scale_of(wr, hr)
            g = GRIDS[SCALE_NAMES[s]]
            cj = min(int(cx * g), g - 1); ci = min(int(cy * g), g - 1)
            a = best_a(wr, hr, anchors[s])
            filled.add((s, ci, cj, a))
        slots += len(filled); total += len(boxes)
    return total, slots


def joint_coverage(per_image, anchors):
    """Upper bound: assign each GT to the globally best (scale, anchor) by IoU."""
    flat = np.vstack(anchors)                      # [9,2]
    owner = []                                     # (scale_idx, local_a) per flat row
    for s, arr in enumerate(anchors):
        owner += [(s, j) for j in range(len(arr))]
    total = slots = 0
    for iw, ih, boxes in per_image:
        filled = set()
        for (_c, x0, y0, x1, y1) in boxes:
            wr, hr = (x1 - x0) / iw, (y1 - y0) / ih
            cx, cy = (x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih
            k = int(wh_iou(np.array([[wr, hr]]), flat)[0].argmax())
            s, a = owner[k]
            g = GRIDS[SCALE_NAMES[s]]
            cj = min(int(cx * g), g - 1); ci = min(int(cy * g), g - 1)
            filled.add((s, ci, cj, a))
        slots += len(filled); total += len(boxes)
    return total, slots


def singlescale_coverage(per_image, anchors, grid):
    total = slots = 0
    for iw, ih, boxes in per_image:
        filled = set()
        for (_c, x0, y0, x1, y1) in boxes:
            wr, hr = (x1 - x0) / iw, (y1 - y0) / ih
            cx, cy = (x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih
            cj = min(int(cx * grid), grid - 1); ci = min(int(cy * grid), grid - 1)
            a = best_a(wr, hr, anchors)
            filled.add((ci, cj, a))
        slots += len(filled); total += len(boxes)
    return total, slots


def save_anchors(anchors, out_dir):
    """Write per-scale k-means anchors as data/visdrone/anchors_fpn_{p3,p4,p5}.txt
    (one 'w_rel h_rel' per line) for the preprocessor + codegen to consume."""
    for s, nm in enumerate(SCALE_NAMES):
        p = Path(out_dir) / f"anchors_fpn_{nm.lower()}.txt"
        with open(p, "w") as f:
            f.write(f"# VisDrone FPN {nm} anchors (grid {GRIDS[nm]}, stride "
                    f"{INPUT_PX//GRIDS[nm]}), size-assigned k-means, w_rel h_rel\n")
            for a in anchors[s]:
                f.write(f"{a[0]:.6f} {a[1]:.6f}\n")
        print(f"  wrote {p}")


def smoke_encoder(per_image, anchors, n=100):
    """Tie preprocess_visdrone.encode_targets_fpn to this (validated) coverage
    logic: on the first n images, its slot count must equal a direct recompute,
    the per-scale target shapes must be [A_s·15, g, g], and the mask must equal
    the obj channel (target[base+4]) exactly (the loader-derives-mask invariant)."""
    from preprocess_visdrone import encode_targets_fpn, PER_ANCHOR, NUM_CLASSES_A
    bad = 0
    for iw, ih, boxes in per_image[:n]:
        tgts, msks, nslots = encode_targets_fpn(iw, ih, boxes, anchors, INPUT_PX)
        # direct recompute of unique slots
        filled = set()
        for (_c, x0, y0, x1, y1) in boxes:
            wr, hr = (x1 - x0) / iw, (y1 - y0) / ih
            cx, cy = (x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih
            s = scale_of(wr, hr); g = GRIDS[SCALE_NAMES[s]]
            cj = min(int(cx * g), g - 1); ci = min(int(cy * g), g - 1)
            filled.add((s, ci, cj, best_a(wr, hr, anchors[s])))
        if nslots != len(filled):
            bad += 1; continue
        for s, g in enumerate(GRIDS.values()):
            A = len(anchors[s])
            if tgts[s].shape != (A * PER_ANCHOR, g, g) or msks[s].shape != (A, g, g):
                bad += 1; break
            obj = tgts[s][4::PER_ANCHOR]                 # [A,g,g] obj channels
            if not np.array_equal((obj > 0).astype(np.float32), msks[s]):
                bad += 1; break
    ok = bad == 0
    print(f"encoder smoke ({n} imgs): {'PASS' if ok else f'FAIL ({bad} bad)'} "
          f"(slot count = coverage recompute, shapes, mask==obj-channel)")
    return ok


def main():
    args = sys.argv[1:]
    save_dir = None
    smoke = False
    if "--save-anchors" in args:
        i = args.index("--save-anchors"); save_dir = args[i + 1]; del args[i:i + 2]
    if "--smoke" in args:
        smoke = True; args.remove("--smoke")
    visdrone_dir = args[0] if args else "data/visdrone"
    print(f"collecting per-image GT from {visdrone_dir} ...")
    per_image = collect_per_image(visdrone_dir)
    nboxes = sum(len(b) for _, _, b in per_image)
    print(f"  {len(per_image)} train images, {nboxes} GT boxes "
          f"({nboxes/max(len(per_image),1):.1f}/img)\n")

    # ── per-scale anchors ──
    anchors, counts = per_scale_anchors(per_image, k=3)
    if save_dir:
        save_anchors(anchors, save_dir)
    print(f"scale assignment (max(w,h)px thresholds {T_LO:.0f}/{T_HI:.0f}):")
    for s, nm in enumerate(SCALE_NAMES):
        pct = 100.0 * counts[s] / max(nboxes, 1)
        print(f"  {nm} (grid {GRIDS[nm]:>2}, stride {INPUT_PX//GRIDS[nm]:>2}): "
              f"{counts[s]:>7} GT ({pct:4.1f}%)  anchors(px@448): "
              + ", ".join(f"({a[0]*INPUT_PX:.0f}x{a[1]*INPUT_PX:.0f})" for a in anchors[s]))
    print()

    # ── single-scale baseline (reproduce ~61%) ──
    a6_path = Path(visdrone_dir).parent / "visdrone" / "anchors_a6.txt"
    if not a6_path.exists():
        a6_path = Path("data/visdrone/anchors_a6.txt")
    tot, sl = singlescale_coverage(per_image, load_anchors(str(a6_path)), 14)
    print(f"single-scale 14x14, A=6 (baseline)     : "
          f"{sl}/{tot} = {100.0*sl/tot:.1f}% encoded")

    # ── FPN multi-scale (the thesis) ──
    tot, sl = multiscale_coverage(per_image, anchors)
    print(f"FPN 3-scale 56/28/14, 3 anchors/scale  : "
          f"{sl}/{tot} = {100.0*sl/tot:.1f}% encoded   <-- thesis")

    tot, sl = joint_coverage(per_image, anchors)
    print(f"  (joint-best-anchor upper bound)      : "
          f"{sl}/{tot} = {100.0*sl/tot:.1f}% encoded")

    if smoke:
        print()
        smoke_encoder(per_image, anchors)


if __name__ == "__main__":
    main()
