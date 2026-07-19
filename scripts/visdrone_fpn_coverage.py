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


# ── center-sampling oracle (planning/yolo_assignment.md bite 0a) ──────────────
#
# Two DIFFERENT ceilings, and the gap between them is the whole point:
#
#   encodable : the GT retains >=1 (scale,cell,anchor) slot after collisions.
#               This is what the existing 88.2% number measures, generalized to
#               many cells per GT. Center sampling can only raise it.
#   reachable : the GT retains >=1 slot from which a box of IoU>=0.5 with the GT
#               is actually DECODABLE. The decode is cx=(j+sigma(tx))/g, and
#               sigma in (0,1) confines a cell's predicted centre STRICTLY INSIDE
#               ITSELF. A ring cell whose neighbour owns the GT centre therefore
#               cannot emit a box on that GT at all -- unless the centre range is
#               widened (YOLOv5 uses 2*sigma-0.5, i.e. span=0.5 cells).
#
# Making a cell positive that cannot represent the target trains it toward an
# unreachable box: a guaranteed false positive with high objectness. So the
# hypothesis needs `reachable` to move, not `encodable`.


def _iou_cwh(cx1, cy1, w1, h1, cx2, cy2, w2, h2):
    ix = min(cx1 + w1 / 2, cx2 + w2 / 2) - max(cx1 - w1 / 2, cx2 - w2 / 2)
    iy = min(cy1 + h1 / 2, cy2 + h2 / 2) - max(cy1 - h1 / 2, cy2 - h2 / 2)
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    ua = w1 * h1 + w2 * h2 - inter
    return inter / ua if ua > 0 else 0.0


def candidate_cells(cx, cy, wr, hr, g, criterion, radius):
    """Cells assigned to one GT. 'center' = today's single cell; 'inside' = FCOS
    (cell CENTRE falls inside the GT box); 'radius' = a (2r+1)^2 square of cells
    around the centre cell. The centre cell is always included."""
    cj0 = min(int(cx * g), g - 1)
    ci0 = min(int(cy * g), g - 1)
    if criterion == "center":
        return [(ci0, cj0)]
    out = set()
    if criterion == "inside":
        # cell j's centre sits at (j+0.5)/g, so require (j+0.5)/g in [cx-w/2, cx+w/2]
        jlo = int(np.ceil((cx - wr / 2) * g - 0.5)); jhi = int(np.floor((cx + wr / 2) * g - 0.5))
        ilo = int(np.ceil((cy - hr / 2) * g - 0.5)); ihi = int(np.floor((cy + hr / 2) * g - 0.5))
        for ci in range(max(0, ilo), min(g - 1, ihi) + 1):
            for cj in range(max(0, jlo), min(g - 1, jhi) + 1):
                out.add((ci, cj))
    elif criterion == "radius":
        R = int(np.floor(radius))
        for di in range(-R, R + 1):
            for dj in range(-R, R + 1):
                ci, cj = ci0 + di, cj0 + dj
                if 0 <= ci < g and 0 <= cj < g:
                    out.add((ci, cj))
    out.add((ci0, cj0))
    return sorted(out)


def reachable_iou(cx, cy, wr, hr, ci, cj, g, span):
    """Best IoU with the GT obtainable from cell (ci,cj): centre clamped to the
    cell (widened by `span` cells each side), w/h free (exp() spans any size)."""
    px = min(max(cx, (cj - span) / g), (cj + 1 + span) / g)
    py = min(max(cy, (ci - span) / g), (ci + 1 + span) / g)
    best = 0.0
    for m in (1.0, 1.25, 1.5, 2.0, 3.0):        # free w/h: search a few scales
        v = _iou_cwh(px, py, wr * m, hr * m, cx, cy, wr, hr)
        if v > best:
            best = v
    return best


def center_sampling_coverage(per_image, anchors, criterion, radius, span,
                             iou_thr=0.5, limit=None, priority=False):
    """-> (total_gt, encodable, reachable, slots_kept). Collisions resolved the
    way encode_targets_fpn does it: later GT overwrites earlier on a shared slot.

    priority=True instead gives every GT's OWN CENTRE cell precedence over any
    other GT's ring cell (ring pass first, centre pass second). Without this the
    naive rule lets a ring cell wipe out a neighbour's centre cell, which would
    make center sampling look bad for a reason that is an encoder detail rather
    than a property of the hypothesis."""
    imgs = per_image if limit is None else per_image[:limit]
    total = enc = reach = slots = 0
    for iw, ih, boxes in imgs:
        owner, cand, gts, ctr = {}, [], [], []
        for gi, (_c, x0, y0, x1, y1) in enumerate(boxes):
            wr, hr = (x1 - x0) / iw, (y1 - y0) / ih
            cx, cy = (x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih
            s = scale_of(wr, hr); g = GRIDS[SCALE_NAMES[s]]
            a = best_a(wr, hr, anchors[s])
            cells = candidate_cells(cx, cy, wr, hr, g, criterion, radius)
            gts.append((cx, cy, wr, hr, s, g, a)); cand.append(cells)
            ctr.append((min(int(cy * g), g - 1), min(int(cx * g), g - 1)))
            if not priority:
                for (ci, cj) in cells:
                    owner[(s, ci, cj, a)] = gi      # last write wins
        if priority:
            for gi, (cx, cy, wr, hr, s, g, a) in enumerate(gts):
                for (ci, cj) in cand[gi]:           # pass 1: ring cells
                    if (ci, cj) != ctr[gi]:
                        owner[(s, ci, cj, a)] = gi
            for gi, (cx, cy, wr, hr, s, g, a) in enumerate(gts):
                owner[(s, ctr[gi][0], ctr[gi][1], a)] = gi   # pass 2: centres win
        total += len(boxes)
        for gi, (cx, cy, wr, hr, s, g, a) in enumerate(gts):
            kept = [c for c in cand[gi] if owner[(s, c[0], c[1], a)] == gi]
            slots += len(kept)
            if kept:
                enc += 1
                if any(reachable_iou(cx, cy, wr, hr, ci, cj, g, span) >= iou_thr
                       for (ci, cj) in kept):
                    reach += 1
    return total, enc, reach, slots


def report_center_sampling(per_image, anchors, limit=None):
    print("\n" + "=" * 78)
    print("CENTER-SAMPLING ORACLE (yolo_assignment.md bite 0a)")
    print("=" * 78)
    print("  encodable = GT keeps >=1 slot after collisions (today's 88.2% metric)")
    print("  reachable = GT keeps >=1 slot that can DECODE a box of IoU>=0.5")
    print("  span 0.0 = today's cx=(j+sigma)/g (centre locked inside its own cell)")
    print("  span 0.5 = YOLOv5 cx=(j+2*sigma-0.5)/g (centre may reach 1/2 cell out)\n")
    variants = [("center", 0, "baseline (1 cell/GT)"),
                ("inside", 0, "FCOS: cell centre inside GT box"),
                ("radius", 1, "radius 1 -> 3x3 cells"),
                ("radius", 2, "radius 2 -> 5x5 cells")]
    for priority in (False, True):
        rule = ("collision: naive last-write-wins (today's encoder)" if not priority
                else "collision: own-centre beats any other GT's ring (best case)")
        print(f"  --- {rule} ---")
        print(f"  {'assignment':<34} {'span':>5} {'slots/GT':>9} {'encodable':>11} {'reachable':>11}")
        print("  " + "-" * 74)
        for (crit, rad, label) in variants:
            for span in (0.0, 0.5):
                tot, enc, reach, slots = center_sampling_coverage(
                    per_image, anchors, crit, rad, span, limit=limit, priority=priority)
                print(f"  {label:<34} {span:>5.1f} {slots/max(tot,1):>9.2f} "
                      f"{100.0*enc/max(tot,1):>10.1f}% {100.0*reach/max(tot,1):>10.1f}%")
        print()


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
    cs = False
    limit = None
    if "--center-sampling" in args:
        cs = True; args.remove("--center-sampling")
    if "--limit" in args:
        i = args.index("--limit"); limit = int(args[i + 1]); del args[i:i + 2]
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

    if cs:
        report_center_sampling(per_image, anchors, limit=limit)

    if smoke:
        print()
        smoke_encoder(per_image, anchors)


if __name__ == "__main__":
    main()
