#!/usr/bin/env python3
"""Pre-process NEU-DET (hot-rolled steel surface defects) → the YOLO detection
.bin formats the VisDrone detector reads.

NEU-DET is 1,800 grayscale 200×200 crops, 300 per class across six defect
types, with Pascal-VOC boxes (Song & Yan 2013, Appl. Surf. Sci. 285; boxes from
He et al. 2020, IEEE TIM 69). It is VisDrone's opposite regime — one or two
LARGE defects per crop instead of seventy 20-px cars per frame — which is the
point of running the same detector on it (planning/neu_det_fpn_demo.md).

Two on-disk formats, byte-identical to scripts/datasets/preprocess_visdrone.py's, so the Lean
loaders, the FPN codegen and scripts/demos/yolo_map_visdrone.py run UNCHANGED:

  (default)   single-grid YOLOv1   perCell = 2*5 + 20   [lean_f32_load_voc_dims]
              + the uncapped GT sidecar val.full_gt.bin the scorer reads
  --fpn DIR   FPN multi-scale      Ntot = Σ_s A_s·15·g_s²  [lean_f32_load_voc_fpn]

The encoders are IMPORTED from scripts/datasets/preprocess_visdrone.py, not copied: the FPN
per-anchor width (5 + 10 classes = 15) is baked into the `fpnDetect` codegen
(LeanMlir/Types.lean), so NEU's six classes go into ids 0–5 of the ten-slot
one-hot and four slots never see a positive. The scorer averages AP over the
classes present in the GT, so those slots cost nothing.

Classes (alphabetical, the order every NEU paper uses):
  0 crazing  1 inclusion  2 patches  3 pitted_surface  4 rolled-in_scale  5 scratches

Split: NEU-DET ships no official split. This uses 1,080 / 360 / 360
(train / val / test), STRATIFIED by class — 180 / 60 / 60 of each class's 300
crops — from a fixed seed, so it is reproducible and every split is balanced.
It is the split size the published Faster R-CNN / YOLO rows use, so the
comparison is loose (different draws) but not meaningless. `split_stems()` is
the one place the draw lives; scripts/probes/neu_anchors.py imports it so the priors
are fitted on exactly the training images.

Geometry: the 200-px crop is upsampled to 448 (2.24×) so the R34 stem is the
same byte prefix of `.lake/build/jax_r34_imagenet.bin` as on VisDrone and the
bootstrap self-check passes as-is. The JPEGs are 3-channel files with grey
content, so `convert("RGB")` is the identity.

Usage:
  python3 scripts/datasets/preprocess_neu_det.py data/neu_det data/neu_det_fpn --size 448 --grid 14 --fpn data/neu_det
  python3 scripts/datasets/preprocess_neu_det.py data/neu_det data/neu_det448 --size 448 --grid 14
  python3 scripts/datasets/preprocess_neu_det.py data/neu_det --stats        # box statistics only

Flags: --size N --grid N --fpn DIR --seed N --stats --splits train,val,test
"""
import os, sys, struct
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow + numpy required.", file=sys.stderr); sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess_visdrone as pv   # the encoders, byte-for-byte the VisDrone ones

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
               "rolled-in_scale", "scratches"]
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
N_PER_CLASS = 300
SPLIT_PER_CLASS = {"train": 180, "val": 60, "test": 60}   # 1080 / 360 / 360
SEED = 0
INPUT_PX = 448


# ── annotations ──────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path):
    """Return (img_w, img_h, [(cid, xmin, ymin, xmax, ymax)]) in pixel coords.
    Boxes with an unknown class name are an ERROR, not a skip: a mirror whose
    names differ (underscores, case) would otherwise silently write a file with
    holes in it. Degenerate boxes are dropped and counted by the caller."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.find("width").text); h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_ID:
            raise ValueError(f"{xml_path}: unknown class name {name!r} "
                             f"(want one of {CLASS_NAMES})")
        bb = obj.find("bndbox")
        x0 = float(bb.find("xmin").text); y0 = float(bb.find("ymin").text)
        x1 = float(bb.find("xmax").text); y1 = float(bb.find("ymax").text)
        # clip to the frame, then drop only what is degenerate afterwards
        x0 = min(max(x0, 0.0), w); x1 = min(max(x1, 0.0), w)
        y0 = min(max(y0, 0.0), h); y1 = min(max(y1, 0.0), h)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        boxes.append((CLASS_ID[name], x0, y0, x1, y1))
    return w, h, boxes


def all_stems(neu_dir):
    """Every image stem with a matching XML, grouped by class from the filename
    (`crazing_17`). Checks the 1,800 / six-class census loudly."""
    imgs = Path(neu_dir) / "IMAGES"
    anns = Path(neu_dir) / "ANNOTATIONS"
    if not imgs.is_dir() or not anns.is_dir():
        print(f"ERROR: expected {imgs} and {anns} (run scripts/datasets/download_neu.sh)", file=sys.stderr)
        sys.exit(1)
    by_class = {n: [] for n in CLASS_NAMES}
    for p in sorted(imgs.glob("*.jpg")):
        cls = p.stem.rsplit("_", 1)[0]
        if cls not in by_class:
            print(f"ERROR: image {p.name} has no class prefix in {CLASS_NAMES}", file=sys.stderr)
            sys.exit(1)
        if not (anns / f"{p.stem}.xml").exists():
            print(f"ERROR: {p.name} has no {p.stem}.xml", file=sys.stderr); sys.exit(1)
        by_class[cls].append(p.stem)
    for n, stems in by_class.items():
        if len(stems) != N_PER_CLASS:
            print(f"ERROR: class {n} has {len(stems)} images, expected {N_PER_CLASS}",
                  file=sys.stderr)
            sys.exit(1)
    return by_class


def split_stems(neu_dir, seed=SEED):
    """{split: [stem, ...]} — stratified 180/60/60 per class from one seeded
    permutation of each class's stems (numeric order first, so the draw does not
    depend on directory listing order). Record order within a split is sorted,
    a stable content-derived order."""
    by_class = all_stems(neu_dir)
    out = {s: [] for s in SPLIT_PER_CLASS}
    for n in CLASS_NAMES:
        stems = sorted(by_class[n], key=lambda s: int(s.rsplit("_", 1)[1]))
        rng = np.random.RandomState(seed + CLASS_ID[n])
        perm = [stems[i] for i in rng.permutation(len(stems))]
        off = 0
        for s, k in SPLIT_PER_CLASS.items():
            out[s] += perm[off:off + k]; off += k
    for s in out:
        out[s].sort()
    return out


def load_split(neu_dir, stems):
    """[(stem, iw, ih, boxes)] for the given stems; images without a kept box
    are skipped (none in the shipped set, but say so if a mirror differs)."""
    anns = Path(neu_dir) / "ANNOTATIONS"
    per_image = []
    n_empty = 0
    for stem in stems:
        w, h, boxes = parse_voc_xml(anns / f"{stem}.xml")
        if not boxes:
            n_empty += 1; continue
        per_image.append((stem, w, h, boxes))
    if n_empty:
        print(f"  WARN: {n_empty} images with no kept box skipped", file=sys.stderr)
    return per_image


# ── statistics (Gate 0 of planning/neu_det_fpn_demo.md) ──────────────────────

def report_stats(per_image, label, input_px=INPUT_PX):
    """The numbers that decide the anchor table and the regime claim: the (w,h)
    distribution in source and 448-px pixels, boxes per image, and the fraction
    fpn_scale_of routes to P3 / P4 / P5. Expectation: P3 ≈ 0."""
    wh_src, wh_rel, cls, per_img = [], [], [], []
    for (_stem, iw, ih, boxes) in per_image:
        per_img.append(len(boxes))
        for (cid, x0, y0, x1, y1) in boxes:
            wh_src.append((x1 - x0, y1 - y0))
            wh_rel.append(((x1 - x0) / iw, (y1 - y0) / ih))
            cls.append(cid)
    wh_src = np.array(wh_src); wh_rel = np.array(wh_rel); cls = np.array(cls)
    px = wh_rel * input_px
    n = len(per_image); nb = len(cls)
    print(f"{label}: {n} images, {nb} boxes ({nb / max(n, 1):.2f}/img); "
          f"boxes/img histogram: " +
          ", ".join(f"{k}:{sum(1 for v in per_img if v == k)}" for k in range(1, 6)) +
          f", 6+:{sum(1 for v in per_img if v >= 6)}")
    mx = wh_src.max(axis=1)
    print(f"  source px  w median {np.median(wh_src[:, 0]):.0f} (p10 {np.percentile(wh_src[:, 0], 10):.0f}, "
          f"p90 {np.percentile(wh_src[:, 0], 90):.0f})  h median {np.median(wh_src[:, 1]):.0f} "
          f"(p10 {np.percentile(wh_src[:, 1], 10):.0f}, p90 {np.percentile(wh_src[:, 1], 90):.0f})  "
          f"max(w,h) median {np.median(mx):.0f} of 200")
    print(f"  @{input_px} px w median {np.median(px[:, 0]):.0f}  h median {np.median(px[:, 1]):.0f}  "
          f"max(w,h) median {np.median(px.max(axis=1)):.0f}; "
          f"{100.0 * np.mean(wh_rel.max(axis=1) > 0.5):.1f}% of boxes span > half the frame")
    scales = np.array([pv.fpn_scale_of(w, h, input_px) for (w, h) in wh_rel])
    frac = [100.0 * np.mean(scales == s) for s in range(3)]
    print(f"  FPN routing (max(w,h) < {pv.FPN_T_LO:.0f} / < {pv.FPN_T_HI:.0f} px @{input_px}): "
          f"P3 {frac[0]:.1f}%  P4 {frac[1]:.1f}%  P5 {frac[2]:.1f}%   "
          f"({int((scales == 0).sum())} / {int((scales == 1).sum())} / {int((scales == 2).sum())} boxes)")
    print(f"  {'class':>16} {'boxes':>5} {'/img':>5} {'med w':>6} {'med h':>6} "
          f"{'P3':>5} {'P4':>5} {'P5':>5}   (px @{input_px})")
    for c, name in enumerate(CLASS_NAMES):
        m = cls == c
        if not m.any():
            continue
        nimg = sum(1 for (_s, _w, _h, b) in per_image if any(cb[0] == c for cb in b))
        sc = scales[m]
        print(f"  {name:>16} {int(m.sum()):>5} {m.sum() / max(nimg, 1):>5.2f} "
              f"{np.median(px[m, 0]):>6.0f} {np.median(px[m, 1]):>6.0f} "
              f"{100.0 * np.mean(sc == 0):>4.0f}% {100.0 * np.mean(sc == 1):>4.0f}% "
              f"{100.0 * np.mean(sc == 2):>4.0f}%")


# ── writers (record layouts from scripts/datasets/preprocess_visdrone.py) ─────────────────────

def load_rgb(neu_dir, stem):
    return Image.open(Path(neu_dir) / "IMAGES" / f"{stem}.jpg").convert("RGB")


def process_split(neu_dir, per_image, out_path):
    """Single-grid format + the uncapped GT sidecar, written record-for-record in
    ONE loop so logits row k, val.bin record k and sidecar record k are the same
    image. Geometry comes from pv.IMG_SIZE / pv.GRID_H (set in main)."""
    gt_path = pv.full_gt_path(out_path)
    written = total_boxes = total_cells = 0
    with open(out_path, "wb") as f, open(gt_path, "wb") as g:
        f.write(struct.pack("<I", 0)); g.write(struct.pack("<I", 0))
        for (stem, iw, ih, boxes) in per_image:
            target, mask = pv.encode_targets(iw, ih, boxes)
            nb, blk = pv.pack_raw_boxes(iw, ih, boxes)
            img = load_rgb(neu_dir, stem).resize((pv.IMG_SIZE, pv.IMG_SIZE), Image.BILINEAR)
            chw = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()
            f.write(chw.tobytes()); f.write(target.tobytes()); f.write(mask.tobytes())
            f.write(struct.pack("<i", nb)); f.write(blk)
            g.write(pv.pack_full_gt(iw, ih, boxes))
            written += 1; total_boxes += len(boxes); total_cells += int(mask.sum())
        f.seek(0); f.write(struct.pack("<I", written))
        g.seek(0); g.write(struct.pack("<I", written))
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  wrote {out_path}: {written} records, {mb:.0f} MB | {total_boxes} boxes → "
          f"{total_cells} cells on the {pv.GRID_H}×{pv.GRID_W} grid "
          f"({100.0 * total_cells / max(total_boxes, 1):.1f}% encoded; rest lost to cell collisions)")
    print(f"  wrote {gt_path}: full uncapped GT ({total_boxes} boxes)")


def process_split_fpn(neu_dir, per_image, out_path, anchors_per_scale):
    """FPN format: image u8 + the flat [P3|P4|P5] target (Ntot f32), exactly what
    lean_f32_load_voc_fpn reads. Also writes the GT sidecar next to it so a
    scorer pointed at this val.bin still finds uncapped GT."""
    ntot = sum(len(anchors_per_scale[s]) * pv.PER_ANCHOR * g * g
               for s, g in enumerate(pv.FPN_GRIDS))
    gt_path = pv.full_gt_path(out_path)
    written = total_boxes = total_slots = 0
    hist = [0, 0, 0]
    with open(out_path, "wb") as f, open(gt_path, "wb") as g:
        f.write(struct.pack("<I", 0)); g.write(struct.pack("<I", 0))
        for (stem, iw, ih, boxes) in per_image:
            tgts, _msks, nslots = pv.encode_targets_fpn(iw, ih, boxes, anchors_per_scale, pv.IMG_SIZE)
            for (_c, x0, y0, x1, y1) in boxes:
                hist[pv.fpn_scale_of((x1 - x0) / iw, (y1 - y0) / ih, pv.IMG_SIZE)] += 1
            flat = np.concatenate([t.reshape(-1) for t in tgts]).astype(np.float32)
            assert flat.size == ntot, f"flat {flat.size} != Ntot {ntot}"
            img = load_rgb(neu_dir, stem).resize((pv.IMG_SIZE, pv.IMG_SIZE), Image.BILINEAR)
            chw = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()
            f.write(chw.tobytes()); f.write(flat.tobytes())
            g.write(pv.pack_full_gt(iw, ih, boxes))
            written += 1; total_boxes += len(boxes); total_slots += nslots
        f.seek(0); f.write(struct.pack("<I", written))
        g.seek(0); g.write(struct.pack("<I", written))
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  wrote {out_path}: {written} records, {mb:.0f} MB | Ntot={ntot}, "
          f"{total_boxes} GT boxes → {total_slots} multi-scale slots "
          f"({100.0 * total_slots / max(total_boxes, 1):.1f}% encoded); "
          f"routed P3/P4/P5 = {hist[0]}/{hist[1]}/{hist[2]}")
    print(f"  wrote {gt_path}: full uncapped GT ({total_boxes} boxes)")


def main():
    argv = sys.argv[1:]
    size = 224; grid = 7; fpn_dir = None; seed = SEED; stats_only = False
    splits = ("train", "val", "test")
    pos = []
    i = 0
    while i < len(argv):
        if argv[i] == "--size":
            size = int(argv[i + 1]); i += 2
        elif argv[i] == "--grid":
            grid = int(argv[i + 1]); i += 2
        elif argv[i] == "--fpn":
            fpn_dir = argv[i + 1]; i += 2
        elif argv[i] == "--seed":
            seed = int(argv[i + 1]); i += 2
        elif argv[i] == "--splits":
            splits = tuple(argv[i + 1].split(",")); i += 2
        elif argv[i] == "--stats":
            stats_only = True; i += 1
        else:
            pos.append(argv[i]); i += 1
    if not pos or (not stats_only and len(pos) != 2):
        print(__doc__); sys.exit(1)
    neu_dir = pos[0]
    pv.IMG_SIZE = size; pv.GRID_H = grid; pv.GRID_W = grid
    if size % grid != 0:
        print(f"WARN: grid {grid} does not evenly divide size {size}", file=sys.stderr)

    split = split_stems(neu_dir, seed)
    print(f"NEU-DET at {neu_dir}: 1,800 images, six classes; split seed {seed} → "
          + ", ".join(f"{s} {len(split[s])}" for s in split))
    data = {s: load_split(neu_dir, split[s]) for s in splits}
    for s in splits:
        report_stats(data[s], s.upper(), input_px=size)
    if stats_only:
        return

    out_dir = pos[1]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if fpn_dir:
        aps = [pv.load_anchors(os.path.join(fpn_dir, f"anchors_fpn_{p}.txt"))
               for p in ("p3", "p4", "p5")]
        if size != 448:
            print(f"WARN: FPN grids {pv.FPN_GRIDS} assume 448px input, got {size}", file=sys.stderr)
        ntot = sum(len(aps[s]) * pv.PER_ANCHOR * g * g for s, g in enumerate(pv.FPN_GRIDS))
        print(f"FPN encoding: A/scale={[len(a) for a in aps]}, grids={pv.FPN_GRIDS}, "
              f"Ntot={ntot}, {3 * size * size + ntot * 4} bytes/record")
        for s in splits:
            process_split_fpn(neu_dir, data[s], os.path.join(out_dir, f"{s}.bin"), aps)
    else:
        print(f"single-grid encoding at {size}px / {grid}×{grid} ({pv.record_size()} bytes/record)")
        for s in splits:
            process_split(neu_dir, data[s], os.path.join(out_dir, f"{s}.bin"))
    print("Done.")


if __name__ == "__main__":
    main()
