#!/usr/bin/env python3
"""Pre-process VisDrone-DET2019 → the YOLOv1 detection .bin format.

Produces the EXACT on-disk record format of preprocess_pets_det.py
(157,728 bytes/record, perCell=30, 224x224 image, 7x7 grid) so the existing
Lean FFI loader, the YOLOv1 codegen, and scripts/yolo_render.py all work
UNCHANGED. This is the WS-A baseline: the single-grid YOLOv1 detector run on
VisDrone, where it is expected to collapse — 70 tiny objects per image cannot
be resolved by a 7x7 grid at 224x224 (a median 20x25 px box shrinks to ~2x5 px
after the resize, far below one 32 px cell). That collapse is the motivating
result for the multi-scale build; see planning/yolo_drone.md.

VisDrone annotation lines are:
  bbox_left, bbox_top, width, height, score, category, truncation, occlusion
We honor the two evaluation subtleties (or the numbers are silently wrong):
  * score == 0  -> ignored region, skip.
  * category 0 (ignored regions) and category 11 (others) -> excluded, skip.
The 10 kept classes are remapped from file ids 1..10 to 0..9.

Usage: python3 preprocess_visdrone.py <visdrone_dir> <out_dir>
  <visdrone_dir> must contain VisDrone2019-DET-{train,val}/ each with
  images/ and annotations/ (as extracted by download_visdrone.sh).
"""
import os, sys, struct, glob
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow + numpy required.", file=sys.stderr); sys.exit(1)

# Defaults = the 224/7×7 WS-A baseline; overridable with --size / --grid for the
# higher-resolution rung (e.g. --size 448 --grid 14). The Lean loader derives the
# same geometry from the spec's imageH (stride-32 ⇒ grid = size/32), so keep
# size = grid*32.
IMG_SIZE, GRID_H, GRID_W = 224, 7, 7
NUM_BOXES, NUM_CLASSES = 2, 20
PER_CELL = NUM_BOXES * 5 + NUM_CLASSES   # 30
MAX_BBOXES = 56


def record_size():
    return (3*IMG_SIZE*IMG_SIZE + PER_CELL*GRID_H*GRID_W*4
            + GRID_H*GRID_W*4 + 4 + MAX_BBOXES*20)

# file category id (1..10) -> our class id (0..9). 0 (ignored) and 11 (others)
# are dropped before this map is consulted.
VISDRONE_CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
                    "tricycle", "awning-tricycle", "bus", "motor"]


def parse_visdrone_txt(txt_path):
    """Return [(cid0_9, xmin, ymin, xmax, ymax)] in pixel coords, filtered."""
    boxes = []
    for ln in Path(txt_path).read_text().splitlines():
        ln = ln.strip().rstrip(",")
        if not ln:
            continue
        parts = ln.split(",")
        if len(parts) < 6:
            continue
        x, y, w, h, score, cat = (int(float(v)) for v in parts[:6])
        if score == 0:            # ignored region marker
            continue
        if cat == 0 or cat == 11: # ignored-regions / others: excluded from eval
            continue
        if w <= 0 or h <= 0:
            continue
        cid = cat - 1             # 1..10 -> 0..9
        boxes.append((cid, float(x), float(y), float(x + w), float(y + h)))
    return boxes


def encode_targets(img_w, img_h, boxes):
    """Same encoding as preprocess_pets_det.py. One box per cell: a later box
    in the same cell overwrites an earlier one — the coarse-grid limitation that
    makes this the collapse baseline on VisDrone's density."""
    target = np.zeros((PER_CELL, GRID_H, GRID_W), dtype=np.float32)
    mask = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for (cid, xmin, ymin, xmax, ymax) in boxes:
        cx = (xmin + xmax) / 2.0 / img_w; cy = (ymin + ymax) / 2.0 / img_h
        w_rel = (xmax - xmin) / img_w;    h_rel = (ymax - ymin) / img_h
        cj = min(int(cx * GRID_W), GRID_W - 1); ci = min(int(cy * GRID_H), GRID_H - 1)
        target[0, ci, cj] = cx * GRID_W - cj
        target[1, ci, cj] = cy * GRID_H - ci
        target[2, ci, cj] = w_rel
        target[3, ci, cj] = h_rel
        target[4, ci, cj] = 1.0
        target[10:30, ci, cj] = 0.0
        target[10 + cid, ci, cj] = 1.0
        mask[ci, cj] = 1.0
    return target, mask


# ── Anchor-based target encoding (brick #2, WS-C) ─────────────────────────────
# Layout differs from the single-box format: perCell = A*(5+C) with C=10 VisDrone
# classes, plus a per-anchor mask [A,gH,gW]. Each anchor slot a is
# [tx,ty,tw,th, obj, cls(10)]. A GT box is assigned to its cell (by center) and
# its best-shape anchor (by wh-IoU). Consumed by the anchor loader + codegen (bite 3).
NUM_CLASSES_A = 10
PER_ANCHOR = 5 + NUM_CLASSES_A          # 15


def load_anchors(path):
    rows = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        w, h = ln.split()
        rows.append((float(w), float(h)))
    return np.array(rows, dtype=np.float32)


def best_anchor(w_rel, h_rel, anchors):
    """Index of the anchor with max wh-IoU (shape match, origin-aligned)."""
    iw = np.minimum(w_rel, anchors[:, 0])
    ih = np.minimum(h_rel, anchors[:, 1])
    inter = iw * ih
    union = w_rel * h_rel + anchors[:, 0] * anchors[:, 1] - inter
    return int(np.argmax(inter / (union + 1e-12)))


def encode_targets_anchor(img_w, img_h, boxes, anchors):
    """Return (target[A*15,gH,gW], mask[A,gH,gW], n_slots_filled). Later GTs
    overwrite earlier on a (cell, anchor) collision — measured by the caller."""
    A = len(anchors)
    target = np.zeros((A * PER_ANCHOR, GRID_H, GRID_W), dtype=np.float32)
    mask = np.zeros((A, GRID_H, GRID_W), dtype=np.float32)
    filled = set()
    for (cid, xmin, ymin, xmax, ymax) in boxes:
        cx = (xmin + xmax) / 2.0 / img_w; cy = (ymin + ymax) / 2.0 / img_h
        w_rel = (xmax - xmin) / img_w;    h_rel = (ymax - ymin) / img_h
        cj = min(int(cx * GRID_W), GRID_W - 1); ci = min(int(cy * GRID_H), GRID_H - 1)
        a = best_anchor(w_rel, h_rel, anchors)
        base = a * PER_ANCHOR
        target[base + 0, ci, cj] = cx * GRID_W - cj
        target[base + 1, ci, cj] = cy * GRID_H - ci
        target[base + 2, ci, cj] = w_rel
        target[base + 3, ci, cj] = h_rel
        target[base + 4, ci, cj] = 1.0
        target[base + 5: base + 5 + NUM_CLASSES_A, ci, cj] = 0.0
        target[base + 5 + cid, ci, cj] = 1.0
        mask[a, ci, cj] = 1.0
        filled.add((a, ci, cj))
    return target, mask, len(filled)


def process_split_anchor(split_dir, out_path, anchors):
    A = len(anchors)
    imgs_dir = Path(split_dir) / "images"
    anns_dir = Path(split_dir) / "annotations"
    img_paths = sorted(glob.glob(str(imgs_dir / "*.jpg")))
    written = skipped = 0
    total_boxes = total_slots = 0
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 0))
        for img_path in img_paths:
            stem = Path(img_path).stem
            txt = anns_dir / f"{stem}.txt"
            if not txt.exists():
                skipped += 1; continue
            try:
                boxes = parse_visdrone_txt(txt)
                if not boxes:
                    skipped += 1; continue
                iw, ih = Image.open(img_path).size
                target, mask, nslots = encode_targets_anchor(iw, ih, boxes, anchors)
                nb, blk = pack_raw_boxes(iw, ih, boxes)
                img = Image.open(img_path).convert("RGB").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                chw = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()
                f.write(chw.tobytes()); f.write(target.tobytes()); f.write(mask.tobytes())
                f.write(struct.pack("<i", nb)); f.write(blk)
                written += 1; total_boxes += len(boxes); total_slots += nslots
            except Exception as e:
                print(f"  skip {stem}: {e}", file=sys.stderr); skipped += 1
        f.seek(0); f.write(struct.pack("<I", written))
    mb = os.path.getsize(out_path) / 1024 / 1024
    cover = 100.0 * total_slots / max(total_boxes, 1)
    print(f"  wrote {out_path}: {written} records ({skipped} skipped), {mb:.0f} MB | "
          f"A={A}, {total_boxes} GT boxes → {total_slots} anchor slots "
          f"({cover:.1f}% encoded; rest lost to (cell,anchor) collisions)")


def pack_raw_boxes(img_w, img_h, boxes):
    out = bytearray(MAX_BBOXES * 20)
    n = min(len(boxes), MAX_BBOXES)
    for i in range(n):
        cid, xmin, ymin, xmax, ymax = boxes[i]
        struct.pack_into("<i", out, i * 20, int(cid))
        struct.pack_into("<ffff", out, i * 20 + 4,
                         xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h)
    return n, bytes(out)


def process_split(split_dir, out_path):
    imgs_dir = Path(split_dir) / "images"
    anns_dir = Path(split_dir) / "annotations"
    img_paths = sorted(glob.glob(str(imgs_dir / "*.jpg")))
    written = skipped = 0
    total_boxes = 0
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 0))
        for img_path in img_paths:
            stem = Path(img_path).stem
            txt = anns_dir / f"{stem}.txt"
            if not txt.exists():
                skipped += 1; continue
            try:
                boxes = parse_visdrone_txt(txt)
                if not boxes:
                    skipped += 1; continue      # no eval-relevant object
                iw, ih = Image.open(img_path).size
                target, mask = encode_targets(iw, ih, boxes)
                nb, blk = pack_raw_boxes(iw, ih, boxes)
                img = Image.open(img_path).convert("RGB").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                chw = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()
                f.write(chw.tobytes()); f.write(target.tobytes()); f.write(mask.tobytes())
                f.write(struct.pack("<i", nb)); f.write(blk)
                written += 1; total_boxes += len(boxes)
            except Exception as e:
                print(f"  skip {stem}: {e}", file=sys.stderr); skipped += 1
        f.seek(0); f.write(struct.pack("<I", written))
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  wrote {out_path}: {written} records ({skipped} skipped), "
          f"{mb:.0f} MB | {total_boxes} boxes kept "
          f"({total_boxes / max(written,1):.1f}/img, of which "
          f"<={GRID_H * GRID_W} survive the {GRID_H}x{GRID_W} target per image)")


def main():
    global IMG_SIZE, GRID_H, GRID_W
    argv = [a for a in sys.argv[1:]]
    size = 224; grid = 7; anchors_path = None
    pos = []
    i = 0
    while i < len(argv):
        if argv[i] == "--size":
            size = int(argv[i+1]); i += 2
        elif argv[i] == "--grid":
            grid = int(argv[i+1]); i += 2
        elif argv[i] == "--anchors":
            anchors_path = argv[i+1]; i += 2
        else:
            pos.append(argv[i]); i += 1
    if len(pos) != 2:
        print(__doc__); sys.exit(1)
    IMG_SIZE = size; GRID_H = grid; GRID_W = grid
    if IMG_SIZE % GRID_H != 0:
        print(f"WARN: grid {GRID_H} does not evenly divide size {IMG_SIZE} "
              f"(stride = size/grid must be integer)", file=sys.stderr)
    print(f"encoding at {IMG_SIZE}px / {GRID_H}×{GRID_W} grid "
          f"({record_size()} bytes/record)")
    visdrone_dir, out_dir = pos[0], pos[1]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train_dir = Path(visdrone_dir) / "VisDrone2019-DET-train"
    val_dir = Path(visdrone_dir) / "VisDrone2019-DET-val"
    if not train_dir.exists() or not val_dir.exists():
        print(f"ERROR: expected {train_dir} and {val_dir}", file=sys.stderr); sys.exit(1)
    if anchors_path:
        anchors = load_anchors(anchors_path)
        A = len(anchors)
        rec = 3*IMG_SIZE*IMG_SIZE + A*PER_ANCHOR*GRID_H*GRID_W*4 + A*GRID_H*GRID_W*4 + 4 + MAX_BBOXES*20
        print(f"anchor encoding: {A} anchors, perCell={A*PER_ANCHOR}, {rec} bytes/record")
        process_split_anchor(train_dir, os.path.join(out_dir, "train.bin"), anchors)
        process_split_anchor(val_dir,   os.path.join(out_dir, "val.bin"), anchors)
    else:
        process_split(train_dir, os.path.join(out_dir, "train.bin"))
        process_split(val_dir,   os.path.join(out_dir, "val.bin"))
    print("Done.")


if __name__ == "__main__":
    main()
