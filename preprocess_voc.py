#!/usr/bin/env python3
"""Pre-process Pascal VOC 2007 to raw binary format for the Lean YOLOv1 loader.

Usage: python3 preprocess_voc.py <voc_root> <output_dir> [--max-images N] [--split trainval|test|both]

  <voc_root> must contain the VOCdevkit/VOC2007 subtree:
    VOCdevkit/VOC2007/JPEGImages/*.jpg
    VOCdevkit/VOC2007/Annotations/*.xml
    VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
    VOCdevkit/VOC2007/ImageSets/Main/test.txt

Writes train.bin (5011 images, trainval) and val.bin (4952 images, test):
  Header: count (4 bytes, little-endian uint32)
  Per record (total 156,604 bytes):
    image  : 3*224*224     bytes uint8   (channel-first RGB, ImageNet-normalized on Lean read)
    target : 30*7*7 * 4    bytes float32 (NCHW: [perCell=30, gridH=7, gridW=7])
    mask   :    7*7 * 4    bytes float32 (per-cell objectness: 1.0 if a GT box's center falls here)

Target channel layout (perCell = numBoxes*5 + numClasses = 2*5 + 20 = 30):
  [0..2)   box 0 (x, y)     cell-relative center in [0, 1]
  [2..4)   box 0 (w, h)     image-relative (unscaled — codegen applies √ with ε floor)
  [4..5)   box 0 confidence (unused by codegen; set to 1.0 where mask=1 for clarity)
  [5..9)   box 1 (x, y, w, h) — never optimized (Option A: predictor 0 always)
  [9..10)  box 1 confidence — always noobj per Option A; unused by codegen
  [10..30) per-cell class one-hot (20 VOC classes in alphabetical order)

VOC classes (alphabetical, IDs 0..19):
  aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
  diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
  train, tvmonitor

If multiple GT boxes have their centers in the same cell, the last-encountered
box wins (deterministic — matches the order XML parser yields).

See planning/yolo_demo_v2.md Phase 1 and planning/yolo_demo_v3.md Phase 2.
"""
import argparse
import os
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow + numpy required. Install with: pip install Pillow numpy", file=sys.stderr)
    sys.exit(1)

IMG_SIZE     = 224
GRID_H       = 7
GRID_W       = 7
NUM_BOXES    = 2
NUM_CLASSES  = 20
PER_CELL     = NUM_BOXES * 5 + NUM_CLASSES  # 30

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_ID = {c: i for i, c in enumerate(VOC_CLASSES)}


def parse_annotation(xml_path):
    """Return (img_w, img_h, [(class_id, xmin, ymin, xmax, ymax), ...]) in pixel coords.

    Skips boxes with difficult=1 (paper convention)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue
        name = obj.find("name").text
        if name not in CLASS_TO_ID:
            continue  # silently skip unknown classes (shouldn't happen for VOC 2007)
        cid = CLASS_TO_ID[name]
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        boxes.append((cid, xmin, ymin, xmax, ymax))
    return img_w, img_h, boxes


def encode_targets(img_w, img_h, boxes):
    """Build (target [30, 7, 7], mask [7, 7]) numpy arrays for a single image.

    Box coords in the input are PIXEL coords on the ORIGINAL image size; we
    convert to [0, 1] image-relative since the model sees a resized 224x224."""
    target = np.zeros((PER_CELL, GRID_H, GRID_W), dtype=np.float32)
    mask = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for (cid, xmin, ymin, xmax, ymax) in boxes:
        cx = (xmin + xmax) / 2.0 / img_w   # [0, 1]
        cy = (ymin + ymax) / 2.0 / img_h
        w_rel = (xmax - xmin) / img_w
        h_rel = (ymax - ymin) / img_h
        # Cell containing the center.
        cell_j = int(cx * GRID_W)
        cell_i = int(cy * GRID_H)
        if cell_j >= GRID_W: cell_j = GRID_W - 1
        if cell_i >= GRID_H: cell_i = GRID_H - 1
        # Cell-relative center (in [0, 1]).
        cx_cell = cx * GRID_W - cell_j
        cy_cell = cy * GRID_H - cell_i
        # Fill box 0 slots. Last write wins if multiple boxes share a cell.
        target[0, cell_i, cell_j] = cx_cell
        target[1, cell_i, cell_j] = cy_cell
        target[2, cell_i, cell_j] = w_rel
        target[3, cell_i, cell_j] = h_rel
        target[4, cell_i, cell_j] = 1.0  # box 0 conf (unused by codegen but informative)
        # Class one-hot (clear other classes first so last-write-wins works).
        target[10:30, cell_i, cell_j] = 0.0
        target[10 + cid, cell_i, cell_j] = 1.0
        mask[cell_i, cell_j] = 1.0
    return target, mask


def process_split(voc_root, split_name, out_path, max_images=None):
    images_dir = voc_root / "JPEGImages"
    annot_dir = voc_root / "Annotations"
    split_path = voc_root / "ImageSets" / "Main" / f"{split_name}.txt"

    if not split_path.exists():
        print(f"  ERROR: missing split list {split_path}", file=sys.stderr)
        return False

    names = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if max_images:
        names = names[:max_images]
    print(f"  {split_name}: {len(names)} images")

    skipped = 0
    record_size = 3 * IMG_SIZE * IMG_SIZE + PER_CELL * GRID_H * GRID_W * 4 + GRID_H * GRID_W * 4

    with open(out_path, "wb") as f:
        # Header — count placeholder, rewrite at end.
        f.write(struct.pack("<I", 0))

        written = 0
        for name in names:
            img_path = images_dir / f"{name}.jpg"
            xml_path = annot_dir / f"{name}.xml"
            if not img_path.exists() or not xml_path.exists():
                skipped += 1
                continue
            try:
                img_w, img_h, boxes = parse_annotation(xml_path)
                if not boxes:
                    # Empty GT → mask all-zero, still write the record (model
                    # should predict no objects). Don't skip; the noobj-conf
                    # term still trains.
                    pass
                target, mask = encode_targets(img_w, img_h, boxes)

                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                img_arr = np.asarray(img, dtype=np.uint8)        # (H, W, 3)
                img_chw = img_arr.transpose(2, 0, 1).copy()      # (3, H, W)

                img_bytes = img_chw.tobytes()
                target_bytes = target.tobytes()
                mask_bytes = mask.tobytes()
                assert len(img_bytes) == 3 * IMG_SIZE * IMG_SIZE
                assert len(target_bytes) == PER_CELL * GRID_H * GRID_W * 4
                assert len(mask_bytes) == GRID_H * GRID_W * 4
                assert len(img_bytes) + len(target_bytes) + len(mask_bytes) == record_size

                f.write(img_bytes)
                f.write(target_bytes)
                f.write(mask_bytes)
                written += 1
            except Exception as e:
                print(f"  skipping {name}: {e}", file=sys.stderr)
                skipped += 1
                continue

        # Rewrite header with final count.
        f.seek(0)
        f.write(struct.pack("<I", written))

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  wrote {out_path}: {written} records, {size_mb:.1f} MB (skipped {skipped})")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("voc_root", help="VOCdevkit/VOC2007 directory")
    ap.add_argument("out_dir", help="output directory (will be created if missing)")
    ap.add_argument("--max-images", type=int, default=None, help="cap per split (for dev)")
    ap.add_argument("--split", choices=["trainval", "test", "both"], default="both")
    args = ap.parse_args()

    voc_root = Path(args.voc_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow either VOCdevkit/VOC2007/ or VOC2007/ as the root.
    if (voc_root / "VOCdevkit" / "VOC2007").exists():
        voc_root = voc_root / "VOCdevkit" / "VOC2007"
    elif (voc_root / "VOC2007").exists():
        voc_root = voc_root / "VOC2007"
    print(f"voc_root resolved to: {voc_root}")

    ok = True
    if args.split in ("trainval", "both"):
        ok &= process_split(voc_root, "trainval", out_dir / "train.bin", args.max_images)
    if args.split in ("test", "both"):
        ok &= process_split(voc_root, "test", out_dir / "val.bin", args.max_images)
    if not ok:
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
