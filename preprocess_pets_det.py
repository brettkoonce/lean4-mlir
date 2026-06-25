#!/usr/bin/env python3
"""Pre-process Oxford-IIIT Pets HEAD boxes → the YOLOv1 detection .bin format.

A cat-vs-dog single-head detector. Reuses the EXACT on-disk record format of
preprocess_voc.py (157,728 bytes/record, perCell=30, 20-class layout) so the
Lean FFI loader, the YOLOv1 codegen, and scripts/yolo_render.py all work
unchanged. Cat/dog are mapped to their VOC class ids (cat=7, dog=11) so the
renderer's class names line up for free.

Why Pets head-detect instead of VOC-person: every image has exactly ONE
prominent, always-present head box → no empty images, no 20-way class collapse,
and the head is large enough for the 7×7 grid. It's the clean "detection
obviously works" demo on real photos.

Usage: python3 preprocess_pets_det.py <pets_dir> <out_dir>
  <pets_dir> must contain:
    images/                      JPEG photos
    annotations/xmls/            PASCAL-VOC-format head boxes (~3686 imgs)
    annotations/trainval.txt     train split list
    annotations/test.txt         val/test split list
  Only images that HAVE a head-box xml are kept.
"""
import os, sys, struct
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow + numpy required.", file=sys.stderr); sys.exit(1)

IMG_SIZE, GRID_H, GRID_W = 224, 7, 7
NUM_BOXES, NUM_CLASSES = 2, 20
PER_CELL = NUM_BOXES * 5 + NUM_CLASSES   # 30
MAX_BBOXES = 56
RECORD_SIZE = 3*IMG_SIZE*IMG_SIZE + PER_CELL*GRID_H*GRID_W*4 + GRID_H*GRID_W*4 + 4 + MAX_BBOXES*20
assert RECORD_SIZE == 157728, RECORD_SIZE

# Map Pets' <name> (cat/dog) onto the VOC class ids so yolo_render labels match.
VOC_CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
               "chair","cow","diningtable","dog","horse","motorbike","person",
               "pottedplant","sheep","sofa","train","tvmonitor"]
CLASS_TO_ID = {c: i for i, c in enumerate(VOC_CLASSES)}   # cat=7, dog=11


def parse_head_xml(xml_path):
    """Return (img_w, img_h, [(cid, xmin, ymin, xmax, ymax)]) from a Pets head xml."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w = int(size.find("width").text); img_h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip().lower()
        if name not in CLASS_TO_ID:                 # expect cat/dog
            continue
        bb = obj.find("bndbox")
        boxes.append((CLASS_TO_ID[name],
                      float(bb.find("xmin").text), float(bb.find("ymin").text),
                      float(bb.find("xmax").text), float(bb.find("ymax").text)))
    return img_w, img_h, boxes


def encode_targets(img_w, img_h, boxes):
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


def pack_raw_boxes(img_w, img_h, boxes):
    out = bytearray(MAX_BBOXES * 20)
    n = min(len(boxes), MAX_BBOXES)
    for i in range(n):
        cid, xmin, ymin, xmax, ymax = boxes[i]
        struct.pack_into("<i", out, i * 20, int(cid))
        struct.pack_into("<ffff", out, i * 20 + 4,
                         xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h)
    return n, bytes(out)


def split_names(pets_dir, split):
    p = Path(pets_dir) / "annotations" / f"{split}.txt"
    return [ln.split()[0] for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def process_names(pets_dir, names, out_path):
    pets_dir = Path(pets_dir)
    imgs = pets_dir / "images"; xmls = pets_dir / "annotations" / "xmls"
    written = skipped = 0
    counts = {"cat": 0, "dog": 0}
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 0))
        for name in names:
            img_path = imgs / f"{name}.jpg"; xml_path = xmls / f"{name}.xml"
            if not img_path.exists() or not xml_path.exists():
                skipped += 1; continue          # no head box → skip
            try:
                iw, ih, boxes = parse_head_xml(xml_path)
                if not boxes:
                    skipped += 1; continue
                for b in boxes:
                    counts["cat" if b[0] == 7 else "dog"] += 1
                target, mask = encode_targets(iw, ih, boxes)
                nb, blk = pack_raw_boxes(iw, ih, boxes)
                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                chw = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()
                f.write(chw.tobytes()); f.write(target.tobytes()); f.write(mask.tobytes())
                f.write(struct.pack("<i", nb)); f.write(blk)
                written += 1
            except Exception as e:
                print(f"  skip {name}: {e}", file=sys.stderr); skipped += 1
        f.seek(0); f.write(struct.pack("<I", written))
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  wrote {out_path}: {written} records ({skipped} skipped), "
          f"{mb:.0f} MB | boxes: {counts['cat']} cat, {counts['dog']} dog")


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    pets_dir, out_dir = sys.argv[1], sys.argv[2]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Pets ships head-box xmls for the trainval split only (test has none), so
    # build train/val by holding out every 10th xml-annotated image as val.
    xmls = Path(pets_dir) / "annotations" / "xmls"
    names = [n for n in split_names(pets_dir, "trainval") if (xmls / f"{n}.xml").exists()]
    names.sort()
    val   = names[::10]                       # ~10% holdout, deterministic
    train = [n for n in names if n not in set(val)]
    print(f"head-box images: {len(names)} → train {len(train)} / val {len(val)}")
    process_names(pets_dir, train, os.path.join(out_dir, "train.bin"))
    process_names(pets_dir, val,   os.path.join(out_dir, "val.bin"))
    print("Done.")


if __name__ == "__main__":
    main()
