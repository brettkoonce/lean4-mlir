#!/usr/bin/env python3
"""Oxford-IIIT Pets head detection via 2x2 MOSAIC (YOLOv4-style).

Each training image is 4 random pets squashed into the 4 quadrants of a 224
canvas, with each head box transformed into quadrant coords. This keeps the
images IN-DISTRIBUTION (natural pet backgrounds — the ImageNet backbone stays
happy, unlike a flat gray canvas) while spreading head locations across all
quadrants → kills the central marginal AND gives multi-object (real NMS).

Same 157,728-byte record format → reuses the Lean loader / YOLOv1 codegen /
yolo_render unchanged. cat=7, dog=11 (VOC ids). 4 boxes/record.

Usage: python3 preprocess_pets_mosaic.py <pets_dir> <out_dir> [n_train] [n_val]
"""
import os, sys, struct
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow + numpy required.", file=sys.stderr); sys.exit(1)

IMG, GH, GW = 224, 7, 7
HALF = IMG // 2                                    # 112 quadrant size
PER_CELL = 2 * 5 + 20
MAX_BBOXES = 56
RECORD_SIZE = 3*IMG*IMG + PER_CELL*GH*GW*4 + GH*GW*4 + 4 + MAX_BBOXES*20
assert RECORD_SIZE == 157728
VOC = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
       "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
CID = {c: i for i, c in enumerate(VOC)}            # cat=7, dog=11
QUAD = [(0, 0), (HALF, 0), (0, HALF), (HALF, HALF)]


def parse_head(xml_path):
    root = ET.parse(xml_path).getroot()
    s = root.find("size"); iw = int(s.find("width").text); ih = int(s.find("height").text)
    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip().lower()
        if name in CID:
            bb = obj.find("bndbox")
            return iw, ih, CID[name], (float(bb.find("xmin").text), float(bb.find("ymin").text),
                                       float(bb.find("xmax").text), float(bb.find("ymax").text))
    return None


def mosaic(pets_dir, names4, rng):
    """Build a 2x2 mosaic canvas + its 4 head boxes (canvas-224 px)."""
    canvas = Image.new("RGB", (IMG, IMG), (128, 128, 128))
    boxes = []
    pets_dir = Path(pets_dir)
    for (qx, qy), name in zip(QUAD, names4):
        p = parse_head(pets_dir / "annotations" / "xmls" / f"{name}.xml")
        if p is None:
            continue
        iw, ih, cid, (x0, y0, x1, y1) = p
        img = Image.open(pets_dir / "images" / f"{name}.jpg").convert("RGB").resize((HALF, HALF), Image.BILINEAR)
        canvas.paste(img, (qx, qy))
        sx, sy = HALF / iw, HALF / ih
        boxes.append((cid, qx + x0*sx, qy + y0*sy, qx + x1*sx, qy + y1*sy))
    chw = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1).copy()
    return chw, boxes


def encode(boxes):
    target = np.zeros((PER_CELL, GH, GW), dtype=np.float32)
    mask = np.zeros((GH, GW), dtype=np.float32)
    for (cid, x0, y0, x1, y1) in boxes:
        cx = (x0+x1)/2/IMG; cy = (y0+y1)/2/IMG
        cj = min(int(cx*GW), GW-1); ci = min(int(cy*GH), GH-1)
        target[0, ci, cj] = cx*GW - cj
        target[1, ci, cj] = cy*GH - ci
        target[2, ci, cj] = (x1-x0)/IMG
        target[3, ci, cj] = (y1-y0)/IMG
        target[4, ci, cj] = 1.0
        target[10:30, ci, cj] = 0.0
        target[10+cid, ci, cj] = 1.0
        mask[ci, cj] = 1.0
    return target, mask


def raw_boxes(boxes):
    out = bytearray(MAX_BBOXES * 20)
    for i, (cid, x0, y0, x1, y1) in enumerate(boxes[:MAX_BBOXES]):
        struct.pack_into("<i", out, i*20, int(cid))
        struct.pack_into("<ffff", out, i*20+4, x0/IMG, y0/IMG, x1/IMG, y1/IMG)
    return len(boxes[:MAX_BBOXES]), bytes(out)


def build(pets_dir, cat_pool, dog_pool, n, out_path, seed0):
    written = 0; cat = dog = 0
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 0))
        for k in range(n):
            rng = np.random.default_rng(seed0 + k)
            # Balanced: each of the 4 quadrant pets is cat-or-dog 50/50, so the
            # box distribution is ~1:1 (kills the dog-majority class collapse).
            names4 = []
            for _ in range(4):
                pool = cat_pool if int(rng.integers(0, 2)) == 0 else dog_pool
                names4.append(pool[int(rng.integers(0, len(pool)))])
            chw, boxes = mosaic(pets_dir, names4, rng)
            if not boxes:
                continue
            for b in boxes:
                cat += (b[0] == 7); dog += (b[0] == 11)
            target, mask = encode(boxes); nb, blk = raw_boxes(boxes)
            f.write(chw.tobytes()); f.write(target.tobytes()); f.write(mask.tobytes())
            f.write(struct.pack("<i", nb)); f.write(blk)
            written += 1
        f.seek(0); f.write(struct.pack("<I", written))
    mb = os.path.getsize(out_path)/1024/1024
    print(f"  wrote {out_path}: {written} mosaics ({mb:.0f} MB) | boxes: {cat} cat, {dog} dog")


def main():
    pets_dir, out_dir = sys.argv[1], sys.argv[2]
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 3500
    n_val   = int(sys.argv[4]) if len(sys.argv) > 4 else 384
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    xmls = Path(pets_dir) / "annotations" / "xmls"
    names = sorted(n.split()[0] for n in
                   (Path(pets_dir)/"annotations"/"trainval.txt").read_text().splitlines()
                   if n.strip() and not n.startswith("#"))
    names = [n for n in names if (xmls / f"{n}.xml").exists()]
    val_pool = names[::10]; train_pool = [n for n in names if n not in set(val_pool)]
    # Oxford Pets convention: capitalized filename = cat, lowercase = dog.
    def split(pool): return ([x for x in pool if x[0].isupper()],
                             [x for x in pool if not x[0].isupper()])
    tcat, tdog = split(train_pool); vcat, vdog = split(val_pool)
    print(f"head-box pets: {len(names)} | train cat {len(tcat)}/dog {len(tdog)} | "
          f"val cat {len(vcat)}/dog {len(vdog)}; building {n_train} train / {n_val} val mosaics (balanced)")
    build(pets_dir, tcat, tdog, n_train, os.path.join(out_dir, "train.bin"), seed0=2000)
    build(pets_dir, vcat, vdog, n_val,   os.path.join(out_dir, "val.bin"),   seed0=8_000_000)
    print("Done.")


if __name__ == "__main__":
    main()
