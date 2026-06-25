#!/usr/bin/env python3
"""Oxford-IIIT Pets head detection, RANDOM-PLACED on a canvas.

Same as preprocess_pets_det.py, but each pet is resized to a random scale and
pasted at a random position on a 224 gray canvas (the head box is transformed to
match). This makes the head-location *marginal uniform*, so a center-prior is
useless — the model is forced to actually localize the head per image. Direct
fix for the marginal-dominance plateau that centered datasets (VOC-person, plain
Pets) hit on the coarse 7×7 grid.

Same 157,728-byte record format → reuses the Lean loader / YOLOv1 codegen /
yolo_render unchanged. cat=7, dog=11 (VOC ids).

Usage: python3 preprocess_pets_canvas.py <pets_dir> <out_dir>
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
NUM_BOXES, NUM_CLASSES = 2, 20
PER_CELL = NUM_BOXES * 5 + NUM_CLASSES
MAX_BBOXES = 56
RECORD_SIZE = 3*IMG*IMG + PER_CELL*GH*GW*4 + GH*GW*4 + 4 + MAX_BBOXES*20
assert RECORD_SIZE == 157728
VOC = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
       "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
CID = {c: i for i, c in enumerate(VOC)}            # cat=7, dog=11
SCALE_MIN, SCALE_MAX = 80, 170                     # pet longer-side px on the 224 canvas


def parse_head_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    s = root.find("size"); iw = int(s.find("width").text); ih = int(s.find("height").text)
    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip().lower()
        if name not in CID:
            continue
        bb = obj.find("bndbox")
        return iw, ih, CID[name], (float(bb.find("xmin").text), float(bb.find("ymin").text),
                                   float(bb.find("xmax").text), float(bb.find("ymax").text))
    return None


def encode_one(cid, hx0, hy0, hx1, hy1):
    """Encode a single head box (canvas-224 pixel coords) → (target[30,7,7], mask[7,7])."""
    target = np.zeros((PER_CELL, GH, GW), dtype=np.float32)
    mask = np.zeros((GH, GW), dtype=np.float32)
    cx = (hx0 + hx1) / 2.0 / IMG; cy = (hy0 + hy1) / 2.0 / IMG
    w_rel = (hx1 - hx0) / IMG;    h_rel = (hy1 - hy0) / IMG
    cj = min(int(cx * GW), GW - 1); ci = min(int(cy * GH), GH - 1)
    target[0, ci, cj] = cx * GW - cj
    target[1, ci, cj] = cy * GH - ci
    target[2, ci, cj] = w_rel
    target[3, ci, cj] = h_rel
    target[4, ci, cj] = 1.0
    target[10 + cid, ci, cj] = 1.0
    mask[ci, cj] = 1.0
    return target, mask


def raw_boxes(cid, hx0, hy0, hx1, hy1):
    out = bytearray(MAX_BBOXES * 20)
    struct.pack_into("<i", out, 0, int(cid))
    struct.pack_into("<ffff", out, 4, hx0/IMG, hy0/IMG, hx1/IMG, hy1/IMG)
    return 1, bytes(out)


def place(pets_dir, name, rng):
    """Load pet, paste at random scale+pos on gray canvas, return (chw_uint8, cid, box224)."""
    img = Image.open(Path(pets_dir) / "images" / f"{name}.jpg").convert("RGB")
    parsed = parse_head_xml(Path(pets_dir) / "annotations" / "xmls" / f"{name}.xml")
    if parsed is None:
        return None
    iw, ih, cid, (xmin, ymin, xmax, ymax) = parsed
    # random scale (longer side -> [SCALE_MIN, SCALE_MAX]), preserve aspect
    longer = max(iw, ih)
    s = rng.uniform(SCALE_MIN, SCALE_MAX) / longer
    pw, ph = max(1, int(iw * s)), max(1, int(ih * s))
    pet = img.resize((pw, ph), Image.BILINEAR)
    # random paste position
    px = rng.integers(0, max(1, IMG - pw + 1)); py = rng.integers(0, max(1, IMG - ph + 1))
    canvas = Image.new("RGB", (IMG, IMG), (128, 128, 128))
    canvas.paste(pet, (int(px), int(py)))
    # transform head box -> canvas coords
    hx0 = px + xmin * s; hy0 = py + ymin * s
    hx1 = px + xmax * s; hy1 = py + ymax * s
    chw = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1).copy()
    return chw, cid, (hx0, hy0, hx1, hy1)


def process(pets_dir, names, out_path, seed0):
    written = 0; counts = {7: 0, 11: 0}
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 0))
        for k, name in enumerate(names):
            rng = np.random.default_rng(seed0 + k)
            r = place(pets_dir, name, rng)
            if r is None:
                continue
            chw, cid, (hx0, hy0, hx1, hy1) = r
            target, mask = encode_one(cid, hx0, hy0, hx1, hy1)
            nb, blk = raw_boxes(cid, hx0, hy0, hx1, hy1)
            f.write(chw.tobytes()); f.write(target.tobytes()); f.write(mask.tobytes())
            f.write(struct.pack("<i", nb)); f.write(blk)
            counts[cid] += 1; written += 1
        f.seek(0); f.write(struct.pack("<I", written))
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  wrote {out_path}: {written} ({mb:.0f} MB) | cat={counts[7]} dog={counts[11]}")


def main():
    pets_dir, out_dir = sys.argv[1], sys.argv[2]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    xmls = Path(pets_dir) / "annotations" / "xmls"
    names = sorted(n.split()[0] for n in
                   (Path(pets_dir) / "annotations" / "trainval.txt").read_text().splitlines()
                   if n.strip() and not n.startswith("#"))
    names = [n for n in names if (xmls / f"{n}.xml").exists()]
    val = names[::10]; train = [n for n in names if n not in set(val)]
    print(f"head-box images: {len(names)} → train {len(train)} / val {len(val)} (random-placed)")
    process(pets_dir, train, os.path.join(out_dir, "train.bin"), seed0=1000)
    process(pets_dir, val,   os.path.join(out_dir, "val.bin"),   seed0=9_000_000)
    print("Done.")


if __name__ == "__main__":
    main()
