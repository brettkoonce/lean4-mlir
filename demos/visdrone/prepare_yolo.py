#!/usr/bin/env python3
"""VisDrone-DET -> YOLO layout, for the standalone PyTorch reference detector.

The filter here is byte-for-byte the same rule as the Lean pipeline's
`parse_visdrone_txt` (../preprocess_visdrone.py): drop score==0 rows, drop
categories 0 (ignored regions) and 11 (others), remap file category 1..10 to
class id 0..9. That is what makes the two arms' mAP numbers comparable -- if
this filter drifts, the whole point of the reference is lost.

Labels are written as a SIBLING of the real image directories, not into a
separate tree. Ultralytics derives the label path by string-replacing "/images/"
with "/labels/" on the resolved image path, so a symlinked image directory sends
it looking inside the source tree anyway -- writing the labels where it will
actually look is the robust arrangement, and it leaves the VisDrone
`annotations/` directory untouched.

    ./.venv/bin/python3 prepare_yolo.py
"""
import os
import sys
from pathlib import Path

from PIL import Image

SRC = Path("../data/visdrone")
DST = Path("data")

CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
           "tricycle", "awning-tricycle", "bus", "motor"]

SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val"}


def parse(txt_path):
    """[(cid0_9, xmin, ymin, xmax, ymax)] in pixels. Mirrors parse_visdrone_txt."""
    boxes = []
    for ln in Path(txt_path).read_text().splitlines():
        ln = ln.strip().rstrip(",")
        if not ln:
            continue
        parts = ln.split(",")
        if len(parts) < 6:
            continue
        x, y, w, h, score, cat = (int(float(v)) for v in parts[:6])
        if score == 0 or cat == 0 or cat == 11 or w <= 0 or h <= 0:
            continue
        boxes.append((cat - 1, float(x), float(y), float(x + w), float(y + h)))
    return boxes


def main():
    total_img = total_box = 0
    for split, srcdir in SPLITS.items():
        img_src = SRC / srcdir / "images"
        ann_src = SRC / srcdir / "annotations"
        if not img_src.is_dir():
            sys.exit(f"missing {img_src} -- run download_visdrone.sh first")

        lbl_dst = SRC / srcdir / "labels"
        lbl_dst.mkdir(parents=True, exist_ok=True)

        n_img = n_box = n_empty = 0
        per_class = [0] * len(CLASSES)
        for jpg in sorted(img_src.glob("*.jpg")):
            txt = ann_src / (jpg.stem + ".txt")
            boxes = parse(txt) if txt.exists() else []
            with Image.open(jpg) as im:
                iw, ih = im.size
            lines = []
            for cid, x0, y0, x1, y1 in boxes:
                # clip to frame: VisDrone boxes can run past the edge
                x0, y0 = max(0.0, x0), max(0.0, y0)
                x1, y1 = min(float(iw), x1), min(float(ih), y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                cx, cy = (x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih
                bw, bh = (x1 - x0) / iw, (y1 - y0) / ih
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                per_class[cid] += 1
            (lbl_dst / (jpg.stem + ".txt")).write_text("\n".join(lines) + "\n")
            n_img += 1
            n_box += len(lines)
            n_empty += (len(lines) == 0)

        print(f"{split}: {n_img} images, {n_box} boxes "
              f"({n_box/max(n_img,1):.1f}/img), {n_empty} empty")
        for name, n in sorted(zip(CLASSES, per_class), key=lambda t: -t[1]):
            print(f"    {name:<16} {n:7d} ({100*n/max(n_box,1):5.1f}%)")
        total_img += n_img
        total_box += n_box

    DST.mkdir(parents=True, exist_ok=True)
    yaml = DST / "visdrone.yaml"
    yaml.write_text(
        f"path: {SRC.resolve()}\n"
        f"train: {SPLITS['train']}/images\n"
        f"val: {SPLITS['val']}/images\n"
        "\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    print(f"\nwrote {yaml}  ({total_img} images, {total_box} boxes)")


if __name__ == "__main__":
    main()
