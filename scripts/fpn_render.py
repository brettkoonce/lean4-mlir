#!/usr/bin/env python3
"""Draw the FPN detector's predictions over VisDrone val images.

The Pets-era `scripts/yolo_render.py` cannot read this head: it assumes a single
7x7 grid and a 1470-wide output, where the multi-scale head emits 185,220 across
P3/P4/P5 with 3 anchors per scale. Rather than reimplement the decode (and risk a
picture that disagrees with the metric), this imports `decode_fpn` from
`yolo_map_visdrone.py`, so the boxes drawn here are by construction the boxes
scored there.

Images come straight out of the FPN record: `process_split_fpn` writes each record
as uint8 CHW 3x448x448 followed by the flat target, so there is no normalization to
undo and no separate images.bin to keep in sync.

Usage:
    python3 scripts/fpn_render.py runs/fpn_x/logits.bin data/visdrone_fpn/val.bin \\
        --fpn data/visdrone --out demos/figures/visdrone_fpn.png --n 6

    # side-by-side with ground truth, which is what makes the density legible
    ... --gt data/visdrone448/val.full_gt.bin
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_map_visdrone import (CLASS_NAMES, decode_fpn, load_anchors_file,
                               read_gt_full)

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("ERROR: Pillow required", file=sys.stderr)
    sys.exit(1)

IMG_PX = 448
IMG_BYTES = 3 * IMG_PX * IMG_PX
NTOT = 185220
FPN_GRIDS = (56, 28, 14)

# One colour per VisDrone class. Distinct hues rather than a ramp: at 70 boxes an
# image the reader is tracing individual boxes, not reading a magnitude.
COLORS = {
    0: (255, 82, 82),    1: (255, 158, 40),   2: (255, 235, 59),
    3: (76, 217, 100),   4: (0, 200, 190),    5: (64, 156, 255),
    6: (140, 122, 255),  7: (214, 106, 255),  8: (255, 92, 170),
    9: (170, 170, 170),
}


def read_image(val_path, idx):
    """Record `idx`'s image as HWC uint8. Records are fixed-size after a 4-byte count."""
    rec = 4 + idx * (IMG_BYTES + NTOT * 4)
    with open(val_path, "rb") as f:
        f.seek(rec)
        buf = f.read(IMG_BYTES)
    chw = np.frombuffer(buf, dtype=np.uint8).reshape(3, IMG_PX, IMG_PX)
    return np.ascontiguousarray(chw.transpose(1, 2, 0))


def draw(img_hwc, boxes, width=2, label=None):
    """boxes: list of (cid, score_or_None, (xmin,ymin,xmax,ymax)) normalized to [0,1]."""
    pil = Image.fromarray(img_hwc).convert("RGB")
    d = ImageDraw.Draw(pil)
    for cid, _score, (x0, y0, x1, y1) in boxes:
        c = COLORS.get(cid, (255, 255, 255))
        d.rectangle([x0 * IMG_PX, y0 * IMG_PX, x1 * IMG_PX, y1 * IMG_PX],
                    outline=c, width=width)
    if label:
        d.rectangle([0, 0, 8 * len(label) + 8, 16], fill=(0, 0, 0))
        d.text((4, 3), label, fill=(255, 255, 255))
    return pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits")
    ap.add_argument("val_bin", help="data/visdrone_fpn/val.bin (images live here)")
    ap.add_argument("--fpn", default="data/visdrone", help="dir with anchors_fpn_*.txt")
    ap.add_argument("--gt", default=None, help="full-GT sidecar; adds a GT column")
    ap.add_argument("--out", default="demos/figures/visdrone_fpn.png")
    ap.add_argument("--n", type=int, default=6, help="images in the figure")
    ap.add_argument("--conf-thresh", type=float, default=0.05)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=300,
                    help="cap drawn dets; the metric's 1000 is unreadable as a picture")
    ap.add_argument("--indices", default=None,
                    help="comma-separated record indices; default = densest images")
    args = ap.parse_args()

    scales = [(g, load_anchors_file(str(Path(args.fpn) / f"anchors_fpn_{p}.txt")))
              for g, p in zip(FPN_GRIDS, ("p3", "p4", "p5"))]

    logits = np.fromfile(args.logits, dtype=np.float32)
    n_rec = logits.size // NTOT
    logits = logits.reshape(n_rec, NTOT)
    print(f"{n_rec} records of {NTOT} logits")

    gts = None
    if args.gt:
        _n, gts = read_gt_full(args.gt)

    if args.indices:
        picks = [int(x) for x in args.indices.split(",")]
    elif gts is not None:
        # Densest images first — the whole point of VisDrone is the density, and a
        # sparse frame makes the detector look better than it is.
        picks = sorted(range(min(n_rec, len(gts))),
                       key=lambda i: -len(gts[i]))[:args.n]
    else:
        picks = list(range(min(args.n, n_rec)))
    print("records:", picks)

    cols = []
    for i in picks:
        img = read_image(args.val_bin, i)
        dets = decode_fpn(logits[i], scales, args.conf_thresh, args.nms_iou,
                          topk=args.topk)
        # decode_fpn returns (cid, score, xyxy) triples ranked by score
        pred_boxes = [(d[0], d[1], d[2]) for d in dets]
        pred = draw(img, pred_boxes, label=f"pred {len(pred_boxes)}")
        if gts is not None:
            gt_boxes = [(c, None, b) for (c, b) in gts[i]]
            col = Image.new("RGB", (IMG_PX, IMG_PX * 2 + 4), (16, 16, 16))
            col.paste(draw(img, gt_boxes, label=f"truth {len(gt_boxes)}"), (0, 0))
            col.paste(pred, (0, IMG_PX + 4))
        else:
            col = pred
        cols.append(col)

    w, h = cols[0].size
    sheet = Image.new("RGB", (w * len(cols) + 4 * (len(cols) - 1), h), (16, 16, 16))
    for k, c in enumerate(cols):
        sheet.paste(c, (k * (w + 4), 0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
