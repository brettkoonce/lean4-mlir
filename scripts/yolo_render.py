#!/usr/bin/env python3
"""Render YOLOv1 predictions over the val images dumped by yolov1-voc-infer.

Reads:
  <dump_dir>/logits.bin    [N, 1470] float32
  <dump_dir>/images.bin    [N, 3, 224, 224] float32 (ImageNet-normalized)
  <dump_dir>/indices.txt   N lines of VOC test IDs (for labeling only)

Writes:
  <dump_dir>/grid.png      4x4 (or floor(√N)²) grid with boxes drawn
  <dump_dir>/det_<i>.png   per-image PNGs (small)

Decoding (per cell at grid (i, j) in [0, 7)):
  pred = logits.reshape(N, 30, 7, 7); per-cell channel layout
    ch 0..2    box 0 (x_cell, y_cell) in [0, 1] of cell
    ch 2..4    box 0 (w_rel, h_rel)   in [0, 1] of image (training applied √ to both
                                       pred and target so the model's raw output IS
                                       the width/height estimate, clamped to ≥ 0)
    ch 4       box 0 confidence
    ch 5..10   box 1 (ignored — "always predictor 0" rule, never trained)
    ch 10..30  per-cell class scores (model trained with MSE on one-hot; we soft-
                                      max here to get nice probabilities)
  detection score = pred_conf * max(softmax(class_scores))
  cx = (j + x_cell) / 7,   cy = (i + y_cell) / 7   (image-relative center)
  xmin = cx - w/2,  ymin = cy - h/2,  xmax/ymax sym

Per-class greedy NMS with IoU > 0.5 dropped; drops detections with score < 0.1.

See planning/yolo_demo_v3.md Phase 5.
"""
import argparse
import math
import os
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: numpy + Pillow required (.venv/bin/pip install Pillow numpy)", file=sys.stderr)
    sys.exit(1)

GRID = 7
NUM_BOXES = 2
NUM_CLASSES = 20
PER_CELL = NUM_BOXES * 5 + NUM_CLASSES  # 30

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Bright contrasting colors per class.
CLASS_COLORS = [
    (255,  50,  50), (255, 165,   0), (255, 215,   0), (124, 252,   0),
    (  0, 255, 127), (  0, 255, 255), (  0, 191, 255), ( 65, 105, 225),
    (138,  43, 226), (199,  21, 133), (255,  20, 147), (255, 105, 180),
    (160,  82,  45), (139,  69,  19), (210, 105,  30), (188, 143, 143),
    (255, 250, 205), (220, 220, 220), (128, 128, 128), ( 64,  64,  64),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iou(a, b):
    """IoU between two [xmin, ymin, xmax, ymax] boxes."""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def decode(pred_1470, score_thresh=0.1, nms_iou=0.5, sigmoid_conf=False):
    """Decode one [1470] flat YOLOv1 prediction → list of (cid, score, [xmin,ymin,xmax,ymax]).

    `sigmoid_conf=True` for models trained with the focal-BCE objectness head
    (useFocal): there the conf channel is a raw logit, so apply sigmoid to recover
    P(object). For the raw-MSE objectness head the conf already lives in [0,1]."""
    pred = pred_1470.reshape(PER_CELL, GRID, GRID)
    dets = []
    for i in range(GRID):
        for j in range(GRID):
            x_cell = pred[0, i, j]
            y_cell = pred[1, i, j]
            w_rel  = max(pred[2, i, j], 0.0)
            h_rel  = max(pred[3, i, j], 0.0)
            conf   = sigmoid(pred[4, i, j]) if sigmoid_conf else pred[4, i, j]
            class_logits = pred[10:30, i, j]
            class_probs = softmax(class_logits)
            cid = int(class_probs.argmax())
            cls_prob = float(class_probs[cid])
            score = float(conf) * cls_prob
            if score < score_thresh:
                continue
            cx = (j + x_cell) / GRID
            cy = (i + y_cell) / GRID
            xmin = float(cx - w_rel / 2)
            ymin = float(cy - h_rel / 2)
            xmax = float(cx + w_rel / 2)
            ymax = float(cy + h_rel / 2)
            dets.append((cid, score, [xmin, ymin, xmax, ymax]))

    # Per-class greedy NMS.
    kept = []
    for c in range(NUM_CLASSES):
        cdets = [d for d in dets if d[0] == c]
        cdets.sort(key=lambda d: -d[1])
        while cdets:
            top = cdets.pop(0)
            kept.append(top)
            cdets = [d for d in cdets if iou(top[2], d[2]) < nms_iou]
    return kept


def denormalize_chw(img_chw):
    """Reverse ImageNet normalization, return HWC uint8 RGB image."""
    img = img_chw.copy()                          # (3, 224, 224)
    img = img.transpose(1, 2, 0)                  # (224, 224, 3)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def draw_dets(img_rgb, dets, voc_id=None):
    """Draw boxes + labels on a 224×224 PIL image; returns annotated PIL.Image."""
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
    H, W = img_rgb.shape[:2]
    for (cid, score, box) in dets:
        x0 = max(int(box[0] * W), 0)
        y0 = max(int(box[1] * H), 0)
        x1 = min(int(box[2] * W), W - 1)
        y1 = min(int(box[3] * H), H - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        color = CLASS_COLORS[cid % len(CLASS_COLORS)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        label = f"{VOC_CLASSES[cid]} {score:.2f}"
        try:
            bbox = draw.textbbox((x0, y0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = 70, 12
        ty = y0 - text_h - 2 if y0 - text_h - 2 >= 0 else y0 + 2
        draw.rectangle([x0, ty, x0 + text_w + 4, ty + text_h + 2], fill=color)
        draw.text((x0 + 2, ty), label, fill=(0, 0, 0), font=font)
    if voc_id is not None:
        draw.text((4, H - 16), voc_id, fill=(255, 255, 255), font=font)
    return pil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir", help="output dir of yolov1-voc-infer (default figures/yolo_voc)")
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--sigmoid-conf", action="store_true",
                    help="apply sigmoid to the conf channel (use for focal-BCE objectness models)")
    ap.add_argument("--max-per-image", type=int, default=0,
                    help="keep only the top-K highest-score detections per image (0 = no cap)")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    logits_path = dump_dir / "logits.bin"
    images_path = dump_dir / "images.bin"
    indices_path = dump_dir / "indices.txt"
    for p in (logits_path, images_path, indices_path):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            sys.exit(1)

    logits = np.fromfile(logits_path, dtype=np.float32)
    images = np.fromfile(images_path, dtype=np.float32)
    indices = [ln.strip() for ln in indices_path.read_text().splitlines() if ln.strip()]
    n = logits.size // 1470
    logits = logits.reshape(n, 1470)
    images = images.reshape(n, 3, 224, 224)
    indices = indices[:n]
    print(f"loaded N={n} predictions from {dump_dir}")

    # Per-image PNGs.
    grid_imgs = []
    for i in range(n):
        dets = decode(logits[i], args.score_thresh, args.nms_iou, args.sigmoid_conf)
        if args.max_per_image > 0:                       # keep only the top-K by score
            dets = sorted(dets, key=lambda d: -d[1])[:args.max_per_image]
        img_rgb = denormalize_chw(images[i])
        pil = draw_dets(img_rgb, dets, voc_id=indices[i] if i < len(indices) else None)
        per_path = dump_dir / f"det_{i:03d}.png"
        pil.save(per_path)
        grid_imgs.append(pil)
        print(f"  {i+1}/{n}: {indices[i] if i < len(indices) else '?'} → {len(dets)} detections → {per_path.name}")

    # Grid: floor(√n)² images.
    side = max(1, int(math.floor(math.sqrt(n))))
    grid_w = grid_imgs[0].size[0] * side
    grid_h = grid_imgs[0].size[1] * side
    grid = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))
    for k in range(side * side):
        gx = (k % side) * grid_imgs[0].size[0]
        gy = (k // side) * grid_imgs[0].size[1]
        grid.paste(grid_imgs[k], (gx, gy))
    grid_path = dump_dir / "grid.png"
    grid.save(grid_path)
    print(f"wrote {grid_path} ({side}×{side} grid)")


if __name__ == "__main__":
    main()
