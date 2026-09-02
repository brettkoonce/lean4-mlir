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

    # truth | before | after, on visually distinct frames, with a class legend
    python3 scripts/fpn_render.py runs/after/logits.bin data/visdrone_fpn/val.bin \\
        --gt data/visdrone448/val.full_gt.bin --compare runs/before/logits.bin \\
        --labels "truth,before 0.1961,after 0.2363" --diverse --scale 2 --n 3

⚠ `--diverse` exists because the default "densest frames first" pick lands inside a
single VisDrone SEQUENCE: val records are consecutive video frames, so the four
densest are four views of one street corner, and the figure reads as a duplicate.
`--diverse` takes the densest frame in a mid-density band, then greedily adds the
frames least similar to those already chosen (8x8 thumbnail cosine).

⚠ At ~70 boxes an image a raw dump is a wall of rectangles. `--topk-per-gt` draws
each model's K highest-scoring detections where K is that frame's GT count — the
same rule for every model, so the comparison stays fair while staying readable.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_map_visdrone import (CLASS_NAMES, decode_fpn, iou,
                               load_anchors_file, read_gt_full)

try:
    from PIL import Image, ImageDraw, ImageFont
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


def _font(size):
    """DejaVu if present, else PIL's bitmap default. The default is ~11 px and
    unreadable on a 2x sheet, which is the only reason this bothers."""
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def legend_strip(width, height=44, pad=12, match=False):
    """One swatch + name per class. Ten distinct hues are useless without a key."""
    strip = Image.new("RGB", (width, height), (16, 16, 16))
    d = ImageDraw.Draw(strip)
    f = _font(17)
    x = pad
    items = ([("tp", "correct"), ("fp", "false positive"), ("fn", "missed")]
             if match else sorted(CLASS_NAMES.items()))
    for cid, name in items:
        sw = 18
        y = (height - sw) // 2
        fill = MATCH_COLORS[cid] if isinstance(cid, str) else COLORS.get(cid, (255, 255, 255))
        d.rectangle([x, y, x + sw, y + sw], fill=fill)
        x += sw + 6
        d.text((x, y + 1), name, fill=(235, 235, 235), font=f)
        x += int(d.textlength(name, font=f)) + pad + 6
    return strip


def pick_diverse(val_path, n_rec, gts, n, lo=35, hi=110):
    """Densest frame in the [lo, hi] GT band, then greedily the least-similar ones.

    Similarity is cosine over an 8x8x3 thumbnail — crude, but the failure it has to
    catch is "the same intersection four times", which is not a subtle one."""
    counts = np.array([len(g) for g in gts[:n_rec]])
    thumbs = np.zeros((n_rec, 3 * 8 * 8), np.float32)
    with open(val_path, "rb") as f:
        for i in range(n_rec):
            f.seek(4 + i * (IMG_BYTES + NTOT * 4))
            a = np.frombuffer(f.read(IMG_BYTES), np.uint8).reshape(3, IMG_PX, IMG_PX)
            thumbs[i] = a[:, ::56, ::56][:, :8, :8].reshape(-1).astype(np.float32)
    thumbs /= np.linalg.norm(thumbs, axis=1, keepdims=True) + 1e-6
    band = np.where((counts >= lo) & (counts <= hi))[0]
    if len(band) == 0:
        band = np.arange(n_rec)
    picked = [int(band[np.argmax(counts[band])])]
    while len(picked) < n:
        sim = thumbs[band] @ thumbs[picked].T
        cand = band[np.argsort(sim.max(axis=1))[:20]]
        picked.append(int(sorted(cand, key=lambda i: -counts[i])[0]))
    return picked


# --match colours. Class hues answer "what did it say"; these answer "was it right",
# which is the question a before/after figure is actually asking.
MATCH_COLORS = {"tp": (60, 220, 90), "fp": (255, 70, 70), "fn": (255, 210, 0)}


def match_dets(dets, gt, thr=0.5):
    """Greedy score-ordered IoU matching, same rule the AP uses: a detection takes
    the best free GT of its own class at IoU >= thr, else it is a false positive.

    ⚠ This RE-DERIVES the metric's matching rather than importing it — the mAP loop
    is fused with the AP accumulation and has no reusable matcher. It shares `iou`
    and the rule, so a picture and a number can still drift apart here; treat the
    counts as illustrative, and the scorer as the source of truth."""
    used = [False] * len(gt)
    out = []
    for cid, score, box in sorted(dets, key=lambda d: -d[1]):
        best, best_j = thr, -1
        for j, (gcid, gbox) in enumerate(gt):
            if used[j] or gcid != cid:
                continue
            v = iou(box, gbox)
            if v >= best:
                best, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            out.append(("tp", cid, box))
        else:
            out.append(("fp", cid, box))
    misses = [("fn", gt[j][0], gt[j][1]) for j in range(len(gt)) if not used[j]]
    return out + misses


def read_image(val_path, idx):
    """Record `idx`'s image as HWC uint8. Records are fixed-size after a 4-byte count."""
    rec = 4 + idx * (IMG_BYTES + NTOT * 4)
    with open(val_path, "rb") as f:
        f.seek(rec)
        buf = f.read(IMG_BYTES)
    chw = np.frombuffer(buf, dtype=np.uint8).reshape(3, IMG_PX, IMG_PX)
    return np.ascontiguousarray(chw.transpose(1, 2, 0))


def draw(img_hwc, boxes, width=2, label=None, scale=1):
    """boxes: list of (cid, score_or_None, (xmin,ymin,xmax,ymax)) normalized to [0,1].

    `scale` upsamples the IMAGE before drawing, so box outlines stay 1 device pixel
    wide against a bigger frame. VisDrone objects are 2-5 px at 448; drawn at 1x a
    2 px box IS its own outline and the picture shows nothing."""
    pil = Image.fromarray(img_hwc).convert("RGB")
    px = IMG_PX * scale
    if scale != 1:
        pil = pil.resize((px, px), Image.LANCZOS)
    d = ImageDraw.Draw(pil)
    for cid, _score, (x0, y0, x1, y1) in boxes:
        c = MATCH_COLORS[cid] if isinstance(cid, str) else COLORS.get(cid, (255, 255, 255))
        d.rectangle([x0 * px, y0 * px, x1 * px, y1 * px], outline=c, width=width)
    if label:
        f = _font(16 + 4 * scale)
        tw = int(d.textlength(label, font=f))
        d.rectangle([0, 0, tw + 16, 14 + 8 * scale], fill=(0, 0, 0))
        d.text((8, 4), label, fill=(255, 255, 255), font=f)
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
    ap.add_argument("--compare", default=None, metavar="LOGITS",
                    help="a second model's logits; adds a column so the sheet reads "
                         "truth | this | that. Both are decoded identically.")
    ap.add_argument("--labels", default=None,
                    help="comma-separated panel labels, e.g. 'truth,before,after'")
    ap.add_argument("--diverse", action="store_true",
                    help="pick visually DISTINCT frames instead of the densest. The "
                         "densest are consecutive frames of one sequence.")
    ap.add_argument("--scale", type=int, default=1, help="upsample factor per panel")
    ap.add_argument("--topk-per-gt", action="store_true",
                    help="draw each model's K best detections, K = that frame's GT "
                         "count. Same rule per model, so the comparison stays fair.")
    ap.add_argument("--match", action="store_true",
                    help="colour by CORRECTNESS (green TP / red FP / yellow missed GT) "
                         "instead of by class, and drop the truth column — the class "
                         "view cannot show what a before/after actually changed")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--layout", choices=("rows", "cols"), default="rows",
                    help="rows: one FRAME per row, panels across (reads left-to-right "
                         "for a 3-way compare). cols: one frame per column, panels "
                         "stacked — the original landscape shape, better on a page.")
    ap.add_argument("--no-legend", action="store_true")
    args = ap.parse_args()

    scales = [(g, load_anchors_file(str(Path(args.fpn) / f"anchors_fpn_{p}.txt")))
              for g, p in zip(FPN_GRIDS, ("p3", "p4", "p5"))]

    def load(path):
        a = np.fromfile(path, dtype=np.float32)
        return a.reshape(a.size // NTOT, NTOT)

    logits = load(args.logits)
    n_rec = logits.shape[0]
    others = [load(args.compare)] if args.compare else []
    print(f"{n_rec} records of {NTOT} logits" + (f" (+1 compare set)" if others else ""))

    gts = None
    if args.gt:
        _n, gts = read_gt_full(args.gt)

    if args.indices:
        picks = [int(x) for x in args.indices.split(",")]
    elif args.diverse and gts is not None:
        picks = pick_diverse(args.val_bin, n_rec, gts, args.n)
    elif gts is not None:
        # Densest images first. ⚠ See --diverse: these come from one sequence.
        picks = sorted(range(min(n_rec, len(gts))),
                       key=lambda i: -len(gts[i]))[:args.n]
    else:
        picks = list(range(min(args.n, n_rec)))
    print("records:", picks, "GT:", [len(gts[i]) for i in picks] if gts else "n/a")

    labels = args.labels.split(",") if args.labels else None
    rows = []
    for i in picks:
        img = read_image(args.val_bin, i)
        panels = []
        if gts is not None and not args.match:
            gt_boxes = [(c, None, b) for (c, b) in gts[i]]
            lab = (labels[0] if labels else "truth") + f"  ({len(gt_boxes)})"
            panels.append(draw(img, gt_boxes, label=lab, scale=args.scale))
        for k, src in enumerate([logits] + others):
            dets = decode_fpn(src[i], scales, args.conf_thresh, args.nms_iou,
                              topk=args.topk)
            if args.topk_per_gt and gts is not None:
                # decode_fpn returns triples ranked by score; K = this frame's GT count.
                dets = dets[:len(gts[i])]
            if args.match and gts is not None:
                m = match_dets(dets, gts[i], args.match_iou)
                boxes = [(kind, None, b) for (kind, _c, b) in m]
                n_tp = sum(1 for k, _, _ in m if k == "tp")
                n_fp = sum(1 for k, _, _ in m if k == "fp")
                n_fn = sum(1 for k, _, _ in m if k == "fn")
                tag = f"  {n_tp} hit / {n_fp} false / {n_fn} missed"
            else:
                boxes = [(d[0], d[1], d[2]) for d in dets]
                tag = f"  ({len(boxes)})"
            if labels and len(labels) > k + 1:
                lab = labels[k + 1]
            else:
                lab = "pred" if k == 0 else f"pred {k + 1}"
            panels.append(draw(img, boxes, label=lab + tag, scale=args.scale))
        rows.append(panels)

    if args.layout == "cols":
        rows = [list(col) for col in zip(*rows)]   # frames become columns
    pw, ph = rows[0][0].size
    ncol, nrow, gap = len(rows[0]), len(rows), 6
    sheet_w = pw * ncol + gap * (ncol - 1)
    legend = (None if args.no_legend else
              legend_strip(sheet_w, height=30 + 14 * args.scale, match=args.match))
    sheet_h = ph * nrow + gap * (nrow - 1) + (legend.size[1] + gap if legend else 0)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 16))
    for r, panels in enumerate(rows):
        for c, panel in enumerate(panels):
            sheet.paste(panel, (c * (pw + gap), r * (ph + gap)))
    if legend:
        sheet.paste(legend, (0, sheet_h - legend.size[1]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
