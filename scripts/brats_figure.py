#!/usr/bin/env python3
"""Label a `brats-predict` strip: column headers, a region legend, black margins trimmed.

`lake exe brats-predict` writes a bare PPM grid — one slice per row, panels of
`--panel` px across (T1gd | +truth | +one prediction per arm) and no text, since
the Lean side has no font. This adds what a reader needs to read it without the
caption: a header over each column and the three region colours. It draws
nothing over the panels themselves; the pixels are the renderer's.

Each row is cropped to the union of its panels' non-black extent (plus a small
margin), and every panel in the row gets the same crop, so the columns stay
pixel-aligned and a boundary can be compared across them.

    lake exe brats-predict net=r34 arm=scratch,r34 out.ppm
    python3 scripts/brats_figure.py out.ppm demos/figures/brats_r34_skip_transfer.png \\
        --labels "T1gd,ground truth,from scratch,ImageNet R34"

Needs Pillow.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# MainBratsPredict.lean `regionColor`: the overlay hues, un-blended.
REGIONS = [((60, 200, 60), "edema"),
           ((220, 50, 50), "necrotic / non-enhancing core"),
           ((255, 215, 0), "enhancing tumour")]
BG = (0, 0, 0)          # the MRI background, so panel edges vanish
FG = (235, 235, 235)


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strip", help="brats-predict output (.ppm, or a PNG of it)")
    ap.add_argument("out")
    ap.add_argument("--labels", required=True, help="comma-separated, one per column")
    ap.add_argument("--panel", type=int, default=224, help="panel side in px")
    ap.add_argument("--scale", type=int, default=2, help="nearest-neighbour upsample")
    ap.add_argument("--margin", type=int, default=6, help="px kept around the anatomy")
    args = ap.parse_args()

    a = np.asarray(Image.open(args.strip).convert("RGB"))
    p = args.panel
    ncol, nrow = a.shape[1] // p, a.shape[0] // p
    labels = args.labels.split(",")
    if len(labels) != ncol:
        raise SystemExit(f"{ncol} columns in the strip, {len(labels)} labels given")

    rows = []
    for r in range(nrow):
        cells = [a[r * p:(r + 1) * p, c * p:(c + 1) * p] for c in range(ncol)]
        lit = np.any(np.stack(cells).max(axis=3) > 12, axis=0)
        ys, xs = np.where(lit)
        y0, y1 = max(ys.min() - args.margin, 0), min(ys.max() + args.margin + 1, p)
        x0, x1 = max(xs.min() - args.margin, 0), min(xs.max() + args.margin + 1, p)
        rows.append([c[y0:y1, x0:x1] for c in cells])

    # One width for every column: the widest row's crop, each row centred in it.
    s, gap = args.scale, 4 * args.scale
    pw = max(r[0].shape[1] for r in rows) * s
    # Sized to stay legible when the book prints the sheet at half a text width.
    head_f, leg_f = _font(14 * s), _font(12 * s)
    head_h, leg_h = 22 * s, 22 * s
    heights = [r[0].shape[0] * s for r in rows]
    W = ncol * pw + (ncol - 1) * gap
    H = head_h + sum(heights) + gap * (nrow - 1) + gap + leg_h
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)

    for c, lab in enumerate(labels):
        tw = d.textlength(lab, font=head_f)
        d.text((c * (pw + gap) + (pw - tw) / 2, 3 * s), lab, fill=FG, font=head_f)
    y = head_h
    for r, cells in enumerate(rows):
        for c, cell in enumerate(cells):
            im = Image.fromarray(np.ascontiguousarray(cell)).resize(
                (cell.shape[1] * s, cell.shape[0] * s), Image.NEAREST)
            sheet.paste(im, (c * (pw + gap) + (pw - im.width) // 2, y))
        y += heights[r] + gap

    x, sw = 4 * s, 11 * s
    ly = H - leg_h + (leg_h - sw) // 2
    for rgb, name in REGIONS:
        d.rectangle([x, ly, x + sw, ly + sw], fill=rgb)
        x += sw + 3 * s
        d.text((x, ly - s // 2), name, fill=FG, font=leg_f)
        x += int(d.textlength(name, font=leg_f)) + 8 * s

    sheet.save(args.out)
    print(f"wrote {args.out}  ({W}x{H}, {nrow} rows x {ncol} columns)")


if __name__ == "__main__":
    main()
