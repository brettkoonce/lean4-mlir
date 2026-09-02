#!/usr/bin/env python3
"""Compose the DDPM trajectory figure from the sampler's own two-row strip.

`mnist-ddpm-sample trajectory` writes a PPM whose top row is the FORWARD process
(a real MNIST digit corrupted toward noise) and whose bottom row is the REVERSE
process (the sampler's own intermediates). Both rows are already aligned column by
column to the same ᾱ, which is the property that makes them worth stacking. This
script only upscales and labels — it does no diffusion arithmetic, so it cannot
disagree with the sampler it illustrates.

    lake exe mnist-ddpm-sample trajectory data=data img=7
    python3 scripts/ddpm_trajectory_figure.py \
        runs/2026-09-02-mnist-ddpm/trajectory.ppm \
        --out demos/figures/ddpm_mnist_trajectory.png
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (18, 18, 20)
FWD = (96, 165, 235)      # forward process
REV = (167, 139, 235)     # generation


def _font(size, bold=True):
    for p in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def arrow(d, y, x0, x1, colour, label, font, rtl=False):
    """A rule with a head at one end and the label centred above it."""
    d.line([(x0, y), (x1, y)], fill=colour, width=3)
    hx = x0 if rtl else x1
    s = -1 if rtl else 1
    d.polygon([(hx, y), (hx - s * 14, y - 7), (hx - s * 14, y + 7)], fill=colour)
    tw = d.textlength(label, font=font)
    cx = (x0 + x1) / 2
    d.rectangle([cx - tw / 2 - 10, y - 13, cx + tw / 2 + 10, y + 13], fill=BG)
    d.text((cx - tw / 2, y - 11), label, fill=colour, font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ppm")
    ap.add_argument("--out", default="demos/figures/ddpm_mnist_trajectory.png")
    ap.add_argument("--tile", type=int, default=96, help="rendered size of each 28px frame")
    ap.add_argument("--gap", type=int, default=6)
    args = ap.parse_args()

    strip = Image.open(args.ppm).convert("RGB")
    # The strip is two rows of `n` tiles separated by a 4px gutter the sampler wrote.
    src_gap, tile_src = 4, 28
    n = (strip.width + src_gap) // (tile_src + src_gap)
    rows = []
    for r in range(2):
        y = r * (tile_src + src_gap)
        rows.append([strip.crop((c * (tile_src + src_gap), y,
                                 c * (tile_src + src_gap) + tile_src, y + tile_src))
                     for c in range(n)])

    T, G = args.tile, args.gap
    strip_w = n * T + (n - 1) * G
    pad, head, between = 26, 46, 34
    W = strip_w + 2 * pad
    H = pad + head + T + between + head + T + pad
    out = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(out)
    f = _font(19)

    y = pad
    for r, (label, colour, rtl) in enumerate(
            [("data  →  noise      (forward process)", FWD, False),
             ("new generated data  ←  noise      (generation process)", REV, True)]):
        arrow(d, y + head // 2, pad, pad + strip_w, colour, label, f, rtl=rtl)
        y += head
        for c, tile in enumerate(rows[r]):
            out.paste(tile.resize((T, T), Image.NEAREST), (pad + c * (T + G), y))
        y += T + between

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"wrote {args.out}  ({out.size[0]}x{out.size[1]}, {n} frames per row)")


if __name__ == "__main__":
    main()
