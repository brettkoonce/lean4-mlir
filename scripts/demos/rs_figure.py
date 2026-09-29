#!/usr/bin/env python3
"""The remote-sensing demo's figure — planning/remote_sensing_wavelengths_demo.md §6.

A dense grid of 640 m chips over a named window of a fetched scene, scored by the arms,
drawn as a coloured chip grid beside the true-colour render and the MapBiomas map.

  make:    .venv-rs/bin/python scripts/demos/rs_figure.py make --name rondonia --scene <scene id> --center -10.62,-62.21 [--size 16]
           writes data/rs/fig_<name>.bin (size² chips, row-major), labels_fig_<name>.bin (seven-class code, 9 = unmapped),
           meta_fig_<name>.npz (true/false-colour renders of the window, the MapBiomas grid, origin, date); score with
           CUDA_VISIBLE_DEVICES=0 lake exe rs-bands arm=<arm> eval tag=<tag> out=<dir> score=fig_<name>
  render:  .venv-rs/bin/python scripts/demos/rs_figure.py render --panel rondonia:rgb=<logits>,all=<logits> \\
               [--panel cerrado_dry:rgb=<logits>,ir=<logits> --panel cerrado_wet:...] --out demos/figures/remote_sensing_wavelengths.png
           one row per --panel: true colour | (false colour) | each arm's chip grid | MapBiomas, with a legend strip.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_rs_brazil import load_scene, read_offsets, Labeller, CODE_MAP, BANDS, CHIP, ALIGN  # noqa: E402
from rs_score import SHARED, collapse  # noqa: E402

PALETTE = {0: (200, 80, 200), 1: (140, 60, 200), 2: (30, 110, 50), 3: (190, 190, 90), 4: (240, 220, 120),
           5: (210, 50, 40), 6: (40, 60, 220), 7: (120, 190, 110), 8: (120, 90, 20), 9: (128, 128, 128)}
NAMES = SHARED + ["savanna", "plantation", "unmapped"]
QL_MAX = 2750.0


def make(args):
    import json
    from rasterio.warp import transform as tf
    lat, lon = [float(v) for v in args.center.split(",")]
    d = os.path.join(args.data, "s2", args.scene)
    quant, offs, baseline = read_offsets(os.path.join(d, "MTD_MSIL1C.xml"))
    X, transform, crs = load_scene(d)
    xs, ys = tf("EPSG:4326", crs, [lon], [lat])
    col, row = ~transform * (xs[0], ys[0])
    n = args.size * CHIP
    r0 = int(max(0, min(X.shape[1] - n, round(row - n / 2)))) // ALIGN * ALIGN
    c0 = int(max(0, min(X.shape[2] - n, round(col - n / 2)))) // ALIGN * ALIGN
    win = X[:, r0:r0 + n, c0:c0 + n]
    if (win[1] == 0).any():
        sys.exit(f"fig_{args.name}: the window at row {r0} col {c0} touches the scene's nodata wedge ({(win[1] == 0).mean():.0%}); pick another centre")
    off = np.array([offs[b] for b in BANDS], dtype=np.float32)[:, None, None]
    refl = np.clip((win.astype(np.float32) + off) / quant, 0, None)
    man = json.load(open(os.path.join(args.data, "manifest_rs.json")))
    mean = np.array(man["band_mean"], dtype=np.float32)[:, None, None]
    std = np.array(man["band_std"], dtype=np.float32)[:, None, None]
    z = (refl - mean) / std
    chips = z.reshape(13, args.size, CHIP, args.size, CHIP).transpose(1, 3, 0, 2, 4).reshape(-1, 13, CHIP, CHIP)
    chips.astype(np.float32).tofile(os.path.join(args.data, f"fig_{args.name}.bin"))
    lab = Labeller(os.path.join(args.data, "mapbiomas", f"brazil_coverage-col4_10m_{args.year}.tif"))
    grid = np.full((args.size, args.size), 9, dtype=np.int32)
    purity = np.zeros((args.size, args.size), dtype=np.float32)
    codes = np.zeros((args.size, args.size), dtype=np.int32)
    for i in range(args.size):
        for j in range(args.size):
            code, pur, hist, _, _ = lab.label(crs, transform, r0 + i * CHIP, c0 + j * CHIP)
            codes[i, j], purity[i, j] = code, pur
            grid[i, j] = CODE_MAP.get(code, 9)
    grid.ravel().astype(np.int32).tofile(os.path.join(args.data, f"labels_fig_{args.name}.bin"))
    dn = refl * 10000.0                                   # offset removed: EuroSAT's own 0–2750 quicklook scaling applies
    tc = np.clip(dn[[3, 2, 1]].transpose(1, 2, 0) / QL_MAX * 255, 0, 255).astype(np.uint8)
    fc = np.clip(dn[[7, 3, 2]].transpose(1, 2, 0) / QL_MAX * 255, 0, 255).astype(np.uint8)
    np.savez_compressed(os.path.join(args.data, f"meta_fig_{args.name}.npz"), true_colour=tc, false_colour=fc, grid=grid, codes=codes,
                        purity=purity, origin=np.array([r0, c0]), size=args.size, scene=args.scene, baseline=str(baseline), center=np.array([lat, lon]))
    print(f"fig_{args.name}: {args.size}×{args.size} chips at row {r0} col {c0} of {args.scene}; MapBiomas classes in the window: "
          + ", ".join(f"{NAMES[c]} {int((grid == c).sum())}" for c in np.unique(grid)))


GAMMA = 0.7      # display only: Amazon canopy sits at reflectance 0.03–0.05 in the visible and is black at EuroSAT's linear 0–2750 stretch


def font(px):
    from PIL import ImageFont
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, px)
    return ImageFont.load_default()


def gamma(img):
    return (255.0 * (img.astype(np.float32) / 255.0) ** GAMMA).astype(np.uint8)


def colour_grid(pred, size, alpha_over=None):
    img = np.zeros((size * CHIP, size * CHIP, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            img[i * CHIP:(i + 1) * CHIP, j * CHIP:(j + 1) * CHIP] = PALETTE[int(pred[i * size + j])]
    if alpha_over is not None:
        img = (0.55 * img + 0.45 * alpha_over).astype(np.uint8)
    return img


def render(args):
    rows = []
    for panel in args.panel:
        name, arms = panel.split(":", 1)
        meta = np.load(os.path.join(args.data, f"meta_fig_{name}.npz"))
        size = int(meta["size"])
        tc, fc, grid = gamma(meta["true_colour"]), gamma(meta["false_colour"]), meta["grid"]
        tiles = [("true colour", tc)]
        if args.false_colour:
            tiles.append(("false colour B08/B04/B03", fc))
        for spec in arms.split(","):
            arm, path = spec.split("=")
            logits = np.fromfile(path, dtype=np.float32).reshape(size * size, -1)
            pred = collapse(logits).argmax(axis=1)
            tiles.append((f"{arm} arm", colour_grid(pred, size, tc)))
        tiles.append(("MapBiomas 2023", colour_grid(grid.ravel(), size, tc)))
        w = size * CHIP
        hdr = 44
        strip = Image.new("RGB", (len(tiles) * (w + 8) - 8, w + hdr), (255, 255, 255))
        dr = ImageDraw.Draw(strip)
        f = font(30)
        for k, (title, im) in enumerate(tiles):
            strip.paste(Image.fromarray(im), (k * (w + 8), hdr))
            dr.text((k * (w + 8) + 6, 6), f"{name.replace('_', ' ')}: {title}" if k == 0 else title, fill=(0, 0, 0), font=f)
        rows.append(strip)
    W = max(r.width for r in rows)
    legend = Image.new("RGB", (W, 48), (255, 255, 255))
    dr = ImageDraw.Draw(legend)
    f = font(30)
    x = 8
    for c, nm in enumerate(NAMES):
        dr.rectangle([x, 10, x + 28, 38], fill=PALETTE[c])
        dr.text((x + 36, 8), nm, fill=(0, 0, 0), font=f)
        x += 36 + int(dr.textlength(nm, font=f)) + 28
    out = Image.new("RGB", (W, sum(r.height for r in rows) + legend.height + 8 * len(rows)), (255, 255, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + 8
    out.paste(legend, (0, y))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.save(args.out)
    print(f"wrote {args.out} ({out.width}×{out.height})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "render"])
    ap.add_argument("--data", default="data/rs")
    ap.add_argument("--name")
    ap.add_argument("--scene")
    ap.add_argument("--center", help="lat,lon of the window centre")
    ap.add_argument("--size", type=int, default=16, help="chips per side (16 = 10.24 km)")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--panel", action="append", default=[], help="name:arm=logits[,arm=logits]")
    ap.add_argument("--false-colour", action="store_true")
    ap.add_argument("--out", default="demos/figures/remote_sensing_wavelengths.png")
    args = ap.parse_args()
    if args.mode == "make":
        if not (args.name and args.scene and args.center):
            sys.exit("make needs --name --scene --center")
        make(args)
    else:
        if not args.panel:
            sys.exit("render needs --panel")
        render(args)


if __name__ == "__main__":
    main()
