#!/usr/bin/env python3
"""MOCK of the PlantVillage→PlantDoc figure, before any training (planning/plant_lab_to_field_demo.md §6).
Real leaves, real leaf masks (PlantVillage's `segmented` twin), and heatmaps DRAWN TO THE EXPECTED
STORY — not a network's output. Every CAM panel is stamped "mock". Template for scripts/plant_figure.py.

(a) four PlantVillage test leaves: input | base CAM | +backgrounds CAM, mass-inside-leaf under each CAM
    — the mock base heat sits on the leaf edge and the grey corners, the fixed heat on the darkest patch
    of the leaf (a crude lesion finder);
(b) the same four crops from PlantDoc: input | base CAM | +field-labels CAM — base heat on sky/soil,
    fixed heat on the greenest region's darkest patch;
(c) two training leaves beside their Imagenette-background composites (what the background fix trains on).
Usage: .venv/bin/python runs/2026-09-17-plant/mock_figure.py [out.png]   (from the repo root)"""
import glob
import os
import random
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

random.seed(4)
np.random.seed(4)
PV = "data/plant/PlantVillage-Dataset/raw"
PD = "data/plant/PlantDoc-Dataset/train"
IMNET = "data/imagenette/imagenette2-320/train"
OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/2026-09-17-plant/mock_figure.png"
PAIRS = [("Tomato___Early_blight", "Tomato Early blight leaf", "tomato, early blight"),
         ("Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight", "corn, leaf blight"),
         ("Apple___Apple_scab", "Apple Scab Leaf", "apple, scab"),
         ("Grape___Black_rot", "grape leaf black rot", "grape, black rot")]
T, GAP, PAD = 150, 6, 12
BG, FG, ACCENT, COOL, WARN = (16, 16, 16), (235, 235, 235), (255, 200, 60), (120, 200, 255), (255, 110, 110)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
font_r = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
VIRIDIS = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], dtype=np.float32)


def viridis(h):
    """h in [0,1] → RGB uint8 via a 5-stop viridis."""
    x = np.clip(h, 0, 1) * 4
    i = np.minimum(x.astype(int), 3)
    t = (x - i)[..., None]
    return (VIRIDIS[i] * (1 - t) + VIRIDIS[i + 1] * t).astype(np.uint8)


def center224(im):
    w, h = im.size
    s = 256 / min(w, h)
    im = im.convert("RGB").resize((max(224, round(w * s)), max(224, round(h * s))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return im.crop((l, t, l + 224, t + 224))


def cam_grid(heat):
    """Make a 224×224 heat look like a 7×7 CAM: average-pool to 7×7, ReLU, normalise, bilinear back."""
    g = heat.reshape(7, 32, 7, 32).mean(axis=(1, 3))
    g = np.maximum(g, 0)
    g = g / (g.sum() + 1e-9)
    up = np.asarray(Image.fromarray((g / g.max() * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
    return g, up


def overlay(img, up, alpha=0.45):
    base = np.asarray(img, dtype=np.float32)
    col = viridis(up).astype(np.float32)
    return Image.fromarray(np.clip(base * (1 - alpha) + col * alpha, 0, 255).astype(np.uint8))


def mass_inside(g, mask):
    m7 = mask.reshape(7, 32, 7, 32).mean(axis=(1, 3))
    return float((g * m7).sum() / (g.sum() + 1e-9))


def gauss_at(cy, cx, sigma=28):
    yy, xx = np.mgrid[0:224, 0:224]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def pv_example(cls):
    fs = sorted(glob.glob(f"{PV}/color/{cls}/*"))
    f = random.choice(fs)
    stem = os.path.splitext(os.path.basename(f))[0]
    seg = f"{PV}/segmented/{cls}/{stem}_final_masked.jpg"
    img = center224(Image.open(f))
    segim = center224(Image.open(seg)) if os.path.exists(seg) else img
    mask = (np.asarray(segim).sum(axis=2) > 40).astype(np.float32)
    return img, mask


def mock_base_heat(mask, corner=True):
    """Heat outside the leaf: a band along the leaf edge on the background side, plus grey-corner blobs."""
    d_out = distance_transform_edt(1 - mask)          # distance from leaf, on the background
    band = np.exp(-(d_out / 18.0) ** 2) * (1 - mask)
    heat = band
    if corner:
        for cy, cx in ((20, 20), (200, 205)):
            heat = heat + 0.8 * gauss_at(cy, cx, 22) * (1 - mask)
    heat = heat + 0.15 * mask * np.random.rand(224, 224)   # a little on the leaf, as a real CAM would
    return gaussian_filter(heat, 4)


def mock_fixed_heat(img, mask):
    """Heat on the darkest / brownest patch inside the leaf — the lesion, crudely."""
    a = np.asarray(img, dtype=np.float32)
    brown = (a[..., 0] - a[..., 1]) - 0.5 * a.mean(axis=2)     # red over green, dark
    score = gaussian_filter(brown * mask, 9)
    score[mask < 0.5] = -1e9
    cy, cx = np.unravel_index(score.argmax(), score.shape)
    return gauss_at(cy, cx, 30) * (0.35 + 0.65 * mask) + 0.05 * mask


def pd_example(cls):
    fs = sorted(glob.glob(f"{PD}/{cls}/*"))
    img = center224(Image.open(random.choice(fs)))
    a = np.asarray(img, dtype=np.float32)
    green = ((a[..., 1] > a[..., 0] + 8) & (a[..., 1] > a[..., 2] + 8)).astype(np.float32)
    green = (gaussian_filter(green, 6) > 0.4).astype(np.float32)
    return img, green


def composite(cls):
    img, mask = pv_example(cls)
    bgf = random.choice(glob.glob(f"{IMNET}/*/*.JPEG"))
    bg = np.asarray(center224(Image.open(bgf)), dtype=np.float32)
    m = gaussian_filter(mask, 1.5)[..., None]
    out = np.asarray(img, dtype=np.float32) * m + bg * (1 - m)
    maskim = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
    return img, maskim, Image.fromarray(out.astype(np.uint8))


def main():
    cols = 6
    W = PAD * 2 + cols * T + (cols - 1) * GAP
    ROW = T + 34
    rows = [("(a)  PlantVillage test leaves — input | base CAM | +backgrounds CAM, with the share of CAM mass inside the leaf mask  [MOCK heat]", 2),
            ("(b)  the same four crops on PlantDoc field photos — input | base CAM | +field-labels CAM  [MOCK heat, MOCK labels]", 2),
            ("(c)  what the background fix trains on — a training leaf, its PlantVillage mask, the leaf on an Imagenette background", 1)]
    H = PAD + sum(30 + n * ROW for _, n in rows) + PAD
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    y = PAD

    def put(x, y, im, l1, l2, col=ACCENT):
        sheet.paste(im.resize((T, T), Image.LANCZOS), (x, y))
        d.text((x + 2, y + T + 2), l1, fill=col, font=small)
        if l2:
            d.text((x + 2, y + T + 16), l2, fill=FG, font=small)

    # (a)
    d.text((PAD, y), rows[0][0], fill=FG, font=font_r); y += 30
    for k, (pvc, pdc, nice) in enumerate(PAIRS):
        r, c = divmod(k, 2)
        img, mask = pv_example(pvc)
        gb, ub = cam_grid(mock_base_heat(mask))
        gf, uf = cam_grid(mock_fixed_heat(img, mask))
        x0 = PAD + c * 3 * (T + GAP)
        yy = y + r * ROW
        put(x0, yy, img, nice, "PlantVillage test")
        put(x0 + (T + GAP), yy, overlay(img, ub), "base (mock CAM)", f"{100 * mass_inside(gb, mask):.0f}% of mass in leaf", WARN)
        put(x0 + 2 * (T + GAP), yy, overlay(img, uf), "+backgrounds (mock)", f"{100 * mass_inside(gf, mask):.0f}% of mass in leaf", COOL)
    y += 2 * ROW
    # (b)
    d.text((PAD, y), rows[1][0], fill=FG, font=font_r); y += 30
    wrong = ["tomato healthy", "corn healthy", "grape healthy", "apple healthy"]
    right = ["tomato early blight", "corn leaf blight", "apple scab", "grape black rot"]
    for k, (pvc, pdc, nice) in enumerate(PAIRS):
        r, c = divmod(k, 2)
        img, green = pd_example(pdc)
        gb, ub = cam_grid(mock_base_heat(green, corner=False) + 0.6 * (1 - green) * np.random.rand(224, 224))
        gf, uf = cam_grid(mock_fixed_heat(img, green))
        x0 = PAD + c * 3 * (T + GAP)
        yy = y + r * ROW
        put(x0, yy, img, nice, "PlantDoc field photo")
        put(x0 + (T + GAP), yy, overlay(img, ub), "base (mock CAM)", f"→ {wrong[k]} ✗", WARN)
        put(x0 + 2 * (T + GAP), yy, overlay(img, uf), "+field labels (mock)", f"→ {right[k]} ✓", COOL)
    y += 2 * ROW
    # (c)
    d.text((PAD, y), rows[2][0], fill=FG, font=font_r); y += 30
    for k, cls in enumerate(("Tomato___Late_blight", "Apple___Cedar_apple_rust")):
        src, m, comp = composite(cls)
        x0 = PAD + k * 3 * (T + GAP)
        put(x0, y, src, cls.split("___")[1].replace("_", " "), "PlantVillage colour")
        put(x0 + (T + GAP), y, m, "leaf mask", "the segmented twin")
        put(x0 + 2 * (T + GAP), y, comp, "composite", "on an Imagenette image", COOL)
    sheet.save(OUT)
    print(OUT, sheet.size)


if __name__ == "__main__":
    main()
