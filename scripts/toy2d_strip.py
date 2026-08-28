#!/usr/bin/env python3
"""Render the reverse-process strip — planning/diffusion_2d_demo.md §5.

⭐ This is the figure images cannot carry. On a 28x28 MNIST DDPM the states
between noise and sample are grey mush; here every panel is legible, and the
cloud visibly contracts from an isotropic Gaussian onto the manifold.

Reads the manifest `lake exe diffusion-2d <target> strip` writes, so the panel
ORDER and the t each panel was taken at are data rather than a filename sort.

⚠ Every panel shares one axis range (`--lim`, default 3.0). That is the whole
point: rescale per panel and the contraction — the thing the figure exists to
show — disappears. The first panel is pure N(0,1) noise, so a few tenths of a
percent of it falls outside the frame and is not drawn.

Usage: python3 scripts/toy2d_strip.py [target] [--lim=3.0] [--size=400]
"""
import json, os, sys
import numpy as np

flags  = [a for a in sys.argv[1:] if a.startswith("--")]
args   = [a for a in sys.argv[1:] if not a.startswith("--")]
TARGET = args[0] if args else "eight_gaussians"


def flag(name, default, cast):
    return cast(next((f.split("=", 1)[1] for f in flags
                      if f.startswith(f"--{name}=")), default))


LIM  = flag("lim", 3.0, float)
SIZE = flag("size", 400, int)
DATA = "data/toy2d"
MAN  = f".lake/build/diffusion2d_strip_{TARGET}.txt"

if not os.path.exists(MAN):
    sys.exit(f"{MAN} missing — run: lake exe diffusion-2d {TARGET} reuse strip")

with open(f"{DATA}/manifest.json") as f:
    label = json.load(f)["targets"][TARGET]["label"]

frames = []
for line in open(MAN):
    idx, t, sig, path = line.split()
    frames.append((int(idx), int(t), float(sig),
                   np.fromfile(path, dtype=np.float32).reshape(-1, 2)))
frames.sort()
support = np.fromfile(f"{DATA}/{TARGET}_support.bin", dtype=np.float32).reshape(-1, 2)


def to_px(pts):
    px = ((pts[:, 0] + LIM) / (2 * LIM) * (SIZE - 1)).astype(int)
    py = ((LIM - pts[:, 1]) / (2 * LIM) * (SIZE - 1)).astype(int)
    ok = (px >= 0) & (px < SIZE) & (py >= 0) & (py < SIZE)
    return px[ok], py[ok]


def plot(img, ox, pts, colour, fat=True):
    px, py = to_px(pts)
    offs = ((0, 0), (1, 0), (0, 1), (1, 1)) if fat else ((0, 0),)
    for dx, dy in offs:
        img[np.clip(py + dy, 0, SIZE - 1), ox + np.clip(px + dx, 0, SIZE - 1)] = colour


BAR = 6                                  # the t progress bar under each panel
H   = SIZE + BAR
img = np.full((H, SIZE * len(frames), 3), 18, dtype=np.uint8)
tmax = max(t for _, t, _, _ in frames) or 1

for panel, (idx, t, sig, pts) in enumerate(frames):
    ox = panel * SIZE
    for g in range(0, SIZE, SIZE // 6):                       # faint grid
        img[:SIZE, ox + g] = np.maximum(img[:SIZE, ox + g], 32)
        img[g, ox:ox + SIZE] = np.maximum(img[g, ox:ox + SIZE], 32)
    # The true support, drawn faintly in EVERY panel so the eye has a fixed
    # target to watch the cloud collapse onto.
    plot(img, ox, support, (52, 52, 60), fat=False)
    # Cold at t = T, warm at t = 0: the colour ramp carries the time axis even
    # in greyscale, where the panels would otherwise be indistinguishable.
    a = panel / max(len(frames) - 1, 1)
    colour = (int(110 + 145 * a), int(190 - 20 * a), int(255 - 165 * a))
    plot(img, ox, pts, colour)
    fill = int(SIZE * (1.0 - t / tmax))
    img[SIZE:, ox:ox + SIZE] = (30, 30, 34)
    img[SIZE + 1:H - 1, ox:ox + max(fill, 1)] = colour
    img[:H, ox] = (70, 70, 78)                                # panel divider

ppm = f".lake/build/diffusion2d_strip_{TARGET}.ppm"
with open(ppm, "wb") as f:
    f.write(f"P6\n{img.shape[1]} {H}\n255\n".encode())
    f.write(img.tobytes())
print(f"wrote {ppm}  ({len(frames)} panels, {label}, lim={LIM})")
print("  t per panel:     " + " ".join(f"{t:>4d}" for _, t, _, _ in frames))
print("  sigma per panel: " + " ".join(f"{s:>4.2f}" for _, _, s, _ in frames))
print("  spread (std of |x|): " +
      " ".join(f"{np.sqrt((p**2).sum(1)).std():>4.2f}" for _, _, _, p in frames))

try:
    from PIL import Image
    png = ppm.replace(".ppm", ".png")
    Image.fromarray(img).save(png)
    print(f"wrote {png}")
except ImportError:
    print("  (PIL absent — PPM only)")
