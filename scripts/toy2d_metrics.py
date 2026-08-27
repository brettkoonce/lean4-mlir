#!/usr/bin/env python3
"""Score the 2-D diffusion demo — planning/diffusion_2d_demo.md §4.

The point of the 2-D demo: unlike every image DDPM in this repo, correctness
here is a NUMBER, not a judgement. Three of them:

  1. Mode recall   (8-gaussians) — nearest-centre assignment, count modes
                    holding >= 1% of mass. 8/8 or it collapsed. The single
                    most diagnostic number in the demo.
  2. Energy distance vs an INDEPENDENT true draw. Standard two-sample
                    statistic: 2E|X-Y| - E|X-X'| - E|Y-Y'|, >= 0, and 0 iff
                    the distributions match.
  3. Off-manifold  fraction — share of samples further than tau from any mode.

Also writes a side-by-side PPM scatter (no matplotlib in the pinned venv).

Usage: python3 scripts/toy2d_metrics.py [samples.bin] [outdir]
"""
import sys
import numpy as np

SAMP = sys.argv[1] if len(sys.argv) > 1 else ".lake/build/diffusion2d_samples.bin"
OUT  = sys.argv[2] if len(sys.argv) > 2 else ".lake/build"

gen     = np.fromfile(SAMP, dtype=np.float32).reshape(-1, 2)
ref     = np.fromfile("data/toy2d/eight_gaussians_ref.bin", dtype=np.float32).reshape(-1, 2)
centres = np.fromfile("data/toy2d/eight_gaussians_centres.bin", dtype=np.float32).reshape(-1, 2)
STD = 0.05


def energy_distance(x, y, cap=2048, seed=0):
    """2E|X-Y| - E|X-X'| - E|Y-Y'|. O(n^2); subsample so it stays in ms."""
    rng = np.random.default_rng(seed)
    if len(x) > cap: x = x[rng.choice(len(x), cap, replace=False)]
    if len(y) > cap: y = y[rng.choice(len(y), cap, replace=False)]
    d = lambda a, b: np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(2 * d(x, y).mean() - d(x, x).mean() - d(y, y).mean())


def mode_stats(pts, centres, std, thresh=0.01):
    dist = np.sqrt(((pts[:, None, :] - centres[None, :, :]) ** 2).sum(-1))
    nearest = dist.argmin(1)
    counts = np.bincount(nearest, minlength=len(centres))
    frac = counts / len(pts)
    hit = int((frac >= thresh).sum())
    # "off-manifold" = further than 4 sigma from EVERY mode centre
    off = float((dist.min(1) > 4 * std).mean())
    return hit, frac, off


hit, frac, off = mode_stats(gen, centres, STD)
hit_r, frac_r, off_r = mode_stats(ref, centres, STD)
ed_gen = energy_distance(gen, ref)
# Control: two independent draws of the TRUE density. This is the noise floor —
# an energy distance near it means "as close as two real samples are".
half = len(ref) // 2
ed_ref = energy_distance(ref[:half], ref[half:])

print(f"samples: {len(gen)}   reference: {len(ref)} (independent draw)")
print()
print(f"  mode recall        {hit}/8      (true sample: {hit_r}/8)")
print(f"  energy distance    {ed_gen:.5f}  (true-vs-true floor: {ed_ref:.5f})")
print(f"  off-manifold       {off*100:.2f}%   (true sample: {off_r*100:.2f}%)")
print()
print("  per-mode mass (%):  " + " ".join(f"{f*100:5.1f}" for f in frac))
print("  true               :  " + " ".join(f"{f*100:5.1f}" for f in frac_r))
print()
# Regression gate (plan phase 6), NOT a quality bar. Set from measurement, not
# taste: the 2026-08-27 sweep lands at energy 0.022-0.036 across train/sampler
# settings, while a collapsed model scores an order of magnitude worse because
# whole modes go missing. 0.05 sits above the observed spread and far below
# collapse, so it catches a regression without failing the honest current state.
GATE_ED = 0.05
ok = (hit == 8) and (ed_gen < GATE_ED)
print(f"  => gate: mode recall {hit}/8 (need 8) and energy {ed_gen:.4f} < {GATE_ED}"
      f"  [{'PASS' if ok else 'FAIL'}]")
print(f"     (energy is {ed_gen / max(ed_ref,1e-9):.0f}x the true-vs-true floor — "
      f"the residual is per-mode MASS imbalance, not missed modes)")
sys.exit(0 if ok else 1)


def scatter_ppm(path, clouds, colours, size=480, lim=1.6):
    """Side-by-side scatter, no matplotlib. clouds/colours are parallel lists."""
    w = size * len(clouds)
    img = np.full((size, w, 3), 18, dtype=np.uint8)
    for panel, (pts, col) in enumerate(zip(clouds, colours)):
        ox = panel * size
        for gx in range(0, size, size // 8):          # faint grid
            img[:, ox + gx] = np.maximum(img[:, ox + gx], 34)
            img[gx, ox:ox + size] = np.maximum(img[gx, ox:ox + size], 34)
        px = ((pts[:, 0] + lim) / (2 * lim) * (size - 1)).astype(int)
        py = ((lim - pts[:, 1]) / (2 * lim) * (size - 1)).astype(int)
        ok = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(py[ok] + dy, 0, size - 1),
                    ox + np.clip(px[ok] + dx, 0, size - 1)] = col
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {size}\n255\n".encode())
        f.write(img.tobytes())
    print(f"  wrote {path}  (left: generated, right: true)")


scatter_ppm(f"{OUT}/diffusion2d_scatter.ppm", [gen, ref],
            [(120, 200, 255), (255, 170, 90)])
