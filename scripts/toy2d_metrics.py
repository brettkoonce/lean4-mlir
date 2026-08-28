#!/usr/bin/env python3
"""Score the 2-D diffusion demo — planning/diffusion_2d_demo.md §4.

The point of the 2-D demo: unlike every image DDPM in this repo, correctness
here is a NUMBER, not a judgement. Three of them, on all four targets:

  1. Cell recall    — assign each sample to its nearest support cell, count
                      cells holding >= 1% of mass. Every target's cells are
                      EQUAL-MASS by construction (preprocess_toy2d.py), so the
                      threshold means the same thing on each. 8-gaussians has
                      8 modes, checkerboard 8 squares, two-moons 2 moons,
                      spiral 8 equal-mass arcs. K/K or it collapsed.
  2. Energy distance vs an INDEPENDENT true draw. Standard two-sample
                      statistic: 2E|X-Y| - E|X-X'| - E|Y-Y'|, >= 0, and 0 iff
                      the distributions match.
  3. Off-support    — share of samples further than tau from the support.
                      ⭐ On checkerboard this is EXACT (in an empty square or
                      not) rather than a tau test, which is why that target is
                      in the list: its failure mode has a closed form.

Also writes a side-by-side PPM scatter (no matplotlib in the pinned venv).

Usage: python3 scripts/toy2d_metrics.py [--target=NAME] [samples.bin] [outdir]
       NAME in eight_gaussians (default) | spiral | two_moons | checkerboard
"""
import json, sys
import numpy as np

args   = [a for a in sys.argv[1:] if not a.startswith("--")]
flags  = [a for a in sys.argv[1:] if a.startswith("--")]
TARGET = next((f.split("=", 1)[1] for f in flags if f.startswith("--target=")),
              "eight_gaussians")
SAMP = args[0] if len(args) > 0 else f".lake/build/diffusion2d_samples_{TARGET}.bin"
OUT  = args[1] if len(args) > 1 else ".lake/build"
DATA = "data/toy2d"

with open(f"{DATA}/manifest.json") as f:
    known = json.load(f)["targets"]
if TARGET not in known:
    sys.exit(f"unknown target {TARGET!r} — have {', '.join(sorted(known))}")
geom = known[TARGET]
NCELL, TAU = geom["ncells"], geom["tau"]

gen = np.fromfile(SAMP, dtype=np.float32).reshape(-1, 2)
ref = np.fromfile(f"{DATA}/{TARGET}_ref.bin", dtype=np.float32).reshape(-1, 2)
sup = np.fromfile(f"{DATA}/{TARGET}_support.bin", dtype=np.float32).reshape(-1, 2)
cel = np.fromfile(f"{DATA}/{TARGET}_cells.bin", dtype=np.int32)

# Regression gates (plan phase 6), NOT quality bars. 0.05 on every target,
# which is where 8-gaussians' gate already sat and which the other three clear
# by 4-9x. Measured 2026-08-28 at 20k train steps / 200 sampler steps / eta 0.25:
#
#   8-gaussians 0.0055 | spiral 0.0116 | two-moons 0.0058 | checkerboard 0.0053
#
# ⚠ These are NOT tightened to sit just above those numbers, and the reason is
# that nobody has measured retrain variance: training is not reproducible run to
# run (see the `reuse` flag in demos/MainDiffusion2d.lean), so one sample per
# target is a value, not a spread. A gate set from one point would be flaky in
# the direction that costs the most — failing on an honest run. Collapse scores
# an order of magnitude worse (0.17 for the eta = 1 arm that lost a mode), so
# 0.05 still separates the two. ▶ Tightening wants a retrain study first.
#
# ⚠⚠ The gate does NOT look at off-support, and on checkerboard that is visible:
# the trained model leaks 31% of its mass into provably empty squares and passes
# on 8/8 recall and a 0.0053 energy distance. Both halves of the gate are
# genuinely insensitive to it. Recorded rather than papered over — the demo
# exists to make failures countable, and this is one the current gate cannot
# count.
GATE_ED = {
    "eight_gaussians": 0.05,
    "spiral":          0.05,
    "two_moons":       0.05,
    "checkerboard":    0.05,
}


def energy_distance(x, y, cap=2048, seed=0):
    """2E|X-Y| - E|X-X'| - E|Y-Y'|. O(n^2); subsample so it stays in ms."""
    rng = np.random.default_rng(seed)
    if len(x) > cap: x = x[rng.choice(len(x), cap, replace=False)]
    if len(y) > cap: y = y[rng.choice(len(y), cap, replace=False)]
    d = lambda a, b: np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(2 * d(x, y).mean() - d(x, x).mean() - d(y, y).mean())


def cell_stats_points(pts):
    """Nearest support POINT gives both the cell and the off-support distance.
    For 8-gaussians the support is the 8 centres, so this is exactly the
    nearest-centre assignment and the 4-sigma test the demo shipped with."""
    best = np.full(len(pts), np.inf)
    near = np.zeros(len(pts), np.int64)
    # Chunked so 8192 x 4096 pairs never materialise as one array.
    for lo in range(0, len(sup), 512):
        blk = sup[lo:lo + 512]
        d = np.sqrt(((pts[:, None, :] - blk[None, :, :]) ** 2).sum(-1))
        j = d.argmin(1)
        dm = d[np.arange(len(pts)), j]
        upd = dm < best
        best = np.where(upd, dm, best)
        near = np.where(upd, cel[lo:lo + 512][j], near)
    counts = np.bincount(near, minlength=NCELL)
    return counts / len(pts), float((best > TAU).mean())


def cell_stats_checker(pts):
    """EXACT membership: which square, and is it one of the occupied ones."""
    lim, grid = geom["lim"], geom["grid"]
    cell = 2.0 * lim / grid
    i = np.floor((pts[:, 0] + lim) / cell).astype(int)
    j = np.floor((pts[:, 1] + lim) / cell).astype(int)
    inside = (i >= 0) & (i < grid) & (j >= 0) & (j < grid)
    occupied = inside & (((i + j) % 2) == 0)
    order = {c: k for k, c in enumerate(
        [(a, b) for a in range(grid) for b in range(grid) if (a + b) % 2 == 0])}
    idx = np.array([order.get((a, b), 0) for a, b in zip(i[occupied], j[occupied])])
    counts = np.bincount(idx, minlength=NCELL) if len(idx) else np.zeros(NCELL, int)
    return counts / len(pts), float((~occupied).mean())


def stats(pts):
    return cell_stats_checker(pts) if geom["kind"] == "checker" else cell_stats_points(pts)


frac,   off   = stats(gen)
frac_r, off_r = stats(ref)
hit   = int((frac   >= 0.01).sum())
hit_r = int((frac_r >= 0.01).sum())
ed_gen = energy_distance(gen, ref)
# Control: two independent draws of the TRUE density. This is the noise floor —
# an energy distance near it means "as close as two real samples are".
half = len(ref) // 2
ed_ref = energy_distance(ref[:half], ref[half:])

off_kind = "exact" if geom["kind"] == "checker" else f"tau={TAU:.3f}"
print(f"target: {geom['label']}   samples: {len(gen)}   "
      f"reference: {len(ref)} (independent draw)")
print()
print(f"  cell recall        {hit}/{NCELL}      (true sample: {hit_r}/{NCELL})")
print(f"  energy distance    {ed_gen:.5f}  (true-vs-true floor: {ed_ref:.5f})")
print(f"  off-support        {off*100:.2f}%   (true sample: {off_r*100:.2f}%, {off_kind})")
print()
print("  per-cell mass (%):  " + " ".join(f"{f*100:5.1f}" for f in frac))
print("  true               :  " + " ".join(f"{f*100:5.1f}" for f in frac_r))
print()


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


# ⚠ This call used to sit BELOW the sys.exit and so never ran — the scatter on
# disk was a development leftover, not an output of the committed script.
scatter_ppm(f"{OUT}/diffusion2d_scatter_{TARGET}.ppm", [gen, ref],
            [(120, 200, 255), (255, 170, 90)])

gate = GATE_ED[TARGET]
ok = (hit == NCELL) and (ed_gen < gate)
print(f"  => gate: cell recall {hit}/{NCELL} (need {NCELL}) and "
      f"energy {ed_gen:.4f} < {gate}  [{'PASS' if ok else 'FAIL'}]")
print(f"     (energy is {ed_gen / max(ed_ref,1e-9):.0f}x the true-vs-true floor)")
sys.exit(0 if ok else 1)
