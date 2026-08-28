#!/usr/bin/env python3
"""2-D toy target distributions for the diffusion demo.

Writes flat float32-LE [N, 2] point clouds that the Lean side reads with a
plain `IO.FS.readBinFile` (no header, no loader needed — the bytes ARE the
array). See planning/diffusion_2d_demo.md §2.

Targets are scaled to roughly unit radius so the data sits in the range a
cosine-schedule DDPM expects (x0 ~ N(0,1)-ish); an 8-gaussians ring at the
literature's radius 2 would need the schedule retuned.

Per target this writes, into `outdir`:
  <name>.bin          training draw     (seed 0)
  <name>_ref.bin      independent draw  (seed 1) — the metric's ground truth
  <name>_support.bin  [M, 2] points ON the support, for the off-support test
  <name>_cells.bin    [M] int32 cell label per support point, for cell recall
and one `manifest.json` carrying the geometry the scorer needs, so the
constants live in ONE file instead of being retyped in the metric script.

⭐ Every target has EQUAL-MASS cells, so the 1%-of-mass recall threshold in
`scripts/toy2d_metrics.py` means the same thing on all four.

Usage: python3 preprocess_toy2d.py [n=8192] [outdir=data/toy2d]
"""
import json, os, sys
import numpy as np

N      = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
OUT    = sys.argv[2] if len(sys.argv) > 2 else "data/toy2d"
RADIUS = 1.0
STD    = 0.05
NSUP   = 4096          # support-curve resolution; spacing must be << tau
os.makedirs(OUT, exist_ok=True)


def eight_gaussians(n, rng):
    """Ring of 8 isotropic modes. Mode index is returned so the metric
    script can score recall without re-deriving the assignment."""
    centres = np.stack([
        [RADIUS * np.cos(2 * np.pi * k / 8), RADIUS * np.sin(2 * np.pi * k / 8)]
        for k in range(8)
    ]).astype(np.float32)
    idx = rng.integers(0, 8, size=n)
    pts = centres[idx] + rng.normal(0, STD, size=(n, 2)).astype(np.float32)
    return pts.astype(np.float32), centres, idx


def eight_gaussians_support():
    """The support IS the 8 centres — one cell each, 12.5% of mass each."""
    _, centres, _ = eight_gaussians(1, np.random.default_rng(0))
    return centres, np.arange(8, dtype=np.int32)


# ── spiral: one long thin manifold, high curvature ──────────────────────────
# Parameterised by u ~ U(0,1); t = sqrt(u)·3π puts equal MASS in equal u, so
# eight equal-u cells are eight equal-mass arcs.
SPIRAL_STD = 0.01


def _spiral_curve(u):
    t = np.sqrt(u) * 3.0 * np.pi
    r = t / (3.0 * np.pi) * RADIUS
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def spiral(n, rng):
    pts = _spiral_curve(rng.uniform(0, 1, size=n))
    pts += rng.normal(0, SPIRAL_STD, size=(n, 2))
    return pts.astype(np.float32), None, None


def spiral_support():
    u = np.linspace(0, 1, NSUP)
    return _spiral_curve(u).astype(np.float32), np.minimum((u * 8).astype(np.int32), 7)


# ── two-moons: two curved, interleaved manifolds ────────────────────────────
# Two unit semicircles, the lower one offset by (1, 0.5) and flipped — the
# standard construction — then centred and scaled into the same unit-ish box
# as the other three. The gap between the moons is what the model bridges when
# it over-smooths, and the off-support fraction is exactly that mass.
MOON_C     = np.array([0.5, 0.25], dtype=np.float64)
MOON_SCALE = 1.0 / 1.5
MOON_STD   = 0.05


def _moon_curve(theta, lower):
    if lower:
        raw = np.stack([1.0 - np.cos(theta), 0.5 - np.sin(theta)], axis=1)
    else:
        raw = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return (raw - MOON_C) * MOON_SCALE


def two_moons(n, rng):
    lower = rng.integers(0, 2, size=n).astype(bool)
    theta = rng.uniform(0, np.pi, size=n)
    pts = np.where(lower[:, None], _moon_curve(theta, True), _moon_curve(theta, False))
    pts += rng.normal(0, MOON_STD, size=(n, 2))
    return pts.astype(np.float32), None, lower.astype(np.int32)


def two_moons_support():
    theta = np.linspace(0, np.pi, NSUP // 2)
    pts = np.concatenate([_moon_curve(theta, False), _moon_curve(theta, True)])
    lab = np.concatenate([np.zeros(NSUP // 2, np.int32), np.ones(NSUP // 2, np.int32)])
    return pts.astype(np.float32), lab


# ── checkerboard: disconnected support, sharp edges ─────────────────────────
# 4×4 grid over [-1,1]², the 8 squares with (i+j) even occupied, uniform
# within. Membership is EXACT — no tau, no nearest-neighbour — so leaked mass
# is counted rather than estimated. That is the one target here whose failure
# mode has a closed-form test.
CHECK_LIM  = 1.0
CHECK_GRID = 4
CHECK_CELL = 2.0 * CHECK_LIM / CHECK_GRID


def _check_cells():
    return [(i, j) for i in range(CHECK_GRID) for j in range(CHECK_GRID)
            if (i + j) % 2 == 0]


def checkerboard(n, rng):
    cells = _check_cells()
    pick = rng.integers(0, len(cells), size=n)
    lo = np.array([[-CHECK_LIM + cells[c][0] * CHECK_CELL,
                    -CHECK_LIM + cells[c][1] * CHECK_CELL] for c in pick])
    pts = lo + rng.uniform(0, CHECK_CELL, size=(n, 2))
    return pts.astype(np.float32), None, pick.astype(np.int32)


def checkerboard_support():
    """A dense grid inside the occupied squares. Unused by the scorer (which
    tests membership exactly) but written so the strip renderer can draw the
    true support the same way it does for the curve targets."""
    cells = _check_cells()
    per = max(1, NSUP // len(cells))
    side = int(np.sqrt(per))
    g = (np.arange(side) + 0.5) / side * CHECK_CELL
    gx, gy = np.meshgrid(g, g)
    out, lab = [], []
    for k, (i, j) in enumerate(cells):
        blk = np.stack([gx.ravel() + (-CHECK_LIM + i * CHECK_CELL),
                        gy.ravel() + (-CHECK_LIM + j * CHECK_CELL)], axis=1)
        out.append(blk)
        lab.append(np.full(len(blk), k, np.int32))
    return np.concatenate(out).astype(np.float32), np.concatenate(lab)


TARGETS = {
    "eight_gaussians": dict(draw=eight_gaussians, support=eight_gaussians_support,
                            kind="points", tau=4 * STD,     ncells=8, label="8-gaussians"),
    "spiral":          dict(draw=spiral,          support=spiral_support,
                            kind="points", tau=4 * SPIRAL_STD, ncells=8, label="spiral"),
    "two_moons":       dict(draw=two_moons,       support=two_moons_support,
                            kind="points", tau=4 * MOON_STD,  ncells=2, label="two-moons"),
    "checkerboard":    dict(draw=checkerboard,    support=checkerboard_support,
                            kind="checker", tau=0.0,          ncells=8, label="checkerboard"),
}

manifest = {"n": N, "targets": {}}
for name, spec in TARGETS.items():
    # ⚠ Each target gets its OWN rng, so adding a target never shifts the draw
    # of one that already has committed numbers against it. A consequence worth
    # knowing before it reads as a copy-paste error: 8-gaussians and
    # checkerboard both open with `rng.integers(0, 8, n)` on seed 0, so their
    # per-cell mass columns are IDENTICAL on the true draw. Same first draw
    # from the same seed, not the same data.
    pts, _, _ = spec["draw"](N, np.random.default_rng(0))
    pts.tofile(f"{OUT}/{name}.bin")
    # A second, independent draw: the metric script needs a TRUE sample to
    # compare against, and reusing the training set would flatter a model that
    # memorised.
    ref, _, _ = spec["draw"](N, np.random.default_rng(1))
    ref.tofile(f"{OUT}/{name}_ref.bin")
    sup, lab = spec["support"]()
    sup.tofile(f"{OUT}/{name}_support.bin")
    lab.tofile(f"{OUT}/{name}_cells.bin")
    manifest["targets"][name] = {
        "label": spec["label"], "kind": spec["kind"],
        "tau": float(spec["tau"]), "ncells": int(spec["ncells"]),
        "lim": CHECK_LIM, "grid": CHECK_GRID,
    }
    print(f"wrote {OUT}/{name}.bin / _ref.bin  {N} pts each, "
          f"{len(sup)} support pts, {spec['ncells']} cells, tau={spec['tau']:.3f}")
    print(f"      range: x[{pts[:,0].min():+.3f},{pts[:,0].max():+.3f}] "
          f"y[{pts[:,1].min():+.3f},{pts[:,1].max():+.3f}]")

# Kept for compatibility: the pre-2026-08-28 scorer read the centres directly.
_, centres, _ = eight_gaussians(1, np.random.default_rng(0))
centres.tofile(f"{OUT}/eight_gaussians_centres.bin")

with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"wrote {OUT}/manifest.json  ({len(TARGETS)} targets)")
