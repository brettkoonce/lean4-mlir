#!/usr/bin/env python3
"""2-D toy target distributions for the diffusion demo.

Writes flat float32-LE [N, 2] point clouds that the Lean side reads with a
plain `IO.FS.readBinFile` (no header, no loader needed — the bytes ARE the
array). See planning/diffusion_2d_demo.md §2.

Targets are scaled to roughly unit radius so the data sits in the range a
cosine-schedule DDPM expects (x0 ~ N(0,1)-ish); an 8-gaussians ring at the
literature's radius 2 would need the schedule retuned.

Usage: python3 preprocess_toy2d.py [n=8192] [outdir=data/toy2d]
"""
import os, sys
import numpy as np

N      = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
OUT    = sys.argv[2] if len(sys.argv) > 2 else "data/toy2d"
RADIUS = 1.0
STD    = 0.05
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


def spiral(n, rng):
    """One long thin manifold — catches corner-cutting."""
    t = rng.uniform(0, 1, size=n) ** 0.5 * 3.0 * np.pi
    r = t / (3.0 * np.pi) * RADIUS
    pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    pts += rng.normal(0, 0.01, size=(n, 2))
    return pts.astype(np.float32), None, None


rng = np.random.default_rng(0)
pts, centres, idx = eight_gaussians(N, rng)
pts.tofile(f"{OUT}/eight_gaussians.bin")
centres.tofile(f"{OUT}/eight_gaussians_centres.bin")
# A second, independent draw: the metric script needs a TRUE sample to compare
# against, and reusing the training set would flatter any model that memorised.
ref, _, _ = eight_gaussians(N, np.random.default_rng(1))
ref.tofile(f"{OUT}/eight_gaussians_ref.bin")

sp, _, _ = spiral(N, rng)
sp.tofile(f"{OUT}/spiral.bin")
spref, _, _ = spiral(N, np.random.default_rng(1))
spref.tofile(f"{OUT}/spiral_ref.bin")

print(f"wrote {OUT}/eight_gaussians.bin      {N} pts, radius {RADIUS}, std {STD}")
print(f"wrote {OUT}/eight_gaussians_ref.bin  {N} pts (independent draw, seed 1)")
print(f"wrote {OUT}/eight_gaussians_centres.bin  8 centres")
print(f"wrote {OUT}/spiral.bin / spiral_ref.bin  {N} pts each")
print(f"range: x[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
      f"y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]")
