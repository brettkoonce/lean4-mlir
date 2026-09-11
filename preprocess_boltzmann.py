#!/usr/bin/env python3
"""Müller-Brown data for the Boltzmann-generator demo — planning/boltzmann_generator_demo.md §3.

The data is the physics. Nothing is downloaded: the target is the density
exp(-U/kT) on the Müller-Brown surface (Müller & Brown 1979), so this script
writes the training set an overdamped Langevin integrator manages to produce in
a fixed budget, an INDEPENDENT draw from the exact density by quadrature (the
energy distance's reference), and the grid the scorer integrates over. Every
column of the demo's tables is exact for the first row because of the grid.

Writes, into `outdir` (default data/boltzmann):
  mb_kT20.bin            Langevin training set, standardised coords, f32 [N, 2]
  mb_kT20_ref.bin        independent draw from the quadrature density, f32 [nref, 2]
  mb_kT12_langevinB.bin  a fresh chain at kT = 12 started in well B, 2e5 steps, thinned
  mb_kT8_langevinB.bin   the same at kT = 8 — the MCMC control of table 2
  mb_grid.npz            the 360x360 grid, U, basin label per cell, the exact
                         populations / mean energy / dF / log Z at kT = 20, 12, 8
  manifest.json          kT, the standardisation, the minima, the constants;
                         the scorer reads it rather than retyping

Coordinates in every .bin are STANDARDISED, z = (x - c) / s with c = (-0.2, 0.75)
and s = 0.8, so N(0, I) covers the box; the same reason preprocess_toy2d.py
scales its targets to unit radius. U is always evaluated in the original
coordinates.

Basins are assigned by gradient DESCENT to a minimum, never by nearest
minimum: the wells are elongated and Voronoi cells put part of A's basin in C.

⚠ The Langevin set is deliberately not equilibrated. At kT = 20, 8 chains x 1e5
steps still over-weight A by a few points; the model trained on it inherits
that, the table shows it, and it is the reason the quadrature row exists.

Usage: python3 preprocess_boltzmann.py [nref=4096] [outdir=data/boltzmann]
"""
import json, os, sys
import numpy as np

NREF = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/boltzmann"
os.makedirs(OUT, exist_ok=True)

# ── Müller-Brown, the textbook constants ────────────────────────────────────
A = np.array([-200., -100., -170., 15.])
a = np.array([-1., -1., -6.5, 0.7])
b = np.array([0., 0., 11., 0.6])
c = np.array([-10., -10., -6.5, 0.7])
X0 = np.array([1., 0., -0.5, -1.])
Y0 = np.array([0., 0.5, 1.5, 1.])

CEN = np.array([-0.2, 0.75])   # standardisation centre
SC = 0.8                       # standardisation scale
BOX = [[-1.7, 1.3], [-0.7, 2.2]]
GRID = 360
KTS = [20.0, 12.0, 8.0]
KT_TRAIN = 20.0
DT = 1e-4          # curvature in the deepest well is ~2200, so dt < 9e-4 for stability
CHAINS = 8
STEPS = 100_000
BURN = 10_000
THIN = 20


def U(P):
    x, y = P[..., 0:1], P[..., 1:2]
    dx, dy = x - X0, y - Y0
    return (A * np.exp(a * dx * dx + b * dx * dy + c * dy * dy)).sum(-1)


def gradU(P):
    x, y = P[..., 0:1], P[..., 1:2]
    dx, dy = x - X0, y - Y0
    e = A * np.exp(a * dx * dx + b * dx * dy + c * dy * dy)
    return np.stack([(e * (2 * a * dx + b * dy)).sum(-1),
                     (e * (b * dx + 2 * c * dy)).sum(-1)], -1)


def descend(P, steps=1500, eta=1e-4, cap=0.01):
    """Capped gradient descent; the cap keeps the huge gradients far from the
    wells from throwing a point out of the box. 1500 steps of at most 0.01
    covers the box three times over."""
    P = P.copy()
    for _ in range(steps):
        g = gradU(P)
        n = np.linalg.norm(g, axis=-1, keepdims=True)
        P -= eta * g * np.minimum(1.0, cap / (eta * n + 1e-12))
    return P


# The three minima, polished from the literature values.
minima = descend(np.array([[-0.558, 1.442], [0.623, 0.028], [-0.050, 0.467]]),
                 steps=6000, eta=5e-5)
Umin = U(minima)
saddles = np.array([[-0.822, 0.624], [0.212, 0.293]])
print("minima  A B C:", np.round(minima, 4).tolist(), "U =", np.round(Umin, 2).tolist())
print("saddles U:", np.round(U(saddles), 2).tolist())


def basin(P):
    Q = descend(P)
    d = ((Q[:, None, :] - minima[None]) ** 2).sum(-1)
    lab = d.argmin(1)
    far = np.sqrt(d.min(1)) > 0.05
    if far.any():
        print(f"  ⚠ {far.sum()} of {len(P)} points did not descend to within 0.05 of a minimum")
    return lab


to_z = lambda P: ((P - CEN) / SC).astype(np.float32)

# ── the grid: quadrature is the ground truth ────────────────────────────────
xs = np.linspace(BOX[0][0], BOX[0][1], GRID)
ys = np.linspace(BOX[1][0], BOX[1][1], GRID)
X, Y = np.meshgrid(xs, ys)
G = np.stack([X, Y], -1).reshape(-1, 2)
UG = U(G)
dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
print(f"grid {GRID}x{GRID}, assigning basins by descent ...")
BG = basin(G)

pops, meanU, dF, logZ = [], [], [], []
for kT in KTS:
    w = np.exp(-(UG - UG.min()) / kT)
    Z = w.sum()
    p = w / Z
    pk = np.array([p[BG == k].sum() for k in range(3)])
    pops.append(pk)
    meanU.append((p * UG).sum())
    dF.append(-kT * np.log(pk[0] / pk[1]))
    # log Z in the ORIGINAL coordinates: ∫ exp(-U/kT) dx dy by the midpoint rule.
    logZ.append(np.log(Z * dA) - UG.min() / kT)
    print(f"kT={kT:4.0f}: p(A/B/C) = {pk[0]:.3f} / {pk[1]:.3f} / {pk[2]:.3f}   "
          f"<U> = {meanU[-1]:7.1f}   dF_AB = {dF[-1]:6.1f}   logZ = {logZ[-1]:.3f}")
    # Mass outside the box is what the quadrature cannot see; report it via the
    # edge cells so a box that is too small fails loudly.
    edge = np.zeros((GRID, GRID), bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    print(f"          mass on the box edge: {p.reshape(GRID, GRID)[edge].sum():.2e}")

np.savez(f"{OUT}/mb_grid.npz", xs=xs, ys=ys, U=UG.reshape(GRID, GRID),
         basin=BG.reshape(GRID, GRID).astype(np.int8), minima=minima, Umin=Umin,
         saddles=saddles, kT=np.array(KTS), pops=np.array(pops), meanU=np.array(meanU),
         dF=np.array(dF), logZ=np.array(logZ), dA=dA)

# ── the reference draw: multinomial over cells, jittered inside the cell ────
rng = np.random.default_rng(1)
w = np.exp(-(UG - UG.min()) / KT_TRAIN)
idx = rng.choice(len(w), size=NREF, p=w / w.sum())
jit = (rng.random((NREF, 2)) - 0.5) * np.array([xs[1] - xs[0], ys[1] - ys[0]])
ref = G[idx] + jit
to_z(ref).tofile(f"{OUT}/mb_kT20_ref.bin")
pr = np.bincount(basin(ref), minlength=3) / NREF
print(f"wrote {OUT}/mb_kT20_ref.bin  {NREF} pts, p = {pr[0]:.3f} / {pr[1]:.3f} / {pr[2]:.3f}")


# ── Langevin: the Euler-Maruyama integrator doing its actual job ────────────
def langevin(P0, kT, steps, thin, burn, rng):
    P = P0.copy()
    out = []
    s = np.sqrt(2 * kT * DT)
    for i in range(steps):
        P = P - gradU(P) * DT + s * rng.standard_normal(P.shape)
        if i >= burn and i % thin == 0:
            out.append(P.copy())
    return np.array(out)          # [T, chains, 2]


rng = np.random.default_rng(0)
P0 = np.stack([rng.uniform(-1.5, 1.1, CHAINS), rng.uniform(-0.4, 2.0, CHAINS)], -1)
L = langevin(P0, KT_TRAIN, STEPS, THIN, BURN, rng).reshape(-1, 2)
# ⚠ Shuffled before writing. The trainer takes CONTIGUOUS batches, and a batch of
# 256 consecutive thinned samples from one chain spans 5,120 Langevin steps in
# one well — every minibatch would be a single basin.
L = L[rng.permutation(len(L))]
to_z(L).tofile(f"{OUT}/mb_kT20.bin")
pl = np.bincount(basin(L), minlength=3) / len(L)
print(f"wrote {OUT}/mb_kT20.bin  {len(L)} pts ({CHAINS} chains x {STEPS} steps, burn {BURN}, "
      f"thin {THIN}), p = {pl[0]:.3f} / {pl[1]:.3f} / {pl[2]:.3f}, <U> = {U(L).mean():.1f}")

# The MCMC control of the transfer table: one chain started in well B at the
# transfer temperature, 2e5 steps. At kT = 8 the barrier out of B is 36 units,
# e^4.5 attempts per crossing, and the chain has not equilibrated.
for kT in KTS[1:]:
    rng = np.random.default_rng(int(kT))
    Lc = langevin(minima[1:2].copy(), kT, 200_000, 50, 0, rng)[:, 0, :]
    to_z(Lc).tofile(f"{OUT}/mb_kT{int(kT)}_langevinB.bin")
    pc = np.bincount(basin(Lc), minlength=3) / len(Lc)
    print(f"wrote {OUT}/mb_kT{int(kT)}_langevinB.bin  {len(Lc)} pts, "
          f"p = {pc[0]:.3f} / {pc[1]:.3f} / {pc[2]:.3f}")

manifest = {
    "kT": KT_TRAIN, "kTs": KTS, "centre": CEN.tolist(), "scale": SC, "box": BOX, "grid": GRID,
    "minima": minima.tolist(), "Umin": Umin.tolist(), "saddles": saddles.tolist(),
    "constants": {"A": A.tolist(), "a": a.tolist(), "b": b.tolist(), "c": c.tolist(),
                  "x0": X0.tolist(), "y0": Y0.tolist()},
    "langevin": {"dt": DT, "chains": CHAINS, "steps": STEPS, "burn": BURN, "thin": THIN,
                 "n": int(len(L))},
    "nref": NREF,
    "files": {"train": "mb_kT20.bin", "ref": "mb_kT20_ref.bin", "grid": "mb_grid.npz",
              "langevinB": {"12": "mb_kT12_langevinB.bin", "8": "mb_kT8_langevinB.bin"}},
}
with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"wrote {OUT}/manifest.json")
