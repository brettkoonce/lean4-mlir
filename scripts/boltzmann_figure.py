#!/usr/bin/env python3
"""The Boltzmann-generator figure (planning/boltzmann_generator_demo.md §7), three panels.

  (a) the Müller-Brown surface with the Langevin training set at kT = 20
  (b) the trained flow from N(0, I) into the wells: paths of the first points of
      the cloud, hollow at the noise end, and every sample at t = 0
  (c) kT = 8: the Langevin chain started in well B beside the model's samples
      reweighted to that temperature, with the three population rows in a box

Inputs: data/boltzmann (preprocess_boltzmann.py) and one run directory holding
  samples/<flow NFE-50 samples>.bin, .paths.bin      lake exe diffusion-2d muller_brown flow … logp
  corrected_kT8.bin, transfer.json                   scripts/boltzmann_metrics.py transfer … --out=<run>

  python3 scripts/boltzmann_figure.py runs/2026-09-11-boltzmann-generator out.png
Series colours: blue for the model, orange for Langevin. Needs matplotlib
(the system python3 has it; the pinned .venv does not).
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

run = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "boltzmann_mb.png"
SAMPLES = f"{run}/samples/diffusion2d_samples_muller_brown-flow_fm-euler_s20000_n50_e0"
DATA = "data/boltzmann"

with open(f"{DATA}/manifest.json") as f:
    MAN = json.load(f)
GRID = np.load(f"{DATA}/mb_grid.npz")
CEN, SC = np.array(MAN["centre"]), MAN["scale"]
minima, Umin, saddles = np.array(MAN["minima"]), np.array(MAN["Umin"]), np.array(MAN["saddles"])
xs, ys = GRID["xs"], GRID["ys"]
X, Y = np.meshgrid(xs, ys)
UG = GRID["U"]
to_x = lambda Z: Z.astype(np.float64) * SC + CEN
load = lambda p: np.fromfile(p, dtype=np.float32).reshape(-1, 2)

L20 = to_x(load(f"{DATA}/mb_kT20.bin"))
L8 = to_x(load(f"{DATA}/mb_kT8_langevinB.bin"))
F50 = to_x(load(f"{SAMPLES}.bin"))
paths = np.fromfile(f"{SAMPLES}.paths.bin", dtype=np.float32)
nPts = 128
paths = to_x(paths.reshape(-1, nPts, 2))            # [nSteps+1, 128, 2]
F8 = to_x(load(f"{run}/corrected_kT8.bin"))
with open(f"{run}/transfer.json") as f:
    T8 = json.load(f)["8"]

rng = np.random.default_rng(0)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1e1b", "#6b6963"
greys = LinearSegmentedColormap.from_list("warmgrey", ["#ffffff", "#ecebe6", "#d8d6cf", "#bdbab1", "#9e9b91"])
levels = np.arange(-150, 60, 10)
UGc = np.clip(UG, -160, 60)


def surface(ax, lw=0.35):
    ax.contourf(X, Y, UGc, levels=levels, cmap=greys, extend="max")
    ax.contour(X, Y, UGc, levels=levels, colors="#8d8a80", linewidths=lw)
    ax.set_xlim(-1.6, 1.25); ax.set_ylim(-0.6, 2.1)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.tick_params(length=2, color=MUTED)
    for s in ax.spines.values():
        s.set_color("#bdbab1"); s.set_linewidth(0.6)


def label_wells(ax):
    for (mx, my), u, name, off in zip(minima, Umin, "ABC", [(-0.05, 0.13), (0.06, -0.15), (0.16, 0.02)]):
        ax.text(mx + off[0], my + off[1], f"{name}  U={u:.0f}", ha="center", va="center", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.scatter(saddles[:, 0], saddles[:, 1], marker="x", s=18, c=INK, linewidths=0.8, zorder=5)


fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)

# (a) surface + Langevin training data
ax = axes[0]; surface(ax)
sub = L20[rng.choice(len(L20), 3000, replace=False)]
ax.scatter(sub[:, 0], sub[:, 1], s=3, c=ORANGE, alpha=0.45, linewidths=0, zorder=3)
label_wells(ax)
ax.set_title("(a)  kT = 20: the surface, and the Langevin training set", loc="left")
ax.text(0.02, 0.02, f"{MAN['langevin']['chains']} chains × 10⁵ steps, thinned; orange",
        transform=ax.transAxes, fontsize=7.5, color=MUTED)

# (b) the trained flow's trajectories
ax = axes[1]; surface(ax, lw=0.25)
sel = rng.choice(nPts, 70, replace=False)
for i in sel:
    ax.plot(paths[:, i, 0], paths[:, i, 1], color=BLUE, lw=0.7, alpha=0.55, zorder=3)
ax.scatter(paths[0, sel, 0], paths[0, sel, 1], s=9, facecolors="white", edgecolors=MUTED, linewidths=0.6, zorder=4)
ax.scatter(F50[:, 0], F50[:, 1], s=3, c=BLUE, alpha=0.5, linewidths=0, zorder=4)
ax.set_title(f"(b)  the trained flow, N(0, I) → exp(−U/kT), 70 of {len(F50)} paths", loc="left")
ax.text(0.02, 0.02, "hollow: noise at t = 1;  blue: samples at t = 0, Euler NFE 50",
        transform=ax.transAxes, fontsize=7.5, color=MUTED)

# (c) kT = 8: Langevin stuck in B, the flow reweighted
ax = axes[2]; surface(ax, lw=0.25)
ax.plot(L8[:, 0], L8[:, 1], color=ORANGE, lw=0.4, alpha=0.5, zorder=3)
ax.scatter(L8[::4, 0], L8[::4, 1], s=3, c=ORANGE, alpha=0.6, linewidths=0, zorder=4)
ax.scatter(F8[:, 0], F8[:, 1], s=3, c=BLUE, alpha=0.5, linewidths=0, zorder=4)
label_wells(ax)
row = lambda r: f"{r['pops'][0]:.3f} / {r['pops'][1]:.3f} / {r['pops'][2]:.3f}"
txt = ("well populations A / B / C\n"
       f"exact      {row(T8['exact'])}\n"
       f"Langevin   {row(T8['langevin'])}\n"
       f"flow       {row(T8.get('corrected', T8['reweighted']))}")
ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=7.2, family="DejaVu Sans Mono", va="bottom", color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d8d6cf", lw=0.6))
ax.set_title("(c)  kT = 8: a chain started in B, 2×10⁵ steps, and the flow reweighted", loc="left")
fig.savefig(out, dpi=190, facecolor="white")
print(f"wrote {out}")
