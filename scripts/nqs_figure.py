#!/usr/bin/env python3
"""The neural-quantum-state figure — planning/transformer_wavefunction_demo.md §6.

  python3 scripts/nqs_figure.py runs/2026-09-11-nqs-ising nqs_ising.png

Reads `<run>/n64/*_metrics.json` (the sweep) and `<run>/samples/` (the GPT's
`_samples.bin` dumps from `lake exe nqs-ising`, copied there), brackets against
the closed forms in `scripts/nqs_metrics.py`, and draws, for a reader who has
not met the Ising chain:

  (a) what the network is asked to find — rows of 64 spins drawn from the
      trained GPT wavefunction at h/J = 0.2, 1 and 2: ordered, critical,
      disordered;
  (b) the check — for every one of the 4,096 configurations at N = 12, h = J,
      the probability the trained GPT assigns against the exact ground state's,
      with the mean-field reference it starts from in grey;
  (c) the result — relative energy error against h/J at N = 64 for the ladder,
      the mean-field floor in grey, bars the Monte Carlo standard error.

Series colours: mean field grey, MLP blue, ViT orange, GPT aqua — the same
slots throughout. Needs matplotlib (the system python3, not the JAX venv).
"""
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nqs_metrics as M

run, out = sys.argv[1], sys.argv[2]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
BLUE, ORANGE, AQUA, INK, MUTED, RULE = "#2a78d6", "#eb6834", "#1baf7a", "#1f1e1b", "#6b6963", "#bdbab1"
COL = {"mf": "#9e9b91", "mlp": BLUE, "vit": ORANGE, "gpt": AQUA}
NAME = {"mf": "mean field", "mlp": "MLP residual", "vit": "ViT residual", "gpt": "GPT"}
MARK = {"mlp": "o", "vit": "s", "gpt": "^"}


def load(sub):
    rows = {}
    for f in sorted(glob.glob(f"{run}/{sub}/*_metrics.json")):
        with open(f) as fh:
            m = json.load(fh)
        if not m.get("useRef", True):
            continue
        rows.setdefault(m["arch"], []).append(m)
    for a in rows:
        rows[a].sort(key=lambda m: m["h"])
    return rows


def samples(name, N):
    """`<prefix>_samples.bin`: one row per sample, N spins as ±1 f32, then weight,
    E_loc and <s^x>. At N ≤ 14 the rows are every configuration with its p(σ)."""
    a = np.fromfile(f"{run}/samples/{name}_samples.bin", dtype=np.float32).reshape(-1, N + 3)
    return a[:, :N], a[:, N].astype(np.float64)


def style(ax):
    ax.tick_params(length=2, color=MUTED)
    for s in ax.spines.values():
        s.set_color(RULE); s.set_linewidth(0.6)
    ax.grid(True, which="major", color="#ecebe6", linewidth=0.6)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(13.2, 4.3), constrained_layout=True)
# the panel labels go in a band above the axes, placed after the layout pass (a long
# label on the first raster would otherwise be laid out as that axes' own decoration)
fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1, 1])
gsA = gs[0, 0].subgridspec(1, 3, wspace=0.06)
axA = [fig.add_subplot(gsA[0, i]) for i in range(3)]
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

# ── (a) samples from the trained GPT wavefunction, N = 64 ───────────────────
ROWS = 96
for ax, (name, h, word) in zip(axA, [("nqs_ising_gpt_n64_q1_h20", 0.2, "ordered"),
                                      ("nqs_ising_gpt_n64_q3_h100", 1.0, "critical"),
                                      ("nqs_ising_gpt_n64_r2_h200", 2.0, "disordered")]):
    S, _ = samples(name, 64)
    ax.imshow(S[:ROWS] > 0, cmap=ListedColormap(["white", INK]), aspect="auto",
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(RULE); s.set_linewidth(0.6)
    ax.set_title(f"h = {h:g} J   {word}", fontsize=8.2, loc="center", pad=3)
axA[0].set_ylabel(f"{ROWS} samples, one per row", fontsize=7.6)
axA[1].set_xlabel("the 64 spins of the chain: dark up, light down", fontsize=7.6)

# ── (b) all 4,096 configurations at N = 12: network against exact ───────────
N12, h12 = 12, 1.0
S12, p_gpt = samples("nqs_ising_gpt_n12_g2_h100", N12)
idx = ((S12 > 0).astype(np.int64) << np.arange(N12)).sum(axis=1)   # bit i of the index is spin i
H, Sall = M.hamiltonian(N12, h12)
w, v = np.linalg.eigh(H)
p_exact = v[:, 0] ** 2
p_exact = p_exact[idx]
phi = M.mean_field(N12, h12)["phi"]
up, dn = math.cos(phi / 2) ** 2, math.sin(phi / 2) ** 2
p_mf = np.prod(np.where(S12 > 0, up, dn), axis=1)
axB.scatter(p_exact, p_mf, s=5, color=COL["mf"], alpha=0.55, lw=0, label="mean-field reference (the start)", zorder=2)
axB.scatter(p_exact, p_gpt, s=5, color=COL["gpt"], alpha=0.8, lw=0, label="trained GPT", zorder=3)
lo, hi = p_exact.min() * 0.5, p_exact.max() * 2
axB.plot([lo, hi], [lo, hi], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlim(lo, hi); axB.set_ylim(lo, hi)
axB.set_xlabel("exact ground state:  |ψ(σ)|²")
axB.set_ylabel("network:  |ψ_θ(σ)|²")
axB.legend(loc="upper left", frameon=False, fontsize=7.3, handlelength=1.0, markerscale=2.2)
style(axB)

# ── (c) relative energy error against h, N = 64 ─────────────────────────────
rows = load("n64")
hs = np.arange(0.2, 2.01, 0.2)
mfe = [abs(M.mean_field(64, h)["E"] - M.exact(64, h)["E0"]) / abs(M.exact(64, h)["E0"]) for h in hs]
axC.plot(hs, mfe, color=COL["mf"], lw=1.6, label=NAME["mf"], zorder=2)
for a in ("mlp", "vit", "gpt"):
    if a not in rows:
        continue
    h = np.array([m["h"] for m in rows[a]])
    E0 = np.array([M.exact(64, x)["E0"] for x in h])
    rel = (np.array([m["E"] for m in rows[a]]) - E0) / np.abs(E0)
    se = np.array([math.sqrt(max(m["var"], 0) / max(m["samples"], 1)) for m in rows[a]]) / np.abs(E0)
    axC.errorbar(h, np.abs(rel), yerr=se, color=COL[a], marker=MARK[a], ms=4.5, lw=1.4,
                 capsize=2, elinewidth=0.8, label=NAME[a], zorder=3)
axC.set_yscale("log")
axC.set_xlabel("h / J")
axC.set_xticks(hs[::2])
axC.axvline(1.0, color=RULE, lw=0.8, ls=(0, (3, 3)), zorder=1)
axC.text(1.02, 0.97, "transition", transform=axC.get_xaxis_transform(), fontsize=6.5,
         color=MUTED, ha="left", va="top", rotation=90)
axC.set_ylabel("|E − E₀| / |E₀|")
# bottom middle: the bars at h = 0.2, 0.6 and 1.8 run to the floor, the transition line
# behind the labels is hidden by the white patch
axC.legend(loc="lower left", bbox_to_anchor=(0.27, 0.0), frameon=True, framealpha=0.92,
           edgecolor="none", fontsize=7.3, handlelength=1.6)
axC.text(0.02, 0.97, "N = 64, ceiling by Jordan-Wigner\nbars: Monte Carlo s.e.", transform=axC.transAxes,
         fontsize=6.3, color=MUTED, ha="left", va="top")
style(axC)

fig.canvas.draw()
for ax, label in [(axA[0], "(a)  what the network is asked to find: the ground state of a ring of spins"),
                  (axB, f"(b)  the check: every configuration, N = {N12}, h = J"),
                  (axC, "(c)  the result: energy error against field")]:
    fig.text(ax.get_position().x0, 0.985, label, fontsize=9.5, ha="left", va="top", in_layout=False)
fig.savefig(out, dpi=190, facecolor="white")
print("wrote", out)
