#!/usr/bin/env python3
"""Rung R4 figure — planning/transformer_wavefunction_demo.md §3 (R4), Table 3 drawn.

  python3 scripts/nqs_j1j2_figure.py runs/2026-09-11-nqs-ising/j1j2 nqs_j1j2.png

Reads every `*_metrics.json` (+ `_psi.bin`) the exe wrote with `model=j1j2`, brackets
each against Lanczos in the S_z = 0 sector, and draws (a) the relative energy error
against J2/J1 for the MLP and the ViT with and without the Marshall sign prior, the
uniform × Marshall floor above them; (b) the fidelity to the ED ground space, which is
the column that says whether the phase head found the signs. The Majumdar-Ghosh point
J2 = J1/2, where E0/N = −3/8 exactly and the ground space is two-dimensional, is
marked. MLP blue, ViT orange; solid with the prior, dashed without. Needs matplotlib.
"""
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nqs_metrics as M

run, out = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
BLUE, ORANGE, INK, MUTED, RULE, GREY = "#2a78d6", "#eb6834", "#1f1e1b", "#6b6963", "#bdbab1", "#9e9b91"
COL = {"mlp": BLUE, "vit": ORANGE}
NAME = {"mlp": "MLP", "vit": "ViT"}

rows = {}
for f in sorted(glob.glob(f"{run}/*_metrics.json")):
    m = json.load(open(f))
    if m.get("model") != "j1j2":
        continue
    N, J2 = int(m["N"]), round(float(m["J2"]), 6)
    ex = M.j1j2_exact(N, J2, 1.0)
    rel = (m["E"] - ex["E0"]) / abs(ex["E0"])
    psi = f.replace("_metrics.json", "_psi.bin")
    F = M.fidelity(ex, psi, N) if os.path.exists(psi) else float("nan")
    key = (m["arch"], bool(m.get("useRef", True)))
    rows.setdefault(key, []).append((J2, rel, F, ex["E_floor"], ex["E0"]))
N = int(json.load(open(sorted(glob.glob(f"{run}/*_metrics.json"))[0]))["N"])
J2s = sorted({j for v in rows.values() for (j, *_) in v})
floor = [(M.j1j2_exact(N, j, 1.0)["E_floor"] - M.j1j2_exact(N, j, 1.0)["E0"]) / abs(M.j1j2_exact(N, j, 1.0)["E0"]) for j in J2s]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
for ax in (axA, axB):
    ax.tick_params(length=2, color=MUTED)
    for s in ax.spines.values():
        s.set_color(RULE); s.set_linewidth(0.6)
    ax.grid(True, color="#ecebe6", linewidth=0.6); ax.set_axisbelow(True)
    ax.axvline(0.5, color=RULE, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xlabel("J₂ / J₁"); ax.set_xticks(J2s)
axA.plot(J2s, floor, color=GREY, lw=1.6, label="uniform × Marshall (floor)", zorder=2)
for (arch, prior), v in sorted(rows.items()):
    v.sort()
    j = [x[0] for x in v]
    style = dict(color=COL[arch], marker="o" if arch == "mlp" else "s", ms=4.5, lw=1.4,
                 ls="-" if prior else (0, (4, 2)), markerfacecolor=COL[arch] if prior else "white",
                 markeredgecolor=COL[arch], label=f"{NAME[arch]}, {'with' if prior else 'no'} sign prior", zorder=3)
    axA.plot(j, [abs(x[1]) for x in v], **style)
    axB.plot(j, [x[2] for x in v], **style)
axA.set_yscale("log"); axA.set_ylabel("|E − E₀| / |E₀|")
axA.set_title(f"(a)  J₁-J₂ chain, N = {N}, S_z = 0 sector, ceiling by Lanczos", loc="left")
axA.text(0.51, 0.97, "Majumdar-Ghosh, E₀/N = −3/8", transform=axA.get_xaxis_transform(), fontsize=6.5,
         color=MUTED, ha="left", va="top", rotation=90)
axB.set_ylabel("fidelity to the ED ground space"); axB.set_ylim(0, 1.03)
axB.set_title("(b)  did the phase head find the signs?", loc="left")
axB.legend(loc="lower left", frameon=False, fontsize=7.2, handlelength=2.2)
fig.savefig(out, dpi=190, facecolor="white")
print("wrote", out)
