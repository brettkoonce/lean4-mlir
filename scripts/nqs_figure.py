#!/usr/bin/env python3
"""The neural-quantum-state figure — planning/transformer_wavefunction_demo.md §6.

  python3 scripts/nqs_figure.py runs/2026-09-11-nqs-ising nqs_ising.png

Reads every `<arch>_h<h>_metrics.json` under `<run>/n12/` and `<run>/n64/`
(the files `lake exe nqs-ising` writes, copied there), brackets each against
the closed forms in `scripts/nqs_metrics.py`, and draws:

  (a) the ansatz: a spin configuration split into patch tokens, the transformer,
      its head, and the host adding the mean-field reference log ψ_ref(σ) before
      any ratio is formed — with the GPT variant's logit bias beside it;
  (b) relative energy error against h/J for the ladder (mean field, MLP, ViT,
      GPT), N = 12 by enumeration and N = 64 by Jordan-Wigner side by side;
  (c) the long-range test: ⟨σᶻ₁σᶻ₁₊ᵣ⟩ against r at h = J, N = 64, exact against
      the trained ViT and GPT.

Series colours: mean field grey, MLP blue, ViT orange, GPT aqua — the same
slots throughout. Needs matplotlib (the system python3, not the JAX venv).
"""
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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


def style(ax):
    ax.tick_params(length=2, color=MUTED)
    for s in ax.spines.values():
        s.set_color(RULE); s.set_linewidth(0.6)
    ax.grid(True, which="major", color="#ecebe6", linewidth=0.6)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(13.2, 3.9), constrained_layout=True)
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1])
axA = fig.add_subplot(gs[0, 0])
axB1 = fig.add_subplot(gs[0, 1])
axB2 = fig.add_subplot(gs[0, 2], sharey=axB1)
axC = fig.add_subplot(gs[0, 3])

# ── (a) the ansatz ───────────────────────────────────────────────────────────
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")


def box(x, y, w, h, text, fc="white", ec=RULE, fs=7.6, weight="normal", color=INK):
    axA.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                 fc=fc, ec=ec, lw=0.8))
    axA.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
             color=color, weight=weight, linespacing=1.25)


def arrow(x0, y0, x1, y1, color=INK):
    axA.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                  lw=0.8, color=color, shrinkA=0, shrinkB=0))


spins = "↑↓↓↑ ↑↑↓↑ ↓↑↑↓ ↑↓↑↑"
box(0.3, 8.45, 9.4, 1.15, f"σ ∈ {{±1}}ᴺ        {spins}", fs=8)
arrow(2.55, 8.45, 2.55, 7.5)
arrow(7.45, 8.45, 7.45, 7.5)
box(0.3, 6.25, 4.5, 1.25, "patches of p spins → token ids\n(9, 13, 6, 11)", fs=7.3)
box(5.2, 6.25, 4.5, 1.25, "mean-field reference\nlog ψ_ref(σ) = Σᵢ (a + b σᵢ)", fc="#eef4fc", fs=7.3)
arrow(2.55, 6.25, 2.55, 5.3)
box(0.3, 4.0, 4.5, 1.3, "transformer\nViT: encoder, mean over tokens\nGPT: causal, lmHead", fs=7.1)
arrow(2.55, 4.0, 2.55, 3.05)
box(0.3, 1.85, 4.5, 1.2, "head → f_θ(σ)\nGPT: logits, + log p_ref as bias", fs=7.1)
arrow(4.8, 2.45, 5.2, 2.45)
arrow(7.45, 6.25, 7.45, 3.05)
box(5.2, 1.85, 4.5, 1.2, "host adds them:\nlog ψ_θ = log ψ_ref + f_θ", fc="#eef4fc", fs=7.3, weight="bold")
axA.text(5.0, 0.75, "E_loc from ψ ratios on the eval graph;  ∂E/∂θ through the DDPM MSE block,\ntarget y = out − M·w/2 with w = 2 p(σ)(E_loc − E)",
         ha="center", va="center", fontsize=6.8, color=MUTED, linespacing=1.3)
axA.set_title("(a)  structure × residual: the ansatz", loc="left")

# ── (b) relative energy error against h ──────────────────────────────────────
for ax, sub, N, title in [(axB1, "n12", 12, "(b)  N = 12, ceiling by enumeration"),
                          (axB2, "n64", 64, "N = 64, ceiling by Jordan-Wigner")]:
    rows = load(sub)
    hs = np.arange(0.2, 2.01, 0.2)
    mfe = [abs(M.mean_field(N, h)["E"] - M.exact(N, h)["E0"]) / abs(M.exact(N, h)["E0"]) for h in hs]
    ax.plot(hs, mfe, color=COL["mf"], lw=1.6, label=NAME["mf"], zorder=2)
    for a in ("mlp", "vit", "gpt"):
        if a not in rows:
            continue
        h = np.array([m["h"] for m in rows[a]])
        E0 = np.array([M.exact(N, x)["E0"] for x in h])
        rel = np.array([m["E"] for m in rows[a]]) - E0
        rel = rel / np.abs(E0)
        se = np.array([math.sqrt(max(m["var"], 0) / max(m["samples"], 1)) for m in rows[a]]) / np.abs(E0)
        if N > 14:
            ax.errorbar(h, np.abs(rel), yerr=se, color=COL[a], marker=MARK[a], ms=4.5, lw=1.4,
                        capsize=2, elinewidth=0.8, label=NAME[a], zorder=3)
        else:
            ax.plot(h, np.abs(rel), color=COL[a], marker=MARK[a], ms=4.5, lw=1.4, label=NAME[a], zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("h / J")
    ax.set_xticks(hs[::2])
    ax.axvline(1.0, color=RULE, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_title(title, loc="left")
    style(ax)
axB1.set_ylabel("|E − E₀| / |E₀|")
plt.setp(axB2.get_yticklabels(), visible=False)
axB2.legend(loc="lower right", frameon=False, fontsize=7.3, handlelength=1.6)
axB1.text(1.02, 0.97, "transition", transform=axB1.get_xaxis_transform(), fontsize=6.5,
          color=MUTED, ha="left", va="top", rotation=90)
axB2.text(0.02, 0.97, "bars: Monte Carlo s.e.\nof the sample energy", transform=axB2.transAxes,
          fontsize=6.3, color=MUTED, ha="left", va="top")

# ── (c) the long-range test at h = J, N = 64 ────────────────────────────────
rows64 = load("n64")
ex = M.jw_exact(64, 1.0)
r = np.arange(0, 33)
axC.plot(r, ex["corr"], color=INK, lw=1.6, label="exact (Jordan-Wigner)", zorder=3)
mf = M.mean_field(64, 1.0)["corr"]
axC.plot(r, mf, color=COL["mf"], lw=1.4, ls=(0, (4, 2)), label="mean field", zorder=2)
for a in ("vit", "gpt"):
    m = next((m for m in rows64.get(a, []) if abs(m["h"] - 1.0) < 1e-6), None)
    if m is None:
        continue
    c = np.array(m["corr"])
    axC.plot(r[:len(c)], c, color=COL[a], marker=MARK[a], ms=3.6, lw=0, label=NAME[a], zorder=4,
             markeredgecolor="white", markeredgewidth=0.5)
axC.set_xlabel("r")
axC.set_ylabel("⟨σᶻ₁ σᶻ₁₊ᵣ⟩ at h = J, N = 64")
axC.set_xlim(0, 32); axC.set_ylim(0.3, 1.02)
axC.legend(loc="upper right", frameon=False, fontsize=7.3, handlelength=1.6)
axC.set_title("(c)  the long-range test", loc="left")
style(axC)

fig.savefig(out, dpi=190, facecolor="white")
print("wrote", out)
