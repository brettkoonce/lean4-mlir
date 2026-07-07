#!/usr/bin/env python3
"""ROI-of-width figure for the verified MNIST MLP (784→d→d→10) size sweep.
Reads runs/mlp_grid_results.tsv, plots the d1=d2 diagonal: accuracy vs hidden
neurons (the diminishing-returns knee) and accuracy vs params (the shrink story)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = {}
with open(os.path.join(ROOT, "runs/mlp_grid_results.tsv")) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        try:
            rows[(int(r["d1"]), int(r["d2"]))] = (float(r["acc"]), int(r["floats"]))
        except ValueError:
            pass

# diagonal d1=d2 = the "neurons" axis
diag = sorted(k[0] for k in rows if k[0] == k[1])
d      = diag
acc    = [rows[(x, x)][0] for x in d]
floats = [rows[(x, x)][1] for x in d]
amax   = max(acc)

INK, ACCENT, KNEE, MUTE = "#1a1a2e", "#2563eb", "#dc2626", "#94a3b8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.9})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))
fig.suptitle("Return on width — verified MNIST MLP  784 → d → d → 10",
             fontsize=15, fontweight="bold", color=INK, y=0.99)

# ── Panel A: accuracy vs hidden neurons (the knee) ────────────────────────────
ax1.plot(d, acc, "-o", color=ACCENT, lw=2.2, ms=7, mfc="white", mec=ACCENT, mew=2, zorder=3)
ax1.set_xscale("log", base=2)
ax1.set_xticks(d); ax1.set_xticklabels([str(x) for x in d], rotation=0)
ax1.set_xlabel("hidden neurons per layer  (d)")
ax1.set_ylabel("MNIST test accuracy  (%)")
ax1.set_title("Accuracy plateaus after ~64 neurons", fontsize=12, color=INK, pad=8)
ax1.grid(True, which="major", axis="y", ls=":", color="#e2e8f0", zorder=0)
ax1.set_ylim(90.5, 98.6)

# knee band + annotations
ax1.axvspan(48, 96, color=KNEE, alpha=0.06, zorder=0)
ax1.axhline(amax, color=MUTE, ls="--", lw=1, zorder=1)
ax1.text(4096, amax + 0.06, f"ceiling {amax:.1f}%", ha="right", va="bottom",
         color=MUTE, fontsize=9)
ax1.annotate("ROI knee ≈ 64\n97.2% at 55K params\n(1/364 the size, −0.8 pt)",
             xy=(64, rows[(64,64)][0]), xytext=(150, 94.4),
             fontsize=9.5, color=KNEE, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=KNEE, lw=1.4))
ax1.annotate("32×32 = 96.2%\n26K params",
             xy=(32, rows[(32,32)][0]), xytext=(32, 92.4),
             fontsize=9, color=INK, ha="center", va="top",
             arrowprops=dict(arrowstyle="->", color=INK, lw=1))

# ── Panel B: accuracy vs params (the shrink story) ────────────────────────────
ax2.plot(floats, acc, "-o", color=ACCENT, lw=2.2, ms=7, mfc="white", mec=ACCENT, mew=2, zorder=3)
ax2.set_xscale("log")
ax2.set_xlabel("trainable parameters  (float count)")
ax2.set_ylabel("MNIST test accuracy  (%)")
ax2.set_title("760× fewer params costs only 1.8 points", fontsize=12, color=INK, pad=8)
ax2.grid(True, which="major", axis="both", ls=":", color="#e2e8f0", zorder=0)
ax2.set_ylim(90.5, 98.6)
ax2.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else (f"{v/1e3:.0f}K" if v >= 1e3 else f"{v:.0f}")))

# mark canonical + shrunk points
for x, lab, dx, dy, ha in [(512, "canonical\n512×512", 0.0, -1.6, "center"),
                           (32,  "shrunk\n32×32",     0.0, -1.6, "center")]:
    fx, fa = rows[(x, x)][1], rows[(x, x)][0]
    ax2.annotate(lab, xy=(fx, fa), xytext=(fx*dx if dx else fx, fa+dy),
                 fontsize=9, color=INK, ha=ha, va="top",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1))
ax2.axvspan(rows[(32,32)][1], rows[(4096,4096)][1], color=MUTE, alpha=0.05, zorder=0)

fig.text(0.5, 0.005,
         "Every point trained end-to-end on the proof-rendered StableHLO (Lean → IREE FFI → gfx1100), 12 epochs, mean-loss SGD.  "
         "Off-diagonal note: 8→4096 collapses to chance (9.8%) — a tiny bottleneck feeding a huge layer fails to train.",
         ha="center", fontsize=8, color=MUTE)

fig.tight_layout(rect=[0, 0.03, 1, 0.96])
out = os.path.join(ROOT, "figures/mlp_width_roi.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
