#!/usr/bin/env python3
"""ROI-of-width figure for the verified MNIST CNN classifier head (conv@32 fixed,
dense head …→d→d→10), overlaid on the MLP width sweep for comparison.
Reads runs/cnn_grid_results.tsv + runs/mlp_grid_results.tsv."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(path, keyfn):
    out = {}
    with open(os.path.join(ROOT, path)) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try: out[keyfn(r)] = (float(r["acc"]), int(r["floats"]))
            except ValueError: pass
    return out

cnn = load("runs/cnn_grid_results.tsv", lambda r: int(r["d"]))
mlp_all = load("runs/mlp_grid_results.tsv", lambda r: (int(r["d1"]), int(r["d2"])))
mlp = {k[0]: v for k, v in mlp_all.items() if k[0] == k[1]}   # diagonal only

cd = sorted(cnn); c_acc = [cnn[x][0] for x in cd]; c_fl = [cnn[x][1] for x in cd]
md = sorted(mlp); m_acc = [mlp[x][0] for x in md]; m_fl = [mlp[x][1] for x in md]

CNNC, MLPC, INK, MUTE, KNEE = "#059669", "#2563eb", "#1a1a2e", "#94a3b8", "#dc2626"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.9})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))
fig.suptitle("Return on width — CNN classifier head vs MLP  (MNIST, verified path)",
             fontsize=15, fontweight="bold", color=INK, y=0.99)

# ── Panel A: accuracy vs hidden/FC neurons ────────────────────────────────────
ax1.plot(cd, c_acc, "-o", color=CNNC, lw=2.3, ms=7, mfc="white", mec=CNNC, mew=2,
         label="CNN  (conv@32 fixed, head d→d)", zorder=4)
ax1.plot(md, m_acc, "-s", color=MLPC, lw=2.0, ms=6, mfc="white", mec=MLPC, mew=1.8,
         label="MLP  (784→d→d→10)", zorder=3)
ax1.set_xscale("log", base=2)
ticks = [8,16,32,64,128,256,512,1024,2048,4096]
ax1.set_xticks(ticks); ax1.set_xticklabels([str(t) for t in ticks])
ax1.set_xlabel("hidden neurons per layer  (d)")
ax1.set_ylabel("MNIST test accuracy  (%)")
ax1.set_title("Good conv features ⇒ the head can be tiny", fontsize=12, color=INK, pad=8)
ax1.grid(True, which="major", axis="y", ls=":", color="#e2e8f0", zorder=0)
ax1.set_ylim(90.5, 99.6)
ax1.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
ax1.axvline(16, color=KNEE, ls="--", lw=1, alpha=0.7, zorder=1)
ax1.annotate("CNN knee ≈ 16\n98.3% at 110K params",
             xy=(16, cnn[16][0]), xytext=(30, 95.0),
             fontsize=9.3, color=CNNC, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=CNNC, lw=1.3))
ax1.annotate("MLP knee ≈ 64",
             xy=(64, mlp[64][0]), xytext=(120, 93.3),
             fontsize=9.3, color=MLPC, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=MLPC, lw=1.3))

# ── Panel B: accuracy vs params ───────────────────────────────────────────────
ax2.plot(c_fl, c_acc, "-o", color=CNNC, lw=2.3, ms=7, mfc="white", mec=CNNC, mew=2,
         label="CNN", zorder=4)
ax2.plot(m_fl, m_acc, "-s", color=MLPC, lw=2.0, ms=6, mfc="white", mec=MLPC, mew=1.8,
         label="MLP", zorder=3)
ax2.set_xscale("log")
ax2.set_xlabel("trainable parameters  (float count)")
ax2.set_ylabel("MNIST test accuracy  (%)")
ax2.set_title("CNN reaches higher accuracy at fewer params", fontsize=12, color=INK, pad=8)
ax2.grid(True, which="major", axis="both", ls=":", color="#e2e8f0", zorder=0)
ax2.set_ylim(90.5, 99.6)
ax2.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
ax2.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else (f"{v/1e3:.0f}K" if v >= 1e3 else f"{v:.0f}")))
ax2.annotate("CNN fc16\n98.3%, 110K",
             xy=(cnn[16][1], cnn[16][0]), xytext=(cnn[16][1], 96.2),
             fontsize=9, color=CNNC, ha="center", va="top",
             arrowprops=dict(arrowstyle="->", color=CNNC, lw=1))

fig.text(0.5, 0.005,
         "Every point trained end-to-end on the proof-rendered StableHLO (Lean → IREE FFI → gfx1100).  "
         "CNN: conv stack held at 32 channels, only the dense head swept, 10 epochs.  MLP diagonal (d₁=d₂), 12 epochs.",
         ha="center", fontsize=8, color=MUTE)

fig.tight_layout(rect=[0, 0.03, 1, 0.96])
out = os.path.join(ROOT, "figures/cnn_vs_mlp_width_roi.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
