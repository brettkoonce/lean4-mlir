"""The Pong DQN figure (planning/pong_dqn_demo.md §5), one picture.

Left: the four stacked frames one pixel-DQN input holds, as the net saw them
(the PGM `lake exe pong-dqn mode=pixels` writes at every evaluation). Right:
mean points per game at ε = 0.05 against agent steps, for the state-vector DQN
(three seeds), the four-frame pixel DQN (three seeds) and the one-frame ablation, with the
random and scripted-tracker baselines as lines.

Inputs, all in one run directory (copied from .lake/build/ by the run):
  pong_dqn_state_state_s{1,2,3}_curve.csv
  pong_dqn_pixels_k4_px4_o15{,_s2,_s3}_curve.csv
  pong_dqn_pixels_k1_px1_o15_curve.csv
  pong_dqn_pixels_k4_px4_o15_seen_<round>.pgm

  python scripts/demos/pong_figure.py runs/2026-09-25-pong-dqn out.png [seen_round=20]
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

run = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "pong_dqn.png"
seen_round = sys.argv[3] if len(sys.argv) > 3 else "20"

RANDOM, TRACKER = -17.88, 11.18     # lake exe pong-env 100, the default opponent
STATE, PIX4, PIX1, INK, GREY = "#5598e7", "#eb6834", "#b59a2e", "#14201a", "#8a938d"

def curve(name):
    with open(f"{run}/{name}") as f:
        r = list(csv.DictReader(f))
    return [int(x["agent_steps"]) for x in r], [float(x["eval_mean"]) for x in r]

def pgm(path):
    with open(path, "rb") as f:
        data = f.read()
    parts = data.split(b"\n", 3)
    w, h = map(int, parts[1].split())
    return w, h, parts[3]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
fig = plt.figure(figsize=(11.0, 3.6), constrained_layout=True)
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1])

# ── left: the input, four frames oldest to newest ───────────────────────
w, h, px = pgm(f"{run}/pong_dqn_pixels_k4_px4_o15_seen_{seen_round}.pgm")
img = np.frombuffer(px, dtype=np.uint8).reshape(h, w)
k = w // 84
sub = gridspec.GridSpecFromSubplotSpec(1, k, subplot_spec=gs[0, 0], wspace=0.06)
for c in range(k):
    ax = fig.add_subplot(sub[0, c])
    ax.imshow(img[:, c * 84:(c + 1) * 84], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.set_title(f"t − {k - 1 - c}" if c < k - 1 else "t", color=INK)
    if c == 0:
        ax.set_ylabel("one input: 4 × 84 × 84", color=INK)

# ── right: learning curves ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.axhline(TRACKER, color=INK, lw=1.0, ls="--", zorder=1)
ax.axhline(RANDOM, color=GREY, lw=1.0, ls=":", zorder=1)
ax.axhline(0, color=GREY, lw=0.6, zorder=0)
for i, s in enumerate((1, 2, 3)):
    x, y = curve(f"pong_dqn_state_state_s{s}_curve.csv")
    ax.plot(x, y, color=STATE, lw=1.1, alpha=0.85, label="state (6 numbers), 3 seeds" if i == 0 else None, zorder=2)
for i, s in enumerate(("", "_s2", "_s3")):
    x, y = curve(f"pong_dqn_pixels_k4_px4_o15{s}_curve.csv")
    ax.plot(x, y, color=PIX4, lw=1.3, alpha=0.9, label="pixels, 4 frames, 3 seeds" if i == 0 else None, zorder=3)
x, y = curve("pong_dqn_pixels_k1_px1_o15_curve.csv")
ax.plot(x, y, color=PIX1, lw=1.4, label="pixels, 1 frame", zorder=3)
ax.text(505000, TRACKER + 0.6, "scripted tracker", color=INK, ha="right", va="bottom", fontsize=8)
ax.text(505000, RANDOM + 0.6, "random", color=GREY, ha="right", va="bottom", fontsize=8)
ax.set_xlim(0, 510000)
ax.set_ylim(-21, 21)
ax.set_xlabel("agent steps (4 frames each)")
ax.set_ylabel("points per game, own − opponent's")
ax.set_xticks([0, 100000, 200000, 300000, 400000, 500000])
ax.set_xticklabels(["0", "100k", "200k", "300k", "400k", "500k"])
ax.legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.08))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.savefig(out, dpi=200)
print("wrote", out)
