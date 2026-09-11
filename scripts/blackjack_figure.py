"""The blackjack DQN figure (planning/blackjack_dqn_demo.md §5), one picture.

Left: the exact value of the greedy policy while tabular Q and the DQN train,
against hands played, with the DP optimum and the published casino table as
lines. Right: the exact hit/stick chart, hard and soft hands, with the cells
where the trained DQN and tabular Q disagree with it marked.

Inputs, all written by the two Lean exes into one run directory:
  states.csv       lake exe blackjack-env dump      (exact hit/stick values, opt/tabq/published)
  tabq_curve.csv   lake exe blackjack-env curve     (hands, exact, agreement)
  dqn_curve.csv    lake exe blackjack-dqn           (updates, hands, exact, agreement)
  dqn_policy.csv   lake exe blackjack-dqn           (usable, sum, dealer, dqn)

  python scripts/blackjack_figure.py runs/2026-09-11-blackjack-dqn out.png        # curve + charts
  python scripts/blackjack_figure.py runs/2026-09-11-blackjack-dqn out.png chart  # the two charts alone (the book's)
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

run = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "blackjack_dqn.png"
chart_only = len(sys.argv) > 3 and sys.argv[3] == "chart"

def rows(name):
    with open(f"{run}/{name}") as f:
        return list(csv.DictReader(f))

states = rows("states.csv")
tabq = rows("tabq_curve.csv")
dqn = rows("dqn_curve.csv")
dqnpol = {(int(r["usable"]), int(r["sum"]), int(r["dealer"])): r["dqn"] for r in rows("dqn_policy.csv")}
exact = {(int(r["usable"]), int(r["sum"]), int(r["dealer"])): r for r in states}

OPT, TABLE, RANDOM, MARKOV = -0.0431, -0.0967, -0.3942, -0.2403
HIT, STICK, INK, GREY = "#eb6834", "#5598e7", "#14201a", "#8a938d"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
if chart_only:
    fig = plt.figure(figsize=(7.0, 3.75), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig)
else:
    fig = plt.figure(figsize=(12.6, 4.3), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.9, 1, 1])

# ── left: exact value of the greedy policy vs hands ─────────────────────
ax = None if chart_only else fig.add_subplot(gs[0, 0])
if ax is not None:
  ax.axhline(OPT, color=INK, lw=1.1, ls="--", zorder=1)
  tx = [int(r["hands"]) for r in tabq]; ty = [float(r["exact"]) for r in tabq]
  dx = [int(r["hands"]) for r in dqn]; dy = [float(r["exact"]) for r in dqn]
  ax.axhline(TABLE, color=GREY, lw=1.0, ls=":", zorder=1)
  ax.plot(tx, ty, color=STICK, lw=1.4, label="tabular Q", zorder=3)
  ax.plot(dx, dy, color=HIT, lw=1.4, label="DQN", zorder=3)
  ax.set_xscale("log")
  ax.set_xlim(1e3, 1.05e6)
  ax.set_ylim(-0.125, -0.030)
  ax.set_xlabel("hands played")
  ax.set_ylabel("exact mean reward per hand of the greedy policy")
  ax.text(1.02e6, OPT + 0.002, f"exact optimum  {OPT:+.4f}", ha="right", va="bottom", fontsize=8, color=INK)
  ax.text(1.02e6, TABLE + 0.002, f"published casino table  {TABLE:+.4f}", ha="right", va="bottom", fontsize=8, color=GREY)
  ax.text(1.05e3, -0.1225, f"off the axis: Markov threshold {MARKOV:+.3f}, random {RANDOM:+.3f}", fontsize=7.5, color=GREY, va="bottom")
  ax.grid(True, which="major", color="0.88", lw=0.6)
  ax.grid(True, which="minor", axis="x", color="0.94", lw=0.4)
  ax.set_title("the ceiling is a theorem: value iteration over 200 states", loc="left")
  ax.legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.36))
  for s in ("top", "right"): ax.spines[s].set_visible(False)

# ── right: the exact chart with the learners' disagreements ─────────────
DEALER = ["A"] + [str(i) for i in range(2, 11)]
def chart(ax, usable, title):
    for i, s in enumerate(range(21, 11, -1)):
        for j, d in enumerate(range(1, 11)):
            r = exact[(usable, s, d)]
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=HIT if r["opt"] == "H" else STICK, edgecolor="white", lw=1.2))
            ax.text(j + 0.5, i + 0.5, r["opt"], ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
            if dqnpol[(usable, s, d)] != r["opt"]:
                ax.add_patch(Rectangle((j + 0.12, i + 0.12), 0.76, 0.76, fill=False, edgecolor=INK, lw=1.6))
            if r["tabq"] != r["opt"]:
                ax.plot(j + 0.8, i + 0.2, marker="o", ms=3.2, color=INK, mec="white", mew=0.6)
    ax.set_xlim(0, 10); ax.set_ylim(10, 0)   # 21 at the top, as the Lean chart prints it
    ax.set_xticks([k + 0.5 for k in range(10)]); ax.set_xticklabels(DEALER)
    ax.set_yticks([k + 0.5 for k in range(10)]); ax.set_yticklabels([str(s) for s in range(21, 11, -1)])
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("dealer shows"); ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    for s in ax.spines.values(): s.set_visible(False)

ch, cs = (0, 1) if chart_only else (1, 2)
axh = fig.add_subplot(gs[0, ch]); chart(axh, 0, "hard hands"); axh.set_ylabel("player total")
axs = fig.add_subplot(gs[0, cs]); chart(axs, 1, "soft hands (usable ace)")

n_d = sum(1 for k, r in exact.items() if k[1] >= 12 and dqnpol[k] != r["opt"])
n_q = sum(1 for k, r in exact.items() if k[1] >= 12 and r["tabq"] != r["opt"])
handles = [Line2D([], [], marker="s", ls="", ms=8, color=HIT, label="exact: hit"),
           Line2D([], [], marker="s", ls="", ms=8, color=STICK, label="exact: stick"),
           Line2D([], [], marker="s", ls="", ms=8, mfc="none", mec=INK, mew=1.6, label=f"DQN differs ({n_d} of 200)"),
           Line2D([], [], marker="o", ls="", ms=4, color=INK, label=f"tabular Q differs ({n_q} of 200)")]
if chart_only:
    # anchored below the canvas; savefig's tight bbox grows the image to include it
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, columnspacing=1.4, handletextpad=0.5, bbox_to_anchor=(0.5, -0.005))
else:
    axs.legend(handles=handles, loc="upper left", bbox_to_anchor=(-1.28, -0.13), ncol=4, frameon=False, columnspacing=1.2, handletextpad=0.4)

fig.savefig(out, dpi=190, facecolor="white", bbox_inches="tight", pad_inches=0.06)
print("wrote", out, "| DQN differs", n_d, "| tabular Q differs", n_q)
