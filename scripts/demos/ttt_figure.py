"""The AlphaZero tic-tac-toe figure (planning/alphazero_ttt_demo.md §6), one picture.

Left: the board at the probe position — X in the centre, O to move — with the trained
net's move probabilities over the empty cells and the solved game's optimal moves ringed
(the rings come from `lake exe ttt-env show <index>`, the instrument itself). Middle:
agreement of the net alone with the solved game over the reachable decision positions,
against iteration, one line per board, with the draw rate of net + MCTS against the
perfect player dashed. Right: the value head against the exact value of every reachable
3×3 decision position.

Inputs, one run directory (copied from .lake/build/ by the run):
  n3_curve.csv  n3_policy.csv  n3_sweep.csv        lake exe alphazero-ttt n=3
  n4_curve.csv  [n4_policy.csv]                    lake exe alphazero-ttt n=4 (optional)

  python scripts/demos/ttt_figure.py runs/2026-09-29-alphazero-ttt out.png [probe_index=81]
"""
import csv, os, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle
import numpy as np

run = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "alphazero_ttt.png"
probe = int(sys.argv[3]) if len(sys.argv) > 3 else 81
N3, N4, INK, GREY, HEAT = "#5598e7", "#eb6834", "#14201a", "#8a938d", "Blues"

def rows(name):
    p = f"{run}/{name}"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))

def optimal_set(index, n):
    """The instrument's optimal moves at a position, parsed from `ttt-env show`."""
    exe = ".lake/build/bin/ttt-env"
    txt = subprocess.run([exe, "show", str(index), f"n={n}"], capture_output=True, text=True).stdout
    opt, board = [], []
    for line in txt.splitlines():
        if line.startswith("  cell"):
            if line.rstrip().endswith("optimal"):
                opt.append(int(line.split()[1]))
        elif line and line[0] in "XO.":
            board.append(line.split())
    return opt, board

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
fig = plt.figure(figsize=(12.0, 3.9), constrained_layout=True)
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.35, 1.0])

# ── left: the board, the net's policy, the theorem's rings ──────────────
ax = fig.add_subplot(gs[0, 0])
pol = {int(r["index"]): r for r in rows("n3_policy.csv")}
r = pol[probe]
n = 3
opt, board = optimal_set(probe, n)
probs = np.array([float(r[f"p{c}"]) for c in range(n * n)])
cmap = plt.get_cmap(HEAT)
for c in range(n * n):
    y, x = divmod(c, n)
    cell = board[y][x]
    if cell == ".":
        ax.add_patch(Rectangle((x, n - 1 - y), 1, 1, facecolor=cmap(0.15 + 0.8 * probs[c] / max(probs.max(), 1e-9)),
                               edgecolor="white", lw=2))
        ax.text(x + 0.5, n - 1 - y + 0.5, f"{100 * probs[c]:.0f}%", ha="center", va="center", fontsize=9,
                color=INK if probs[c] < 0.5 * probs.max() else "white")
    else:
        ax.add_patch(Rectangle((x, n - 1 - y), 1, 1, facecolor="#f2f2f2", edgecolor="white", lw=2))
        ax.text(x + 0.5, n - 1 - y + 0.5, cell, ha="center", va="center", fontsize=22, color=INK, weight="bold")
    if c in opt:
        ax.add_patch(Circle((x + 0.5, n - 1 - y + 0.5), 0.42, fill=False, edgecolor=N4, lw=2.2, zorder=5))
ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal"); ax.axis("off")
mover = r["mover"]
ax.set_title(f"{mover} to move — the net's policy, its value {float(r['value']):+.2f}\n"
             f"rings: the optimal moves (exact value {int(r['exact']):+d})")

# ── middle: agreement with the theorem against iteration ────────────────
ax = fig.add_subplot(gs[0, 1])
firsts = {}
for name, color, label in (("n3_curve.csv", N3, "3×3"), ("n4_curve.csv", N4, "4×4")):
    cur = rows(name)
    if cur is None:
        continue
    it = [int(x["iter"]) for x in cur]
    agree = [100 * float(x["agree"]) for x in cur]
    def draw_rate(x):
        g = sum(int(x[k]) for k in ("mcts_x_w", "mcts_x_d", "mcts_x_l", "mcts_o_w", "mcts_o_d", "mcts_o_l"))
        return 100 * (int(x["mcts_x_d"]) + int(x["mcts_o_d"])) / g
    ax.plot(it, agree, color=color, lw=1.6, label=f"{label}: net alone agrees with the theorem")
    dr = [draw_rate(x) for x in cur]
    ax.plot(it, dr, color=color, lw=1.2, ls="--", label=f"{label}: net + MCTS draws vs perfect")
    # the first iteration from which every match against perfect is a draw
    first = next((i for i, d in zip(it, dr) if d >= 100 and all(x >= 100 for x in dr[it.index(i):])), None)
    if first is not None:
        firsts[label] = (first, color)
if firsts:
    if len(set(f for f, _ in firsts.values())) == 1:
        f0 = next(iter(firsts.values()))[0]
        ax.annotate(f"both boards unbeaten with search from iteration {f0}", (f0, 100), xytext=(f0 + 1.5, 93.5),
                    fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    else:
        for j, (label, (f0, color)) in enumerate(firsts.items()):
            ax.annotate(f"{label} unbeaten with search from {f0}", (f0, 100), xytext=(f0 + 1.5, 93.5 - 4 * j),
                        fontsize=7.5, color=color, arrowprops=dict(arrowstyle="-", color=color, lw=0.7))
ax.axhline(100, color=INK, lw=0.8, ls=":")
ax.set_xlabel("iteration"); ax.set_ylabel("%"); ax.set_ylim(50, 101.5)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.legend(loc="lower right", frameon=False)
ax.set_title("net alone vs the theorem, and net + MCTS vs the perfect player")
ax.spines[["top", "right"]].set_visible(False)

# ── right: the value head against the exact value, 3×3 ───────────────────
ax = fig.add_subplot(gs[0, 2])
sw = rows("n3_sweep.csv")
rng = np.random.default_rng(0)
for z, label in ((-1, "loss"), (0, "draw"), (1, "win")):
    vals = np.array([float(x["value"]) for x in sw if int(x["exact"]) == z])
    xs = z + rng.uniform(-0.28, 0.28, len(vals))
    ax.scatter(xs, vals, s=4, color=N3, alpha=0.25, lw=0, zorder=2)
    ax.plot([z - 0.35, z + 0.35], [np.median(vals)] * 2, color=INK, lw=1.4, zorder=3)
    ax.text(z, 1.04, f"{label}, n = {len(vals)}", ha="center", va="bottom", fontsize=8, color=GREY)
ax.plot([-1, 1], [-1, 1], color=GREY, lw=0.8, ls=":", zorder=1)
ax.set_xticks([-1, 0, 1]); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.1, 1.25)
ax.set_xlabel("exact value for the mover (the theorem)"); ax.set_ylabel("the net's value, tanh v")
ax.set_title("value head vs the theorem, all 4,520 3×3 positions")
ax.spines[["top", "right"]].set_visible(False)

fig.savefig(out, dpi=150)
print(f"wrote {out}")
