"""The Deep CFR / Leduc figure (planning/leduc_deep_cfr_demo.md §5a), one picture.

(a) One hand: the six cards, P0's jack under a public queen, and round 2's betting tree after
    check-check from P0's seat with the trained profile's probability on each of P0's edges — the
    bluff edge (raising the worst hand) marked. (b) Exploitability against nodes touched, log–log:
    tabular ES-MCCFR, Deep CFR as SD-CFR's average after every iteration (mean over seeds), the
    strategy net's final point, CFR+ at 1,000 iterations and the uniform policy as horizontal
    lines; r = 3 solid, r = 13 faded when its runs exist. (c) Raise probability per (private,
    public) at round 2's first decision after check-check: the trained profile beside CFR+, the
    worst-hand cells ringed.

Inputs, one run directory (copied from .lake/build/ by the run; see its README):
  r3_s<seed>_curve.csv   r3_s<seed>.log   r3_s1_sdcfr.txt   r3_esmccfr.csv   r3_cfrplus.txt
  r13_s<seed>_curve.csv  r13_esmccfr.csv                                      (optional)

  python scripts/demos/leduc_figure.py runs/2026-10-08-deep-cfr-leduc out.png [table=r3_s1_sdcfr.txt]
"""
import csv, glob, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

run = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "deep_cfr_leduc.png"
table_name = next((a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("table=")), "r3_s1_sdcfr.txt")
INK, GREY, NET, TAB, EQ, BLUFF = "#14201a", "#8a938d", "#5598e7", "#eb6834", "#5aa46a", "#d1342f"
R = 3
RANKS = ["J", "Q", "K"]
STATES = {"": 0, "c": 1, "r": 2, "cr": 3, "rr": 4, "crr": 5}

def info_index(round_, closing, s, a, pub):
    return s * R + a if round_ == 0 else 6 * R + (((closing * 6 + s) * R + a) * R + pub)

def load_table(path):
    """A trained table: `<name>.txt` rows `idx σ_fold σ_call σ_raise` (what the run directory
    keeps; `runs/**/*.bin` is ignored) or the f32 `[nInfo, 3]` `.bin` the trainer writes."""
    if path.endswith(".bin"):
        return np.fromfile(f"{run}/{path}", dtype="<f4").reshape(-1, 3)
    tab = np.zeros((6 * R + 30 * R * R, 3))
    with open(f"{run}/{path}") as f:
        for line in f:
            v = line.split()
            tab[int(v[0])] = [float(v[1]), float(v[2]), float(v[3])]
    return tab

def load_txt_table(path):
    tab = np.zeros((6 * R + 30 * R * R, 3))
    with open(f"{run}/{path}") as f:
        for line in f:
            v = line.split()
            tab[int(v[0])] = [float(v[6]), float(v[7]), float(v[8])]
    return tab

def rows(name):
    p = f"{run}/{name}"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
fig = plt.figure(figsize=(13.0, 4.1), constrained_layout=True)
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.3, 1.1, 1.12])

# ── (a) the hand and round 2's tree ──
net = load_table(table_name)
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 11.6); ax.set_ylim(-1.35, 10); ax.axis("off")
ax.set_title("(a) one hand: P0 holds J under a public Q — round 2 after check-check", loc="left")
cards = [("J", "♠"), ("J", "♥"), ("Q", "♠"), ("Q", "♥"), ("K", "♠"), ("K", "♥")]
for i, (rk, su) in enumerate(cards):
    x = 0.4 + i * 1.05
    mine, pub, theirs = (i == 0), (i == 3), (i == 1)
    face = "#fff6e5" if mine else ("#e8f1ff" if pub else "white")
    ax.add_patch(FancyBboxPatch((x, 8.3), 0.8, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=face, ec=INK if (mine or pub) else GREY, lw=1.4 if (mine or pub) else 0.8))
    col = "#c0392b" if su == "♥" else INK
    ax.text(x + 0.4, 8.95, f"{rk}{su}", ha="center", va="center", fontsize=10, color=col)
    if mine: ax.text(x + 0.4, 7.95, "P0", ha="center", va="center", fontsize=7.5, color=INK)
    if pub: ax.text(x + 0.4, 7.95, "public", ha="center", va="center", fontsize=7.5, color=INK)
ax.text(7.2, 8.95, "P1: one of the\nother four", ha="left", va="center", fontsize=7.5, color=GREY)
# P0's sets in this round: (closing cc, state s, private J, public Q)
def p0(s):
    return net[info_index(1, 0, STATES[s], 0, 1)]
# decision nodes: name -> (x, y, who); terminals: name -> (x, y) of the label's left edge
nodes = {"": (0.7, 4.0, 0), "c": (3.0, 5.9, 1), "r": (3.0, 2.0, 1), "cr": (5.4, 5.0, 0),
         "rr": (5.4, 0.9, 0), "crr": (7.8, 4.0, 1)}
terms = {"cc": (5.4, 7.1), "fold-cr": (7.8, 6.3), "crc": (7.8, 5.2), "fold-crr": (9.7, 4.5),
         "crrc": (9.7, 3.5), "fold-r": (5.4, 3.1), "rc": (5.4, 2.1), "fold-rr": (7.8, 1.6), "rrc": (7.8, 0.3)}
# the opponent's terminals say what P1 did; P0's carry the trained profile's probability
ACT = {"f": 0, "c": 1, "r": 2}
NAME = {"f": "fold", "c": "call", "r": "raise"}
outcome = {"cc": "P1 checks: showdown 1+1", "fold-cr": "−1", "crc": "showdown 5+5", "fold-crr": "P1 folds: +5",
           "crrc": "P1 calls: showdown 9+9", "fold-r": "P1 folds: +1", "rc": "P1 calls: showdown 5+5",
           "fold-rr": "−5", "rrc": "showdown 9+9"}
# (from, to, action, who moves); a decision-node destination gets a mid-edge label, a terminal gets its label at the tip
edges = [("", "c", "c", 0), ("", "r", "r", 0), ("c", "cc", "c", 1), ("c", "cr", "r", 1),
         ("r", "fold-r", "f", 1), ("r", "rc", "c", 1), ("r", "rr", "r", 1),
         ("cr", "fold-cr", "f", 0), ("cr", "crc", "c", 0), ("cr", "crr", "r", 0),
         ("rr", "fold-rr", "f", 0), ("rr", "rrc", "c", 0),
         ("crr", "fold-crr", "f", 1), ("crr", "crrc", "c", 1)]
for a, b, act, who in edges:
    x0, y0, _ = nodes[a]
    terminal = b in terms
    x1, y1 = terms[b] if terminal else nodes[b][:2]
    bluff = (a == "" and act == "r")
    col = BLUFF if bluff else (INK if who == 0 else GREY)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8 if bluff else (1.1 if who == 0 else 0.8),
                                shrinkA=9, shrinkB=3 if terminal else 9))
    if terminal:
        if who == 0:
            txt = f"{NAME[act]} {p0(a)[ACT[act]]:.2f}: {outcome[b]}"
        else:
            txt = outcome[b]
        ax.text(x1 + 0.1, y1, txt, ha="left", va="center", fontsize=6.8, color=INK if who == 0 else GREY,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    else:
        xm, ym = x0 + 0.45 * (x1 - x0), y0 + 0.45 * (y1 - y0)
        if who == 0:
            p = p0(a)[ACT[act]]
            txt = f"{'check' if (a == '' and act == 'c') else NAME[act]} {p:.2f}" + ("  bluff" if bluff else "")
            above = (act == "c")          # the root's check (no bet to call yet) rides above its edge, every raise below
            ax.text(xm, ym + (0.28 if above else -0.28), txt, ha="center", va="bottom" if above else "top",
                    fontsize=7.5, color=col, fontweight="bold" if bluff else "normal",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        else:
            ax.text(xm, ym - 0.24, "P1 raises", ha="center", va="top", fontsize=6.8, color=GREY)
for name, (x, y, who) in nodes.items():
    ax.add_patch(plt.Circle((x, y), 0.32, fc="white" if who == 0 else "#f0f0f0", ec=INK if who == 0 else GREY, lw=1.2))
    ax.text(x, y, f"P{who}", ha="center", va="center", fontsize=7.5, color=INK if who == 0 else GREY)
ax.text(0.3, -1.3, "P0's edges carry the trained profile's probabilities (SD-CFR, seed 1); P1's are the opponent's.\nContributions 1+1 after round 1's check-check; a round-2 raise is 4; showdown and fold payoffs to P0.",
        fontsize=6.8, color=GREY, va="bottom")

# ── (b) exploitability against nodes touched ──
ax = fig.add_subplot(gs[0, 1])
ax.set_title("(b) exploitability against nodes touched", loc="left")
def es_curve(name):
    rs = rows(name)
    if rs is None:
        return None
    by = {}
    for x in rs:
        key = round(np.log10(int(x["nodes"])), 1)
        by.setdefault(key, []).append((int(x["nodes"]), float(x["exploitability"])))
    pts = sorted((np.mean([n for n, _ in v]), np.mean([e for _, e in v])) for v in by.values())
    return np.array(pts)
def deep_curve(prefix):
    files = sorted(glob.glob(f"{run}/{prefix}_s*_curve.csv"))
    if not files:
        return None, None
    runs_ = []
    for f in files:
        with open(f) as h:
            rs = list(csv.DictReader(h))
        runs_.append(np.array([(int(x["nodes"]), float(x.get("exploit_avg", x.get("exploit_sdcfr")))) for x in rs]))
    n = min(len(r) for r in runs_)
    stack = np.stack([r[:n] for r in runs_])
    return stack[:, :, 0].mean(0), stack[:, :, 1].mean(0)
def strategy_points(prefix):
    pts = []
    for f in sorted(glob.glob(f"{run}/{prefix}_s*.log")):
        txt = open(f).read()
        m = re.search(r"strategy net: exploitability ([0-9.eE+-]+)", txt)
        n = re.search(r"trained: \d+ iterations, (\d+) nodes touched", txt)
        if m and n:
            pts.append((int(n.group(1)), float(m.group(1))))
    return pts
for prefix, alpha, suffix in (("r3", 1.0, ""), ("r13", 0.35, ", r = 13")):
    es = es_curve(f"{prefix}_esmccfr.csv")
    nodes_, ex = deep_curve(prefix)
    if es is not None:
        ax.plot(es[:, 0], es[:, 1], "-o", color=TAB, ms=3.5, alpha=alpha, label=f"tabular ES-MCCFR{suffix}")
    if nodes_ is not None:
        ax.plot(nodes_, ex, "-", color=NET, alpha=alpha, lw=1.6, label=f"Deep CFR, SD-CFR average{suffix}")
    for i, (n, e) in enumerate(strategy_points(prefix)):
        ax.plot([n], [e], "s", color=NET, alpha=alpha, ms=5, label=(f"Deep CFR, strategy net{suffix}" if i == 0 else None))
ax.axhline(2.3736, ls="--", color=GREY, lw=0.9); ax.text(1.2e3, 2.9, "uniform", color=GREY, fontsize=7.5, va="bottom")
ax.axhline(2.38e-4, ls="--", color=EQ, lw=0.9); ax.text(1.2e3, 3.0e-4, "CFR+, 1,000 iterations (exact, every node)", color=EQ, fontsize=7.5, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("nodes touched"); ax.set_ylabel("exploitability (chips / hand)")
ax.set_ylim(1e-4, 5); ax.grid(True, which="major", alpha=0.25)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.13), frameon=False)

# ── (c) raise probability at round 2's first decision after check-check ──
sub = gs[0, 2].subgridspec(1, 2, wspace=0.1)
cfr = load_txt_table("r3_cfrplus.txt")
def raise_grid(tab):
    g = np.full((R, R), np.nan)
    for a in range(R):
        for pub in range(R):
            g[a, pub] = tab[info_index(1, 0, 0, a, pub)][2]
    return g
worst = [(a, pub) for pub in range(R) for a in range(R)
         if a != pub and a == min(x for x in range(R) if x != pub)]
for j, (name, tab) in enumerate((("Deep CFR (SD-CFR, seed 1)", net), ("CFR+", cfr))):
    ax = fig.add_subplot(sub[0, j])
    g = raise_grid(tab)
    ax.imshow(g, cmap="Blues", vmin=0, vmax=1, origin="lower")
    for a in range(R):
        for pub in range(R):
            ax.text(pub, a, f"{100 * g[a, pub]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if g[a, pub] > 0.55 else INK)
    for a, pub in worst:
        ax.add_patch(Rectangle((pub - 0.47, a - 0.47), 0.94, 0.94, fill=False, ec=BLUFF, lw=1.8))
    ax.set_xticks(range(R)); ax.set_xticklabels(RANKS); ax.set_xlabel("public card")
    ax.set_yticks(range(R)); ax.set_yticklabels(RANKS if j == 0 else [])
    if j == 0: ax.set_ylabel("private card")
    ax.set_title(("(c) P(raise) opening round 2 after check-check\n" if j == 0 else "\n") + name, loc="left", fontsize=8.5)
fig.savefig(out, dpi=150)
print("wrote", out)
