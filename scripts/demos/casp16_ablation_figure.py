#!/usr/bin/env python3
"""The distogram demo's ablation chart in the shape of AlphaFold 2's Fig. 4a: one row per arm, the
mean paired difference to the baseline (ESM-2 650M · 64 ch · crop 64 · 30 ep) over the evaluation
units, with a 95 % bootstrap interval over units, in three columns — top-L/5 long-range contact
precision over the 84 units, and the fold's Cβ-lDDT and TM-score over the 78 units folded by every
arm (≤ 512 residues). Arms whose fold used the orientation restraints say so. Rows whose
prediction directory is missing are skipped (an arm still training). The default is the book's
twelve rows, the chain of levers from the baseline to the book run and the swaps beneath it;
`--full` draws every arm of the plan's §10 table.
  .venv-casp/bin/python scripts/demos/casp16_ablation_figure.py [out.png] [--full]"""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path(".lake/build")
BLUE, INK, MUTED, PAPER = "#2a78d6", "#1f1e1b", "#6b6963", "#fbfbf8"
BASE = "r16x64_esm650_train_full_e30-esm650"
# label, prediction dir (under .lake/build/distogram_<…>_targets), fold file
ROWS = [
    ("the book run: the row below at 100 epochs",      "r16x128_esm3b_orient_pair1_train_full_e100-esm3b-crop96-pair1-orient", "fold_scores_orient.csv"),
    ("3B + plane + 128 ch + crop 96 + heads, ω/φ fold", "r16x128_esm3b_orient_pair1_train_full_e30-esm3b-crop96-pair1-orient", "fold_scores_orient.csv"),
    ("3B + plane + 128 ch + crop 96, no heads",         "r16x128_esm3b_pair1_train_full_e30-esm3b-crop96-pair1", "fold_scores.csv"),
    ("3B + 128 ch + crop 96, no plane",                 "r16x128_esm3b_train_full_e30-esm3b-crop96", "fold_scores.csv"),
    ("3B + the 3B contact head's plane, 64 ch",         "r16x64_esm3b_pair1_train_full_e30-esm3b-pair1", "fold_scores.csv"),
    ("+ 128 ch + crop 96",                              "r16x128_esm650_train_full_e30-esm650-crop96", "fold_scores.csv"),
    ("+ the 650M contact head's plane",                 "r16x64_esm650_pair1_train_full_e30-esm650-pair1", "fold_scores.csv"),
    ("Baseline: ESM-2 650M · 64 ch · crop 64 · 30 ep", BASE, "fold_scores.csv"),
    ("seed 2",                                         "r16x64_esm650_train_full_e30-esm650-s2", "fold_scores.csv"),
    ("ESM-2 3B in place of 650M",                      "r16x64_esm3b_train_full_e30-esm3b", "fold_scores.csv"),
    ("ESM-2 35M in place of 650M",                     "r16x64_train_full_e30", "fold_scores.csv"),
    ("one-hot residues, no language model",            "r16x64_onehot_train_full_e30-onehot", "fold_scores.csv"),
]
ROWS_FULL = [
    ("the same at 100 epochs: the book run, ω/φ fold", "r16x128_esm3b_orient_pair1_train_full_e100-esm3b-crop96-pair1-orient", "fold_scores_orient.csv"),
    ("3B + plane + 128 ch + crop 96 + heads, ω/φ fold", "r16x128_esm3b_orient_pair1_train_full_e30-esm3b-crop96-pair1-orient", "fold_scores_orient.csv"),
    ("3B + plane + 128 ch + crop 96", "r16x128_esm3b_pair1_train_full_e30-esm3b-crop96-pair1", "fold_scores.csv"),
    ("+ plane + 128 ch + crop 96 (650M)",               "r16x128_esm650_pair1_train_full_e30-esm650-crop96-pair1", "fold_scores.csv"),
    ("3B + 128 ch + crop 96",                           "r16x128_esm3b_train_full_e30-esm3b-crop96", "fold_scores.csv"),
    ("3B + plane + orientation heads, ω/φ fold",        "r16x64_esm3b_orient_pair1_train_full_e30-esm3b-pair1-orient", "fold_scores_orient.csv"),
    ("3B + the 3B contact head's plane",                "r16x64_esm3b_pair1_train_full_e30-esm3b-pair1", "fold_scores.csv"),
    ("3B + plane, seed 2",                              "r16x64_esm3b_pair1_train_full_e30-esm3b-pair1-s2", "fold_scores.csv"),
    ("+ plane + orientation heads, ω/φ fold",           "r16x64_esm650_orient_pair1_train_full_e30-esm650-pair1-orient", "fold_scores_orient.csv"),
    ("+ 128 ch + crop 96 (both levers)",               "r16x128_esm650_train_full_e30-esm650-crop96", "fold_scores.csv"),
    ("+ crop 96 + orientation heads, ω/φ fold",        "r16x64_esm650_orient_train_full_e30-esm650-crop96-orient", "fold_scores_orient.csv"),
    ("+ the 650M contact head's logit plane (pair=1)", "r16x64_esm650_pair1_train_full_e30-esm650-pair1", "fold_scores.csv"),
    ("+ 128 channels",                                 "r16x128_esm650_train_full_e30-esm650", "fold_scores.csv"),
    ("+ crop 96",                                      "r16x64_esm650_train_full_e30-esm650-crop96", "fold_scores.csv"),
    ("+ 100 epochs",                                   "r16x64_esm650_train_full_e100-esm650", "fold_scores.csv"),
    ("+ orientation heads, ω/φ fold",                  "r16x64_esm650_orient_train_full_e30-esm650-orient", "fold_scores_orient.csv"),
    ("ensemble of five 650M arms",                     "ens-esm650x5", "fold_scores.csv"),
    ("Baseline: ESM-2 650M · 64 ch · crop 64 · 30 ep", BASE, "fold_scores.csv"),
    ("seed 2",                                         "r16x64_esm650_train_full_e30-esm650-s2", "fold_scores.csv"),
    ("purged training list (no template chains)",     "r16x64_esm650_train_e30-esm650-purged", "fold_scores.csv"),
    ("ESM-2 3B in place of 650M",                      "r16x64_esm3b_train_full_e30-esm3b", "fold_scores.csv"),
    ("ESM-2 150M in place of 650M",                    "r16x64_esm150_train_full_e30-esm150", "fold_scores.csv"),
    ("ESM-2 35M",                                      "r16x64_train_full_e30", "fold_scores.csv"),
    ("one-hot residues, no language model",            "r16x64_onehot_train_full_e30-onehot", "fold_scores.csv"),
]
PALETTE = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b", "#b279a2", "#1f1e1b",
           "#ff9da6", "#9d755d", "#bab0ac", "#8c6d31", "#637939"]


def load(dirn, foldf):
    d = B / f"distogram_{dirn}_targets"
    if not (d / "table.csv").exists():
        return None
    prec = {r["eu"]: float(r["ours"]) for r in csv.DictReader(open(d / "table.csv"))}
    fold = {}
    if (d / foldf).exists():
        fold = {r["eu"]: (float(r["cb_lddt"]), float(r["tm"])) for r in csv.DictReader(open(d / foldf))}
    return prec, fold


def paired(a, b, rng, n=2000):
    """mean of a − b over the shared keys with a 95 % bootstrap interval over the units."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diff = np.array([a[k] - b[k] for k in keys])
    bs = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(n)])
    return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(keys)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0] if args else "demos/figures/casp16_ablation.png")
    ROWS = ROWS_FULL if "--full" in sys.argv else ROWS
    rng = np.random.RandomState(0)
    base_prec, base_fold = load(BASE, "fold_scores.csv")
    base_l = {k: v[0] for k, v in base_fold.items()}; base_t = {k: v[1] for k, v in base_fold.items()}
    rows = []
    for label, dirn, foldf in ROWS:
        got = load(dirn, foldf)
        if got is None:
            print(f"skipped (not on disk yet): {label}"); continue
        prec, fold = got
        l = {k: v[0] for k, v in fold.items()}; t = {k: v[1] for k, v in fold.items()}
        rows.append((label, paired(prec, base_prec, rng), paired(l, base_l, rng), paired(t, base_t, rng)))
    # one column per metric, each with a broken x-axis: the language-model rows sit at −0.1 to −0.6
    # and the levers at the top LM at +0.01 to +0.03; one linear axis cannot show both
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14.0, 0.42 * len(rows) + 2.1)); fig.patch.set_facecolor(PAPER)
    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.08, left=0.26, right=0.985, top=0.9, bottom=0.1)
    titles = ["top-L/5 long-range contact precision, 84 units",
              "fold Cβ-lDDT, 78 units",
              "fold TM-score, 78 units"]
    base_mean = [np.mean(list(base_prec.values())), np.mean(list(base_l.values())), np.mean(list(base_t.values()))]
    first_axes = []
    for c in range(3):
        stats = [r[c + 1] for r in rows]
        big = [st for st in stats if st is not None and st[2] < -0.04]
        small = [st for st in stats if st is not None and st[2] >= -0.04]
        inner = gridspec.GridSpecFromSubplotSpec(1, 2 if big else 1, subplot_spec=outer[c], wspace=0.05,
                                                  width_ratios=[0.42, 1] if big else [1])
        axs = [fig.add_subplot(inner[k]) for k in range(2 if big else 1)]
        if big:
            axs[0].set_xlim(min(st[1] for st in big) - 0.03, -0.04)
        axs[-1].set_xlim(min([st[1] for st in small] + [0]) - 0.006, max([st[2] for st in small] + [0]) + 0.009)
        for ax in axs:
            ax.set_facecolor(PAPER); ax.set_ylim(-0.6, len(rows) - 0.4)
            ax.axvline(0, color="#c9ccc6", lw=1, zorder=1)
            ax.tick_params(colors=MUTED, labelsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ax.spines.values():
                s_.set_color("#d9dcd6")
            lo_, hi_ = ax.get_xlim()
            for r, (label, *st_) in enumerate(rows):
                st = st_[c]
                if st is None:
                    continue
                m, lo, hi, n = st
                y = len(rows) - 1 - r
                colr = INK if label.startswith("Baseline") else PALETTE[r % len(PALETTE)]
                ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="o", ms=5.5, color=colr, ecolor=colr,
                            elinewidth=1.3, capsize=2.5, zorder=3, clip_on=True)
                if lo_ <= m <= hi_:
                    txt = f"{base_mean[c]:.3f}" if label.startswith("Baseline") else f"{m:+.3f}"
                    ax.text(m, y + 0.27, txt, ha="center", va="bottom", fontsize=7, color=colr)
        if big:   # the break marks
            axs[0].spines["right"].set_visible(False); axs[1].spines["left"].set_visible(False)
            axs[1].tick_params(axis="y", length=0)
            kw = dict(transform=axs[0].transAxes, color=MUTED, clip_on=False, lw=0.9)
            axs[0].plot([1 - 0.012, 1 + 0.012], [-0.012, 0.012], **kw)
            kw = dict(transform=axs[1].transAxes, color=MUTED, clip_on=False, lw=0.9)
            axs[1].plot([-0.006, 0.006], [-0.012, 0.012], **kw)
        for ax in axs[1:]:
            ax.set_yticks([])
        axs[0].set_yticks(range(len(rows)))
        axs[0].set_yticklabels([r[0] for r in rows][::-1] if c == 0 else [""] * len(rows), fontsize=8.5, color=INK)
        axs[0].set_title(titles[c], fontsize=9.5, color=INK, loc="left", x=0.0)
        first_axes.append(axs)
    fig.text(0.62, 0.025, "difference to the baseline, mean over units; bars: 95 % bootstrap interval over units",
             ha="center", color=MUTED, fontsize=8.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor=PAPER)
    print(f"-> {out}")
    for label, p, l, t in rows:
        f = lambda s: "—" if s is None else f"{s[0]:+.3f} [{s[1]:+.3f}, {s[2]:+.3f}] n={s[3]}"
        print(f"{label:48s} prec {f(p)}   lDDT {f(l)}   TM {f(t)}")
