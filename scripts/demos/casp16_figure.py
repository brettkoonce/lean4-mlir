#!/usr/bin/env python3
"""The CASP16 distogram demo's figure (planning/casp16_distogram_demo.md §6): three panels.
  (a) the object — one evaluation unit's experimental contact map (Cβ–Cβ < 8 Å), residues on
      both axes, so a reader who has never seen a distogram sees what is being predicted
  (b) the check — the net's expected Cβ–Cβ distance for every pair (upper triangle) against
      the experimental distances (lower), one sequential ramp, with the top-L/5 long-range
      contact precision of that map
  (c) the result — every CASP16 group's model 1 on the featured units, scored at pseudo-Cβ
      exactly as our fold is (Cβ-lDDT), three named groups marked, and our fold's dot
  .venv-casp/bin/python scripts/demos/casp16_figure.py <pred_dir> [out.png] [--eus A B C] [--panel-a EU]
Blue is the model; the field is grey; the three named groups are ink with distinct markers."""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from casp16_labels import EDGES, FAR, UNOBS
from casp16_targets import top_l5_precision
from casp16_score import score

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
BLUE, INK, MUTED, PAPER = "#2a78d6", "#1f1e1b", "#6b6963", "#fbfbf8"
NAMED = {"304": ("AF3-server", "^"), "051": ("MULTICOM", "s"), "145": ("ColabFold", "D")}
CONTACT_MAX = int(np.searchsorted(EDGES, 8.0, side="right") - 2)


def true_dist(t):
    cb = t["cb"]; d = np.sqrt(((cb[:, None, :] - cb[None, :, :]) ** 2).sum(-1))
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_dir")
    ap.add_argument("out", nargs="?", default="demos/figures/casp16_distogram.png")
    ap.add_argument("--eus", nargs="+", default=["T1235-D1", "T1267s1-D1", "T1226-D1"])
    ap.add_argument("--panel-a", default="T1235-D1")
    ap.add_argument("--label", default="ours")
    ap.add_argument("--metric", choices=["lddt", "tm"], default="lddt",
                    help="panel (c)'s axis: Cβ-lDDT (CASP's headline, mirror-blind) or TM-score (sees chirality; the fair column against the official table)")
    ap.add_argument("--all-units", action="store_true",
                    help="add a bottom row to (c) with every folded unit: ours (blue) beside each unit's field median from the official score table (grey)")
    a = ap.parse_args()
    d = Path(a.pred_dir)
    eus = {r["eu"]: r for r in csv.DictReader(open(ROOT / "eu_list.csv"))}

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.3 if a.all_units else 4.6), gridspec_kw=dict(width_ratios=[1, 1.08, 1.55]))
    for ax in axes:
        ax.set_facecolor(PAPER)
    fig.patch.set_facecolor(PAPER)

    # ── (a) the object: the experimental contact map ──
    eu = a.panel_a
    t = np.load(ROOT / "targets" / f"{eu}.npz"); L = len(t["obs"])
    dist = true_dist(t)
    contacts = np.where(np.isnan(dist), np.nan, (dist < 8.0).astype(float))
    ax = axes[0]
    ax.imshow(contacts, cmap=LinearSegmentedColormap.from_list("ink", [PAPER, INK]), vmin=0, vmax=1,
              interpolation="nearest", origin="upper")
    ax.set_title(f"(a) {eu}: which residues touch", loc="left", fontsize=11, color=INK)
    ax.set_xlabel(f"residue j\n{L} residues, {eus[eu]['difficulty']} target; black = Cβ–Cβ under 8 Å\nin the experimental structure",
                  color=MUTED, fontsize=8.5)
    ax.set_ylabel("residue i", color=MUTED)

    # ── (b) the check: expected distance predicted (upper) vs true (lower) ──
    p = np.load(d / f"{eu}.pred.npz")
    edist, pcontact = p["edist"].astype(float), p["pcontact"].astype(float)
    both = np.where(np.triu(np.ones((L, L), bool), 1), edist, np.where(np.isnan(dist), np.nan, np.minimum(dist, 22.0)))
    np.fill_diagonal(both, np.nan)
    ax = axes[1]
    ramp = LinearSegmentedColormap.from_list("blue", ["#0b2f5e", BLUE, "#d9e6f7", PAPER])   # near = dark, far = paper
    im = ax.imshow(both, cmap=ramp, vmin=2, vmax=22, interpolation="nearest", origin="upper")
    ax.plot([0, L - 1], [0, L - 1], color=MUTED, lw=0.6)
    prec, n = top_l5_precision(pcontact, t["cls"], t["obs"])
    ax.set_title("(b) predicted (above) vs measured (below)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel(f"residue j\n{a.label}: expected distance of the 64-bin prediction;\ntop-L/5 long-range contact precision {prec:.2f} ({n} pairs)",
                  color=MUTED, fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Cβ–Cβ, Å", color=MUTED, fontsize=8)
    cb.ax.tick_params(labelsize=8, colors=MUTED)

    # ── (c) the result: the CASP16 field at pseudo-Cβ with our dot ──
    ax = axes[2]
    rng = np.random.RandomState(0)
    col = "ca_lddt" if a.metric == "lddt" else "tm"
    nrows = len(a.eus) + (1 if a.all_units else 0)
    ours = {}
    for k, eu in enumerate(a.eus):
        fld = list(csv.DictReader(open(ROOT / "work" / f"field_{eu}_m1_cb.csv")))
        vals = np.array([float(r[col]) for r in fld])
        y = nrows - 1 - k
        ax.scatter(vals, y + rng.normal(0, 0.07, len(vals)), s=14, color=MUTED, alpha=0.45, lw=0, zorder=2)
        med = np.median(vals)
        ax.plot([med, med], [y - 0.28, y + 0.28], color=MUTED, lw=1.2, zorder=3)
        for r in fld:
            if r["group"].zfill(3) in NAMED:
                nm, mk = NAMED[r["group"].zfill(3)]
                ax.scatter([float(r[col])], [y], marker=mk, s=42, facecolor=PAPER, edgecolor=INK, lw=1.1, zorder=4)
        fold = d / f"{eu}.fold.pdb"
        if fold.exists():
            sc = score(fold, eu, pseudo_cb=True, tag=f"fig-{eu}")
            ours[eu] = sc
            ax.scatter([sc[col]], [y], s=110, color=BLUE, edgecolor=PAPER, lw=1.2, zorder=5)
            ax.text(sc[col], y - 0.36, f"ours {sc[col]:.2f}", ha="center", fontsize=8, color=BLUE, fontweight="bold")
        ax.text(0.0, y + 0.33, f"{eu} · {eus[eu]['difficulty']} · {eus[eu]['length']} aa · {len(fld)} groups · field median {med:.2f}",
                ha="left", va="bottom", fontsize=8.5, color=INK)
    if a.all_units:
        # every folded unit against the official score table: TM is the fair column (ours is over
        # pseudo-Cβ atoms and reads 0.01–0.02 under the official Cα TM-score); the official LDDT is
        # all-atom, which our trace cannot be scored on, so that comparison is approximate
        from casp16_table import field_percentiles
        rows = field_percentiles(d, "")
        o = np.array([r[2] if a.metric == "lddt" else r[5] for r in rows])
        fm = np.array([r[3] if a.metric == "lddt" else r[6] for r in rows])
        y = 0; jit = rng.normal(0, 0.09, len(rows))
        ax.scatter(fm, y + jit, s=11, color=MUTED, alpha=0.5, lw=0, zorder=2)
        ax.scatter(o, y + jit, s=11, color=BLUE, alpha=0.75, lw=0, zorder=3)
        feat = [k for k, r in enumerate(rows) if r[0] in a.eus]
        ax.scatter(o[feat], y + jit[feat], s=34, facecolor=BLUE, edgecolor=INK, lw=0.9, zorder=4)
        for v, c, lab in ((np.median(fm), MUTED, "field"), (np.median(o), BLUE, "ours")):
            ax.plot([v, v], [y - 0.3, y + 0.3], color=c, lw=1.4, zorder=5)
            ax.text(v, y - 0.4, f"{lab} {v:.2f}", ha="center", fontsize=8, color=c, fontweight="bold")
        src = "official LDDT, approximate" if a.metric == "lddt" else "official TM"
        ax.text(0.0, y + 0.36, f"all {len(rows)} units · grey: field median per unit ({src}) · blue: ours, ringed: the 3 above",
                ha="left", va="bottom", fontsize=8, color=INK)
    for nm, mk in NAMED.values():
        ax.scatter([], [], marker=mk, s=42, facecolor=PAPER, edgecolor=INK, lw=1.1, label=nm)
    ax.scatter([], [], s=14, color=MUTED, alpha=0.6, lw=0, label="one CASP16 group, model 1")
    ax.scatter([], [], s=90, color=BLUE, label=f"{a.label}")
    ax.legend(loc="upper center", fontsize=7.5, frameon=False, ncol=5, bbox_to_anchor=(0.5, -0.16), handletextpad=0.3, columnspacing=1.0)
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.75, nrows - 0.3 + 0.7)
    ax.set_yticks([])
    ax.set_xlabel(("Cβ-lDDT" if a.metric == "lddt" else "TM-score") + " against the experimental structure, one scorer for every dot"
                  + (" in the named rows" if a.all_units else ""), color=MUTED, fontsize=8.5)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.set_title("(c) scored against the CASP16 field", loc="left", fontsize=11, color=INK)
    for ax in axes:
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#d9dcd6")
    fig.tight_layout(w_pad=1.8)
    fig.subplots_adjust(bottom=0.27)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180, facecolor=PAPER)
    print(f"-> {a.out}  panel (b) precision {prec:.3f}; ours on the strip:", {k: round(v['ca_lddt'], 3) for k, v in ours.items()})
