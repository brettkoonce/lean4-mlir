#!/usr/bin/env python3
"""The ablation table of planning/casp16_distogram_demo.md §10 from what is on disk: one row per
`.lake/build/distogram_*_targets/` directory — the run's final val top-L/5 long-range precision
(<pfx>_curve.csv), the 84-EU precision by difficulty (table.csv, from casp16_predict.py), the
fold's mean Cβ-lDDT / TM over the folded EUs (fold_scores.csv, from casp16_fold_score.py; the
`.orient` variant beside it when present), and for predictions with orientation heads their
accuracy on the EUs: the fraction of binned long-range pairs whose argmax ω / θ / φ bin is within
one bin of the truth. The ESM-2 heads are the first rows."""
import csv, glob, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from casp16_labels import OBINS, PBINS
ROOT = Path("data/casp16")
meta = {r["eu"]: r for r in csv.DictReader(open(ROOT / "targets" / "summary.csv"))}


def by_diff(rows, key):
    out = []
    for d in (None, "easy", "medium", "hard"):
        sel = [float(r[key]) for r in rows if d is None or r["difficulty"] == d]
        out.append(np.mean(sel) if sel else np.nan)
    return out


def orient_acc(d):
    """Within-one-bin accuracy of the argmax ω / θ / φ over binned long-range pairs, all EUs."""
    hit = np.zeros(3); n = np.zeros(3)
    files = sorted(d.glob("*.pred.npz"))
    if not files:          # an arm whose .pred.npz were dropped for disk (regenerable): no orientation column
        return None
    for f in files:
        p = np.load(f)
        if "omega" not in p:
            return None
        t = np.load(ROOT / "targets" / f"{f.name[:-9]}.npz")
        L = len(t["obs"]); i, j = np.triu_indices(L, 24)
        for k, (name, nb, periodic) in enumerate((("omega", OBINS, True), ("theta", OBINS, True), ("phi", PBINS, False))):
            truth = t[name][i, j]; ok = truth < nb
            pred = p[name][i, j][ok].astype(np.float32)[:, :nb].argmax(-1)
            diff = np.abs(pred - truth[ok].astype(int))
            if periodic:
                diff = np.minimum(diff, nb - diff)
            hit[k] += (diff <= 1).sum(); n[k] += ok.sum()
    return hit / np.maximum(n, 1)


def field_percentiles(d, sfx=""):
    """Our fold against the CASP16 field's model-1 rows per EU (raw/CASP16_prot_domains.scores.csv):
    the fraction of groups our Cβ-lDDT / TM-score exceeds. TM is the fair column (ours is over
    pseudo-Cβ atoms and reads 0.01–0.02 under the official Cα TM-score); the official LDDT is
    all-atom, which our trace cannot be scored on, and Cβ-lDDT read ~0.03 high on nine good models,
    so that percentile is approximate. Returns rows (eu, difficulty, lddt, field median, pct, tm,
    field median, pct, n groups)."""
    import pandas as pd
    sc = pd.read_csv(ROOT / "raw" / "CASP16_prot_domains.scores.csv", sep=r"\s+")
    sc.columns = [c.strip("#").strip() for c in sc.columns]
    m = sc["Model"].str.extract(r"^(T\d+s?\d*(?:v\d)?)TS(\d+)_(\d)-(D\d+)$")
    sc["eu"] = m[0] + "-" + m[3]; sc = sc[m[2] == "1"]
    sc["LDDT"] = pd.to_numeric(sc["LDDT"], errors="coerce"); sc["TMscore"] = pd.to_numeric(sc["TMscore"], errors="coerce")
    rows = []
    for r in csv.DictReader(open(d / f"fold_scores{sfx}.csv")):
        f = sc[sc["eu"] == r["eu"]]
        if len(f) < 5:
            continue
        rows.append((r["eu"], r["difficulty"], float(r["cb_lddt"]), f["LDDT"].median(), (f["LDDT"] < float(r["cb_lddt"])).mean(),
                     float(r["tm"]), f["TMscore"].median(), (f["TMscore"] < float(r["tm"])).mean(), len(f)))
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", action="store_true", help="also place each arm's folds in the CASP16 field (field_percentiles)")
    a = ap.parse_args()
    print(f"{'arm':44s} {'val':>5s} {'EU L/5':>6s} {'easy':>5s} {'med':>5s} {'hard':>5s} | {'fold Cβ-lDDT':>12s} {'TM':>6s} {'n':>3s} | {'ω/θ/φ ±1 bin':>14s}")
    for head, key in (("ESM-2 35M contact head", "targets"), ("ESM-2 150M contact head", "targets_esm150")):
        f = ROOT / key / "summary.csv"
        if f.exists():
            rows = list(csv.DictReader(open(f)))
            m = by_diff(rows, "esm_p")
            print(f"{head:44s} {'':>5s} {m[0]:6.3f} {m[1]:5.3f} {m[2]:5.3f} {m[3]:5.3f} |")
    for d in sorted(glob.glob(".lake/build/distogram_*_targets")):
        d = Path(d); pfx = str(d)[:-len("_targets")]
        arm = d.name[len("distogram_"):-len("_targets")]
        val = ""
        cf = Path(pfx + "_curve.csv")
        if cf.exists():
            last = list(csv.DictReader(open(cf)))[-1]
            val = f"{100 * float(last['val_top_l5_lr_precision']):5.1f}"
        tf = d / "table.csv"
        m = by_diff(list(csv.DictReader(open(tf))), "ours") if tf.exists() else [np.nan] * 4
        line = f"{arm:44s} {val:>5s} {m[0]:6.3f} {m[1]:5.3f} {m[2]:5.3f} {m[3]:5.3f} |"
        for sfx in ("", "_orient"):
            ff = d / f"fold_scores{sfx}.csv"
            if ff.exists():
                rows = list(csv.DictReader(open(ff)))
                if rows and "cb_lddt" in rows[0]:
                    line += f" {np.mean([float(r['cb_lddt']) for r in rows]):12.3f} {np.mean([float(r['tm']) for r in rows]):6.3f} {len(rows):3d}" + (" (orient)" if sfx else "")
                    if not sfx: line += " |"
        acc = orient_acc(d)
        if acc is not None:
            line += f"  {acc[0]:.2f} / {acc[1]:.2f} / {acc[2]:.2f}"
        print(line)
        if a.field:
            for sfx in ("", "_orient"):
                if (d / f"fold_scores{sfx}.csv").exists():
                    rows = field_percentiles(d, sfx)
                    if rows:
                        print(f"    {'fold' + sfx:12s} vs the field, {len(rows)} EUs: TM percentile median {np.median([x[7] for x in rows]):.2f} mean {np.mean([x[7] for x in rows]):.2f};"
                              f" Cβ-lDDT percentile median {np.median([x[4] for x in rows]):.2f} mean {np.mean([x[4] for x in rows]):.2f} (approximate)")
                        for diff in ("easy", "medium", "hard"):
                            sel = [x for x in rows if x[1] == diff]
                            if sel:
                                print(f"      {diff:6s} n={len(sel):2d}: TM {np.mean([x[5] for x in sel]):.3f} vs field median {np.mean([x[6] for x in sel]):.3f};"
                                      f" Cβ-lDDT {np.mean([x[2] for x in sel]):.3f} vs {np.mean([x[3] for x in sel]):.3f}")
