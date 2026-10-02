#!/usr/bin/env python3
"""Step 10 of planning/casp16_distogram_demo.md §3 for a whole prediction directory: score every
<EU>.fold.pdb (and its mirror) against the experimental structure at pseudo-Cβ — Cβ-lDDT and the
TM-score over those atoms, the two numbers the field is rescored on — and write
<dir>/fold_scores.csv (eu, L, difficulty, cb_lddt, tm, mirror_cb_lddt, mirror_tm) with means by
difficulty on stdout. The mirror columns measure the hand rule: TM sees chirality, lDDT does not,
so mirror_tm > tm is a wrong hand. Each prep file is tagged by directory so two arms can be scored
back to back without colliding in data/casp16/work/prep."""
import argparse, csv, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from casp16_score import ROOT, eu_ranges, score

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="prediction directory with <EU>.fold.pdb")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--suffix", default="", help="score <EU>.<suffix>.fold.pdb instead (e.g. orient) -> fold_scores_<suffix>.csv")
    a = ap.parse_args()
    d = Path(a.dir)
    sfx = f".{a.suffix}" if a.suffix else ""
    meta = {r["eu"]: r for r in csv.DictReader(open(ROOT / "targets" / "summary.csv"))}
    eus = eu_ranges()
    tag = d.name.replace("distogram_", "")
    rows = []
    for f in sorted(d.glob(f"*{sfx}.fold.pdb")):
        eu = f.name[:-len(f"{sfx}.fold.pdb")]
        if "." in eu:          # another variant's file (.truth, .orient, …)
            continue
        r = score(f, eu, eus=eus, tag=f"{tag}-{eu}{sfx}", pseudo_cb=True)
        row = dict(eu=eu, L=int(meta[eu]["L"]), difficulty=meta[eu]["difficulty"],
                   cb_lddt=round(r["ca_lddt"], 4), tm=round(r["tm"], 4), mirror_cb_lddt="", mirror_tm="")
        m = d / f"{eu}{sfx}.fold_mirror.pdb"
        if m.exists() and not a.no_mirror:
            rm = score(m, eu, eus=eus, tag=f"{tag}-{eu}{sfx}-mirror", pseudo_cb=True)
            row.update(mirror_cb_lddt=round(rm["ca_lddt"], 4), mirror_tm=round(rm["tm"], 4))
        rows.append(row)
        print(f"{eu:12s} L {row['L']:4d} {row['difficulty']:6s} Cβ-lDDT {row['cb_lddt']:.3f} TM {row['tm']:.3f}"
              + (f"   mirror {row['mirror_cb_lddt']:.3f} / {row['mirror_tm']:.3f}" if row["mirror_tm"] != "" else ""), flush=True)
    out_csv = d / (f"fold_scores_{a.suffix}.csv" if a.suffix else "fold_scores.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for diff in ("easy", "medium", "hard", None):
        sel = [r for r in rows if diff is None or r["difficulty"] == diff]
        if sel:
            wrong = sum(1 for r in sel if r["mirror_tm"] != "" and r["mirror_tm"] > r["tm"])
            print(f"{diff or 'all':6s}: Cβ-lDDT mean {np.mean([r['cb_lddt'] for r in sel]):.3f} median {np.median([r['cb_lddt'] for r in sel]):.3f}"
                  f"  TM mean {np.mean([r['tm'] for r in sel]):.3f} median {np.median([r['tm'] for r in sel]):.3f}"
                  f"  ({len(sel)} EUs, mirror better on {wrong})")
    print(f"-> {out_csv}")
