#!/usr/bin/env python3
"""An ensemble arm for the distogram demo (planning/casp16_distogram_demo.md §10a): average the
per-EU class probabilities of several prediction directories into a new
.lake/build/distogram_ens-<name>_targets/ that casp16_fold.py, casp16_fold_score.py and
casp16_table.py read like any trained arm. The 66 distance classes are averaged in probability
space and renormalized; P(contact) and the expected distance are recomputed from the mean; the
ω / θ / φ planes are averaged over the members that have them (absent when none does).
table.csv is written as casp16_predict.py writes it, and each member's 84-EU mean is printed
beside the ensemble's."""
import argparse, csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from casp16_predict import ROOT, NC, CENTRES, CONTACT_MAX, ORIENT, top_l5_precision

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="-> .lake/build/distogram_ens-<name>_targets/")
    ap.add_argument("dirs", nargs="+", help="member prediction directories (<EU>.pred.npz from casp16_predict.py)")
    a = ap.parse_args()
    out = Path(f".lake/build/distogram_ens-{a.name}_targets"); out.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in a.dirs]
    base = {r["eu"]: r for r in csv.DictReader(open(ROOT / "targets" / "summary.csv"))}
    eus = sorted(set.intersection(*[{f.name[:-9] for f in d.glob("*.pred.npz")} for d in dirs]))
    rows = []
    for eu in eus:
        preds = [np.load(d / f"{eu}.pred.npz") for d in dirs]
        p = np.mean([q["probs"].astype(np.float32) for q in preds], 0)
        p /= p.sum(-1, keepdims=True)
        pcontact = p[:, :, : CONTACT_MAX + 1].sum(-1).astype(np.float32)
        edist = (p[:, :, :NC - 1] * CENTRES[:NC - 1]).sum(-1).astype(np.float32)
        t = np.load(ROOT / "targets" / f"{eu}.npz")
        prec, _ = top_l5_precision(pcontact, t["cls"], t["obs"])
        extra = {}
        for name, _, _ in ORIENT:
            have = [q[name].astype(np.float32) for q in preds if name in q]
            if have:
                extra[name] = np.mean(have, 0).astype(np.float16)
        np.savez_compressed(out / f"{eu}.pred.npz", probs=p.astype(np.float16), pcontact=pcontact, edist=edist,
                            windows=np.float32(len(dirs)), **extra)
        b = base.get(eu, {})
        rows.append(dict(eu=eu, L=len(t["obs"]), ours=prec, esm=float(b.get("esm_p", "nan")), difficulty=b.get("difficulty", "")))
    with open(out / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for d in dirs:
        m = [float(r["ours"]) for r in csv.DictReader(open(d / "table.csv")) if r["eu"] in set(eus)]
        print(f"{d.name:60s} mean P@L/5 {np.mean(m):.3f}")
    for diff in ("easy", "medium", "hard", None):
        sel = [r["ours"] for r in rows if diff is None or r["difficulty"] == diff]
        print(f"ensemble {diff or 'all':6s}: {np.mean(sel):.3f} ({len(sel)} EUs)")
    print(f"-> {out}")
