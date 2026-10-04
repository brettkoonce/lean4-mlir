#!/usr/bin/env python3
"""The BraTS tail table: per-patient CSVs from `lake exe brats-eval` (2D / 2.5D) and
`jax/scripts/unet3d_brats.py` (3D) side by side, under the same post-processing.

The decision in planning/brats_25d_3d.md §4c is about the patients a model fails, not the mean,
so per model and region this prints the mean, the mean over the worst tenth of the patients, and
how many score under 0.7 and under 0.5; then the ET-absent patients' false calls and the per-slice
ET false-alarm rate (slices with no ground-truth ET that get >= 1 / >= 10 predicted ET pixels).

`--min-et N` applies the standard BraTS post-process to every CSV alike: a volume whose predicted
ET is under N voxels has its ET relabelled to necrosis/non-enhancing core. That moves ET pixels
inside TC, so WT and TC are unchanged and only ET's intersection and prediction go to zero — the
CSV's counts are enough to apply it, and a volume relabelled has no ET false alarms left. Pass
several (`--min-et 0,200,500`) for one table each; 0 is the raw model.

`--paired` adds, against the first CSV, the mean over that model's worst tenth on each region
for every other model on the SAME patients, and how many patients each model wins on.

    python3 scripts/probes/brats_tail.py a.csv b.csv --names "2D s0,3D 20k" --min-et 0,200 --paired
"""
import argparse
import csv

import numpy as np

REGIONS = ["WT", "TC", "ET"]


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def dice(i, g, p):
    if g == 0:
        return 1.0 if p == 0 else 0.0
    return 2.0 * i / (g + p)


def post(rows, min_et):
    """Per volume: {region: dice}, ET_gt, ET_pred, and the false-alarm counts after the post-process."""
    out = []
    for r in rows:
        et_i, et_g, et_p = int(r["ET_inter"]), int(r["ET_gt"]), int(r["ET_pred"])
        fa1 = int(r["ET_fa1"]) if "ET_fa1" in r else None
        fa10 = int(r["ET_fa10"]) if "ET_fa10" in r else None
        if min_et and et_p < min_et:
            et_i, et_p = 0, 0
            fa1 = 0 if fa1 is not None else None
            fa10 = 0 if fa10 is not None else None
        d = {k: dice(int(r[f"{k}_inter"]), int(r[f"{k}_gt"]), int(r[f"{k}_pred"])) for k in ("WT", "TC")}
        d["ET"] = dice(et_i, et_g, et_p)
        out.append({"dice": d, "et_gt": et_g, "et_pred": et_p,
                    "clear": int(r["ET_clear_slices"]) if "ET_clear_slices" in r else None,
                    "fa1": fa1, "fa10": fa10})
    return out


def table(names, models, k):
    w = max(len(n) for n in names) + 2
    hdr = "".join(f"{reg:>26}" for reg in REGIONS)
    print(f"{'':{w}}{hdr}   ET-absent: pred vox   ET false alarms (>=1 / >=10 px)")
    print(f"{'':{w}}" + "".join(f"{'mean  worst10  <.7 <.5':>26}" for _ in REGIONS))
    for name, m in zip(names, models):
        cells = ""
        for reg in REGIONS:
            d = np.array([v["dice"][reg] for v in m])
            worst = np.sort(d)[:k].mean()
            cells += f"{d.mean():>10.3f}{worst:>9.3f}{(d < 0.7).sum():>4d}{(d < 0.5).sum():>3d}"
        absent = ",".join(str(v["et_pred"]) for v in m if v["et_gt"] == 0) or "-"
        if m[0]["clear"] is None or m[0]["fa1"] is None:
            fa = "(no columns)"
        else:
            clear = sum(v["clear"] for v in m)
            f1, f10 = sum(v["fa1"] for v in m), sum(v["fa10"] for v in m)
            fa = f"{f1}/{clear} = {f1 / clear:.4f} / {f10 / clear:.4f}"
        print(f"{name:{w}}{cells}   {absent:>20}   {fa}")


def paired(names, models, k):
    base = models[0]
    print(f"  paired against {names[0]}: mean on {names[0]}'s worst {k} patients per region; wins / losses over all")
    for name, m in zip(names[1:], models[1:]):
        parts = []
        for reg in REGIONS:
            b = np.array([v["dice"][reg] for v in base])
            o = np.array([v["dice"][reg] for v in m])
            idx = np.argsort(b)[:k]
            parts.append(f"{reg} {b[idx].mean():.3f} -> {o[idx].mean():.3f}  ({(o > b).sum()}/{(o < b).sum()})")
        print(f"    {name}: " + "   ".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--names", default=None, help="comma-separated, one per CSV")
    ap.add_argument("--min-et", default="0", help="comma-separated voxel thresholds (0 = no post-process)")
    ap.add_argument("--paired", action="store_true")
    args = ap.parse_args()
    names = args.names.split(",") if args.names else args.csvs
    assert len(names) == len(args.csvs), "one name per CSV"
    raw = [load(p) for p in args.csvs]
    n = {len(r) for r in raw}
    assert len(n) == 1, f"CSVs score different patient counts: {n}"
    k = max(1, n.pop() // 10)
    for t in [int(x) for x in args.min_et.split(",")]:
        print(f"\n== min-ET {t}" + (" (raw)" if t == 0 else f": ET under {t} voxels in a volume -> core"))
        models = [post(r, t) for r in raw]
        table(names, models, k)
        if args.paired and len(models) > 1:
            paired(names, models, k)


if __name__ == "__main__":
    main()
