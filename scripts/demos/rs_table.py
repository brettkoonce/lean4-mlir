#!/usr/bin/env python3
"""Tables 1 and 2 of the remote-sensing demo from `rs_score.py --json` outputs — planning/remote_sensing_wavelengths_demo.md §4.

  .venv-rs/bin/python scripts/demos/rs_table.py runs/2026-09-29-rs-phase3 [--arms rgb,rgbn,ms10,all,ir] [--md]

Reads score_<arm>_s<seed>_<part>.json and pair_<arm>_s<seed>.json; prints accuracy and macro-F1 per arm × part as
mean ± sd over seeds (Table 1), the per-class recall of forest and pasture, the wet − dry swing from the
pair files, and the diagnostic rows (Table 2: what each arm calls savanna formation and forest plantation).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

PARTS = ["eurosat_test", "amazon_dry", "cerrado_dry", "cerrado_wet"]
SHARED = ["annual crop", "perennial crop", "forest", "herbaceous", "pasture", "built", "water"]


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "   —   "
    return f"{np.mean(v):5.2f} ± {np.std(v):4.2f}" if len(v) > 1 else f"{v[0]:5.2f}       "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--arms", default="rgb,rgbn,ms10,all,ir")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    arms = args.arms.split(",")
    rows = {}
    for f in glob.glob(os.path.join(args.dir, "score_*_s*_*.json")):
        m = re.match(r"score_(\w+?)_s(\d+)_(\w+)\.json", os.path.basename(f))
        if not m:
            continue
        arm, seed, part = m.group(1), int(m.group(2)), m.group(3)
        txt = open(f).read()
        j = json.loads(txt[txt.index("{"):])
        rows[(arm, seed, part)] = j
    pairs = {}
    for f in glob.glob(os.path.join(args.dir, "pair_*_s*.json")):
        m = re.match(r"pair_(\w+?)_s(\d+)\.json", os.path.basename(f))
        txt = open(f).read()
        j = json.loads(txt[txt.index("{"):])
        if "pair" in j:       # scored on the part named first (dry); the pair part is wet
            pairs[(m.group(1), int(m.group(2)))] = dict(dry=j["scored"]["acc"], wet=j["pair"]["other_scored"]["acc"],
                                                        agreement=j["pair"]["agreement"], swing=j["pair"]["recall_swing"])
    seeds = sorted({k[1] for k in rows})
    sep = " | " if args.md else "  "
    print("Table 1 — accuracy (EuroSAT 10-way; Brazil 7-way restricted argmax), mean ± sd over seeds " + str(seeds))
    print("arm   " + sep + sep.join(f"{p:>14s}" for p in PARTS) + sep + "wet − dry")
    for arm in arms:
        cells = [ms([rows[(arm, s, p)]["scored"]["acc"] for s in seeds if (arm, s, p) in rows]) for p in PARTS]
        swing = ms([pairs[(arm, s)]["wet"] - pairs[(arm, s)]["dry"] for s in seeds if (arm, s) in pairs])
        print(f"{arm:5s}" + sep + sep.join(cells) + sep + swing)
    print("\ngrass merged (herbaceous ∪ pasture as one class), accuracy")
    for arm in arms:
        cells = [ms([rows[(arm, s, p)]["grass_merged"]["acc"] for s in seeds if (arm, s, p) in rows and "grass_merged" in rows[(arm, s, p)]]) for p in PARTS[1:]]
        print(f"{arm:5s}" + sep + sep.join(cells))
    print("\nmacro-F1 on the Brazil parts")
    for arm in arms:
        cells = [ms([rows[(arm, s, p)]["scored"]["macro_f1"] for s in seeds if (arm, s, p) in rows]) for p in PARTS[1:]]
        print(f"{arm:5s}" + sep + sep.join(cells))
    print("\nrecall of forest / pasture / water (Brazil parts)")
    for arm in arms:
        cells = []
        for p in PARTS[1:]:
            r = [rows[(arm, s, p)]["scored"]["recall"] for s in seeds if (arm, s, p) in rows]
            cells.append(" / ".join(ms([x[c] * 100 if x.get(c) is not None else None for x in r]).split()[0] if r else "—" for c in ("forest", "pasture", "water")))
        print(f"{arm:5s}" + sep + sep.join(f"{c:>22s}" for c in cells))
    print("\nTable 2 — what each arm calls the classes Europe does not have (% of chips, seed mean)")
    for part in ("cerrado_dry", "cerrado_wet", "amazon_dry"):
        for diag in ("savanna formation", "forest plantation"):
            any_ = [rows[(a, s, part)]["diagnostic"].get(diag) for a in arms for s in seeds if (a, s, part) in rows]
            if not any(any_):
                continue
            n = next(d["n"] for d in any_ if d)
            print(f"  {part}, {diag} (n={n}):")
            for arm in arms:
                ds = [rows[(arm, s, part)]["diagnostic"].get(diag) for s in seeds if (arm, s, part) in rows]
                ds = [d for d in ds if d]
                if not ds:
                    continue
                pred = {c: np.mean([d["predicted"][c] for d in ds]) * 100 for c in SHARED}
                print(f"    {arm:5s} " + "  ".join(f"{c} {v:4.1f}" for c, v in pred.items() if v >= 0.5))
    # the Brazil-trained ceiling and the fine-tune rows: fold means over brazil_all (score_ceil_*, score_ft_*)
    def fold_rows(pattern, key):
        acc = {}
        for f in glob.glob(os.path.join(args.dir, pattern)):
            m = re.match(key, os.path.basename(f))
            if not m:
                continue
            txt = open(f).read()
            try:
                j = json.loads(txt[txt.index("{"):])
            except ValueError:
                continue
            acc.setdefault(m.groups()[:-1], []).append((j["scored"]["acc"], j["scored"]["macro_f1"], j["scored"]["recall"]))
        return acc
    ceil = fold_rows("score_ceil_*_fold*.json", r"score_ceil_(\w+?)_fold(\d)\.json")
    ft = fold_rows("score_ft_*_n*_fold*.json", r"score_ft_(\w+?)_n(\d+)_fold(\d)\.json")
    if ceil or ft:
        print("\nTable 3 — trained or fine-tuned on Brazil labels, five folds by chip id (brazil_all: acc / macro-F1, fold mean ± sd; n folds)")
        for (arm,), v in sorted(ceil.items()):
            print(f"  ceiling, {arm:5s} from scratch, all labels     {ms([x[0] for x in v])} / {ms([x[1] for x in v])}  ({len(v)})")
        for (arm, n), v in sorted(ft.items(), key=lambda kv: (kv[0][0], int(kv[0][1]) or 10 ** 9)):
            lab = "all labels" if n == "0" else f"{n} labels "
            print(f"  fine-tune, {arm:5s} from EuroSAT s1, {lab:10s} {ms([x[0] for x in v])} / {ms([x[1] for x in v])}  ({len(v)})")
    print("\nper-chip agreement between the seasons, and the recall swing per class (dry − wet, seed mean):")
    for arm in arms:
        sw = [pairs[(arm, s)]["swing"] for s in seeds if (arm, s) in pairs]
        ag = [pairs[(arm, s)]["agreement"] for s in seeds if (arm, s) in pairs]
        if ag:
            print(f"  {arm:5s} same prediction on {100 * np.mean(ag):5.1f}% of chips")
        if sw:
            print(f"  {arm:5s} " + "  ".join(f"{c} {np.mean([x[c] for x in sw if x[c] is not None]):+5.1f}" for c in SHARED if any(x[c] is not None for x in sw)))


if __name__ == "__main__":
    main()
