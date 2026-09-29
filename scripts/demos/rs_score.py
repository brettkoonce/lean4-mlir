#!/usr/bin/env python3
"""Score an `rs-bands` logits file — planning/remote_sensing_wavelengths_demo.md §5.

  .venv-rs/bin/python scripts/demos/rs_score.py <prefix>_logits_<part>.bin --part <part> [--data data/rs]
                                                [--pair <prefix>_logits_<other>.bin --pair-part <other>] [--json]

A EuroSAT part (`eurosat_*`) is scored 10-way: accuracy with a Wilson 95% interval,
per-class recall, the confusion matrix's top pairs. A Brazil part carries labels in the
seven-class map (0 annual crop, 1 perennial crop, 2 forest, 3 herbaceous, 4 pasture,
5 built, 6 water) plus two DIAGNOSTIC codes never scored (7 savanna formation, 8 forest
plantation); ten European logits collapse into the seven — built = max(Industrial,
Residential), water = max(River, SeaLake), Highway masked out (the plant demo's restricted
argmax) — and the diagnostic rows report the fraction of chips each predicted class
receives. Logits already seven wide (a Brazil-trained arm, `classes=7`) skip the
collapse. `--pair` joins two parts on chip id (`meta_<part>.npz`, `chip_id`), the
wet/dry instrument: per-chip agreement and the per-class swing in recall; it refuses
parts whose chip sets differ.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

EURO = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
SHARED = ["annual crop", "perennial crop", "forest", "herbaceous", "pasture", "built", "water"]
DIAG = {7: "savanna formation", 8: "forest plantation"}
# shared class → the EuroSAT logits it takes the max over
COLLAPSE = [[0], [6], [1], [2], [5], [4, 7], [8, 9]]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(logits_path, part, data, fold=None):
    lbl = np.fromfile(os.path.join(data, f"labels_{part}.bin"), dtype=np.int32)
    if fold is not None:                      # a fold run's logits cover only the fold's chips, in part order
        folds = np.fromfile(os.path.join(data, f"folds_{part}.bin"), dtype=np.int32)
        lbl = lbl[folds == fold]
    raw = np.fromfile(logits_path, dtype=np.float32)
    if len(raw) % len(lbl):
        sys.exit(f"{logits_path}: {len(raw)} floats is not a multiple of {len(lbl)} labels")
    return raw.reshape(len(lbl), -1), lbl


HEAD_BRAZIL = False      # --head brazil: a 10-wide head fine-tuned on Brazil labels; outputs 0–6 ARE the seven classes


def collapse(logits):
    if logits.shape[1] == len(SHARED):
        return logits
    if HEAD_BRAZIL:
        return logits[:, :len(SHARED)]
    if logits.shape[1] != len(EURO):
        sys.exit(f"logits are {logits.shape[1]} wide: expected 10 (EuroSAT head) or 7 (Brazil head)")
    return np.stack([logits[:, g].max(axis=1) for g in COLLAPSE], axis=1)


def report(pred, lbl, names):
    n = len(lbl)
    acc = float((pred == lbl).mean()) if n else 0.0
    lo, hi = wilson(int((pred == lbl).sum()), n)
    cm = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(cm, (lbl, pred), 1)
    rec = {names[c]: (float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None) for c in range(len(names))}
    prec = [cm[c, c] / cm[:, c].sum() if cm[:, c].sum() else 0.0 for c in range(len(names))]
    f1 = [2 * p * r / (p + r) if (p + r) else 0.0 for p, r in zip(prec, [rec[nm] or 0.0 for nm in names])]
    present = [c for c in range(len(names)) if cm[c].sum()]
    macro_f1 = float(np.mean([f1[c] for c in present])) if present else 0.0
    off = [(int(cm[a, b]), names[a], names[b]) for a in range(len(names)) for b in range(len(names)) if a != b and cm[a, b]]
    off.sort(reverse=True)
    return dict(n=n, acc=100 * acc, wilson=[lo, hi], macro_f1=100 * macro_f1, recall=rec,
                support={names[c]: int(cm[c].sum()) for c in range(len(names))}, top_confusions=off[:8],
                confusion=cm.tolist())


def print_report(r, names, title):
    print(f"{title}: n={r['n']}  acc {r['acc']:.2f}% [{r['wilson'][0]:.2f}, {r['wilson'][1]:.2f}]  macro-F1 {r['macro_f1']:.2f}")
    for nm in names:
        rc = r["recall"][nm]
        print(f"   {nm:22s} n={r['support'][nm]:6d}  recall {'   -' if rc is None else f'{100 * rc:5.1f}'}")
    if r["top_confusions"]:
        print("   top confusions (true → predicted): " + "; ".join(f"{a}→{b} {k}" for k, a, b in r["top_confusions"][:5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits")
    ap.add_argument("--part", required=True)
    ap.add_argument("--data", default="data/rs")
    ap.add_argument("--pair", help="another logits file, joined on chip id")
    ap.add_argument("--pair-part")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fold", type=int, help="score a fold run: only the chips whose folds_<part>.bin id is this")
    ap.add_argument("--head", choices=["eurosat", "brazil"], default="eurosat",
                    help="brazil: the 10-wide head was fine-tuned on Brazil labels, so outputs 0–6 are the seven classes (no collapse)")
    args = ap.parse_args()
    global HEAD_BRAZIL
    HEAD_BRAZIL = args.head == "brazil"
    logits, lbl = load(args.logits, args.part, args.data, args.fold)
    out = dict(logits=args.logits, part=args.part, fold=args.fold)

    if args.part.startswith("eurosat"):
        pred = logits.argmax(axis=1)
        r = report(pred, lbl, EURO)
        print_report(r, EURO, f"{args.part} (10-way)")
        out["scored"] = r
    else:
        seven = collapse(logits)
        pred = seven.argmax(axis=1)
        keep = lbl < len(SHARED)
        r = report(pred[keep], lbl[keep], SHARED)
        print_report(r, SHARED, f"{args.part} (7-way, restricted argmax; {int((~keep).sum())} diagnostic chips aside)")
        # the coarser reading: EuroSAT's Pasture (a European meadow) and HerbaceousVegetation against
        # MapBiomas's pasture (grazed grass) and grassland are one "grass" class either side
        g_pred = np.where(pred == 3, 4, pred)
        g_lbl = np.where(lbl == 3, 4, lbl)
        rg = report(g_pred[keep], g_lbl[keep], SHARED)
        print(f"   grass merged (herbaceous ∪ pasture): acc {rg['acc']:.2f}% [{rg['wilson'][0]:.2f}, {rg['wilson'][1]:.2f}]  "
              f"macro-F1 {rg['macro_f1']:.2f}  grass recall {100 * (rg['recall']['pasture'] or 0):.1f}")
        out["grass_merged"] = dict(acc=rg["acc"], wilson=rg["wilson"], macro_f1=rg["macro_f1"], grass_recall=rg["recall"]["pasture"])
        if logits.shape[1] == len(EURO):
            hw = float((logits.argmax(axis=1) == 3).mean())
            print(f"   (10-way argmax would have said Highway for {100 * hw:.1f}% of chips)")
            out["highway_rate_10way"] = hw
        out["scored"] = r
        diag = {}
        for code, nm in DIAG.items():
            m = lbl == code
            if m.any():
                frac = np.bincount(pred[m], minlength=len(SHARED)) / m.sum()
                diag[nm] = dict(n=int(m.sum()), predicted={SHARED[c]: float(frac[c]) for c in range(len(SHARED))})
                print(f"   {nm} (n={int(m.sum())}, diagnostic): " + "  ".join(f"{SHARED[c]} {100 * frac[c]:.0f}%" for c in range(len(SHARED)) if frac[c] >= 0.005))
        out["diagnostic"] = diag

        if args.pair:
            if not args.pair_part:
                sys.exit("--pair needs --pair-part")
            l2, lbl2 = load(args.pair, args.pair_part, args.data)
            ids = [np.load(os.path.join(args.data, f"meta_{p}.npz"))["chip_id"] for p in (args.part, args.pair_part)]
            if len(ids[0]) != len(ids[1]) or not np.array_equal(np.sort(ids[0]), np.sort(ids[1])):
                sys.exit(f"--pair: {args.part} and {args.pair_part} do not hold the same chips ({len(ids[0])} vs {len(ids[1])})")
            order = np.argsort(ids[1])[np.searchsorted(np.sort(ids[1]), ids[0])]
            p2 = collapse(l2).argmax(axis=1)[order]
            lbl2 = lbl2[order]
            assert np.array_equal(lbl, lbl2), "labels differ across the pair — the annual label should be identical"
            k2 = keep
            agree = float((pred[k2] == p2[k2]).mean())
            r2 = report(p2[k2], lbl[k2], SHARED)
            swing = {nm: (None if r["recall"][nm] is None else 100 * (r["recall"][nm] - r2["recall"][nm])) for nm in SHARED}
            print(f"pair {args.part} vs {args.pair_part}: same prediction on {100 * agree:.1f}% of chips; "
                  f"acc {r['acc']:.2f} vs {r2['acc']:.2f}; recall swing (this − other): "
                  + "  ".join(f"{nm} {s:+.1f}" for nm, s in swing.items() if s is not None))
            out["pair"] = dict(other=args.pair_part, agreement=agree, other_scored=r2, recall_swing=swing)
    if args.json:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
