#!/usr/bin/env python3
"""Score the PlantVillage → PlantDoc demo — planning/plant_lab_to_field_demo.md §5.

One scorer for both datasets and every arm. Reads the f32 [N, 38] logits `plant-leaf`
writes for a part, the part's labels from its own records, and `meta_plant.npz`:

  accuracy with a 95 % Wilson interval, per logits file and pooled over seeds
  PlantDoc parts (`pd_all`, `pd_fold<k>_test`, `pd_test`): 38-way AND restricted to the 28
      mapped classes (`--restrict` picks which one the headline is); fold hygiene — a
      `pd_fold<k>_test` logits file must come from a run tagged `field<k>`
  PlantVillage test parts: the leak audit read from meta (test → nearest train image,
      the ArASL instrument) and accuracy on the leaked vs the other images
  per class, and the most confused pairs

  .venv/bin/python scripts/plant_score.py LOGITS.bin [LOGITS2.bin ...] --part pd_all|pv_test|pvg_test|...
                   [--restrict] [--data=data/plant] [--json=OUT.json] [--top=8]
"""
import argparse
import json
import math
import os
import re

import numpy as np

N_CLASSES = 38
LEAK_THR = 6.0


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def acc_str(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:6.2f}% [{lo:.2f}, {hi:.2f}]"


def part_labels(path, side=224):
    """Labels of an Imagenette-format part: the byte before each record's pixels."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n = int(np.frombuffer(bytes(raw[:4]), dtype=np.uint32)[0])
    rec = 1 + 3 * side * side
    return np.asarray(raw[4:4 + n * rec:rec]).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits", nargs="+")
    ap.add_argument("--part", required=True)
    ap.add_argument("--restrict", action="store_true", help="PlantDoc: argmax over the 28 mapped classes only")
    ap.add_argument("--data", default="data/plant")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    meta = np.load(os.path.join(args.data, "meta_plant.npz"))
    classes = [str(c) for c in meta["classes"]]
    short = [c.replace("___", " ").replace("_", " ")[:34] for c in classes]
    labels = part_labels(os.path.join(args.data, f"{args.part}.bin"))
    n = len(labels)
    is_pd = args.part.startswith("pd_")
    mapped = np.zeros(N_CLASSES, dtype=bool)
    mapped[meta["pd_to_pv"]] = True

    preds38, predsR = [], []
    for path in args.logits:
        lg = np.fromfile(path, dtype=np.float32)
        assert lg.size == n * N_CLASSES, f"{path}: {lg.size} floats, expected {n} × {N_CLASSES} for {args.part}"
        lg = lg.reshape(n, N_CLASSES)
        preds38.append(lg.argmax(axis=1))
        lr = lg.copy()
        lr[:, ~mapped] = -np.inf
        predsR.append(lr.argmax(axis=1))
        # fold hygiene: a fold's held-out logits must come from that fold's run
        mfold = re.match(r"pd_fold(\d)_test", args.part)
        if mfold and f"field{mfold.group(1)}" not in os.path.basename(path):
            raise SystemExit(f"{path} scored on {args.part} but its name does not say field{mfold.group(1)} — refusing")
    preds38, predsR = np.stack(preds38), np.stack(predsR)
    use = predsR if (is_pd and args.restrict) else preds38
    correct = use == labels[None, :]
    runs = len(args.logits)

    print(f"== {args.part}, {n} images, {runs} run(s){', restricted to the 28 mapped classes' if is_pd and args.restrict else ''} ==")
    accs = []
    for path, pr in zip(args.logits, use):
        accs.append(100 * (pr == labels).mean())
        print(f"  {os.path.basename(path):64s} {acc_str(int((pr == labels).sum()), n)}")
    if runs > 1:
        print(f"  mean over {runs} runs: {np.mean(accs):.2f} ± {np.std(accs, ddof=1):.2f}")
    k_all = int(correct.sum())
    print(f"  pooled: {acc_str(k_all, correct.size)}")
    out = dict(part=args.part, n=n, runs=[os.path.basename(p) for p in args.logits], acc=accs,
               pooled=dict(k=k_all, n=int(correct.size), acc=100 * k_all / correct.size, wilson=wilson(k_all, correct.size)))
    if is_pd:
        k38, kR = int((preds38 == labels).sum()), int((predsR == labels).sum())
        print(f"  38-way argmax: {acc_str(k38, correct.size)}   restricted to mapped: {acc_str(kR, correct.size)}   "
              f"predictions of an unmapped class: {100 * (~mapped[preds38]).mean():.1f}%")
        out["acc38"], out["accR"] = 100 * k38 / correct.size, 100 * kR / correct.size

    # the leak audit for a PlantVillage test part
    stem = args.part.split("_test")[0] if "_test" in args.part else None
    if stem in ("pv", "pvg") and f"{stem}_test_nn_mad" in meta:
        nd, nd64 = meta[f"{stem}_test_nn_mad"], meta[f"{stem}_test_nn_mad64"]
        te, nn = meta[f"{stem}_test_index"], meta[f"{stem}_test_nn_train_index"]
        assert len(te) == n
        leaked = nd < LEAK_THR
        same_leaf = (meta["pv_leaf"][nn] == meta["pv_leaf"][te]) & (meta["pv_leaf"][te] >= 0)
        print(f"leak audit ({stem}): {100 * leaked.mean():.1f}% of test images have a train image within {LEAK_THR:g} grey levels "
              f"at 16×16 ({100 * (nd64 < LEAK_THR).mean():.1f}% at 64×64); nearest is the same leaf {100 * same_leaf.mean():.1f}%")
        out["leak"] = dict(frac16=float(leaked.mean()), frac64=float((nd64 < LEAK_THR).mean()), same_leaf=float(same_leaf.mean()))
        for name, mask in (("leaked", leaked), ("not leaked", ~leaked), ("same-leaf neighbour", same_leaf), ("other", ~same_leaf)):
            if mask.sum():
                k, m = int(correct[:, mask].sum()), int(mask.sum()) * runs
                out["leak"][name] = dict(n=int(mask.sum()), acc=100 * k / m)
                print(f"  accuracy on {name:20s} n={int(mask.sum()):6d}: {acc_str(k, m)}")
        k_nn = int((meta["pv_label"][nn] == labels).sum())
        print(f"  1-NN on 16×16 thumbnails: {acc_str(k_nn, n)}")
        out["leak"]["nn_acc"] = 100 * k_nn / n

    # per class
    per_class = []
    for c in range(N_CLASSES):
        m = labels == c
        if not m.any():
            continue
        k, t = int(correct[:, m].sum()), int(m.sum()) * runs
        per_class.append(dict(cls=classes[c], n=int(m.sum()), acc=100 * k / t))
    order = sorted(per_class, key=lambda r: r["acc"])
    print(f"per class ({len(per_class)} present), worst to best:")
    for i in range(0, len(order), 3):
        print("  " + "   ".join(f"{short[classes.index(r['cls'])]:34s} {r['acc']:5.1f} (n={r['n']})" for r in order[i:i + 3]))
    out["per_class"] = per_class

    # confusions
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for pr in use:
        np.add.at(cm, (labels, pr), 1)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j] + off[j, i], i, j) for i in range(N_CLASSES) for j in range(i + 1, N_CLASSES) if off[i, j] + off[j, i]), reverse=True)
    print(f"most confused pairs (both directions, per run):")
    out["confused"] = []
    for cnt, i, j in pairs[:args.top]:
        out["confused"].append(dict(a=classes[i], b=classes[j], a_as_b=int(off[i, j]), b_as_a=int(off[j, i]), per_run=cnt / runs))
        print(f"  {short[i]:34s} ↔ {short[j]:34s} {cnt / runs:7.1f}  ({off[i, j]} / {off[j, i]})")
    out["confusion"] = cm.tolist()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
