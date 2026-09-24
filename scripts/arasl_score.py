#!/usr/bin/env python3
"""Score the ArASL demo — planning/arasl_people_watching_demo.md §5.

One scorer for both protocols, so the two columns of Table 1 are one code path with
one constant changed. Reads the f32 [N, 32] test logits `arasl-signs` writes, the
labels of the split's test part, and the leak audit `preprocess_arasl.py` stored in
`meta_<split>.npz` (for every test image, its nearest train image at 16×16 and the
distance there and at 64×64). Prints:

  accuracy with a 95 % Wilson interval, per logits file, and the mean ± sd over seeds
  accuracy on the LEAKED test images (a train image within 6 grey levels) and on the
      rest — the number the gap between the two columns is explained by
  the 1-NN-on-thumbnails floor: the nearest train image's label, no parameters
  per-class accuracy on the test part, and the most confused letter pairs

  .venv/bin/python scripts/arasl_score.py LOGITS.bin [LOGITS2.bin ...] --split=random|blocked
                   [--data=data/arasl] [--json=OUT.json] [--top=5]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _stats import wilson, acc_str  # noqa: E402

N_CLASSES = 32
LEAK_THR = 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits", nargs="+")
    ap.add_argument("--split", required=True, choices=("random", "blocked"))
    ap.add_argument("--data", default="data/arasl")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    labels = np.fromfile(os.path.join(args.data, f"labels_{args.split}_test.bin"), dtype=np.int32)
    meta = np.load(os.path.join(args.data, f"meta_{args.split}.npz"))
    classes = [str(c) for c in meta["classes"]]
    n = len(labels)
    # the audit rows are in the test part's order; cross-check against the labels file
    glabel = np.zeros(int(meta["index"].max()) + 1, dtype=np.int32)
    glabel[meta["index"]] = meta["label"]
    te, nn = meta["test_index"], meta["test_nn_train_index"]
    assert len(te) == n and np.array_equal(glabel[te], labels), "meta and labels disagree"
    leaked16 = meta["test_nn_mad"] < LEAK_THR
    leaked64 = meta["test_nn_mad64"] < LEAK_THR
    nn_label = glabel[nn]

    preds, accs = [], []
    for path in args.logits:
        lg = np.fromfile(path, dtype=np.float32)
        assert lg.size == n * N_CLASSES, f"{path}: {lg.size} floats, expected {n} × {N_CLASSES}"
        pr = lg.reshape(n, N_CLASSES).argmax(axis=1)
        preds.append(pr)
        accs.append(100 * (pr == labels).mean())
    preds = np.stack(preds)
    correct = preds == labels[None, :]

    print(f"== ArASL {args.split} split, {n} test images, {len(args.logits)} run(s) ==")
    for path, pr in zip(args.logits, preds):
        print(f"  {os.path.basename(path):48s} {acc_str(int((pr == labels).sum()), n)}")
    if len(accs) > 1:
        print(f"  mean over {len(accs)} seeds: {np.mean(accs):.2f} ± {np.std(accs, ddof=1):.2f}")
    # pooled over seeds: every (seed, image) pair is one trial
    k_all, n_all = int(correct.sum()), correct.size
    print(f"  pooled: {acc_str(k_all, n_all)}")

    print(f"leak audit ({args.split}): {100 * leaked16.mean():.1f}% of test images have a train image within "
          f"{LEAK_THR:g} grey levels at 16×16, {100 * leaked64.mean():.1f}% at 64×64")
    rows = {}
    for name, mask in (("leaked (16×16)", leaked16), ("not leaked (16×16)", ~leaked16),
                       ("leaked (64×64)", leaked64), ("not leaked (64×64)", ~leaked64)):
        if mask.sum() == 0:
            continue
        k, m = int(correct[:, mask].sum()), int(mask.sum()) * len(accs)
        rows[name] = dict(n=int(mask.sum()), acc=100 * k / m, wilson=wilson(k, m))
        print(f"  accuracy on {name:20s} n={int(mask.sum()):5d}: {acc_str(k, m)}")
    k_nn = int((nn_label == labels).sum())
    print(f"  1-NN on 16×16 thumbnails (nearest train image's label): {acc_str(k_nn, n)}")

    # per class, pooled over seeds
    print("per class (test, pooled over seeds):")
    per_class = []
    for c in range(N_CLASSES):
        m = labels == c
        k, t = int(correct[:, m].sum()), int(m.sum()) * len(accs)
        per_class.append(dict(cls=classes[c], n=int(m.sum()), acc=100 * k / t,
                              leaked=100 * leaked16[m].mean()))
    order = sorted(range(N_CLASSES), key=lambda c: per_class[c]["acc"])
    for i in range(0, N_CLASSES, 4):
        print("  " + "   ".join(f"{per_class[c]['cls']:5s} {per_class[c]['acc']:5.1f}" for c in order[i:i + 4]))
    weakest = ", ".join(f"{per_class[c]['cls']} {per_class[c]['acc']:.1f}%" for c in order[:5])
    print(f"  weakest: {weakest}")

    # confusions, pooled: directed (true → predicted) and the symmetrised pairs
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for pr in preds:
        np.add.at(cm, (labels, pr), 1)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    sym = off + off.T
    pairs = [(sym[i, j], i, j) for i in range(N_CLASSES) for j in range(i + 1, N_CLASSES) if sym[i, j]]
    pairs.sort(reverse=True)
    print(f"most confused pairs (both directions, pooled over {len(accs)} run(s)):")
    top_pairs = []
    for cnt, i, j in pairs[:args.top]:
        top_pairs.append(dict(a=classes[i], b=classes[j], a_as_b=int(off[i, j]), b_as_a=int(off[j, i]),
                              total=int(cnt), per_run=cnt / len(accs)))
        print(f"  {classes[i]:5s} ↔ {classes[j]:5s}  {cnt / len(accs):6.1f} per run  "
              f"({classes[i]}→{classes[j]} {off[i, j]}, {classes[j]}→{classes[i]} {off[j, i]})")

    if args.json:
        out = dict(split=args.split, n_test=n, runs=[os.path.basename(p) for p in args.logits],
                   acc=accs, mean=float(np.mean(accs)), sd=float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                   pooled=dict(k=k_all, n=n_all, acc=100 * k_all / n_all, wilson=wilson(k_all, n_all)),
                   leak=dict(frac16=float(leaked16.mean()), frac64=float(leaked64.mean()), by_leak=rows,
                             nn_acc=100 * k_nn / n, nn_wilson=wilson(k_nn, n)),
                   per_class=per_class, confused=top_pairs, confusion=cm.tolist())
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
