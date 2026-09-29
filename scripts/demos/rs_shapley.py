#!/usr/bin/env python3
"""Exact Shapley values over five band groups — planning/remote_sensing_wavelengths_demo.md §5.

Players: visible (B02 B03 B04), red edge (B05 B06 B07), NIR (B08 B8A), SWIR (B11 B12),
atmospheric (B01 B09 B10). A removed group is set to 0 in the standardised record, i.e. to
its EuroSAT training mean. Thirty-two coalitions per chip, scored by the `all` arm's eval
graph, so the value is exact and efficiency (Σφ = v(N) − v(∅)) holds to the float.

  make:   .venv-rs/bin/python scripts/demos/rs_shapley.py make --part eurosat_test [--n 500] [--seed 0] [--data data/rs]
          writes data/rs/shap_<part>.bin (n × 32 chips, coalition-major within a chip), labels_shap_<part>.bin,
          meta_shap_<part>.npz; then score it with
          CUDA_VISIBLE_DEVICES=0 lake exe rs-bands arm=all eval tag=<tag> out=<dir> score=shap_<part>
  score:  .venv-rs/bin/python scripts/demos/rs_shapley.py score --part eurosat_test --logits <dir>/rs_cifar8w_all_<tag>_logits_shap_<part>.bin
          the value is the true class's log-probability (a Brazil part: after the seven-class collapse;
          a diagnostic chip: the class the full coalition predicts); reports φ per group pooled and per class,
          the share of |φ|, the top group per chip, and the efficiency residual.
"""
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_score import EURO, SHARED, COLLAPSE, collapse  # noqa: E402

GROUPS = {"visible": [1, 2, 3], "red edge": [4, 5, 6], "NIR": [7, 12], "SWIR": [10, 11], "atmospheric": [0, 8, 9]}
NAMES = list(GROUPS)
NP = len(NAMES)
COALITIONS = list(range(1 << NP))          # bit g set = group g present


def make(args):
    lbl = np.fromfile(os.path.join(args.data, f"labels_{args.part}.bin"), dtype=np.int32)
    X = np.fromfile(os.path.join(args.data, f"{args.part}.bin"), dtype=np.float32).reshape(len(lbl), 13, 64, 64)
    rng = np.random.default_rng(args.seed)
    classes = np.unique(lbl)
    per = max(1, args.n // len(classes))
    pick = np.concatenate([rng.permutation(np.where(lbl == c)[0])[:per] for c in classes])
    pick.sort()
    out = np.empty((len(pick), len(COALITIONS), 13, 64, 64), dtype=np.float32)
    for j, m in enumerate(COALITIONS):
        keep = np.zeros(13, dtype=bool)
        for g, nm in enumerate(NAMES):
            if m >> g & 1:
                keep[GROUPS[nm]] = True
        out[:, j] = X[pick] * keep[None, :, None, None]
    out.reshape(-1, 13, 64, 64).tofile(os.path.join(args.data, f"shap_{args.part}.bin"))
    np.repeat(lbl[pick], len(COALITIONS)).astype(np.int32).tofile(os.path.join(args.data, f"labels_shap_{args.part}.bin"))
    np.savez(os.path.join(args.data, f"meta_shap_{args.part}.npz"), chip=pick, coalitions=np.array(COALITIONS), groups=json.dumps(GROUPS))
    print(f"shap_{args.part}: {len(pick)} chips × {len(COALITIONS)} coalitions = {len(pick) * len(COALITIONS)} records "
          f"({out.nbytes / 1e9:.2f} GB); score with: lake exe rs-bands arm=all eval score=shap_{args.part}")


def shapley(v):
    """Exact Shapley of a value table v[coalition mask] over NP players."""
    phi = np.zeros(NP)
    for g in range(NP):
        for m in COALITIONS:
            if m >> g & 1:
                continue
            s = bin(m).count("1")
            w = math.factorial(s) * math.factorial(NP - s - 1) / math.factorial(NP)
            phi[g] += w * (v[m | (1 << g)] - v[m])
    return phi


def score(args):
    meta = np.load(os.path.join(args.data, f"meta_shap_{args.part}.npz"))
    pick = meta["chip"]
    lbl_all = np.fromfile(os.path.join(args.data, f"labels_{args.part}.bin"), dtype=np.int32)
    lbl = lbl_all[pick]
    raw = np.fromfile(args.logits, dtype=np.float32)
    K = len(raw) // (len(pick) * len(COALITIONS))
    logits = raw.reshape(len(pick), len(COALITIONS), K)
    euro = args.part.startswith("eurosat")
    if not euro:
        logits = np.stack([collapse(logits[:, j]) for j in range(len(COALITIONS))], axis=1)
    names = EURO if euro else SHARED
    full = (1 << NP) - 1
    lp = logits - np.log(np.exp(logits - logits.max(axis=2, keepdims=True)).sum(axis=2, keepdims=True)) - logits.max(axis=2, keepdims=True)
    target = lbl.copy()
    diag = np.logical_and(not euro, lbl >= len(SHARED))
    target[diag] = logits[diag, full].argmax(axis=1)          # a diagnostic chip: the class the full coalition predicts
    v = lp[np.arange(len(pick)), :, target]                    # [n, 32] log-probability of the target class
    phi = np.stack([shapley(v[i]) for i in range(len(pick))])
    resid = phi.sum(axis=1) - (v[:, full] - v[:, 0])
    share = np.abs(phi) / np.abs(phi).sum(axis=1, keepdims=True).clip(1e-12)
    top = phi.argmax(axis=1)
    print(f"{args.part}: {len(pick)} chips, value = log p(target) ; v(N) − v(∅) mean {np.mean(v[:, full] - v[:, 0]):.3f} ; "
          f"efficiency residual max |Σφ − (v(N) − v(∅))| = {np.abs(resid).max():.2e}")
    print("  group         mean φ    share of |φ|   top group on")
    for g, nm in enumerate(NAMES):
        print(f"  {nm:12s}  {phi[:, g].mean():+7.3f}   {100 * share[:, g].mean():5.1f}%        {100 * (top == g).mean():5.1f}% of chips")
    print("  per class (mean φ):")
    for c in np.unique(target):
        m = target == c
        print(f"    {names[c]:22s} n={m.sum():4d}  " + "  ".join(f"{nm} {phi[m, g].mean():+.2f}" for g, nm in enumerate(NAMES)))
    if args.json:
        print(json.dumps(dict(part=args.part, n=int(len(pick)), groups=NAMES, mean_phi=phi.mean(axis=0).tolist(),
                              share=share.mean(axis=0).tolist(), top=np.bincount(top, minlength=NP).tolist(),
                              residual_max=float(np.abs(resid).max())), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "score"])
    ap.add_argument("--part", required=True)
    ap.add_argument("--data", default="data/rs")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logits")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.mode == "make":
        make(args)
    else:
        if not args.logits:
            sys.exit("score needs --logits")
        score(args)


if __name__ == "__main__":
    main()
