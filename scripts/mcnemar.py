#!/usr/bin/env python3
"""McNemar's test between two per-example correctness bitmaps — the PAIRED accuracy comparison.

Reads two `LEAN_MLIR_DUMP_CORRECT` bitmaps (one byte per validation image, 1 = top-1 correct, in
eval order) and answers: **is model B actually better than model A, or is the gap inside the
validation set's own resolution?**

⭐⭐ **WHY NOT JUST COMPARE THE TWO ACCURACIES.** Because a single accuracy on a finite val set has
a confidence interval, and on these sets it is wide enough to swallow most of the gaps that get
quoted:

    Imagenette,  n =  3,925, p ≈ 0.85  ->  95% CI  ±1.11 pt
    ImageNet,    n = 50,000, p ≈ 0.77  ->  95% CI  ±0.37 pt

⛔ **A worked example from this repo, and it is a headline.** RSB-A3 is reported as the verified
path BEATING its JAX reference, 77.43% vs 77.22%. That gap is 0.21 pt — **0.79σ** treated as two
independent measurements, i.e. not significant as stated.

But the two were scored on the SAME 50,000 images, so treating them as independent throws away
almost everything. The gap is 105 net label flips; McNemar looks only at the images where the two
models DISAGREE and asks whether the flips are lopsided:

    chi2 = (b - c)^2 / (b + c)      b = A right & B wrong,  c = A wrong & B right

With 105 net flips that clears p < 0.05 as long as the two models disagree on fewer than ~2,900 of
the 50,000 — which two R50s trained on the same recipe comfortably do. **So the claim is probably
fine and has never been tested.** This script is what tests it.

⚠⚠ **WHAT THIS DOES NOT ANSWER.** It compares two TRAINED MODELS, not two recipes. "Does the head
LayerNorm help ConvNeXt" is a statement about the distribution over seeds, and no paired test on
one pair of runs can answer it — that needs n runs per arm and a mean ± std. McNemar tells you
whether *these two checkpoints* differ on *this* val set; it is silent about whether a re-roll of
the seed would reverse the sign.

⚠ Both bitmaps must come from the same val set in the same order. They will, if both were produced
by `scoreCheckpoint` on the same `dataDir` — the eval loader is deterministic and θ does not move
during scoring. Mismatched lengths are refused rather than truncated.

    lake build convnext-imagenet-verified
    LEAN_MLIR_DUMP_CORRECT=/tmp/a .lake/build/bin/... --score   # model A
    LEAN_MLIR_DUMP_CORRECT=/tmp/b .lake/build/bin/... --score   # model B
    scripts/mcnemar.py /tmp/a.bin /tmp/b.bin
"""
import argparse, math, sys
import numpy as np


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="bitmap of model A")
    ap.add_argument("b", help="bitmap of model B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    A = np.fromfile(args.a, dtype=np.uint8).astype(bool)
    B = np.fromfile(args.b, dtype=np.uint8).astype(bool)
    if A.shape != B.shape:
        sys.exit(f"length mismatch: {args.a} has {A.size}, {args.b} has {B.size} — these are not "
                 f"the same val set (or not the same order)")
    n = A.size
    ka, kb = int(A.sum()), int(B.sum())
    la, lb = wilson(ka, n), wilson(kb, n)
    print(f"  n = {n}")
    print(f"  {args.label_a:10s} {ka}/{n} = {100*ka/n:.2f}%   95% CI {la[0]:.2f}–{la[1]:.2f}")
    print(f"  {args.label_b:10s} {kb}/{n} = {100*kb/n:.2f}%   95% CI {lb[0]:.2f}–{lb[1]:.2f}")
    print(f"  unpaired gap: {100*(kb-ka)/n:+.2f} pt"
          f"   (the CIs {'OVERLAP — unpaired, this is inconclusive' if la[1] > lb[0] and lb[1] > la[0] else 'are disjoint'})")

    b = int((A & ~B).sum())    # A right, B wrong
    c = int((~A & B).sum())    # A wrong, B right
    both = int((A & B).sum())
    neither = int((~A & ~B).sum())
    print(f"\n  paired table:  both right {both}   both wrong {neither}   "
          f"only {args.label_a} {b}   only {args.label_b} {c}")
    print(f"  discordant: {b + c} of {n} ({100*(b+c)/n:.1f}%) — the only images the test looks at")
    if b + c == 0:
        print("  the two models are IDENTICAL on this set; nothing to test")
        return 0

    # ⚠ Exact binomial, not the chi-square approximation. The chi2 form is unreliable when the
    # discordant count is small, which is exactly the regime two near-identical checkpoints land in.
    k = min(b, c)
    p_exact = min(1.0, 2.0 * sum(math.comb(b + c, i) for i in range(k + 1)) / 2.0 ** (b + c))
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)      # with continuity correction
    print(f"  McNemar exact two-sided p = {p_exact:.4g}   (chi2 w/ continuity = {chi2:.2f})")
    better = args.label_b if c > b else args.label_a
    if p_exact < 0.05:
        print(f"  ✓ SIGNIFICANT at p<0.05 — {better} is better on this val set, and the finite-set")
        print(f"    CI above is NOT the right reason to doubt it (the comparison is paired)")
    else:
        print(f"  ✗ NOT significant at p<0.05 — the {abs(c-b)} net flips do not separate these two")
        print(f"    models on {n} images, whatever the two point estimates look like")
    print("\n  ⚠ This is about these two CHECKPOINTS, not about the recipe. A claim of the form")
    print("    'change X is worth Y points' needs n seeds per arm, not a paired test on one pair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
