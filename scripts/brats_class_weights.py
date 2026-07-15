#!/usr/bin/env python3
"""Compute class-weight vectors for `LossKind.perPixelWeightedCE` from BraTS.

Reads the *training* split's masks (never val — weights are a modelling choice
fitted to the training distribution, and reading val to set them is a leak,
however mild) and prints weight vectors under several standard schemes, along
with the number that actually decides whether a scheme works:

    share_c = w_c·f_c / Σ_c w_c·f_c

the fraction of the total loss class c owns. Unweighted CE is the `w_c = 1` row,
and on BraTS it hands background ~97% of the loss — which is the whole reason
the net collapses to predicting background and nothing else.

Note the loss is **invariant to the overall scale of the weights** (the
reduction divides by Σ_k w_{y_k}; see LossKind.perPixelWeightedCE), so each
vector below is printed normalized to w_0 = 1 purely for readability. Only the
ratios are load-bearing.

Usage:
    python3 scripts/brats_class_weights.py [data/brats/train.bin]
"""
import sys

import numpy as np

SIZE = 240
MODALITIES = 4
NUM_CLASSES = 4
CLASS_NAMES = ["background", "edema", "non-enhancing", "enhancing"]


def class_histogram(path):
    """Voxel count per class over the split. Seeks past the image block of each
    record rather than reading it — masks are 1/5 of the file."""
    img_bytes = MODALITIES * SIZE * SIZE
    mask_bytes = SIZE * SIZE
    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    with open(path, "rb") as f:
        count = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        for i in range(count):
            f.seek(img_bytes, 1)
            buf = f.read(mask_bytes)
            if len(buf) != mask_bytes:
                sys.exit(f"short read at record {i}/{count}")
            hist += np.bincount(np.frombuffer(buf, dtype=np.uint8),
                                minlength=NUM_CLASSES)
    return count, hist


def shares(w, f):
    """Fraction of total loss each class owns under weights w."""
    wf = w * f
    return wf / wf.sum()


def report(name, w, f, note=""):
    w = w / w[0]                      # scale-invariant; normalize for reading
    sh = shares(w, f)
    print(f"\n  {name}{'  — ' + note if note else ''}")
    print("    " + "  ".join(f"{CLASS_NAMES[c]}: w={w[c]:8.2f} share={sh[c]*100:5.2f}%"
                             for c in range(NUM_CLASSES)))
    lean = ", ".join(f"{v:.4f}" for v in w)
    print(f"    Lean: [{lean}]")
    return w


def loss_floors(f):
    """What each loss scores for a few reference predictors.

    Reading a segmentation loss curve without these is how the CE row got
    misread once already: CE's epoch-1 loss of 0.1328 sits BELOW the 0.1417 of
    the best constant predictor, which looks like evidence of learning and is
    not -- the net got under the floor purely by being confident on
    trivially-separable background while its argmax never fired a tumour class
    (planning/brats_demo.md Workstream A). The floors are what make a loss
    number mean something.
    """
    NC = len(f)

    def wce_of(qfun, w):
        num = sum(f[c] * w[c] * (-np.log(max(qfun(c)[c], 1e-300))) for c in range(NC))
        den = sum(f[c] * w[c] for c in range(NC))
        return num / den

    ones = np.ones(NC)
    inv = 1.0 / f
    const = lambda q: (lambda c: q)
    uniform_q = np.ones(NC) / NC
    allbg = np.zeros(NC); allbg[0] = 1.0
    # Knows WHERE the tumour is but not which type: exact on background,
    # uniform over all NC at tumour pixels.
    loc = lambda c: (allbg if c == 0 else uniform_q)

    print("\nloss floors — what a given predictor scores under each loss:")
    print(f"\n  {'predictor':<38}{'plain CE':>12}{'weighted CE':>14}")
    print("  " + "-" * 64)
    for name, qf in (("predict the class PRIOR everywhere", const(f)),
                     ("predict UNIFORM everywhere", const(uniform_q)),
                     ("predict BACKGROUND everywhere", const(allbg)),
                     ("knows tumour location, uniform there", loc)):
        a = wce_of(qf, ones)
        b = wce_of(qf, inv)
        fmt = lambda v: (f"{v:>12.4f}" if v < 100 else f"{'inf':>12}")
        print(f"  {name:<38}{fmt(a)}{fmt(b):>14}")

    print(f"""
  The best CONSTANT predictor flips, and that is the whole mechanism:

    under plain CE     it is "predict the prior"  ({wce_of(const(f), ones):.4f}), i.e. mostly
                       background -- the doorway to the collapse, and cheaper
                       than uniform ({wce_of(const(uniform_q), ones):.4f}) by 10x. Descent walks straight in.
    under weighted CE  it is "predict UNIFORM"    ({wce_of(const(uniform_q), inv):.4f}), and predicting the
                       prior is now {wce_of(const(f), inv)/wce_of(const(uniform_q), inv):.1f}x WORSE ({wce_of(const(f), inv):.4f}). Predicting background
                       everywhere is unbounded. The trivial answer is not merely
                       discouraged, it is the worst place in the space.

  So on the weighted-CE curve the number to beat is {wce_of(const(uniform_q), inv):.4f} (any constant),
  and {wce_of(loc, inv):.4f} is roughly "found the tumour, hasn't typed it yet".

  Caveat, in the same spirit as the one above: these make the curve READABLE,
  not conclusive. No scalar in a training log can tell you a thin class
  survived -- only the per-class IoU / region Dice at eval can.""")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/brats/train.bin"
    print(f"reading masks from {path} ...")
    count, hist = class_histogram(path)
    total = int(hist.sum())
    f = hist / total
    print(f"\n{count} slices, {total} voxels")
    print("  per-class voxels: " + "  ".join(
        f"{CLASS_NAMES[c]}={int(hist[c])} ({f[c]*100:.3f}%)" for c in range(NUM_CLASSES)))

    print("\nweight schemes (normalized to w_background = 1):")

    report("uniform (= plain perPixelCE)", np.ones(NUM_CLASSES), f,
           "the status quo: background owns almost all of the loss")
    report("inverse sqrt frequency", 1.0 / np.sqrt(f), f,
           "conservative; tumour still a minority of the loss")
    inv = report("inverse frequency", 1.0 / f, f,
                 "every class contributes equally — Dice's stated goal, via a "
                 "gradient that does not vanish")

    # Median-frequency balancing (SegNet) is w_c = median(f)/f_c, which is
    # inverse frequency times a constant. Under a /N reduction that constant is
    # a real difference — it rescales every gradient. Under this loss's
    # weighted-mean reduction it cancels exactly, so the two schemes are not
    # merely similar, they are the same scheme. Printing it as a third option
    # would imply a choice that does not exist.
    med = float(np.median(f))
    assert np.allclose((med / f) / (med / f)[0], inv / inv[0]), \
        "median-freq should be inverse-freq up to scale"
    print("\n  median frequency balancing (SegNet) — omitted: it is inverse frequency")
    print(f"    times a constant (median f = {med:.6f}), and this loss is scale-invariant")
    print("    in the weights, so the two are identical here. Under a /N reduction they")
    print("    would differ; under Σw they cannot.")

    print("\n  Recommendation: inverse frequency.")
    print("  It is the scheme that makes the per-class shares exactly equal, which")
    print("  is precisely what soft Dice was reaching for and failed to deliver —")
    print("  Dice's gradient carries a p_i factor and vanishes on a collapsed")
    print("  class, while CE's is flat at p->0 (scripts/seg_dice_vanishing_grad_probe.py).")
    print("  The usual objection to inverse frequency is that a ~200x dynamic range")
    print("  destabilizes training. That objection assumes a /N reduction, where the")
    print("  weights inflate the gradient scale. This emitter divides by the sum of")
    print("  the weights actually present in the batch, so the loss stays a weighted")
    print("  MEAN, on the same scale as unweighted CE, and self-normalizes per batch.")
    lean = ", ".join(f"{v:.4f}" for v in inv)
    print(f"\n  Paste into demos/MainUnetBratsTrain.lean:\n    [{lean}]")

    loss_floors(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
