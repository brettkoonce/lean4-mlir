#!/usr/bin/env python3
"""Check the confusion-matrix region-Dice identity used by the eval harness.

`runTraining`'s segmentation eval reports Dice on the BraTS regions (WT/TC/ET)
by reading them off the `[NC*NC]` confusion matrix it already accumulates for
mIoU, rather than making a second pass over the val set:

    inter_R = sum_{g in R} sum_{p in R} C[g][p]
    |gt_R|  = sum_{g in R} sum_p C[g][p]
    |pr_R|  = sum_{p in R} sum_g C[g][p]
    Dice_R  = 2*inter_R / (|gt_R| + |pr_R|)

That is an identity, not an approximation — but it is the kind of identity that
is easy to typo (transposed index, region membership tested on the wrong axis)
and whose failure looks like a plausible number rather than a crash. So: build
random label fields, compute Dice the obvious per-pixel way, compute it the
confusion-matrix way, and require exact agreement.

Part 2 computes the ground-truth region voxel counts over the real val.bin.
Those are what the harness must print as `gt=` — they depend only on the data,
not on the model, so they pin the row-sum axis against real numbers.

Usage:
    python3 scripts/seg_region_dice_check.py [data/brats/val.bin]
"""
import sys

import numpy as np

SIZE = 240
MODALITIES = 4
NUM_CLASSES = 4

# Must match `bratsIO.segRegions` in LeanMlir/Train.lean. MSD's remap of the
# native BraTS 1/2/4 -> 2/1/3 is already applied by preprocess_brats.py, so
# these are MSD label ids: 1 = edema, 2 = non-enhancing/necrotic, 3 = enhancing.
REGIONS = {"WT": (1, 2, 3), "TC": (2, 3), "ET": (3,)}


def dice_direct(gt, pred, region):
    """Dice for a region, straight from the label fields."""
    g = np.isin(gt, region)
    p = np.isin(pred, region)
    den = g.sum() + p.sum()
    if den == 0:
        return 1.0
    return 2.0 * np.logical_and(g, p).sum() / den


def dice_from_confusion(conf, region):
    """Dice for a region, from conf[true][pred] — the harness's formula."""
    nc = conf.shape[0]
    inter = gt = pr = 0
    for g in range(nc):
        for p in range(nc):
            v = int(conf[g][p])
            if g in region and p in region:
                inter += v
            if g in region:
                gt += v
            if p in region:
                pr += v
    den = gt + pr
    if den == 0:
        return 1.0
    return 2.0 * inter / den


def check_identity():
    rng = np.random.default_rng(0)
    worst = 0.0
    trials = 0
    # Sweep class priors from uniform to brutally imbalanced, because the
    # imbalanced end is the only one this demo actually operates in.
    for alpha in (10.0, 1.0, 0.1, 0.02):
        for trial in range(25):
            prior = rng.dirichlet([alpha] * NUM_CLASSES)
            gt = rng.choice(NUM_CLASSES, size=4096, p=prior)
            # Correlate pred with gt so intersections are non-trivial, but keep
            # a tunable error rate so we exercise real confusion structure.
            pred = gt.copy()
            flip = rng.random(gt.shape) < rng.uniform(0.0, 1.0)
            pred[flip] = rng.choice(NUM_CLASSES, size=int(flip.sum()), p=prior)
            conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            np.add.at(conf, (gt, pred), 1)
            for name, region in REGIONS.items():
                a = dice_direct(gt, pred, region)
                b = dice_from_confusion(conf, region)
                worst = max(worst, abs(a - b))
                trials += 1
    print(f"part 1: confusion-matrix identity over {trials} region-trials")
    print(f"  max |direct - from_confusion| = {worst:.3e}")
    ok = worst < 1e-12
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check_degenerate():
    """The all-background predictor: what the collapsed checkpoint does."""
    rng = np.random.default_rng(1)
    gt = rng.choice(NUM_CLASSES, size=4096, p=[0.83, 0.10, 0.05, 0.02])
    pred = np.zeros_like(gt)
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(conf, (gt, pred), 1)
    print("part 2: all-background predictor (the collapse the harness must show)")
    ok = True
    for name, region in REGIONS.items():
        d = dice_from_confusion(conf, region)
        direct = dice_direct(gt, pred, region)
        ok &= (d == 0.0) and (direct == 0.0)
        print(f"  Dice {name} = {d}  (direct {direct})")
    print(f"  {'PASS' if ok else 'FAIL'} — collapse must read 0, not 1 (gt is non-empty)")
    return ok


def check_val_counts(path):
    """Ground-truth region counts over the real val set."""
    rec = MODALITIES * SIZE * SIZE + SIZE * SIZE
    with open(path, "rb") as f:
        count = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        print(f"part 3: ground-truth region counts over {path} ({count} slices)")
        hist = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _ in range(count):
            buf = f.read(rec)
            if len(buf) != rec:
                print("  FAIL: short read")
                return False
            m = np.frombuffer(buf[MODALITIES * SIZE * SIZE:], dtype=np.uint8)
            hist += np.bincount(m, minlength=NUM_CLASSES)
    total = int(hist.sum())
    print(f"  per-class voxels: " + "  ".join(
        f"c{i}={int(hist[i])}" for i in range(NUM_CLASSES)))
    print(f"  brain-vs-background sanity: background is {100.0*hist[0]/total:.1f}% of voxels")
    print("  the harness must print these as `gt=`:")
    for name, region in REGIONS.items():
        gt = int(sum(hist[c] for c in region))
        print(f"    val Dice {name}: gt={gt}")
    return True


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/brats/val.bin"
    ok = check_identity()
    print()
    ok &= check_degenerate()
    print()
    try:
        check_val_counts(path)
    except FileNotFoundError:
        print(f"part 3: skipped — {path} not found (run ./download_brats.sh)")
    print()
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
