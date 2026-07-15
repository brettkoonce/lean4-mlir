#!/usr/bin/env python3
"""How much class balance can foreground oversampling actually buy on BraTS?

Asked BEFORE building it, because the answer decides whether it is worth
building — the same move that made `seg_grad_scorecard.py` worth more than the
GPU-week it replaced.

nnU-Net forces ~33% of training patches to contain foreground. That lever is
enormous when most patches are pure background. But `preprocess_brats.py` keeps
only slices with >=1 tumour voxel, so on this dataset EVERY training slice
already contains foreground and the nnU-Net rule is a no-op by construction.

The lever that remains is weaker and different: slices vary enormously in how
much tumour they hold (a polar cross-section is a few voxels; an equatorial one
is thousands), so we can bias sampling toward the tumour-RICH slices. This
measures the ceiling on that: what class balance is reachable, and therefore how
it compares to the levers already built (wce's static 196x, focal's dynamic one).

Usage:
    python3 scripts/brats_oversample_probe.py [data/brats/train.bin]
"""
import sys

import numpy as np

SIZE = 240
MODALITIES = 4
NUM_CLASSES = 4
CLASS_NAMES = ["background", "edema", "non-enhancing", "enhancing"]


def per_slice_histograms(path):
    """[N, NUM_CLASSES] voxel counts per slice. Seeks past each image block."""
    img_bytes = MODALITIES * SIZE * SIZE
    mask_bytes = SIZE * SIZE
    out = []
    with open(path, "rb") as f:
        count = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        for _ in range(count):
            f.seek(img_bytes, 1)
            buf = f.read(mask_bytes)
            out.append(np.bincount(np.frombuffer(buf, dtype=np.uint8),
                                   minlength=NUM_CLASSES))
    return np.array(out, dtype=np.int64)


def balance(hist_sum):
    return hist_sum / hist_sum.sum()


def report(name, f, note=""):
    print(f"\n  {name}{'  — ' + note if note else ''}")
    print("    " + "  ".join(f"{CLASS_NAMES[c]}={100*f[c]:6.3f}%" for c in range(NUM_CLASSES)))
    return f


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/brats/train.bin"
    print(f"reading per-slice masks from {path} ...")
    H = per_slice_histograms(path)
    n = H.shape[0]
    tumour = H[:, 1:].sum(axis=1)
    frac = tumour / (SIZE * SIZE)

    print(f"\n{n} slices. Per-slice tumour fraction:")
    for q in (0, 10, 25, 50, 75, 90, 100):
        print(f"    p{q:<3d} = {100*np.percentile(frac, q):7.3f}%")
    print(f"\n  slices with ZERO tumour: {int((tumour == 0).sum())}"
          f"  <- nnU-Net's 'force 33% foreground' rule has nothing to fix here")

    base = report("uniform sampling (status quo)", balance(H.sum(axis=0)))
    base_rare = base[3]

    # Two-bucket oversampling: with probability `f`, draw from the tumour-rich
    # bucket (top `top` fraction by burden); otherwise draw uniformly. Expected
    # balance is the mixture of the two buckets' mean histograms.
    print("\n  two-bucket oversampling — draw a fraction `f` of each batch from")
    print("  the top-`top` slices by tumour burden, rest uniform:")
    order = np.argsort(-tumour)
    rows = []
    for top in (0.5, 0.25, 0.1):
        rich = order[:int(n * top)]
        rich_mean = H[rich].mean(axis=0)
        all_mean = H.mean(axis=0)
        for f in (0.33, 0.5, 0.67):
            mix = f * rich_mean + (1 - f) * all_mean
            b = balance(mix)
            rows.append((top, f, b))
            print(f"    top={top:<5} f={f:<5} -> " +
                  "  ".join(f"{CLASS_NAMES[c][:4]}={100*b[c]:6.3f}%"
                            for c in range(NUM_CLASSES)) +
                  f"   enhancing x{b[3]/base_rare:.2f}")

    # The ceiling: sample ONLY the richest slices. Nothing this lever can do
    # beats this, and it is not a usable configuration (it throws away most of
    # the training set), so it bounds the whole idea.
    best = max(rows, key=lambda r: r[2][3])
    ceil_b = balance(H[order[:int(n * 0.1)]].mean(axis=0))
    print(f"\n  CEILING (train on ONLY the richest 10% — not a real config, a bound):")
    print("    " + "  ".join(f"{CLASS_NAMES[c][:4]}={100*ceil_b[c]:6.3f}%"
                             for c in range(NUM_CLASSES)) +
          f"   enhancing x{ceil_b[3]/base_rare:.2f}")

    print(f"""
VERDICT

  Best practical setting moves enhancing tumour {100*base_rare:.3f}% -> {100*best[2][3]:.3f}%
  of voxels, a factor of {best[2][3]/base_rare:.2f}x. Even the ceiling — training on
  nothing but the richest decile, which is not a real configuration — is
  {ceil_b[3]/base_rare:.2f}x.

  Compare the levers already built and FD-verified:
    wce        196x   (exactly w_3/w_0; static, live from step 0)
    focal      ~4e6x  at a collapsed net (dynamic; a no-op at init)
    prior-bias ~19000x for focal at step 0 (one bias vector)

  So foreground oversampling is a ~2x lever where the loss-side ones are 100x
  to 10000x. That is not an argument against it — it is orthogonal, it composes
  with all of them, and 2x is 2x. It IS an argument against it being a headline
  arm of the ablation, and against spending the demo's remaining GPU on it
  before the arms that move the number by two orders of magnitude more.

  The reason it is weak here is structural, not incidental: preprocess_brats.py
  already filtered to slices with >=1 tumour voxel, so the big win nnU-Net gets
  (most patches are pure background; force a third to contain something) was
  banked at preprocessing time. What is left is reweighting within an
  already-filtered set.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
