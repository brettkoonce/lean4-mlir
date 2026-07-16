#!/usr/bin/env bash
# run_brats_ablation.sh — reproduce the BraTS loss ablation and its money slide.
#
# The demo's thesis is the thin-class collapse, taken seriously: on brain-tumour
# MRI the class you care about (enhancing tumour) is 0.5% of voxels, and the
# choice of loss + init + LR schedule is the difference between a model that
# predicts nothing, one that paints the whole brain, and one that segments.
#
# This script trains the three arms that tell that story and renders them into
# one figure. Full write-up: planning/brats_demo.md (read "State of play" first).
#
# Prerequisites:
#   ./download_brats.sh                 # MSD Task01, ~7.6 GB, no account needed
#   lake build unet-brats-train brats-predict
#   an IREE GPU backend (this box: IREE_BACKEND=rocm on gfx1100)
#
# Cost: ~40 min/epoch on one gfx1100. The three arms below at 10 epochs are
# ~20 h total, or ~7 h if you run them one per GPU (they don't contend — each
# tags its own artifacts, see NetSpec.buildTag). Drop to `EPOCHS=3` for a smoke.

set -euo pipefail
cd "$(dirname "$0")"

DATA="${1:-data/brats}"
EPOCHS="${2:-10}"
GPU="${IREE_GPU:-0}"
export IREE_BACKEND="${IREE_BACKEND:-rocm}"
FIG="demos/figures/brats_ce_vs_wce_vs_fix.ppm"

run () {  # run <label> <args...>
  echo "=== training: $* ==="
  HIP_VISIBLE_DEVICES="$GPU" ./.lake/build/bin/unet-brats-train "$DATA" "$EPOCHS" "$@"
}

# ── Arm 1: plain per-pixel cross-entropy — the collapse ──────────────────────
# Predicts background on every pixel. mIoU ~0.243, which is the score of a model
# that has learned nothing (predict-background-everywhere scores 0.2434). CE at
# 10 epochs is WORSE than at 1: the collapse is an absorbing state, not
# underfitting. This is the baseline the other two have to beat.
run ce

# ── Arm 2: inverse-frequency weighted CE — the inversion ─────────────────────
# Amplifies the rare class 196x. It escapes the collapse and overshoots into the
# mirror failure: ~99% enhancing recall at ~2% precision, painting ~29% of every
# brain as tumour. The weighting is doing exactly what we asked — w3/w0 = 196
# prices one miss at 196 false alarms — just not what we wanted.
run wce

# ── Arm 3: the fix — sqrt-freq weights + prior-bias init + cosine LR ──────────
# All three levers, and all three are needed (planning/brats_demo.md B'):
#   * sqrt-frequency weights (β=0.5): amplify 14x, not 196x — the exchange rate
#     that finds the tumour instead of painting the brain. The usable band is a
#     narrow cliff near β=0.5.
#   * prior-bias init (pb): start the head predicting the class prior, i.e. on
#     the tumour side of the knife-edge. Without it, cosine LR commits to the
#     nearest wide basin, which from a cold start is background (it collapses).
#   * cosine LR (cos): damp the oscillation. At constant LR these weights carve
#     a narrow, unstable basin and the model rotates which class it predicts
#     every epoch; decaying LR settles it in.
run wcesqrt cos pb

# ── The money slide: three arms, one figure, identical slices ────────────────
# T1gd | ground truth | +ce | +wce | +fix. Same brains, same backdrop; the only
# thing that differs across the prediction columns is the training recipe.
echo "=== rendering $FIG ==="
HIP_VISIBLE_DEVICES="$GPU" ./.lake/build/bin/brats-predict \
  arm=ce,wce,wcesqrt_pb_cos "$FIG"

echo
echo "wrote $FIG"
echo "convert to PNG:  convert $FIG ${FIG%.ppm}.png"
echo "per-class IoU + WT/TC/ET Dice for each arm are in the training logs above;"
echo "the eval runs every epoch (TrainConfig.evalEveryNEpochs := 1)."
