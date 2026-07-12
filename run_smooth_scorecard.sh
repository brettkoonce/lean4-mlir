#!/bin/bash
# Fixed-protocol smoothing SCORECARD runs (planning/gaussian_smoothing_next.md 3c-data):
# first-100 test images (SMOOTH_STRIDE=1), sigma=0.5 only, n=10112 (default SMOOTH_N
# rounds up to 79*128), alpha=0.001. The per-image (count, n) columns land in the CSV
# and feed scripts/smooth_scorecard_gen.py -> kernel tail-check corpus.
# GPU0: cifar-smooth. GPU1: mnist-cnn-smooth then mnist-mlp-smooth.
cd "$(dirname "$0")"
export PATH=$PWD/.venv/bin:$PATH
export IREE_BACKEND=rocm
export SMOOTH_SIGMA_MILLI=500
export SMOOTH_MAXCERT=100
export SMOOTH_STRIDE=1

# keep the June frontier CSVs (different protocol: every-50th, all sigmas)
for s in mlp cnn cifar; do
  [ -f runs/smooth_${s}_radii.csv ] && cp -n runs/smooth_${s}_radii.csv runs/smooth_${s}_radii_frontier.csv.bak
done

( export HIP_VISIBLE_DEVICES=0
  echo "GPU0 cifar start $(date)"
  .lake/build/bin/cifar-smooth data > runs/smooth_cifar_scorecard.log 2>&1
  cp runs/smooth_cifar_radii.csv runs/smooth_cifar_scorecard.csv
  echo "GPU0 cifar done $(date)" ) &
P0=$!

( export HIP_VISIBLE_DEVICES=1
  echo "GPU1 cnn start $(date)"
  .lake/build/bin/mnist-cnn-smooth data > runs/smooth_cnn_scorecard.log 2>&1
  cp runs/smooth_cnn_radii.csv runs/smooth_cnn_scorecard.csv
  echo "GPU1 cnn done; mlp start $(date)"
  .lake/build/bin/mnist-mlp-smooth data > runs/smooth_mlp_scorecard.log 2>&1
  cp runs/smooth_mlp_radii.csv runs/smooth_mlp_scorecard.csv
  echo "GPU1 mlp done $(date)" ) &
P1=$!

wait $P0 $P1
echo "ALL DONE $(date)"
