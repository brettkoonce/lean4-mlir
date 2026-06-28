#!/bin/bash
# Randomized-smoothing certify across BOTH gfx1100 GPUs (HIP_VISIBLE_DEVICES pins each stream).
# GPU0: cifar-smooth (heaviest single job). GPU1: mnist-cnn-smooth then mnist-mlp-smooth.
cd "$(dirname "$0")"
export PATH=$PWD/.venv/bin:$PATH
export IREE_BACKEND=rocm

( export HIP_VISIBLE_DEVICES=0
  echo "GPU0 cifar start $(date)"
  .lake/build/bin/cifar-smooth data > runs/smooth_cifar.log 2>&1
  echo "GPU0 cifar done $(date)" ) &
P0=$!

( export HIP_VISIBLE_DEVICES=1
  echo "GPU1 cnn start $(date)"
  .lake/build/bin/mnist-cnn-smooth data > runs/smooth_cnn.log 2>&1
  echo "GPU1 cnn done; mlp start $(date)"
  .lake/build/bin/mnist-mlp-smooth data > runs/smooth_mlp.log 2>&1
  echo "GPU1 mlp done $(date)" ) &
P1=$!

wait $P0 $P1
echo "ALL DONE $(date)"
