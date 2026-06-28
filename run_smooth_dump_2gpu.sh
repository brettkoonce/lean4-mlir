#!/bin/bash
# Re-run MNIST/CIFAR smoothing with the per-image radius dump (→ runs/smooth_<slug>_radii.csv),
# split across both gfx1100 GPUs (distinct slugs ⇒ distinct vmfb, safe to run concurrently).
# Deterministic seeds ⇒ numbers match the committed tightened run; this just also dumps the CSVs.
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
