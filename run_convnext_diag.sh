#!/bin/bash
# Diagnostic: is the ConvNeXt collapse the SGD path itself, or the SGD-lr × noise-variance
# interaction? GPU0 = σ=0 (pure SGD, no noise). GPU1 = σ=0.10 (moderate noise). Few epochs —
# the collapse was immediate, so 3 epochs distinguishes "training" (natural acc >> 10%) from
# "constant" (natural acc ≈ 10%). vmfb is cached, so both share the GPUs safely.
cd "$(dirname "$0")"
export PATH=$PWD/.venv/bin:$PATH
export IREE_BACKEND=rocm
export SMOOTH_EPOCHS=3 SMOOTH_EVAL_BATCHES=4 SMOOTH_N=500 SMOOTH_MAXCERT=15

( export HIP_VISIBLE_DEVICES=0 SMOOTH_SIGMA_MILLI=0
  echo "GPU0 σ=0.00 start $(date)"
  .lake/build/bin/convnext-smooth data > runs/diag_convnext_s000.log 2>&1
  echo "GPU0 σ=0.00 done $(date)" ) &
P0=$!

( export HIP_VISIBLE_DEVICES=1 SMOOTH_SIGMA_MILLI=100
  echo "GPU1 σ=0.10 start $(date)"
  .lake/build/bin/convnext-smooth data > runs/diag_convnext_s010.log 2>&1
  echo "GPU1 σ=0.10 done $(date)" ) &
P1=$!

wait $P0 $P1
echo "ALL DONE $(date)"
