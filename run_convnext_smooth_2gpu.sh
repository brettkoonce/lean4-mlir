#!/bin/bash
# ConvNeXt-T (Imagenette 224²) randomized-smoothing certify, split across both gfx1100 GPUs.
# Same net on both GPUs ⇒ shared vmfb path, so the compileVmfb cache (skip-if-fresh) must be warm
# first (it is — pre-built); both processes then only READ the cached vmfb. σ split per GPU.
cd "$(dirname "$0")"
export PATH=$PWD/.venv/bin:$PATH
export IREE_BACKEND=rocm
export SMOOTH_EPOCHS=8 SMOOTH_EVAL_BATCHES=4 SMOOTH_N=2000 SMOOTH_MAXCERT=20

( export HIP_VISIBLE_DEVICES=0 SMOOTH_SIGMA_MILLI=250
  echo "GPU0 σ=0.25 start $(date)"
  .lake/build/bin/convnext-smooth data > runs/smooth_convnext_s025.log 2>&1
  echo "GPU0 σ=0.25 done $(date)" ) &
P0=$!

( export HIP_VISIBLE_DEVICES=1 SMOOTH_SIGMA_MILLI=500
  echo "GPU1 σ=0.5 start $(date)"
  .lake/build/bin/convnext-smooth data > runs/smooth_convnext_s050.log 2>&1
  echo "GPU1 σ=0.5 done $(date)" ) &
P1=$!

wait $P0 $P1
echo "ALL DONE $(date)"
