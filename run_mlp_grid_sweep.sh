#!/bin/bash
# Width sweep of the verified MNIST MLP (784→d₁→d₂→10) via mnist-mlp-grid.
# Diagonal (d₁=d₂) for the ROI curve + off-diagonal corners for 2D coverage.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2
export PATH=/home/skoonce/lean/claude_max/lean4-jax/.venv/bin:$PATH
export IREE_BACKEND=rocm IREE_CHIP=gfx1100 HIP_VISIBLE_DEVICES=0 LEAN_MLIR_SEED=0
EPOCHS="${1:-12}"
BIN=.lake/build/bin/mnist-mlp-grid
RESULTS=runs/mlp_grid_results.tsv
mkdir -p runs/mlp_grid
: > "$RESULTS"
echo -e "d1\td2\tacc\tfloats\tms_per_ep" >> "$RESULTS"

# diagonal (the neurons axis) + off-diagonal {8,512,4096}² corners
PAIRS="8,8 16,16 32,32 64,64 128,128 256,256 512,512 1024,1024 2048,2048 4096,4096 \
8,512 8,4096 512,8 512,4096 4096,8 4096,512"

for p in $PAIRS; do
  d1="${p%,*}"; d2="${p#*,}"
  LOG="runs/mlp_grid/mlp_${d1}x${d2}.log"
  echo "=== $(date -u +%H:%M:%S) mlp ${d1}x${d2} (${EPOCHS} ep) ==="
  "$BIN" "$d1" "$d2" "$EPOCHS" data > "$LOG" 2>&1
  # final-epoch accuracy + param floats + last-epoch ms
  acc=$(grep -oP 'test_acc = \d+/\d+ = \K[0-9.]+' "$LOG" | tail -1)
  floats=$(grep -oP '\(\d+ params, \K[0-9]+' "$LOG" | head -1)
  msep=$(grep -oP '\(\K[0-9]+(?=ms\))' "$LOG" | tail -1)
  echo -e "${d1}\t${d2}\t${acc:-NA}\t${floats:-NA}\t${msep:-NA}" >> "$RESULTS"
  echo "    -> acc=${acc:-NA}% floats=${floats:-NA} ${msep:-NA}ms/ep"
done
echo "=== sweep done $(date -u +%H:%M:%S) ==="
cat "$RESULTS"
