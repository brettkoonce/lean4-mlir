#!/bin/bash
# FC-width sweep of the verified MNIST CNN (conv@32 fixed, dense head …→d→d→10)
# via mnist-cnn-grid. 1-D sweep of the classifier head; conv extractor held fixed.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2
export PATH=/home/skoonce/lean/claude_max/lean4-jax/.venv/bin:$PATH
export IREE_BACKEND=rocm IREE_CHIP=gfx1100 HIP_VISIBLE_DEVICES=0 LEAN_MLIR_SEED=0
EPOCHS="${1:-10}"
BIN=.lake/build/bin/mnist-cnn-grid
RESULTS=runs/cnn_grid_results.tsv
mkdir -p runs/cnn_grid
: > "$RESULTS"
echo -e "d\tacc\tfloats\tms_per_ep" >> "$RESULTS"

for d in 8 16 32 64 128 256 512 1024 2048 4096; do
  LOG="runs/cnn_grid/cnn_fc${d}.log"
  echo "=== $(date -u +%H:%M:%S) cnn fc${d} (${EPOCHS} ep) ==="
  "$BIN" "$d" "$EPOCHS" data > "$LOG" 2>&1
  acc=$(grep -oP 'test_acc = \d+/\d+ = \K[0-9.]+' "$LOG" | tail -1)
  floats=$(grep -oP '\(\d+ params, \K[0-9]+' "$LOG" | head -1)
  msep=$(grep -oP '\(\K[0-9]+(?=ms\))' "$LOG" | tail -1)
  echo -e "${d}\t${acc:-NA}\t${floats:-NA}\t${msep:-NA}" >> "$RESULTS"
  echo "    -> acc=${acc:-NA}% floats=${floats:-NA} ${msep:-NA}ms/ep"
done
echo "=== sweep done $(date -u +%H:%M:%S) ==="
cat "$RESULTS"
