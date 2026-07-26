#!/bin/bash
# FC-head width sweep of the verified cifar8-BN CNN (AdamW), SPLIT OVER 2 GPUs.
# GPU 0 and GPU 1 each take half the widths (interleaved small/large) and run in parallel.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2
export PATH=/home/skoonce/lean/claude_max/lean4-jax/.venv/bin:$PATH
export IREE_BACKEND=rocm IREE_CHIP=gfx1100 LEAN_MLIR_SEED=0
EPOCHS="${1:-25}"
BIN=.lake/build/bin/cifar8-bn-grid
mkdir -p runs/cifar8bn_grid

# one GPU's worth of widths, sequential; writes a per-GPU results file
run_gpu () {
  local gpu="$1"; shift
  local widths=("$@")
  local res="runs/cifar8bn_grid/results_gpu${gpu}.tsv"
  : > "$res"
  for d in "${widths[@]}"; do
    local log="runs/cifar8bn_grid/cifar8bn_fc${d}.log"
    rm -f ".lake/build/cifar8_bn_${d}_adam_ckpt.bin"*   # fresh run
    echo "=== $(date -u +%H:%M:%S) [gpu${gpu}] cifar8bn fc${d} (${EPOCHS} ep) ==="
    HIP_VISIBLE_DEVICES="$gpu" "$BIN" "$d" "$EPOCHS" data > "$log" 2>&1
    local acc floats msep
    acc=$(grep -oP 'test_acc = \d+/\d+ = \K[0-9.]+' "$log" | tail -1)
    floats=$(grep -oP '\(\d+ params, \K[0-9]+' "$log" | head -1)
    msep=$(grep -oP '\(\K[0-9]+(?=ms\))' "$log" | tail -1)
    echo -e "${d}\t${acc:-NA}\t${floats:-NA}\t${msep:-NA}" >> "$res"
    echo "    [gpu${gpu}] -> fc${d} acc=${acc:-NA}% floats=${floats:-NA}"
  done
}

# interleave so each GPU gets a mix of cheap (small d) and expensive (large d) points
run_gpu 0 8 32 128 512 2048 &
PID0=$!
run_gpu 1 16 64 256 1024 4096 &
PID1=$!
wait $PID0 $PID1

# merge, sorted by width
RESULTS=runs/cifar8bn_grid_results.tsv
echo -e "d\tacc\tfloats\tms_per_ep" > "$RESULTS"
cat runs/cifar8bn_grid/results_gpu0.tsv runs/cifar8bn_grid/results_gpu1.tsv | sort -n >> "$RESULTS"
echo "=== sweep done $(date -u +%H:%M:%S) ==="
cat "$RESULTS"
