#!/usr/bin/env bash
# Crash-safe wrapper for the YOLOv1 Pets MIXED single+mosaic run (Workstream B,
# planning/yolo_demo_v2.md). Checkpoints every 2 epochs (cfg.checkpointEveryN);
# on any non-zero exit it resumes from the newest checkpoint written THIS session
# via LEAN_MLIR_INIT_LOAD / LEAN_MLIR_START_STEP (the ROCm-segfault survival path).
#
# Usage: run_yolo_mixed.sh [gpu] [data_dir]
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2
export PATH=/home/skoonce/lean/claude_max/lean4-jax/.venv/bin:$PATH
export IREE_BACKEND=rocm IREE_CHIP=gfx1100

GPU="${1:-0}"
DATA="${2:-data/pets_mixed}"
export HIP_VISIBLE_DEVICES="$GPU"
EXE=.lake/build/bin/yolov1-pets-train-bootstrap
PFX=.lake/build/resnet_34___yolov1_deep_head__pets_
LOG=runs/yolo_mixed_gpu${GPU}.log
MARKER=runs/.yolo_mixed_start
SPE=218          # steps/epoch = floor(3500 / 16); verified against the log's "N batches/epoch"
MAX=40           # restart cap

mkdir -p runs
touch "$MARKER"  # only checkpoints newer than this count as "this session"

for i in $(seq 1 $MAX); do
  latest=$(find "$(dirname "$PFX")" -name "$(basename "$PFX")_params_e*.bin" -newer "$MARKER" 2>/dev/null \
           | sed -E 's/.*_params_e([0-9]+)\.bin/\1/' | sort -n | tail -1)
  if [ -n "${latest:-}" ]; then
    export LEAN_MLIR_INIT_LOAD="${PFX}_params_e${latest}.bin"
    export LEAN_MLIR_START_STEP=$((latest * SPE))
    echo "=== [restart $i] RESUME from epoch $latest (step $LEAN_MLIR_START_STEP) $(date -u) ===" >> "$LOG"
  else
    unset LEAN_MLIR_INIT_LOAD 2>/dev/null || true
    unset LEAN_MLIR_START_STEP 2>/dev/null || true
    echo "=== [restart $i] FRESH start (R34 bootstrap) $(date -u) ===" >> "$LOG"
  fi
  "$EXE" "$DATA" >> "$LOG" 2>&1
  code=$?
  echo "=== [restart $i] trainer exited code=$code (was ~epoch ${latest:-0}) $(date -u) ===" >> "$LOG"
  if [ "$code" -eq 0 ]; then echo "=== DONE (clean exit) $(date -u) ===" >> "$LOG"; break; fi
  sleep 5
done
echo "WRAPPER_DONE $(date -u)" >> "$LOG"
