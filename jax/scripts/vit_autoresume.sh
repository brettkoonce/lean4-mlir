#!/usr/bin/env bash
# Auto-resume wrapper for the ViT-Tiny ImageNet run on mars: survives the
# intermittent ROCm segfaults (~epoch 8-11) by checkpointing every 2 epochs
# and resuming from the latest checkpoint. Uses the generated trainer's
# built-in env-var resume mechanism (no codegen change). Run in tmux.
set -u
cd /home/skoonce/lean/claude_max/lean4-jax
PY=.venv/bin/python3
SCRIPT=.lake/build/generated_vit_tiny_imagenet.py
CKPT=.lake/build/vit_adamw_ckpt
RUNDIR=runs/2026-05-31-vit-adamw-resumable
LOG="$RUNDIR/train.log"
SPE=2502          # steps/epoch = floor(1,281,167 / 512)
MAX=25            # restart cap (80ep / ~8ep-per-crash ≈ 10 crashes; 25 = margin)
mkdir -p "$RUNDIR"

export LD_PRELOAD=/opt/rocm/lib/librccl.so.1
export PYTHONUNBUFFERED=1
export LEAN_MLIR_PARAMS_OUT="$CKPT"
export LEAN_MLIR_CKPT_EVERY=2

for i in $(seq 1 $MAX); do
  latest=$(ls ${CKPT}_e*.bin 2>/dev/null | sed -E 's/.*_e([0-9]+)\.bin/\1/' | sort -n | tail -1)
  if [ -n "${latest:-}" ]; then
    export LEAN_MLIR_INIT_LOAD="${CKPT}_e${latest}.bin"
    export LEAN_MLIR_START_STEP=$((latest * SPE))
    echo "=== [restart $i] RESUME from epoch $latest (step $LEAN_MLIR_START_STEP) ===" >> "$LOG"
  else
    unset LEAN_MLIR_INIT_LOAD 2>/dev/null || true
    unset LEAN_MLIR_START_STEP 2>/dev/null || true
    echo "=== [restart $i] FRESH start ===" >> "$LOG"
  fi
  "$PY" -u "$SCRIPT" >> "$LOG" 2>&1
  code=$?
  echo "=== [restart $i] python exited code=$code (was ~epoch ${latest:-0}) ===" >> "$LOG"
  if [ "$code" -eq 0 ]; then echo "=== DONE (clean exit) ===" >> "$LOG"; break; fi
  sleep 5
done
echo "WRAPPER_DONE" >> "$LOG"
