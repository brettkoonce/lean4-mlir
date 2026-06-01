#!/usr/bin/env bash
# Auto-resume wrapper for the YOLOv1-VOC bootstrap run on mars (IREE/Lean path).
# Survives the intermittent ROCm segfaults by checkpointing every 2 epochs
# (cfg.checkpointEveryNEpochs) and resuming from the latest checkpoint via the
# trainer's LEAN_MLIR_INIT_LOAD / LEAN_MLIR_START_STEP env vars. Run in tmux.
#
# Resume detection uses `find -newer <marker>` so it only picks up checkpoints
# written *this session* — yesterday's stale e10..e80 (collapsed run) are
# ignored, and the first launch starts fresh from the R34 bootstrap.
set -u
cd /home/skoonce/lean/claude_max/lean4-jax
EXE=.lake/build/bin/yolov1-voc-train-bootstrap
PFX=.lake/build/resnet_34___yolov1_conv_head__person_voc_   # conv-head person detector prefix (trailing _ → double __)
RUNDIR="${1:-runs/2026-05-31-yolov1-voc-gradclip}"   # optional arg: run/log dir
DATA="${2:-data/voc2007}"                            # optional arg: data dir (e.g. data/voc2007_person)
LOG="$RUNDIR/train.log"
MARKER="$RUNDIR/.wrapper_start"
SPE=313           # steps/epoch = floor(5011 / 16)
MAX=30            # restart cap (80ep / ~6ep-per-crash ≈ 14 crashes; 30 = margin)
mkdir -p "$RUNDIR"
touch "$MARKER"   # only checkpoints newer than this count as "this session"

export IREE_BACKEND=rocm
export IREE_CHIP=gfx1100

for i in $(seq 1 $MAX); do
  latest=$(find "$(dirname "$PFX")" -name "$(basename "$PFX")_params_e*.bin" -newer "$MARKER" 2>/dev/null \
           | sed -E 's/.*_params_e([0-9]+)\.bin/\1/' | sort -n | tail -1)
  if [ -n "${latest:-}" ]; then
    export LEAN_MLIR_INIT_LOAD="${PFX}_params_e${latest}.bin"
    export LEAN_MLIR_START_STEP=$((latest * SPE))
    echo "=== [restart $i] RESUME from epoch $latest (step $LEAN_MLIR_START_STEP) ===" >> "$LOG"
  else
    unset LEAN_MLIR_INIT_LOAD 2>/dev/null || true
    unset LEAN_MLIR_START_STEP 2>/dev/null || true
    echo "=== [restart $i] FRESH start ===" >> "$LOG"
  fi
  "$EXE" "$DATA" >> "$LOG" 2>&1
  code=$?
  echo "=== [restart $i] trainer exited code=$code (was ~epoch ${latest:-0}) ===" >> "$LOG"
  if [ "$code" -eq 0 ]; then echo "=== DONE (clean exit) ===" >> "$LOG"; break; fi
  sleep 5
done
echo "WRAPPER_DONE" >> "$LOG"
