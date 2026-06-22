#!/usr/bin/env bash
# Supervised RSB-A2 ResNet-50 ImageNet run with LOSSLESS suspend/resume.
#
# RSB-A2 = timm "ResNet Strikes Back" A2 (300ep, LAMB + BCE + mixup/cutmix +
# RandAugment m7-mstd0.5-inc1 + 3x repeated-aug + stochastic depth) -> 79.8%
# top-1. The committed config (jax/MainResnet50Imagenet.lean) is 300ep; this
# script is schedule-agnostic and stops when training prints "Done."
#
# Resume is full-state: on restart it reloads the newest <CKPT>_e{N}.state.npz
# (params + LAMB m/v/t + EMA shadow + running-BN buffers + global step) via
# LEAN_MLIR_RESUME, so the trajectory continues BIT-FOR-BIT.
#
# Watchdog kills training on a PCIe AER (BadTLP/Hardware Error) before it can
# cascade into a host reset, then auto-resumes from the last full-state ckpt.
#
# Cloned from supervise_vit_80ep.sh. NB R50 is conv-bound: bf16 conv is a
# CUDA/cuDNN win only (the committed config sets bf16Conv=false for ROCm), so
# the CUDA box (ares, BACKEND=cuda) is the faster home for the full run.
set -u

# ── box config (override via env) ────────────────────────────────────────────
BACKEND="${BACKEND:-rocm}"                   # rocm = 2x 7900 XTX (this box) | cuda = ares
JAX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${VENV_PY:-/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python}"
CKPT_BASE="${CKPT_BASE:-/home/skoonce/r50_rsb_a2_imagenet}"     # -> _e{N}.bin + _e{N}.state.npz
PY=.lake/build/generated_resnet50_imagenet.py
RUNLOG=/tmp/r50_imagenet.log                 # full training stdout (current attempt)
MASTER=/tmp/r50_imagenet_master.log          # supervisor narration (persists across attempts)
MAX_ATTEMPTS="${MAX_ATTEMPTS:-120}"

cd "$JAX_DIR" || { echo "no jax dir: $JAX_DIR"; exit 1; }

# Per-backend device + RCCL env. ROCm 2-GPU needs librccl preloaded (TF's bundled
# NCCL otherwise shadows jaxlib's RCCL and the first all-reduce dies). CUDA masks
# the two ares cards that storm PCIe AER (idx 1,5).
if [ "$BACKEND" = "rocm" ]; then
  DEV_ENV=(HIP_VISIBLE_DEVICES=0,1 LD_PRELOAD=/opt/rocm/lib/librccl.so.1)
else
  DEV_ENV=(CUDA_VISIBLE_DEVICES=0,2,3,4)
fi

echo "[sup] $(date '+%F %T') START R50-RSB-A2 ($BACKEND); ckpt=$CKPT_BASE; jax_dir=$JAX_DIR" | tee -a "$MASTER"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt+1))

  # Newest full-state checkpoint to resume from (if any).
  LAST_STATE=""; LAST_EP=0
  for f in ${CKPT_BASE}_e*.state.npz; do
    [ -e "$f" ] || continue
    n=$(echo "$f" | sed -E 's/.*_e([0-9]+)\.state\.npz/\1/')
    if [ "$n" -gt "$LAST_EP" ]; then LAST_EP="$n"; LAST_STATE="$f"; fi
  done

  RESUME_ENV=()
  if [ -n "$LAST_STATE" ]; then
    RESUME_ENV=(LEAN_MLIR_RESUME="$LAST_STATE")
    echo "[sup] $(date '+%T') attempt $attempt: LOSSLESS RESUME from epoch $LAST_EP ($LAST_STATE)" | tee -a "$MASTER"
  else
    echo "[sup] $(date '+%T') attempt $attempt: fresh start (no .state.npz yet)" | tee -a "$MASTER"
  fi

  : > "$RUNLOG"
  START="$(date '+%Y-%m-%d %H:%M:%S')"

  env "${DEV_ENV[@]}" \
      LEAN_MLIR_PARAMS_OUT="$CKPT_BASE" \
      LEAN_MLIR_CKPT_EVERY="${CKPT_EVERY:-1}" \
      "${RESUME_ENV[@]}" \
      "$VENV_PY" -u "$PY" > "$RUNLOG" 2>&1 &
  PYPID=$!
  echo "[sup] $(date '+%T') launched PID=$PYPID" | tee -a "$MASTER"

  result="unknown"
  while kill -0 "$PYPID" 2>/dev/null; do
    if journalctl -k --since "$START" 2>/dev/null | grep -qiE "BadTLP|Hardware Error|AER:"; then
      echo "[sup] $(date '+%T') !!! AER detected — killing PID=$PYPID" | tee -a "$MASTER"
      kill -9 "$PYPID" 2>/dev/null; sleep 2
      result="aer"; break
    fi
    if grep -q "^Done\." "$RUNLOG" 2>/dev/null; then result="done"; break; fi
    sleep 5
  done
  if [ "$result" = "unknown" ]; then
    if grep -q "^Done\." "$RUNLOG" 2>/dev/null; then result="done"; else result="crash"; fi
  fi

  grep -E "^\[Epoch " "$RUNLOG" 2>/dev/null | tail -1 | sed "s/^/[sup]   last: /" | tee -a "$MASTER" >/dev/null

  if [ "$result" = "done" ]; then
    echo "[sup] $(date '+%T') ✅ TRAINING COMPLETE. final=${CKPT_BASE}.bin" | tee -a "$MASTER"
    exit 0
  fi

  echo "[sup] $(date '+%T') attempt $attempt ended ($result); cooling 15s then resuming" | tee -a "$MASTER"
  pkill -9 -f "generated_resnet50_imagenet.py" 2>/dev/null
  sleep 15
done

echo "[sup] $(date '+%T') ⛔ hit MAX_ATTEMPTS=$MAX_ATTEMPTS — giving up" | tee -a "$MASTER"
exit 1
