#!/usr/bin/env bash
# Supervised 80-epoch ConvNeXt-T-ImageNet bf16 run on the 4 clean GPUs (0,2,3,4).
# This is the *validation* tier — bump EPOCHS to 300 in MainConvNeXtImagenet.lean
# and re-emit for the real run (then point this at the same PY).
# - Checkpoints every epoch (LEAN_MLIR_CKPT_EVERY=1) to $CKPT_base_e{N}.bin
# - AER watchdog: kills training the instant a PCIe BadTLP/Hardware Error
#   appears, before it can cascade into a host reset (the box hit AER 5x
#   during the ViT run — see reference_ares_pcie_aer).
# - Auto-resume: after an AER kill, reloads the newest checkpoint and
#   relaunches, continuing the cosine LR schedule (LEAN_MLIR_START_STEP).
# - Stops when training prints "Done." (epoch 80 complete) or after a
#   safety cap on restart attempts.
set -u
# Resolve the jax/ dir from this script's location — works on any checkout
# (ares klawd_max_power, mars claude_max, …) without a hardcoded path.
cd "$(dirname "$0")/.." || exit 1

DEVS="0,2,3,4"
PY=.lake/build/generated_convnext_tiny_imagenet.py
CKPT_BASE=/home/skoonce/convnext_tiny_imagenet_bf16          # -> _e{N}.bin per epoch, .bin final
SPE=5004                                            # steps per epoch (batch 256 = 4x64 -> 5004, same as R34)
RUNLOG=/tmp/convnext_t_80ep.log                           # full training stdout (current attempt)
MASTER=/tmp/convnext_t_80ep_master.log                    # supervisor narration (persists across attempts)
MAX_ATTEMPTS=60

echo "[sup] $(date '+%F %T') START 80-epoch ConvNeXt-T bf16 on GPUs $DEVS" | tee -a "$MASTER"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt+1))

  # Find newest per-epoch checkpoint to resume from (if any).
  LAST_CKPT=""; LAST_EP=0
  for f in ${CKPT_BASE}_e*.bin; do
    [ -e "$f" ] || continue
    n=$(echo "$f" | sed -E 's/.*_e([0-9]+)\.bin/\1/')
    if [ "$n" -gt "$LAST_EP" ]; then LAST_EP="$n"; LAST_CKPT="$f"; fi
  done

  RESUME_ENV=()
  if [ -n "$LAST_CKPT" ]; then
    START_STEP=$(( LAST_EP * SPE ))
    RESUME_ENV=(LEAN_MLIR_INIT_LOAD="$LAST_CKPT" LEAN_MLIR_START_STEP="$START_STEP")
    echo "[sup] $(date '+%T') attempt $attempt: RESUME from epoch $LAST_EP ($LAST_CKPT), start_step=$START_STEP" | tee -a "$MASTER"
  else
    echo "[sup] $(date '+%T') attempt $attempt: fresh start (no checkpoint)" | tee -a "$MASTER"
  fi

  : > "$RUNLOG"
  START="$(date '+%Y-%m-%d %H:%M:%S')"

  # NOTE: use `env` so env-var words from the array expansion are applied as
  # assignments (a quoted "${arr[@]}" prefix is otherwise treated as a command).
  env CUDA_VISIBLE_DEVICES=$DEVS \
      LEAN_MLIR_PARAMS_OUT="$CKPT_BASE" \
      LEAN_MLIR_CKPT_EVERY=1 \
      "${RESUME_ENV[@]}" \
      ../.venv/bin/python -u "$PY" > "$RUNLOG" 2>&1 &
  PYPID=$!
  echo "[sup] $(date '+%T') launched PID=$PYPID" | tee -a "$MASTER"

  # Monitor loop: AER -> kill; "Done." -> success; process-exit -> inspect.
  result="unknown"
  while kill -0 "$PYPID" 2>/dev/null; do
    if journalctl -k --since "$START" 2>/dev/null | grep -qiE "BadTLP|Hardware Error|AER:"; then
      echo "[sup] $(date '+%T') !!! AER detected — killing PID=$PYPID" | tee -a "$MASTER"
      kill -9 "$PYPID" 2>/dev/null; sleep 2
      result="aer"; break
    fi
    if grep -q "^Done\." "$RUNLOG" 2>/dev/null; then
      result="done"; break
    fi
    sleep 5
  done
  # If loop exited because process died on its own:
  if [ "$result" = "unknown" ]; then
    if grep -q "^Done\." "$RUNLOG" 2>/dev/null; then result="done"; else result="crash"; fi
  fi

  # Record latest epoch line seen this attempt.
  grep -E "^\[Epoch " "$RUNLOG" 2>/dev/null | tail -1 | sed "s/^/[sup]   last: /" | tee -a "$MASTER" >/dev/null

  if [ "$result" = "done" ]; then
    echo "[sup] $(date '+%T') ✅ TRAINING COMPLETE (80 epochs). final=${CKPT_BASE}.bin" | tee -a "$MASTER"
    exit 0
  fi

  echo "[sup] $(date '+%T') attempt $attempt ended ($result); cooling 15s then resuming" | tee -a "$MASTER"
  pkill -9 -f "generated_convnext_tiny_imagenet.py" 2>/dev/null
  sleep 15
done

echo "[sup] $(date '+%T') ⛔ hit MAX_ATTEMPTS=$MAX_ATTEMPTS — giving up" | tee -a "$MASTER"
exit 1
