#!/usr/bin/env bash
# Supervised 90-epoch MobileNetV2-ImageNet bf16 run on all 6 GPUs (0,1,2,3,4,5).
# Paper-faithful recipe: RMSProp + crop/flip only (AutoAugment OFF). See
# planning/jax_imagenet_sweep.md — re-run #1 (the old SGD+AA 68.3% number is stale).
# - Checkpoints every epoch (LEAN_MLIR_CKPT_EVERY=1) to $CKPT_base_e{N}.bin
# - AER watchdog: kills training the instant a PCIe BadTLP/Hardware Error
#   appears, before it can cascade into a host reset (the box hit AER 5x
#   during the ViT run — see reference_ares_pcie_aer).
# - Auto-resume: after an AER kill, reloads the newest checkpoint and
#   relaunches, continuing the cosine LR schedule (LEAN_MLIR_START_STEP).
# - Stops when training prints "Done." (epoch 90 complete) or after a
#   safety cap on restart attempts.
set -u
# Resolve the jax/ dir from this script's location — works on any checkout
# (ares klawd_max_power, mars claude_max, …) without a hardcoded path.
cd "$(dirname "$0")/.." || exit 1

DEVS="0,1,2,3,4,5"
PY=.lake/build/generated_mobilenet_v2_imagenet.py
CKPT_BASE=/home/skoonce/mnv2_imagenet_bf16          # -> _e{N}.bin per epoch, .bin final
SPE=5083                                            # 6-GPU: batch 252 (6x42) -> 1281167//252
RUNLOG=/tmp/mnv2_90ep_6gpu.log                      # full training stdout (current attempt)
MASTER=/tmp/mnv2_90ep_6gpu_master.log               # supervisor narration (persists across attempts)
MAX_ATTEMPTS=60

echo "[sup] $(date '+%F %T') START 90-epoch MNv2 bf16 on GPUs $DEVS (6-GPU, post-Gen3)" | tee -a "$MASTER"

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
    echo "[sup] $(date '+%T') ✅ TRAINING COMPLETE (90 epochs). final=${CKPT_BASE}.bin" | tee -a "$MASTER"
    exit 0
  fi

  echo "[sup] $(date '+%T') attempt $attempt ended ($result); cooling 15s then resuming" | tee -a "$MASTER"
  pkill -9 -f "generated_mobilenet_v2_imagenet.py" 2>/dev/null
  sleep 15
done

echo "[sup] $(date '+%T') ⛔ hit MAX_ATTEMPTS=$MAX_ATTEMPTS — giving up" | tee -a "$MASTER"
exit 1
