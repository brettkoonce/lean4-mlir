#!/usr/bin/env bash
# Supervised 300-epoch ConvNeXt-T-ImageNet run on the 4 clean GPUs (0,2,3,4)
# with a thermal duty cycle: 30-minute rest every 30 epochs (9 rests).
# 300ep canonical/paper-faithful tier of the 80->300 ladder; the 80ep
# validation run hit 78.13%/94.05% and its curve was still climbing.
# Derived from supervise_convnext_t_80ep_4gpu_duty.sh; differences:
# - DEVS=0,2,3,4 (batch 256 = 4x64, SPE 5004)
# - Planned rests: when the epoch-30/-60 full-state checkpoint lands, the
#   trainer is killed, the box cools REST_SECS, then training resumes
#   bit-for-bit via LEAN_MLIR_RESUME (params + opt state + EMA + step),
#   so the cosine LR schedule and Adam moments are unaffected.
# - AER watchdog kept (unplanned kills auto-resume the same way).
set -u
cd "$(dirname "$0")/.." || exit 1

DEVS="0,2,3,4"
PY=.lake/build/generated_convnext_tiny_imagenet_full.py
CKPT_BASE=/home/skoonce/convnext_t300_4gpu/convnext_tiny_imagenet
SPE=5004                                   # 4-GPU: 1281167 // 256
REST_EPOCHS="30 60 90 120 150 180 210 240 270"
REST_SECS=1800
RUNLOG=/tmp/convnext_t_300ep_4gpu.log       # per-attempt trainer stdout
MASTER=/tmp/convnext_t_300ep_4gpu_master.log
FULLLOG="${FULLLOG:-${CKPT_BASE}_full.log}"
mkdir -p "$(dirname "$CKPT_BASE")"
MAX_ATTEMPTS=150

# Next rest epoch strictly ahead of the last completed epoch (999 = none left).
next_rest() {
  for e in $REST_EPOCHS; do
    if [ "$1" -lt "$e" ]; then echo "$e"; return; fi
  done
  echo 999
}

echo "[sup] $(date '+%F %T') START 300-epoch ConvNeXt-T on GPUs $DEVS (300ep canonical; duty cycle ${REST_SECS}s rest every 30 epochs)" | tee -a "$MASTER"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt+1))

  # Newest full-state checkpoint (preferred: exact resume). The pruner keeps
  # the 3 newest .state.npz, so if any exist the newest is complete (atomic
  # rename in save_train_state).
  LAST_STATE=""; LAST_EP=0
  for f in ${CKPT_BASE}_e*.state.npz; do
    [ -e "$f" ] || continue
    n=$(echo "$f" | sed -E 's/.*_e([0-9]+)\.state\.npz/\1/')
    if [ "$n" -gt "$LAST_EP" ]; then LAST_EP="$n"; LAST_STATE="$f"; fi
  done

  RESUME_ENV=()
  if [ -n "$LAST_STATE" ]; then
    RESUME_ENV=(LEAN_MLIR_RESUME="$LAST_STATE")
    echo "[sup] $(date '+%T') attempt $attempt: RESUME full state from epoch $LAST_EP ($LAST_STATE)" | tee -a "$MASTER"
  else
    echo "[sup] $(date '+%T') attempt $attempt: fresh start (no checkpoint)" | tee -a "$MASTER"
  fi
  REST_AT=$(next_rest "$LAST_EP")

  : > "$RUNLOG"
  echo "===== $(date '+%F %T') full-run log =====" >> "$FULLLOG"
  START="$(date '+%Y-%m-%d %H:%M:%S')"

  env CUDA_VISIBLE_DEVICES=$DEVS \
      LEAN_MLIR_PARAMS_OUT="$CKPT_BASE" \
      LEAN_MLIR_CKPT_EVERY=1 \
      "${RESUME_ENV[@]}" \
      ../.venv/bin/python -u "$PY" > >(tee -a "$FULLLOG" > "$RUNLOG") 2>&1 &
  PYPID=$!
  echo "[sup] $(date '+%T') launched PID=$PYPID (next rest after epoch $REST_AT)" | tee -a "$MASTER"

  result="unknown"
  while kill -0 "$PYPID" 2>/dev/null; do
    if journalctl -k --since "$START" 2>/dev/null | grep -iE "BadTLP|AER:|Uncorrected|Fatal" | grep -qivE "no action required"; then
      echo "[sup] $(date '+%T') !!! AER detected — killing PID=$PYPID" | tee -a "$MASTER"
      kill -9 "$PYPID" 2>/dev/null; sleep 2
      result="aer"; break
    fi
    if grep -q "^Done\." "$RUNLOG" 2>/dev/null; then
      result="done"; break
    fi
    # Planned rest: the epoch-$REST_AT full-state file is written atomically
    # right after that epoch's .bin, so its existence means it is safe to stop.
    if [ "$REST_AT" -ne 999 ] && [ -e "${CKPT_BASE}_e${REST_AT}.state.npz" ]; then
      echo "[sup] $(date '+%T') 💤 epoch $REST_AT checkpoint landed — stopping for ${REST_SECS}s cooldown" | tee -a "$MASTER"
      kill -9 "$PYPID" 2>/dev/null; sleep 2
      result="rest"; break
    fi
    sleep 5
  done
  if [ "$result" = "unknown" ]; then
    if grep -q "^Done\." "$RUNLOG" 2>/dev/null; then result="done"; else result="crash"; fi
  fi

  grep -E "^\[Epoch " "$RUNLOG" 2>/dev/null | tail -1 | sed "s/^/[sup]   last: /" | tee -a "$MASTER" >/dev/null

  if [ "$result" = "done" ]; then
    echo "[sup] $(date '+%T') ✅ TRAINING COMPLETE (300 epochs). final=${CKPT_BASE}.bin" | tee -a "$MASTER"
    exit 0
  fi

  pkill -9 -f "generated_convnext_tiny_imagenet_full.py" 2>/dev/null
  if [ "$result" = "rest" ]; then
    sleep "$REST_SECS"
    echo "[sup] $(date '+%T') cooldown over — resuming" | tee -a "$MASTER"
  else
    echo "[sup] $(date '+%T') attempt $attempt ended ($result); cooling 15s then resuming" | tee -a "$MASTER"
    sleep 15
  fi
done

echo "[sup] $(date '+%F %T') ⛔ hit MAX_ATTEMPTS=$MAX_ATTEMPTS — giving up" | tee -a "$MASTER"
exit 1
