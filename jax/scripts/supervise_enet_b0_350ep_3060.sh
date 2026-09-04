#!/usr/bin/env bash
# Supervised 350-epoch EfficientNet-B0-ImageNet run on the 4x RTX 3060 box.
#
# Derived from supervise_enet_b0_350ep_4gpu_duty.sh, which is hardcoded to ares.
# Three differences, each of which is fatal on this box:
#   - DEVS=0,1,2,3      — ares' "AER-clean four" (0,2,3,4) name a card index this
#                         box does not have; CUDA_VISIBLE_DEVICES=4 aborts.
#   - PY_BIN            — `../.venv/bin/python` DOES NOT EXIST here. The pinned
#                         env is /home/skoonce/.venv-cuda (jax 0.11.1, xla_cuda13,
#                         tf 2.21.0, tfds 4.9.10).
#   - Rests are TEMPERATURE-driven, not every-30-epochs. This box has fans; the
#     fixed schedule would spend 5.5 h resting a box that may never get hot.
#     Same policy scripts/jobs/enet-default-4gpu.conf already sets for this
#     hardware ("prefer temperature-driven resting over a fixed epoch schedule").
#
# A rest is taken at the next EPOCH BOUNDARY after the trip, not immediately:
# LEAN_MLIR_CKPT_EVERY=1 means a full state lands every epoch, so waiting costs
# at most one epoch (~7 min) and loses no work. Resume is bit-for-bit via
# LEAN_MLIR_RESUME (params + opt state + EMA + BN + step), so the cosine LR
# schedule and the Adam/RMSProp moments are unaffected by any stop.
#
#     nohup bash jax/scripts/supervise_enet_b0_350ep_3060.sh &
set -u
cd "$(dirname "$0")/.." || exit 1

DEVS="${DEVS:-0,1,2,3}"
PY_BIN="${PY_BIN:-/home/skoonce/.venv-cuda/bin/python3}"
PY=.lake/build/generated_efficientnet_b0_imagenet_full.py
CKPT_BASE="${CKPT_BASE:-/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet}"
SPE=5004                                   # 4-GPU: 1281167 // 256
EPOCHS=350                                 # the CONSTANT the trainer anneals over
TEMP_MAX="${TEMP_MAX:-80}"                 # trip: any GPU at or above this
TEMP_RESUME="${TEMP_RESUME:-62}"           # resume once the hottest is back under
REST_MIN_SECS="${REST_MIN_SECS:-600}"      # floor on a rest, even if it cools fast
RUNLOG=/tmp/enet_b0_350ep_3060.log         # per-attempt trainer stdout
MASTER=/tmp/enet_b0_350ep_3060_master.log
FULLLOG="${FULLLOG:-${CKPT_BASE}_full.log}"
mkdir -p "$(dirname "$CKPT_BASE")"
MAX_ATTEMPTS=150

[ -x "$PY_BIN" ] || { echo "⛔ no python at $PY_BIN"; exit 1; }
[ -f "$PY" ]     || { echo "⛔ no trainer at $PY — run scripts/regen_jax_generated.sh sync"; exit 1; }

hottest() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1; }

# Newest full-state checkpoint epoch (0 = none). The pruner keeps the 3 newest
# .state.npz and save_train_state renames atomically, so the newest is complete.
last_state_epoch() {
  local hi=0 n
  for f in ${CKPT_BASE}_e*.state.npz; do
    [ -e "$f" ] || continue
    n=$(echo "$f" | sed -E 's/.*_e([0-9]+)\.state\.npz/\1/')
    [ "$n" -gt "$hi" ] && hi="$n"
  done
  echo "$hi"
}
last_state_path() {
  local ep="$1"; [ "$ep" -gt 0 ] && echo "${CKPT_BASE}_e${ep}.state.npz" || echo ""
}

echo "[sup] $(date '+%F %T') START EfficientNet-B0 ImageNet ${EPOCHS}ep · GPUs $DEVS · bf16 · batch 256 (4x64) · SPE $SPE · rest when any GPU >= ${TEMP_MAX}C" | tee -a "$MASTER"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt+1))

  LAST_EP=$(last_state_epoch)
  LAST_STATE=$(last_state_path "$LAST_EP")
  RESUME_ENV=()
  if [ -n "$LAST_STATE" ]; then
    RESUME_ENV=(LEAN_MLIR_RESUME="$LAST_STATE")
    echo "[sup] $(date '+%T') attempt $attempt: RESUME full state from epoch $LAST_EP ($LAST_STATE)" | tee -a "$MASTER"
  else
    echo "[sup] $(date '+%T') attempt $attempt: fresh start (no checkpoint)" | tee -a "$MASTER"
  fi

  : > "$RUNLOG"
  echo "===== $(date '+%F %T') full-run log =====" >> "$FULLLOG"
  START="$(date '+%Y-%m-%d %H:%M:%S')"

  env CUDA_VISIBLE_DEVICES=$DEVS \
      LEAN_MLIR_PARAMS_OUT="$CKPT_BASE" \
      LEAN_MLIR_CKPT_EVERY=1 \
      "${RESUME_ENV[@]}" \
      "$PY_BIN" -u "$PY" > >(tee -a "$FULLLOG" > "$RUNLOG") 2>&1 &
  PYPID=$!
  echo "[sup] $(date '+%T') launched PID=$PYPID" | tee -a "$MASTER"

  result="unknown"; hot_armed=0
  while kill -0 "$PYPID" 2>/dev/null; do
    if journalctl -k --since "$START" 2>/dev/null | grep -iE "BadTLP|AER:|Uncorrected|Fatal" | grep -qivE "no action required"; then
      echo "[sup] $(date '+%T') !!! AER detected — killing PID=$PYPID" | tee -a "$MASTER"
      kill -9 "$PYPID" 2>/dev/null; sleep 2
      result="aer"; break
    fi
    if grep -q "^Done\." "$RUNLOG" 2>/dev/null; then
      result="done"; break
    fi
    T=$(hottest)
    if [ -n "$T" ] && [ "$hot_armed" -eq 0 ] && [ "$T" -ge "$TEMP_MAX" ]; then
      hot_armed=1
      echo "[sup] $(date '+%T') 🌡  ${T}C >= ${TEMP_MAX}C — will rest at the next epoch checkpoint" | tee -a "$MASTER"
    fi
    # Stop only once a NEW epoch's state has landed: nothing in flight is lost.
    if [ "$hot_armed" -eq 1 ] && [ "$(last_state_epoch)" -gt "$LAST_EP" ]; then
      echo "[sup] $(date '+%T') 💤 epoch $(last_state_epoch) checkpoint landed at ${T}C — cooling to ${TEMP_RESUME}C" | tee -a "$MASTER"
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
    echo "[sup] $(date '+%T') ✅ TRAINING COMPLETE (${EPOCHS} epochs). final=${CKPT_BASE}.bin" | tee -a "$MASTER"
    exit 0
  fi

  pkill -9 -f "generated_efficientnet_b0_imagenet_full.py" 2>/dev/null
  if [ "$result" = "rest" ]; then
    rested=0
    while :; do
      sleep 30; rested=$((rested+30))
      T=$(hottest)
      [ "$rested" -ge "$REST_MIN_SECS" ] && [ -n "$T" ] && [ "$T" -lt "$TEMP_RESUME" ] && break
      [ "$rested" -ge 3600 ] && break     # hard cap: never rest more than an hour
    done
    echo "[sup] $(date '+%T') cooldown over after ${rested}s (now $(hottest)C) — resuming" | tee -a "$MASTER"
  else
    echo "[sup] $(date '+%T') attempt $attempt ended ($result); cooling 15s then resuming" | tee -a "$MASTER"
    sleep 15
  fi
done

echo "[sup] $(date '+%F %T') ⛔ hit MAX_ATTEMPTS=$MAX_ATTEMPTS — giving up" | tee -a "$MASTER"
exit 1
