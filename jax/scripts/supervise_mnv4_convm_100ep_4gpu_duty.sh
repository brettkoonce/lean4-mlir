#!/usr/bin/env bash
# Supervised 100-epoch MobileNetV4-Conv-M ImageNet run on the 4 clean GPUs (0,2,3,4),
# Tier-2 reduced-reg recipe (LR 0.004 AdamW, eff-batch 4096 via grad-accum 8×micro-512,
# RandAug m9, LS 0.1, dropout 0.1, EMA). See planning/mnv4_imagenet.md.
#
# Derived from supervise_convnext_t_80ep_4gpu_duty.sh. MNv4-specific differences:
# - PY = generated_mobilenet_v4_imagenet.py ; SPE = 312 (1281167 // 4096, batch 4096).
# - JAX_COMPILATION_CACHE_DIR set: the cold-cache XLA compile is ~15 min (UIB has many
#   distinct conv shapes). Caching makes it a ONE-TIME cost, so every planned rest / AER
#   resume skips it. The cache is populated on the first run and reused thereafter. This
#   is the single most important addition vs the ConvNeXt template. (The train_step shapes
#   don't depend on epoch count, so a warm cache from any prior micro-512 4-GPU run hits.)
# - Thermal duty cycle: 30-min rest after epochs 33 and 66 (post-fan-repair the box ran a
#   100ep R50-A3 continuously for ~17h with 0 AER kills, so this is cheap insurance, not a
#   hard requirement — set REST_EPOCHS="" to run straight through).
# - AER watchdog kept (unplanned kills auto-resume bit-for-bit via LEAN_MLIR_RESUME).
#
# The benign `ncclCommRegister … Cuda failure 500 'named symbol not found'` warning WILL
# print — it is NCCL user-buffer registration falling back on consumer PCIe cards, not a
# failure. Collectives work (verified). See reference_klawd_nccl_compile.
set -u
cd "$(dirname "$0")/.." || exit 1

DEVS="0,2,3,4"
PY=.lake/build/generated_mobilenet_v4_imagenet.py
CKPT_BASE=/home/skoonce/mnv4_convm_100ep/mobilenet_v4_imagenet   # -> _e{N}.bin + _e{N}.state.npz per epoch, .bin final
SPE=312                                     # 4-GPU: 1281167 // 4096 (eff-batch 4096)
REST_EPOCHS="33 66"                         # planned thermal rests (empty = run straight through)
REST_SECS=1800                              # 30-min cooldown
JAX_CACHE=/home/skoonce/.jax_cache          # persistent XLA compile cache (skip the ~15-min recompile on resume)
RUNLOG=/tmp/mnv4_convm_100ep_4gpu.log       # per-attempt trainer stdout
MASTER=/tmp/mnv4_convm_100ep_4gpu_master.log
FULLLOG="${FULLLOG:-${CKPT_BASE}_full.log}"
mkdir -p "$(dirname "$CKPT_BASE")" "$JAX_CACHE"
MAX_ATTEMPTS=60

# Next rest epoch strictly ahead of the last completed epoch (999 = none left).
next_rest() {
  for e in $REST_EPOCHS; do
    if [ "$1" -lt "$e" ]; then echo "$e"; return; fi
  done
  echo 999
}

echo "[sup] $(date '+%F %T') START 100-epoch MNv4-Conv-M on GPUs $DEVS (duty cycle: ${REST_SECS}s rest after epochs ${REST_EPOCHS:-none})" | tee -a "$MASTER"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt+1))

  # Newest full-state checkpoint (exact bit-for-bit resume). The pruner keeps the 3 newest
  # .state.npz; the newest is complete (atomic rename in save_train_state).
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
      JAX_COMPILATION_CACHE_DIR="$JAX_CACHE" \
      LEAN_MLIR_PARAMS_OUT="$CKPT_BASE" \
      LEAN_MLIR_CKPT_EVERY=1 \
      TFDS_DATA_DIR="${TFDS_DATA_DIR:-/home/skoonce/tensorflow_datasets}" \
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
    # Planned rest: the epoch-$REST_AT full-state file is written atomically right after
    # that epoch's .bin, so its existence means it is safe to stop.
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
    echo "[sup] $(date '+%T') ✅ TRAINING COMPLETE (100 epochs). final=${CKPT_BASE}.bin" | tee -a "$MASTER"
    exit 0
  fi

  pkill -9 -f "generated_mobilenet_v4_imagenet.py" 2>/dev/null
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
