#!/usr/bin/env bash
# Supervised 500-epoch MobileNetV4-Conv-M ImageNet run on the 4 clean GPUs (0,2,3,4),
# Tier-3 PAPER-FAITHFUL recipe (`full`): LR 0.004 AdamW, eff-batch 4096 (grad-accum
# 8×micro-512), RandAug m15, LS 0.1, dropout 0.2, wd 0.1, EMA 0.9999, dropPath 0.075,
# running-BN eval. Paper Conv-M non-distilled target ≈ 79.9% top-1. See
# planning/mnv4_imagenet.md.
#
# Same machinery as supervise_mnv4_convm_100ep_4gpu_duty.sh; differences:
# - PY = generated_mobilenet_v4_imagenet_full.py (500ep, full regularization).
# - CKPT_BASE under mnv4_convm_500ep/.
# - Thermal rests every 30 epochs (matching the ConvNeXt-300ep convention), for a
#   ~3.5-day run. Bit-for-bit resume via LEAN_MLIR_RESUME (params + Adam + EMA + BN +
#   step), so cosine LR / dropPath ramp / running-BN stats all continue exactly.
# - MAX_ATTEMPTS=200 (long run → more restart opportunities).
#
# JAX_COMPILATION_CACHE_DIR keeps the ~15-min XLA compile a one-time cost across the
# ~16 planned rests + any AER resumes. The benign `ncclCommRegister … Cuda failure 500`
# warning prints and is harmless (see reference_klawd_nccl_compile).
set -u
cd "$(dirname "$0")/.." || exit 1

DEVS="0,2,3,4"
PY=.lake/build/generated_mobilenet_v4_imagenet_full.py
CKPT_BASE=/home/skoonce/mnv4_convm_500ep/mobilenet_v4_imagenet
SPE=312                                     # 4-GPU: 1281167 // 4096 (eff-batch 4096)
REST_EPOCHS="30 60 90 120 150 180 210 240 270 300 330 360 390 420 450 480"
REST_SECS=1800                              # 30-min cooldown
JAX_CACHE=/home/skoonce/.jax_cache          # persistent XLA compile cache
RUNLOG=/tmp/mnv4_convm_500ep_4gpu.log
MASTER=/tmp/mnv4_convm_500ep_4gpu_master.log
FULLLOG="${FULLLOG:-${CKPT_BASE}_full.log}"
mkdir -p "$(dirname "$CKPT_BASE")" "$JAX_CACHE"
MAX_ATTEMPTS=200

next_rest() {
  for e in $REST_EPOCHS; do
    if [ "$1" -lt "$e" ]; then echo "$e"; return; fi
  done
  echo 999
}

echo "[sup] $(date '+%F %T') START 500-epoch MNv4-Conv-M (paper full recipe) on GPUs $DEVS (rest ${REST_SECS}s every 30ep)" | tee -a "$MASTER"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt+1))

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
    echo "[sup] $(date '+%T') ✅ TRAINING COMPLETE (500 epochs). final=${CKPT_BASE}.bin" | tee -a "$MASTER"
    exit 0
  fi

  pkill -9 -f "generated_mobilenet_v4_imagenet_full.py" 2>/dev/null
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
