#!/usr/bin/env bash
# Supervised RSB-A3 ResNet-50 ImageNet run (100ep, 160px train / 224px eval)
# with LOSSLESS suspend/resume + a THERMAL DUTY-CYCLE cooldown.
#
# RSB-A3 = timm "ResNet Strikes Back" A3 (100ep, LAMB + BCE + mixup/cutmix,
# train@160 / test@224, no EMA, no stochastic depth) -> ~78.1% top-1. The
# committed short config (regenerate via: resnet50-imagenet short) lives in
# generated_resnet50_imagenet_short.py.
#
# Resume is full-state: on restart it reloads the newest <CKPT>_e{N}.state.npz
# (params + LAMB m/v/t + running-BN buffers + global step) via LEAN_MLIR_RESUME,
# so the trajectory continues BIT-FOR-BIT.
#
# Two reasons training stops and resumes:
#   1. AER watchdog — kills on a PCIe BadTLP/Hardware Error before it cascades
#      into a host reset, then auto-resumes from the last full-state ckpt.
#   2. THERMAL COOLDOWN — the klawd box gets flaky ~24h into *continuous* load,
#      so after epochs in COOLDOWN_AT (25/50/75) we stop, let the GPUs idle for
#      COOLDOWN_SECS (30min), then resume. CKPT_EVERY=5 guarantees a full-state
#      ckpt exists exactly at each cooldown boundary, so the pause is lossless.
set -u

# ── box config (override via env) ────────────────────────────────────────────
BACKEND="${BACKEND:-cuda}"                   # cuda = 4x 4060Ti (this box) | rocm = mars
JAX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${VENV_PY:-/home/skoonce/lean/klawd_max_power/lean4-jax/.venv/bin/python}"
CKPT_BASE="${CKPT_BASE:-/home/skoonce/r50_rsb_a3_imagenet}"   # -> _e{N}.bin + _e{N}.state.npz
PY=.lake/build/generated_resnet50_imagenet_short.py
RUNLOG=/tmp/r50_a3_imagenet.log              # full training stdout (current attempt)
MASTER=/tmp/r50_a3_imagenet_master.log       # supervisor narration (persists across attempts)
# Cumulative trainer stdout for the WHOLE run --- appended live (never truncated),
# kept next to the checkpoints so it survives /tmp clears and host resets, not just
# resumes. RUNLOG stays per-attempt so the detection greps below do not match stale
# lines from an earlier attempt.
FULLLOG="${FULLLOG:-${CKPT_BASE}_full.log}"
mkdir -p "$(dirname "$CKPT_BASE")"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-120}"
CKPT_EVERY="${CKPT_EVERY:-5}"
COOLDOWN_AT="${COOLDOWN_AT:-25 50 75}"       # epochs after which to pause
COOLDOWN_SECS="${COOLDOWN_SECS:-1800}"       # 30 min GPUs-idle cooldown

cd "$JAX_DIR" || { echo "no jax dir: $JAX_DIR"; exit 1; }

# Per-backend device + RCCL env. CUDA masks the two ares/klawd cards that storm
# PCIe AER (idx 1,5); ROCm 2-GPU needs librccl preloaded.
if [ "$BACKEND" = "rocm" ]; then
  DEV_ENV=(HIP_VISIBLE_DEVICES=0,1 LD_PRELOAD=/opt/rocm/lib/librccl.so.1)
else
  DEV_ENV=(CUDA_VISIBLE_DEVICES=0,2,3,4)
fi

echo "[sup] $(date '+%F %T') START R50-RSB-A3 ($BACKEND); ckpt=$CKPT_BASE; every=$CKPT_EVERY; cooldown@[$COOLDOWN_AT] ${COOLDOWN_SECS}s; jax_dir=$JAX_DIR" | tee -a "$MASTER"

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

  # Next thermal-cooldown boundary strictly after the epoch we're resuming from.
  NEXT_COOL=""
  for c in $COOLDOWN_AT; do
    if [ "$c" -gt "$LAST_EP" ]; then NEXT_COOL="$c"; break; fi
  done

  : > "$RUNLOG"
  echo "===== $(date '+%F %T') full-run log =====" >> "$FULLLOG"
  START="$(date '+%Y-%m-%d %H:%M:%S')"

  env "${DEV_ENV[@]}" \
      LEAN_MLIR_PARAMS_OUT="$CKPT_BASE" \
      LEAN_MLIR_CKPT_EVERY="$CKPT_EVERY" \
      "${RESUME_ENV[@]}" \
      "$VENV_PY" -u "$PY" > >(tee -a "$FULLLOG" > "$RUNLOG") 2>&1 &
  PYPID=$!
  echo "[sup] $(date '+%T') launched PID=$PYPID (next cooldown @epoch ${NEXT_COOL:-none})" | tee -a "$MASTER"

  result="unknown"
  while kill -0 "$PYPID" 2>/dev/null; do
    if journalctl -k --since "$START" 2>/dev/null | grep -qiE "BadTLP|Hardware Error|AER:"; then
      echo "[sup] $(date '+%T') !!! AER detected — killing PID=$PYPID" | tee -a "$MASTER"
      kill -9 "$PYPID" 2>/dev/null; sleep 2
      result="aer"; break
    fi
    # Thermal cooldown: the cooldown-boundary full-state ckpt has been written.
    if [ -n "$NEXT_COOL" ] && grep -q "_e${NEXT_COOL}\.state\.npz" "$RUNLOG" 2>/dev/null; then
      echo "[sup] $(date '+%T') ❄️  epoch $NEXT_COOL ckpt written — pausing for cooldown" | tee -a "$MASTER"
      kill -9 "$PYPID" 2>/dev/null; sleep 2
      result="cooldown"; break
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

  pkill -9 -f "generated_resnet50_imagenet_short.py" 2>/dev/null

  if [ "$result" = "cooldown" ]; then
    echo "[sup] $(date '+%T') ❄️  GPUs idle — cooling ${COOLDOWN_SECS}s before resuming past epoch $NEXT_COOL" | tee -a "$MASTER"
    sleep "$COOLDOWN_SECS"
    echo "[sup] $(date '+%T') cooldown done — resuming" | tee -a "$MASTER"
    continue
  fi

  echo "[sup] $(date '+%T') attempt $attempt ended ($result); cooling 15s then resuming" | tee -a "$MASTER"
  sleep 15
done

echo "[sup] $(date '+%T') ⛔ hit MAX_ATTEMPTS=$MAX_ATTEMPTS — giving up" | tee -a "$MASTER"
exit 1
