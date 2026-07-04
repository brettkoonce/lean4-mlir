#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ViT-Tiny DeiT-Ti / ImageNet-1k · 300 epochs · PACED (suspend/resume)
#
# Trains in segments of PAUSE_EVERY epochs, then idles the GPUs for PAUSE_SECS
# to cap sustained load (thermal/power headroom), then resumes bit-for-bit.
#
# HOW IT SUSPENDS: the generated trainer already writes a LOSSLESS full-state
# checkpoint every epoch (params + Adam m/v/t + EMA shadow + global step) to
# <CKPT_BASE>_e{N}.state.npz. So "suspend" = let a segment run to its boundary
# epoch, wait for that epoch's .state.npz to land on disk, kill the process,
# sleep, then relaunch with LEAN_MLIR_RESUME=<newest state> — the trajectory
# continues exactly (Adam moments + EMA survive), not just the weights.
#
# Segment boundaries are at epochs PAUSE_EVERY, 2·PAUSE_EVERY, ... The FINAL
# segment runs to completion ("Done.") with no trailing pause. With the defaults
# (300 ep, every 30) that is 9 pauses (epochs 30,60,…,270) × 30 min = 4.5h idle.
#
# A crash / AER before a boundary is NOT a scheduled pause: the loop just
# resumes immediately from the newest checkpoint (no 30-min wait).
#
# ⚠️  This script does NOT start until you run it:
#        nohup bash run_vit_deit_300ep_paced.sh > runs/vit_deit_300ep_paced_master.log 2>&1 &
#     Tail progress:  tail -f runs/vit_deit_300ep_paced.log
# ─────────────────────────────────────────────────────────────────────────────
set -u

# ── config (override via env) ────────────────────────────────────────────────
REPO="$(cd "$(dirname "$0")" && pwd)"
JAX_DIR="$REPO/jax"
VENV_PY="${VENV_PY:-/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python}"
PY="${PY:-.lake/build/generated_vit_tiny_imagenet.py}"      # relative to JAX_DIR
CKPT_BASE="${CKPT_BASE:-/home/skoonce/vit_tiny_imagenet_bf16}"  # -> _e{N}.bin + _e{N}.state.npz
TFDS_DATA_DIR="${TFDS_DATA_DIR:-/home/skoonce/tensorflow_datasets}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-300}"     # must match the committed vitTinyImagenetConfig
PAUSE_EVERY="${PAUSE_EVERY:-30}"        # pause after every N epochs
PAUSE_SECS="${PAUSE_SECS:-1800}"        # 30 minutes idle
POLL_SECS="${POLL_SECS:-15}"            # how often to check the log for the boundary
MAX_ATTEMPTS="${MAX_ATTEMPTS:-400}"     # safety cap on total segments/retries

RUNLOG="$REPO/runs/vit_deit_300ep_paced.log"          # cumulative training stdout
MASTER="$REPO/runs/vit_deit_300ep_paced_master.log"   # supervisor narration

mkdir -p "$REPO/runs"
cd "$JAX_DIR" || { echo "no jax dir: $JAX_DIR"; exit 1; }

# ROCm 2-GPU needs librccl preloaded (TF's bundled NCCL otherwise shadows
# jaxlib's RCCL and the first all-reduce dies). Same env the supervise scripts use.
DEV_ENV=(HIP_VISIBLE_DEVICES=0,1 LD_PRELOAD=/opt/rocm/lib/librccl.so.1)

log(){ echo "[pace] $(date '+%F %T') $*" | tee -a "$MASTER"; }

# ── preflight ────────────────────────────────────────────────────────────────
[ -f "$PY" ]        || { log "FATAL: generated trainer missing: $JAX_DIR/$PY (build+run vit-tiny-imagenet once to emit it)"; exit 1; }
[ -x "$VENV_PY" ]   || { log "FATAL: venv python missing: $VENV_PY"; exit 1; }
[ -e /opt/rocm/lib/librccl.so.1 ] || { log "FATAL: librccl.so.1 not found — 2-GPU will die"; exit 1; }
grep -q "WD \* msk \* p" "$PY" || log "WARN: trainer has no WD mask — is this the DeiT-faithful build?"
grep -q "_do_val"        "$PY" || log "WARN: trainer has no val-gating — every-epoch validation"
if ls "${CKPT_BASE}"_e*.state.npz >/dev/null 2>&1; then
  log "NOTE: existing checkpoints at ${CKPT_BASE}_e*.state.npz — this run will RESUME from the newest, not start fresh."
fi

log "START  ViT DeiT ${TOTAL_EPOCHS}ep · pause ${PAUSE_SECS}s every ${PAUSE_EVERY} ep · ckpt=$CKPT_BASE · jax=$JAX_DIR"

# newest completed epoch = highest N among <base>_e{N}.state.npz (0 if none)
newest_epoch(){
  local last=0 f n
  for f in "${CKPT_BASE}"_e*.state.npz; do
    [ -e "$f" ] || continue
    n=$(echo "$f" | sed -E 's/.*_e([0-9]+)\.state\.npz/\1/')
    [ "$n" -gt "$last" ] && last="$n"
  done
  echo "$last"
}

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt+1))

  LAST_EP="$(newest_epoch)"
  if [ "$LAST_EP" -ge "$TOTAL_EPOCHS" ]; then
    log "✅ COMPLETE — newest checkpoint is epoch $LAST_EP (final=${CKPT_BASE}.bin)"; exit 0
  fi

  # next boundary strictly greater than what's done; clamp to TOTAL for the last segment
  NEXT_PAUSE=$(( (LAST_EP / PAUSE_EVERY + 1) * PAUSE_EVERY ))
  [ "$NEXT_PAUSE" -gt "$TOTAL_EPOCHS" ] && NEXT_PAUSE="$TOTAL_EPOCHS"
  FINAL_SEG=0; [ "$NEXT_PAUSE" -ge "$TOTAL_EPOCHS" ] && FINAL_SEG=1

  RESUME_ENV=()
  if [ "$LAST_EP" -gt 0 ]; then
    RESUME_ENV=(LEAN_MLIR_RESUME="${CKPT_BASE}_e${LAST_EP}.state.npz")
    log "attempt $attempt: RESUME from epoch $LAST_EP → run to epoch $NEXT_PAUSE$( [ "$FINAL_SEG" = 1 ] && echo ' (final, to Done.)' )"
  else
    log "attempt $attempt: FRESH start → run to epoch $NEXT_PAUSE"
  fi

  env "${DEV_ENV[@]}" \
      TFDS_DATA_DIR="$TFDS_DATA_DIR" PYTHONUNBUFFERED=1 \
      LEAN_MLIR_PARAMS_OUT="$CKPT_BASE" LEAN_MLIR_CKPT_EVERY=1 \
      "${RESUME_ENV[@]}" \
      "$VENV_PY" -u "$PY" >> "$RUNLOG" 2>&1 &
  PID=$!
  log "attempt $attempt: launched PID=$PID"

  # Watch until: (a) the boundary epoch's state file is logged → scheduled pause;
  # (b) "Done." → finished; (c) the process dies → crash, resume with no pause.
  result="crash"
  while kill -0 "$PID" 2>/dev/null; do
    if [ "$FINAL_SEG" = 0 ] && grep -q "saved full train state -> ${CKPT_BASE}_e${NEXT_PAUSE}\.state\.npz" "$RUNLOG" 2>/dev/null; then
      result="boundary"; break
    fi
    if grep -q "^Done\." "$RUNLOG" 2>/dev/null; then result="done"; break; fi
    sleep "$POLL_SECS"
  done

  # Stop the trainer (idempotent; harmless if it already exited on its own).
  kill "$PID" 2>/dev/null; sleep 2; kill -9 "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  # Fallback sweep (the pattern matches the python argv, never this bash script).
  pkill -9 -f "$PY" 2>/dev/null; sleep 3

  grep -E "^\[Epoch " "$RUNLOG" 2>/dev/null | tail -1 | sed 's/^/[pace]   last epoch line: /' | tee -a "$MASTER" >/dev/null

  case "$result" in
    done)
      log "✅ trainer printed Done. — run complete (final=${CKPT_BASE}.bin)"; exit 0 ;;
    boundary)
      log "⏸  hit epoch $NEXT_PAUSE — GPUs idle for ${PAUSE_SECS}s ($(( PAUSE_SECS/60 )) min)"
      sleep "$PAUSE_SECS"
      log "▶  pause over — resuming" ;;
    crash)
      log "⚠  segment ended before epoch $NEXT_PAUSE (crash/AER?) — resuming immediately (no pause)"
      sleep 10 ;;
  esac
done

log "⛔ hit MAX_ATTEMPTS=$MAX_ATTEMPTS — giving up"
exit 1
