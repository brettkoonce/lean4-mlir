#!/usr/bin/env bash
# Supervised 350-epoch MobileNetV2-ImageNet run on the 4 clean GPUs (0,2,3,4),
# PAPER-FAITHFUL recipe (`full`): RMSProp ρ0.9 / μ0.9 / ε1.0, lr 0.045 @ batch 256,
# exp-LR-decay ×0.98/epoch (NOT cosine), wd 4e-5, dropout 0.2, crop+flip only,
# label smoothing 0, running-BN eval, bf16 + bf16Conv. Paper target ≈ 72.0% top-1.
# See planning/jax_imagenet_sweep.md and blueprint §7.3 (this run is that [TODO]).
#
# This is the gap-closer for the −3.23 pt in planning/paper_faithfulness.md. The
# standing 68.77% number is the 90ep *validation* tier and predates three of the
# rows here: it ran cosine (not exp-decay), label smoothing 0.1 (not 0), and 90
# epochs (not 350). So this is not merely a longer run of the same recipe.
#
# Same machinery as supervise_mnv4_convm_500ep_4gpu_duty.sh; differences:
# - PY = generated_mobilenet_v2_imagenet_full.py (350ep).
# - CKPT_BASE under mnv2_350ep/ (the old ~/mnv2/ holds the stale 90ep run).
# - Thermal rests every 30 epochs, matching the ConvNeXt-300ep / MNv4-500ep
#   convention, for a ~2-day run.
# - PY_BIN is explicit and preflighted (see below).
#
# INTERPRETER: `../.venv/bin/python`, the pinned CUDA stack from
# jax/requirements-cuda-lock.txt — jax/jaxlib/jax-cuda12-* 0.10.2, cuDNN 9.23.2.1,
# tf 2.21.0, tfds 4.9.10.
#
# ⚠ Do NOT substitute another interpreter. On 2026-07-28 this venv was found
# holding an unrelated April IREE stack (CPU-only jaxlib, no tfds); running the
# trainer under a conda jax 0.4.26 / cuDNN 8.9.7 env instead crashed on the first
# train_step — `CUDNN_STATUS_EXECUTION_FAILED` from the bf16 1x1
# __cudnn$convBackwardFilter in the inverted-residual blocks, with XLA reporting
# "Results mismatch between different convolution algorithms". It also ran ~2x
# slower where it ran at all. planning/jax_imagenet_sweep.md called this exact
# risk: "a silent jax/cuDNN bump could shift bf16 conv kernel selection and move
# the published numbers." Rebuild with `pip install -r requirements-cuda-lock.txt`.
# Quick fingerprint: the right stack reprs devices as `CudaDevice(id=0)`.
#
# Bit-for-bit resume via LEAN_MLIR_RESUME (params + RMSProp mean-square/momentum +
# BN running stats + step), so the exp-LR schedule and BN stats continue exactly.
# This matters more here than for the SGD nets: the old INIT_LOAD/START_STEP path
# restores weights only, which would silently reset the RMSProp accumulators at
# every AER restart.
#
# JAX_COMPILATION_CACHE_DIR keeps the XLA compile a one-time cost across the ~11
# planned rests + any AER resumes. The benign `ncclCommRegister … Cuda failure 500`
# warning prints and is harmless (see reference_klawd_nccl_compile).
set -u
cd "$(dirname "$0")/.." || exit 1

DEVS="${DEVS:-0,2,3,4}"                     # the 4 clean cards; idx1 (bus 02) and
                                            # idx5 (bus 62) are the BadTLP pair
PY=.lake/build/generated_mobilenet_v2_imagenet_full.py
PY_BIN="${PY_BIN:-../.venv/bin/python}"     # pinned stack — see INTERPRETER above
CKPT_BASE="${CKPT_BASE:-/home/skoonce/mnv2_350ep/mobilenet_v2_imagenet}"
# SPE (documentation only — resume is state-based, no step math needed):
#   4-GPU: batch 256 (4x64) -> 1281167//256 = 5004 steps/epoch
#   6-GPU: batch 252 (6x42) -> 1281167//252 = 5083 steps/epoch
REST_EPOCHS="${REST_EPOCHS:-30 60 90 120 150 180 210 240 270 300 330}"
REST_SECS="${REST_SECS:-1800}"              # 30-min cooldown
JAX_CACHE=/home/skoonce/.jax_cache          # persistent XLA compile cache
TFDS_DIR="${TFDS_DATA_DIR:-/home/skoonce/tensorflow_datasets}"
RUNLOG=/tmp/mnv2_350ep_4gpu.log             # current attempt only (greps below)
MASTER=/tmp/mnv2_350ep_4gpu_master.log      # supervisor narration, persists
FULLLOG="${FULLLOG:-${CKPT_BASE}_full.log}" # cumulative trainer stdout, appended
mkdir -p "$(dirname "$CKPT_BASE")" "$JAX_CACHE"
MAX_ATTEMPTS=200

# ── preflight ────────────────────────────────────────────────────────────────
# Fail loudly here rather than 8 hours in. Checks the emitted trainer exists, the
# interpreter has a CUDA jax + tfds, and the prepared dataset is where we think.
[ -f "$PY" ] || { echo "[sup] MISSING $PY — run: .lake/build/bin/mobilenet-v2-imagenet full"; exit 1; }
[ -x "$PY_BIN" ] || { echo "[sup] PY_BIN not executable: $PY_BIN"; exit 1; }
[ -d "$TFDS_DIR/imagenet2012" ] || { echo "[sup] no imagenet2012 under $TFDS_DIR"; exit 1; }
# ⛔ THE CHECK THIS RUN NEEDED AND DID NOT HAVE. $PY lives in gitignored .lake/build; a warm
# cache serves whatever was emitted last, and the published 71.44 was trained on a Jun-22
# artifact carrying labelSmoothing 0.1 that the source had dropped on 2026-07-07. A diff
# against the committed jax/generated/ copies — no Lean, no GPU, instant.
if ! ../scripts/regen_jax_generated.sh box >/dev/null 2>&1; then
  echo "[sup] ⛔ STALE GENERATED ARTIFACTS on this box:" | tee -a "$MASTER"
  ../scripts/regen_jax_generated.sh box 2>&1 | grep -E "STALE|not built" | sed "s/^/[sup]    /" | tee -a "$MASTER"
  echo "[sup]    Re-emit first: scripts/regen_jax_generated.sh" | tee -a "$MASTER"
  exit 1
fi
if ! CUDA_VISIBLE_DEVICES=$DEVS "$PY_BIN" -c "
import sys, jax, tensorflow_datasets
g = [d for d in jax.devices() if d.platform == 'gpu']
print('[preflight] jax', jax.__version__, len(g), 'gpu devices; tfds', tensorflow_datasets.__version__)
if not g:
    print('[preflight] no GPU devices'); sys.exit(1)
# Version gate: the lock pins jax 0.10.x. An OLDER stack silently changes bf16 conv
# kernel selection -- jax 0.4.26/cuDNN 8.9.7 crashes outright on this net, in the
# bf16 1x1 cudnn convBackwardFilter of the inverted-residual blocks.
#
# 0.11.x ADMITTED 2026-08-29, on evidence rather than convenience. The 4x RTX 3060
# box has no pinned .venv at all (../.venv/bin/python does not exist there) and its
# only CUDA stack is .venv-cuda: jax 0.11.1, CUDA 13.3, driver 610.57.04. Before
# widening this gate, jax/scripts/smoke_mnv2_runningbn_gpu.py was run there on all
# four cards under 0.11.1 -- i.e. THIS net's bf16 convs, forward and backward:
#     [ok] running-BN trains on GPU: loss 6.930 -> 2.021
#     [ok] 52 BN buffers EMA-updated + finite
# No CUDNN_STATUS_EXECUTION_FAILED, no algorithm-mismatch warning, loss descends.
# That smoke then dies in an UNRELATED TypeError: it calls eval_batch(params, bn, x, y)
# but the emitted trainer's signature has since gained a 'take' argument. Test rot,
# not a stack fault; the two [ok] lines above are what this gate rests on.
# The floor stays: anything below 0.10 is still refused, which is the crash this gate
# was built for. A future 0.12 wants its own smoke before being added here.
#
# NOTE: this comment lives inside the double-quoted python -c string. No backticks and
# no dollar signs here -- bash expands both, and doing so broke this script once.
if not (jax.__version__.startswith('0.10.') or jax.__version__.startswith('0.11.')):
    print('[preflight] WRONG JAX: got', jax.__version__, '- expected 0.10.x or 0.11.x')
    sys.exit(1)
sys.exit(0)
"; then
  echo "[sup] PREFLIGHT FAILED: $PY_BIN is not the pinned stack (need jax 0.10.x + GPU + tfds)." | tee -a "$MASTER"
  echo "[sup]   rebuild: python3.12 -m venv .venv && .venv/bin/pip install -r jax/requirements-cuda-lock.txt" | tee -a "$MASTER"
  exit 1
fi

next_rest() {
  for e in $REST_EPOCHS; do
    if [ "$1" -lt "$e" ]; then echo "$e"; return; fi
  done
  echo 999
}

echo "[sup] $(date '+%F %T') START 350-epoch MNv2 (paper full recipe) on GPUs $DEVS (rest ${REST_SECS}s every 30ep)" | tee -a "$MASTER"

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
      TFDS_DATA_DIR="$TFDS_DIR" \
      "${RESUME_ENV[@]}" \
      "$PY_BIN" -u "$PY" > >(tee -a "$FULLLOG" > "$RUNLOG") 2>&1 &
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
    echo "[sup] $(date '+%T') ✅ TRAINING COMPLETE (350 epochs). final=${CKPT_BASE}.bin" | tee -a "$MASTER"
    exit 0
  fi

  pkill -9 -f "generated_mobilenet_v2_imagenet_full.py" 2>/dev/null
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
