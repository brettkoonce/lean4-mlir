#!/usr/bin/env bash
# deploy/orin_imagenette.sh — one Imagenette net on the Orin Nano: the board's recipe, the memory
# cap, and a relaunch that resumes from the checkpoint. The engine behind `lake run imagenette-orin`,
# which runs the six rows of the tier through it; this file is for one net at a time.
#
#     BIN=vit-verified-adam FRACTION=0.25 deploy/orin_imagenette.sh
#     DRY_RUN=1 BIN=… FRACTION=… deploy/orin_imagenette.sh      # the plan: env, command, checkpoint state; no launch
#     EPOCHS=2  BIN=… FRACTION=… deploy/orin_imagenette.sh      # a smoke run (LEAN_MLIR_MAX_EPOCHS)
#     BIN=resnet50-verified-adam FRACTION=0.36 \
#       LEAN_MLIR_VARIANT=acc2x16 LEAN_MLIR_BATCH=16 LEAN_MLIR_G2_STEPS=590 deploy/orin_imagenette.sh
#
# THE RECIPE (deploy/ORIN.md §2 and §4). The train split is streamed from train.bin
# (`LEAN_MLIR_IMAGENETTE_STREAM=1`: labels and an index array resident, each batch `pread`; bit-identical
# to the f32 loader, tests/imagenette_u8_tie.sh), the BFC pool is reserved once (`LEAN_MLIR_PREALLOCATE=1`)
# at a per-net share of the board's one DRAM (`FRACTION`: the pool must hold the step's one activation
# arena, and the rest of the process must fit beside it), and the process runs under a cgroup memory cap
# (`systemd-run --user --scope -p MemoryMax=5800M -p MemorySwapMax=0`), because a process that reaches the
# top of the 8 GB takes tmux down with it, and the cap turns that into a clean failure the next attempt
# recovers from.
#
# THE RELAUNCH. `trainAdamSched` writes [θ|m|v] and an `.epoch` marker at every epoch end and resumes from
# them on start. An attempt that dies is relaunched WHILE EACH ATTEMPT ADVANCES THE EPOCH: a kill mid-epoch
# costs that epoch, and two attempts in a row that complete nothing is a configuration failure (a pool
# fraction that does not hold the step), not bad luck, so the loop stops there. A row whose checkpoint is
# complete is scored by the trainer and exits 0 at once, so re-running the tier resumes where it left off.
#
# Env:
#   BIN        the lean_exe (required); SLUG is its first dash-field unless set
#   FRACTION   LEAN_MLIR_MEM_FRACTION (required; the per-net table in deploy/ORIN.md §4)
#   TAG        LEAN_MLIR_CKPT_TAG (default orin): .lake/build/<SLUG>_<variant>_ckpt_xla_<TAG>.bin
#   EPOCHS     LEAN_MLIR_MAX_EPOCHS (default: the trainer's own schedule)
#   MEMCAP     MemoryMax for the scope (default 5800M; 0 runs without a scope)
#   RUNDIR     logs (default runs/<date>-<BIN>-orin): runner.log and one attempt<N>.log per launch
#   MAX_ATTEMPTS  default 40
#   DRY_RUN=1  print the plan and exit 0 (2 when a precheck fails)
#   Any LEAN_MLIR_* already in the environment passes through (LEAN_MLIR_VARIANT, _BATCH, _G2_STEPS …).
#   The GPU is $LEAN_DEMO_GPU, else $CUDA_VISIBLE_DEVICES, else 0.
set -u
BIN=${BIN:?set BIN=<lean_exe>, e.g. vit-verified-adam}
FRACTION=${FRACTION:?set FRACTION=<LEAN_MLIR_MEM_FRACTION>, the row\'s pool fraction (deploy/ORIN.md §4)}
SLUG=${SLUG:-${BIN%%-*}}
TAG=${TAG:-orin}
EPOCHS=${EPOCHS:-}
MEMCAP=${MEMCAP:-5800M}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-40}
DRY_RUN=${DRY_RUN:-}
VARIANT=${LEAN_MLIR_VARIANT:-adam}
GPU=${LEAN_DEMO_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
RUNDIR=${RUNDIR:-runs/$(date +%F)-$BIN-orin}
BINPATH=.lake/build/bin/$BIN
CKPT=.lake/build/${SLUG}_${VARIANT}_ckpt_xla_${TAG}.bin

fail () { echo "✗ $*" >&2; exit 2; }
[ -f lakefile.lean ] || fail "run from the repo root"
[ -x "$BINPATH" ] || fail "missing $BINPATH — lake build $BIN"
[ -f data/imagenette/train.bin ] || fail "missing data/imagenette/train.bin — scripts/datasets/download_imagenette.sh"
# The shim must be the one built from this tree: one from before the allocator knobs ignores every
# LEAN_MLIR_* allocator variable and reserves most of the board (ORIN.md §2). `lake run` rebuilds it;
# a direct launch does not.
[ -f ffi/libpjrt_ffi.so ] || fail "missing ffi/libpjrt_ffi.so — gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so"
[ ffi/pjrt_ffi.c -nt ffi/libpjrt_ffi.so ] && fail "ffi/libpjrt_ffi.so is older than ffi/pjrt_ffi.c — rebuild it (ffi/README.md); a stale shim ignores the allocator knobs"
# The plugin: the board's env file names it; on a desktop the box's plugin stands in, so the recipe
# can be smoke-tested where the board is not.
if [ -z "${PJRT_PLUGIN:-}" ]; then
  . scripts/jobs/_box.sh
  [ -f "$BOX_PLUG" ] && export PJRT_PLUGIN="$BOX_PLUG"
fi
[ -n "${PJRT_PLUGIN:-}" ] && [ -f "$PJRT_PLUGIN" ] || fail "PJRT_PLUGIN unset or missing — on the board: . ~/pjrt/orin_env.sh"

tegra=0; [ -f /etc/nv_tegra_release ] && tegra=1
clocks=""
if [ "$tegra" = 1 ]; then
  f=$(cat /sys/class/devfreq/*gpu*/min_freq 2>/dev/null | head -1)
  case "$f" in
    918000000) clocks="pinned" ;;
    "")        clocks="unknown" ;;
    *)         clocks="NOT pinned (min_freq $f) — sudo jetson_clocks; unpinned timings are ~2x pessimistic" ;;
  esac
fi
wrap=()
if [ "$MEMCAP" != 0 ]; then
  if command -v systemd-run > /dev/null; then
    wrap=(systemd-run --user --scope -q -p "MemoryMax=$MEMCAP" -p MemorySwapMax=0 --)
  else
    echo "⚠ no systemd-run: running without the $MEMCAP memory cap" >&2
  fi
fi

epoch_now () { if [ -f "$CKPT.epoch" ]; then tr -dc 0-9 < "$CKPT.epoch"; else echo 0; fi; }
export LEAN_MLIR_LOWERER=xla LEAN_MLIR_IMAGENETTE_STREAM=1 LEAN_MLIR_PREALLOCATE=1 \
       LEAN_MLIR_MEM_FRACTION="$FRACTION" LEAN_MLIR_CKPT_TAG="$TAG" CUDA_VISIBLE_DEVICES="$GPU"
[ -n "$EPOCHS" ] && export LEAN_MLIR_MAX_EPOCHS="$EPOCHS"

echo "━━━ $BIN on the Orin recipe ━━━"
echo "   binary      $BINPATH"
echo "   plugin      $PJRT_PLUGIN"
echo "   board       $([ "$tegra" = 1 ] && echo "Tegra, clocks $clocks" || echo "not a Tegra board: the recipe runs, the cap bounds host memory only")"
echo "   cap         ${wrap[*]:-none}"
echo "   env         $(env | grep '^LEAN_MLIR_' | sort | tr '\n' ' ')"
echo "   gpu         $GPU"
echo "   checkpoint  $CKPT$([ -f "$CKPT" ] && echo " — present, epoch $(epoch_now); the run resumes from it (another TAG= starts over)" || echo " — none, a fresh run")"
echo "   logs        $RUNDIR/{runner.log,attempt<N>.log}"
if [ -n "$DRY_RUN" ]; then echo "   dry run — nothing launched"; exit 0; fi

mkdir -p "$RUNDIR"
log () { echo "$(date '+%F %T') $*" | tee -a "$RUNDIR/runner.log"; }
log "start $BIN fraction $FRACTION tag $TAG epochs ${EPOCHS:-default} gpu $GPU clocks ${clocks:-n/a} cap ${MEMCAP}"
attempt=$(ls "$RUNDIR"/attempt*.log 2>/dev/null | wc -l)
before=$(epoch_now)
while :; do
  attempt=$((attempt + 1))
  [ "$attempt" -gt "$MAX_ATTEMPTS" ] && { log "✗ $MAX_ATTEMPTS attempts — stopping at epoch $before"; exit 1; }
  alog=$RUNDIR/attempt$attempt.log
  log "attempt $attempt from epoch $before → $alog"
  ${wrap[@]+"${wrap[@]}"} "$BINPATH" 2>&1 | tee "$alog"
  rc=${PIPESTATUS[0]}
  now=$(epoch_now)
  if [ "$rc" -eq 0 ]; then
    log "✓ done: rc 0 at epoch $now after $attempt attempt(s); $(grep -o 'val_acc = .*' "$alog" | tail -1)"
    exit 0
  fi
  if [ "$now" -gt "$before" ]; then
    log "attempt $attempt ended rc $rc at epoch $now (was $before) — relaunching from the checkpoint"
    before=$now
    continue
  fi
  log "✗ attempt $attempt ended rc $rc without completing an epoch (still $now) — stopping; tail of $alog:"
  tail -5 "$alog" | tee -a "$RUNDIR/runner.log"
  exit "$rc"
done
