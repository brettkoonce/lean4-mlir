#!/usr/bin/env bash
# seed_sweep.sh — every trainer in a SUITE at n seeds, packed across the GPUs.
#
#     scripts/seed_sweep.sh                                 # imagenette, 6 GPUs, seeds 1 2 3
#     SUITE=small scripts/seed_sweep.sh 0,1,2,3,4,5 "1 2 3 4 5"
#     DRY_RUN=1 scripts/seed_sweep.sh                       # print the packing, launch nothing
#     scripts/seed_sweep.sh 0,2,3,4 "1 2 3"                 # the AER-clean four only
#     SWEEP_DIR=runs/2026-08-31-imagenette-n3 scripts/...   # pin the dir (resume a crashed sweep)
#
# SUITE=imagenette (default) — the seven 80-epoch Imagenette trainers.
# SUITE=small — the book's chapter 1-4 trainers: MNIST at 12 epochs, CIFAR at 40 (the ladder
#   12 -> 40 -> 80 standardised 2026-08-31).
#
# ⭐ CIFAR is the WIDE head (`cifar8w-{,bn-}ablation`, d1=512), not the narrow `cifar-{,bn-}verified`
#   this suite first pointed at. Chapter 4 quotes the wide net — §4.1's 77.48% and §4.2's
#   `runs/2026-08-12-cifar8w-6arm-xla-cuda/` medians are both `cifar8w-bn-ablation` — so the narrow
#   pair produced seed statistics for a net the chapter never reports (75.28% vs the quoted 77.48).
#   The wide head is ~1.6x the wall clock per epoch and buys no accuracy (§4.3's head-width sweep is
#   exactly that finding); it is run anyway because it is the net the chapter bridges MNIST to
#   ResNet with.
#
# ⚠ Each ablation binary runs THREE optimizer arms in sequence (SGD / Nesterov / AdamW) on one
#   controlled pipeline, so six arms are two jobs and one job yields three final-epoch numbers.
#   The OK line reports all of them; `LEAN_MLIR_DUMP_CORRECT` writes one bitmap per arm.
#
# ⚠ The no-BN wide net's MOMENTUM arm diverges to exactly 10.00% — 3 of 5 seeds on 2026-08-12,
#   which is §4.2's published finding. Its SGD and AdamW arms finish ~72-73%. That is a RESULT,
#   not a runner failure, so a job containing a diverged arm still counts as done.
#
# `planning/imagenette_error_intervals.md` §2. Seven trainers x 3 seeds = 21 runs, LPT-packed
# one-per-GPU (XLA preallocates ~75% of a card, so two will not share one). ~19.4 GPU-h, so
# ~3.3 h on six cards (19.7 GPU-h).
#
# ⚠ Every trap in that document's §3 is handled HERE and nowhere else — do not run these
# binaries by hand and expect the same numbers:
#
#   1. PJRT_FFI_RESIDENT=1 is set. Off by default (ffi/pjrt_ffi.c:284); absent, ConvNeXt takes
#      2 h 14 m instead of 1 h 19 m and prints no RESIDENT: line to say so.
#   3. LEAN_MLIR_CKPT_TAG=s<seed> gives every seed its OWN checkpoint path. Without it all three
#      seeds share one blob and seeds 2 and 3 resume seed 1's finished epoch 80 and exit clean.
#      It also keeps the sweep off the untagged checkpoints already in .lake/build.
#   4. The `-adam` binaries, exactly — `efficientnet-verified` is a different net's numbers.
#   6. One job per GPU, and the AER watchdog below stops the sweep rather than trusting a card.
#
# Resumable: a job whose done-marker exists is skipped, so a crashed box restarts where it was.
set -uo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:-0,1,2,3,4,5}"
SEEDS="${2:-1 2 3}"
# ⚠ Date-stamped, but a crash-and-restart AFTER midnight would otherwise open a fresh directory
# and lose the done-markers that make this resumable. Reuse the newest existing sweep dir unless
# SWEEP_DIR says otherwise, so "run it again" always means "finish it".
SUITE="${SUITE:-imagenette}"
LOGDIR="${SWEEP_DIR:-$(ls -d runs/*-${SUITE}-seeds 2>/dev/null | tail -1)}"
LOGDIR="${LOGDIR:-runs/$(date +%Y-%m-%d)-${SUITE}-seeds}"
mkdir -p "$LOGDIR"/{done,bitmaps}
IDX="$LOGDIR/.idx"; LOCK="$LOGDIR/.lock"; ABORT="$LOGDIR/.abort"
echo 0 > "$IDX"; : > "$LOCK"; rm -f "$ABORT"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOGDIR/sweep.log"; }

# LPT order — longest first. Hours are the book's measured per-net figures, except resnet50 and
# mobilenetv4, measured 2026-08-31 (the planning doc carried ~2.5 h and ~50 min ESTIMATES for
# those two; both were high). Order only affects packing, never results.
declare -A EXE EPOCHS
case "$SUITE" in
  imagenette)
    NETS=(resnet34 resnet50 convnext efficientnet mobilenetv4 mobilenetv2 vit)
    EXE=( [resnet34]=resnet34-verified-adam       [resnet50]=resnet50-verified-adam
          [convnext]=convnext-verified-adam       [efficientnet]=efficientnet-verified-adam
          [mobilenetv4]=mobilenetv4-verified-adam [mobilenetv2]=mobilenetv2-verified-adam
          [vit]=vit-verified-adam )
    for n in "${NETS[@]}"; do EPOCHS[$n]=80; done ;;
  small)
    # LPT: cifar8w_bn ~12.6 min/seed (3 arms x 40 ep x 6.31 s), cifar8w ~6.2 (3 x 40 x 3.12),
    # then the MNIST three at well under a minute each.
    NETS=(cifar8w_bn cifar8w mnist_cnn mnist_mlp mnist_linear)
    EXE=( [cifar8w_bn]=cifar8w-bn-ablation [cifar8w]=cifar8w-ablation
          [mnist_cnn]=mnist-cnn-verified
          [mnist_mlp]=mnist-mlp-verified [mnist_linear]=mnist-linear-verified )
    EPOCHS=( [cifar8w_bn]=40 [cifar8w]=40 [mnist_cnn]=12 [mnist_mlp]=12 [mnist_linear]=12 ) ;;
  *) echo "⛔ unknown SUITE=$SUITE (imagenette|small)"; exit 1 ;;
esac

JOBS=()
for net in "${NETS[@]}"; do for s in $SEEDS; do JOBS+=("$net:$s"); done; done

# --- precheck ------------------------------------------------------------------------------
for net in "${!EXE[@]}"; do
  [ -x ".lake/build/bin/${EXE[$net]}" ] || { echo "⛔ missing .lake/build/bin/${EXE[$net]} — lake build it"; exit 1; }
done
[ -d data ] || { echo "⛔ no data/"; exit 1; }
command -v flock >/dev/null || { echo "⛔ no flock"; exit 1; }

# ⚠⚠ The watchdog's own gate. `dmesg` is unreadable here (kernel.dmesg_restrict=1) and returns
# EMPTY, so a `dmesg | grep -c AER` watchdog reports "0 events" forever and looks perfectly
# healthy — that false green was live in this script for one revision. journalctl -k is the
# source that works; if it ever stops working, refuse rather than run unwatched.
if [ -z "$(journalctl -k -n 1 2>/dev/null)" ]; then
  echo "⛔ journalctl -k returns nothing — the AER watchdog would be a silent no-op. Refusing."
  exit 1
fi

# --- AER watchdog --------------------------------------------------------------------------
# The kernel journal is the only thing that reports PCIe BadTLP under load. GPUs 1 and 5 are
# excluded from every 4-GPU job here on suspicion; a 1-epoch probe on each ran clean 2026-08-31,
# which is evidence and not a clearance. If anything fires, stop taking new jobs.
# `no action required` is the benign corrected-error chatter (same filter as supervise.sh).
START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
aer_since() {
  journalctl -k --since "$1" 2>/dev/null \
    | grep -iE "BadTLP|AER:|Uncorrected|Fatal" | grep -qivE "no action required"
}

if [ "${DRY_RUN:-0}" != "0" ]; then
  echo "DRY_RUN — SUITE=$SUITE, ${#JOBS[@]} jobs over GPUs $GPUS, seeds [$SEEDS], into $LOGDIR"
  i=0; for j in "${JOBS[@]}"; do printf '  %2d  %-14s seed %s  %2s ep  -> %s\n' "$i" "${j%%:*}" "${j##*:}" "${EPOCHS[${j%%:*}]}" "${EXE[${j%%:*}]}"; i=$((i+1)); done
  echo "  watchdog source: journalctl -k (dmesg is restricted here); baseline clean = $(aer_since "$START_TS" && echo NO || echo yes)"
  exit 0
fi

( while [ ! -f "$ABORT" ]; do
    if aer_since "$START_TS"; then
      say "!!! PCIe AER since $START_TS — no further jobs will start"
      journalctl -k --since "$START_TS" 2>/dev/null \
        | grep -iE "BadTLP|AER:|Uncorrected|Fatal" | grep -ivE "no action required" | tail -5 >> "$LOGDIR/aer.log"
      touch "$ABORT"
    fi
    sleep 60
  done ) &
WATCHDOG=$!
trap 'touch "$ABORT"; kill $WATCHDOG 2>/dev/null' EXIT

run_job() {
  local gpu=$1 net=${2%%:*} seed=${2##*:}
  local tag="${net}_s${seed}" exe="${EXE[$net]}"
  [ -f "$LOGDIR/done/$tag" ] && { say "gpu$gpu  $tag  already done, skipping"; return; }
  say "gpu$gpu  $tag  start  ($exe)"
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES="$gpu" \
  PJRT_FFI_RESIDENT=1 \
  LEAN_MLIR_SEED="$seed" \
  LEAN_MLIR_CKPT_TAG="s$seed" \
  LEAN_MLIR_DUMP_CORRECT="$LOGDIR/bitmaps/$tag" \
    ./.lake/build/bin/"$exe" data > "$LOGDIR/$tag.log" 2>&1
  local rc=$?
  local el=$((SECONDS-t0))
  # ⚠ rc alone is not enough: §3.2's cap makes an already-finished net print `done` and exit 0
  # with no epoch line at all. Require a real final epoch line too.
  if [ $rc -eq 0 ] && grep -q "^done" "$LOGDIR/$tag.log" \
     && grep -qE "epoch +${EPOCHS[$net]}:.*(val_acc|test_acc)" "$LOGDIR/$tag.log"; then
    printf '%s %ds\n' "$tag" "$el" > "$LOGDIR/done/$tag"
    # ⚠ Not `tail -1`: a 3-arm ablation has three final-epoch lines and tail would report only
    # the last (AdamW), quietly hiding the SGD and momentum arms — including a diverged one.
    say "gpu$gpu  $tag  OK   $((el/60))m$((el%60))s  $(awk -v E="${EPOCHS[$net]}" '
        $0 ~ "^  epoch +" E ":" && match($0, /= [0-9.]+%/) {
          a[++n] = substr($0, RSTART + 2, RLENGTH - 2) }
        END { for (i = 1; i <= n; i++) printf "%s%s", (i > 1 ? " / " : ""), a[i] }' "$LOGDIR/$tag.log")"
  else
    say "gpu$gpu  $tag  FAIL rc=$rc after $((el/60))m — see $LOGDIR/$tag.log"
  fi
}

worker() {
  local gpu=$1
  while true; do
    [ -f "$ABORT" ] && { say "gpu$gpu  aborting (watchdog)"; return; }
    local i
    i=$( flock "$LOCK" -c "i=\$(cat '$IDX'); echo \$((i+1)) > '$IDX'; echo \$i" )
    [ "$i" -ge "${#JOBS[@]}" ] && return
    run_job "$gpu" "${JOBS[$i]}"
  done
}

say "sweep start: SUITE=$SUITE, ${#JOBS[@]} jobs, GPUs $GPUS, seeds [$SEEDS] -> $LOGDIR"
for g in ${GPUS//,/ }; do worker "$g" & done
wait
say "sweep complete: $(ls "$LOGDIR/done" | wc -l)/${#JOBS[@]} jobs done"
