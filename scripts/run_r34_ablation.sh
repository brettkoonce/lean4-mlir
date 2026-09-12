#!/usr/bin/env bash
# run_r34_ablation.sh — §5.6's recipe ablation on ResNet-34/Imagenette, one arm per GPU.
#
#     scripts/run_r34_ablation.sh                          # all 8 arms, GPUs 0-3
#     GPUS=1,2,3 ARMS="nowarm nocos noaug" scripts/run_r34_ablation.sh
#     PREC=bf16 scripts/run_r34_ablation.sh          # the precision peer of the whole sweep
#
# ⭐ A WORK QUEUE, not fixed waves. Arms differ in wall clock (the `bare` arm has no warmup and a
# constant rate; `noaug` skips two host ops a step), so a wave barrier idles every card that
# finished early until the slowest one lands. `wait -n` starts the next arm the moment ANY card
# frees, which is also what lets a card busy with another job join the sweep late.
#
# ⚠⚠ LEAN_MLIR_CKPT_TAG IS LOAD-BEARING, PER ARM. Six of the eight arms load the SAME artifact
# (`resnet34_adam`), so by default they share a checkpoint path: the second arm resumes the first
# at epoch 80, prints `done`, and reports the first arm's accuracy as its own. The failure is
# silent — the log reads like a finished run.
# ⚠ `noaug` is `full` plus LEAN_MLIR_NO_AUG=1. The flag is host-side, so it is set here; the
# binary still lists the arm by name so a typo is an error rather than a silent `full`.
set -u
cd "$(dirname "$0")/.." || exit 1

GPUS="${GPUS:-0,1,2,3}"
ARMS="${ARMS:-full nowarm nocos noaug nowd nols noadam bare}"
# ⚠ PREC picks the artifact FAMILY, and it must reach the OUTPUT DIR and the CKPT TAG as well as
# the binary: an fp32 and a bf16 arm of the same name otherwise share a checkpoint and a log, and
# the second silently resumes the first at epoch 80 — the failure this file already warns about,
# one axis over.
PREC="${PREC:-fp32}"
[ "$PREC" = "fp32" ] || [ "$PREC" = "bf16" ] || { echo "⛔ PREC must be fp32 or bf16"; exit 1; }
# SEEDS — planning/archive/r34_ablation_seeds.md: a tour number is mean ± 95% CI over seeds. One seed keeps
# the 2026-09-01 layout (`<arm>.log`); several write `<arm>_s<seed>.log`, queued seed-major so a
# sweep stopped early still holds whole seeds. ⚠ Every seed gets its OWN checkpoint tag — the trap
# `seed_sweep.sh` documents: without it seeds 2.. resume seed 1's checkpoint and report its number.
SEEDS="${SEEDS:-1}"
NSEED=$(echo $SEEDS | wc -w)
# PJRT_FFI_RESIDENT defaults ON here now. It is what made the seed sweep's ResNet-34 runs take 45
# minutes where these arms took 80 (same net, same recipe); the 2026-09-01 tables ran with it off.
RESIDENT="${PJRT_FFI_RESIDENT:-1}"
# ⛔⛔ THE DEFAULT IS RESOLVED **AFTER** PREC, AND THAT ORDER IS THE WHOLE POINT. Setting
# `OUT="${OUT:-…}"` before this line makes the bf16 branch a no-op — `${OUT:-…}` cannot override a
# value already set — so both precisions land in ONE directory and the second sweep's
# `> $OUT/$arm.log` TRUNCATES the first's logs the instant an arm starts. That is not hypothetical:
# it destroyed three completed fp32 curves on 2026-09-01 before the guard below existed.
if [ "$NSEED" -gt 1 ]; then    OUT="${OUT:-runs/$(date +%F)-r34-ablation-$PREC-seeds}"
elif [ "$PREC" = "bf16" ]; then OUT="${OUT:-runs/2026-09-01-r34-ablation-bf16}"
else                            OUT="${OUT:-runs/2026-09-01-r34-ablation}"; fi
logname () {  # arm seed
  if [ "$NSEED" -gt 1 ]; then echo "$OUT/$1_s$2.log"; else echo "$OUT/$1.log"; fi
}
mkdir -p "$OUT"
# ⚠⚠ REFUSE TO OVERWRITE ANOTHER RUN'S LOGS. A log already in this directory is either this
# precision's (a resume the caller must ask for) or another's (the truncation above). Either way
# the caller decides, not the script. `FORCE=1` to proceed.
if [ -z "${FORCE:-}" ]; then
  for a in $ARMS; do for sd in $SEEDS; do
    if [ -s "$(logname $a $sd)" ]; then
      echo "⛔ $(logname $a $sd) already exists and is non-empty."
      echo "   Starting would TRUNCATE it. Move it aside, pick another OUT, or set FORCE=1."
      exit 1
    fi
  done; done
fi
# A stamp, so a directory can say which precision produced it without parsing a log.
# ⚠⚠ REFUSE A LEFTOVER CHECKPOINT TOO. The log guard above does not see `.lake/build/`: on
# 2026-09-11 a killed launch left `resnet34_*_ckpt_xla_abl-bf16-<arm>-s1.bin` at epoch 3, the run
# directory was removed, and the relaunch RESUMED four arms from it — the log looked like a
# finished run, only "resuming from checkpoint at epoch 3" gave it away. `RESUME=1` to proceed.
if [ -z "${FORCE:-}" ] && [ -z "${RESUME:-}" ]; then
  for a in $ARMS; do for sd in $SEEDS; do
    for c in .lake/build/*_ckpt_xla_abl-$PREC-$a-s$sd.bin; do
      [ -e "$c" ] || continue
      echo "⛔ $c exists: this arm/seed would RESUME from it, not train from scratch."
      echo "   Delete it (and its .epoch) for a fresh run, or set RESUME=1 to continue it on purpose."
      exit 1
    done
  done; done
fi
echo "$PREC" > "$OUT/.precision"
echo "$SEEDS" > "$OUT/.seeds"
PLUG="${PJRT_PLUGIN:-/home/skoonce/.venv-cuda/lib/python3.12/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so}"
[ -f "$PLUG" ] || { echo "⛔ plugin not found: $PLUG"; exit 1; }
[ -x .lake/build/bin/resnet34-ablation ] || { echo "⛔ build resnet34-ablation first"; exit 1; }

IFS=',' read -r -a GPUARR <<< "$GPUS"
NSLOT=${#GPUARR[@]}
echo "queue: $(echo $ARMS | wc -w) arm(s) x $NSEED seed(s) [$SEEDS] $PREC resident=$RESIDENT over ${NSLOT} card(s) [$GPUS] -> $OUT"
# DRY_RUN=1 prints the queue and exits before anything waits on a card.
if [ -n "${DRY_RUN:-}" ]; then
  for sd in $SEEDS; do for a in $ARMS; do
    echo "  would run: arm=$a seed=$sd -> $(logname $a $sd)  (ckpt abl-$PREC-$a-s$sd)"; done; done
  exit 0
fi

# ⭐ Block until this CARD is actually idle, not merely unclaimed by THIS script. The queue can be
# started while another job (a chapter re-run, an earlier wave) still holds a GPU; without this the
# arm launches anyway, both jobs halve, and the s/ep figure the chapter quotes is a contention
# artifact rather than the net's cost. Measured: 4 arms one-per-card run at the same 60 s/epoch as
# one arm alone, so the only thing that slows an arm down is sharing a card.
wait_for_gpu () {  # gpu
  local gpu="$1" waited=0
  while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i "$gpu" 2>/dev/null | grep -q .; do
    sleep 30; waited=$((waited+30))
  done
  [ "$waited" -gt 0 ] && echo "    (waited ${waited}s for gpu$gpu)"
  return 0
}

run_arm () {  # arm seed gpu
  local arm="$1" seed="$2" gpu="$3"
  wait_for_gpu "$gpu"
  local t0=$SECONDS
  local extra=()
  [ "$arm" = "noaug" ] && extra=(LEAN_MLIR_NO_AUG=1)
  local log; log="$(logname "$arm" "$seed")"
  env CUDA_VISIBLE_DEVICES="$gpu" PJRT_PLUGIN="$PLUG" PJRT_FFI_RESIDENT="$RESIDENT" \
      LEAN_MLIR_SEED="$seed" LEAN_MLIR_CKPT_TAG="abl-$PREC-$arm-s$seed" "${extra[@]}" \
      .lake/build/bin/resnet34-ablation data "$arm" $([ "$PREC" = bf16 ] && echo bf16) \
      > "$log" 2>&1
  printf '  ✔ %-8s s%-2s gpu%s %5ss  %s\n' "$arm" "$seed" "$gpu" "$((SECONDS-t0))" \
    "$(grep -oE 'val_acc = [0-9]+/[0-9]+ = [0-9.]+%' "$log" | tail -1)"
}

declare -A BUSY=()
for job in $(for sd in $SEEDS; do for a in $ARMS; do echo "$a:$sd"; done; done); do
  arm="${job%%:*}"; seed="${job##*:}"
  # find a free card; if none, block until one frees
  slot=""
  while [ -z "$slot" ]; do
    for g in "${GPUARR[@]}"; do [ -z "${BUSY[$g]:-}" ] && { slot="$g"; break; }; done
    [ -z "$slot" ] && { wait -n; for g in "${GPUARR[@]}"; do
        [ -n "${BUSY[$g]:-}" ] && ! kill -0 "${BUSY[$g]}" 2>/dev/null && unset 'BUSY[$g]'; done; }
  done
  run_arm "$arm" "$seed" "$slot" &
  BUSY[$slot]=$!
done
wait
echo "ALL ARMS DONE -> $OUT"
for f in "$OUT"/*.log; do
  printf '%-14s %s\n' "$(basename "$f" .log)" \
    "$(grep -oE 'val_acc = [0-9]+/[0-9]+ = [0-9.]+%.*' "$f" | tail -1)"
done
