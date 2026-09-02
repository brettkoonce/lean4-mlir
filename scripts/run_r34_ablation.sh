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
# ⛔⛔ THE DEFAULT IS RESOLVED **AFTER** PREC, AND THAT ORDER IS THE WHOLE POINT. Setting
# `OUT="${OUT:-…}"` before this line makes the bf16 branch a no-op — `${OUT:-…}` cannot override a
# value already set — so both precisions land in ONE directory and the second sweep's
# `> $OUT/$arm.log` TRUNCATES the first's logs the instant an arm starts. That is not hypothetical:
# it destroyed three completed fp32 curves on 2026-09-01 before the guard below existed.
if [ "$PREC" = "bf16" ]; then OUT="${OUT:-runs/2026-09-01-r34-ablation-bf16}"
else                         OUT="${OUT:-runs/2026-09-01-r34-ablation}"; fi
mkdir -p "$OUT"
# ⚠⚠ REFUSE TO OVERWRITE ANOTHER RUN'S LOGS. A log already in this directory is either this
# precision's (a resume the caller must ask for) or another's (the truncation above). Either way
# the caller decides, not the script. `FORCE=1` to proceed.
if [ -z "${FORCE:-}" ]; then
  for a in $ARMS; do
    if [ -s "$OUT/$a.log" ]; then
      echo "⛔ $OUT/$a.log already exists and is non-empty."
      echo "   Starting would TRUNCATE it. Move it aside, pick another OUT, or set FORCE=1."
      exit 1
    fi
  done
fi
# A stamp, so a directory can say which precision produced it without parsing a log.
echo "$PREC" > "$OUT/.precision"
PLUG="${PJRT_PLUGIN:-/home/skoonce/.venv-cuda/lib/python3.12/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so}"
[ -f "$PLUG" ] || { echo "⛔ plugin not found: $PLUG"; exit 1; }
[ -x .lake/build/bin/resnet34-ablation ] || { echo "⛔ build resnet34-ablation first"; exit 1; }

IFS=',' read -r -a GPUARR <<< "$GPUS"
NSLOT=${#GPUARR[@]}
echo "queue: $(echo $ARMS | wc -w) arm(s) $PREC over ${NSLOT} card(s) [$GPUS] -> $OUT"

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

run_arm () {  # arm gpu
  local arm="$1" gpu="$2"
  wait_for_gpu "$gpu"
  local t0=$SECONDS
  local extra=()
  [ "$arm" = "noaug" ] && extra=(LEAN_MLIR_NO_AUG=1)
  env CUDA_VISIBLE_DEVICES="$gpu" PJRT_PLUGIN="$PLUG" \
      LEAN_MLIR_CKPT_TAG="abl-$PREC-$arm" "${extra[@]}" \
      .lake/build/bin/resnet34-ablation data "$arm" $([ "$PREC" = bf16 ] && echo bf16) \
      > "$OUT/$arm.log" 2>&1
  printf '  ✔ %-8s gpu%s %5ss  %s\n' "$arm" "$gpu" "$((SECONDS-t0))" \
    "$(grep -oE 'val_acc = [0-9]+/[0-9]+ = [0-9.]+%' "$OUT/$arm.log" | tail -1)"
}

declare -A BUSY=()
for arm in $ARMS; do
  # find a free card; if none, block until one frees
  slot=""
  while [ -z "$slot" ]; do
    for g in "${GPUARR[@]}"; do [ -z "${BUSY[$g]:-}" ] && { slot="$g"; break; }; done
    [ -z "$slot" ] && { wait -n; for g in "${GPUARR[@]}"; do
        [ -n "${BUSY[$g]:-}" ] && ! kill -0 "${BUSY[$g]}" 2>/dev/null && unset 'BUSY[$g]'; done; }
  done
  run_arm "$arm" "$slot" &
  BUSY[$slot]=$!
done
wait
echo "ALL ARMS DONE -> $OUT"
for f in "$OUT"/*.log; do
  printf '%-10s %s\n' "$(basename "$f" .log)" \
    "$(grep -oE 'val_acc = [0-9]+/[0-9]+ = [0-9.]+%.*' "$f" | tail -1)"
done
