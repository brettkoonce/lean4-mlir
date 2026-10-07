#!/bin/bash
# D17 (planning/imagenet_parity.md §9.2): the side-quest ETA re-probe on ares after X2 / X5 / X6, the
# torchvision crop (aa974a95), the exact GELU (§3.8 renders, `geluExact`), the ViT shim fix and the
# uint8 wire — every rate in planning/side_quest_runs.md §1 predates at least one of those. Same
# method as runs/2026-10-04-dimm-fan-thermal/ (queue2/4/5) and the 3060 box's
# runs/2026-10-06-sidequest-probe-3060/, so the tables compare row for row:
#   JAX      = the trainer run directly, 12-min window (MNv4 20), step lines wall-stamped -> winrate.py;
#              no LEAN_MLIR_PARAMS_OUT / CKPT_EVERY, so no checkpoint
#   verified = the §3a smoke: ONCE=1 LEAN_MLIR_MAX_STEPS=600 LEAN_MLIR_PROBE_WARM=200 scripts/supervise.sh <job>
#              -> PROBE (median of steps 201..600) + PROBE-SPREAD. Under LEAN_MLIR_CKPT_TAG=probe: the
#              600 steps are OPTIMIZER steps, and MNv4 `full` (312/epoch at acc 8) crosses an epoch, which
#              is where the stray epoch-1 checkpoint of 10-05 came from (moved to .lake/build/stale/);
#              the tag keeps a probe's checkpoint off every real lineage, and the phase deletes it after.
# DIMM/CPU/GPU temps every 2 s per run -> temps_<name>.tsv (the 10-05 format).
#
#     setsid -f nohup runs/2026-10-07-d17-reprobe/queue.sh jax      > runs/2026-10-07-d17-reprobe/queue_jax.out 2>&1
#     setsid -f nohup runs/2026-10-07-d17-reprobe/queue.sh verified > runs/2026-10-07-d17-reprobe/queue_verified.out 2>&1
#
# Two phases because the verified prechecks run `lake build` (pc_exe): the JAX half needs no Lean
# binary and runs while the tree is being edited; the verified half runs on a built tree.
set -u
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../.." && pwd)
cd "$R"
PHASE=${1:?usage: queue.sh jax|verified}
JSECS=${JSECS:-720}; VSECS=${VSECS:-1800}
logtemps() {
  echo -e "t\tdimm_18\tdimm_19\tdimm_1c\tdimm_1d\ttctl\tgpu_temps\tgpu_util" > "$1"
  local t0=$(date +%s.%N)
  while :; do
    local dimms="" f
    for a in 18 19 1c 1d; do
      f=$(ls /sys/bus/i2c/devices/1-00$a/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
      dimms+=$'\t'$([ -n "$f" ] && awk '{printf "%.2f",$1/1000}' "$f" || echo NA)
    done
    local tctl=$(sensors k10temp-pci-00c3 2>/dev/null | awk '/Tctl/{gsub(/[+°C]/,"",$2);print $2}')
    local g=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{t=t (t?",":"") $1; u=u (u?",":"") $2} END{print t"\t"u}')
    printf "%.1f%s\t%s\t%s\n" "$(echo "$(date +%s.%N) - $t0" | bc)" "$dimms" "$tctl" "$g" >> "$1"
    sleep 2
  done
}
stamp() { while IFS= read -r l; do printf '%s\t%s\n' "$(date +%s.%N)" "$l"; done; }
grouprun() {  # $1 = secs, $2 = log, rest = cmd; waits on the session leader, then kills its session
  local secs=$1 log=$2; shift 2
  local pf="$D/.sid"; rm -f "$pf"
  setsid bash -c 'echo $$ > "$0"; exec "$@" 2>&1' "$pf" "$@" | stamp > "$log" &
  local p=$! end=$(( $(date +%s) + secs )) sid=""
  while [ -z "$sid" ]; do sleep 1; sid=$(cat "$pf" 2>/dev/null); done
  while kill -0 "$sid" 2>/dev/null && [ "$(date +%s)" -lt "$end" ]; do sleep 5; done
  pkill -TERM -s "$sid" 2>/dev/null; sleep 15; pkill -KILL -s "$sid" 2>/dev/null
  wait $p 2>/dev/null
}
gpus_idle() { for i in $(seq 60); do
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || return 0; sleep 5; done; return 1; }
jaxrun() {  # $1 = name, $2 = trainer (relative to jax/)
  gpus_idle || { echo "⛔ GPUs busy before $1"; return 1; }
  logtemps "$D/temps_$1.tsv" & local lp=$!
  sleep 20
  grouprun "$JSECS" "$D/$1.log" env -C jax JAX_COMPILATION_CACHE_DIR=/home/skoonce/.jax_cache \
    TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets ../.venv/bin/python -u "$2"
  sleep 60; kill $lp; sleep 60
}
verrun() {  # $1 = job
  gpus_idle || { echo "⛔ GPUs busy before $1"; return 1; }
  logtemps "$D/temps_v_$1.tsv" & local lp=$!
  sleep 20
  grouprun "$VSECS" "$D/v_$1.log" env ONCE=1 LEAN_MLIR_MAX_STEPS=600 LEAN_MLIR_PROBE_WARM=200 \
    LEAN_MLIR_CKPT_TAG=probe scripts/supervise.sh "$1"
  sleep 60; kill $lp; sleep 60
  rm -f .lake/build/*_ckpt_xla_probe.bin*    # a probe leaves no lineage behind
}
echo "start $PHASE $(date) on $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
case "$PHASE" in
  jax)
    scripts/regen_jax_generated.sh box || { echo "⛔ jax/.lake/build is stale — regen_jax_generated.sh sync"; exit 1; }
    for spec in "a2 .lake/build/generated_resnet50_imagenet_a2accum.py" \
                "a1 .lake/build/generated_resnet50_imagenet_a1.py" \
                "vits .lake/build/generated_vit_s_imagenet.py" \
                "vitb .lake/build/generated_vit_b_imagenet_accum.py" \
                "cnxs .lake/build/generated_convnext_s_imagenet.py" \
                "cnxb .lake/build/generated_convnext_b_imagenet.py"; do
      set -- $spec; echo "jax $1 $(date)"; jaxrun "$1" "$2"
    done
    echo "jax mnv4 $(date)"; JSECS=1200 jaxrun mnv4 .lake/build/generated_mobilenet_v4_imagenet_full.py ;;
  verified)
    for j in vits-default-emabf16-4gpu vitb-default-emabf16-4gpu r50-a2-bf16-4gpu r50-a1-bf16-4gpu \
             cnxs-default-emabf16-4gpu cnxb-default-emabf16-4gpu mnv4-full-4gpu; do
      echo "verified $j $(date)"; verrun "$j"
    done ;;
  *) echo "usage: queue.sh jax|verified"; exit 2 ;;
esac
echo "done $PHASE $(date)"
