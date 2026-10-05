#!/bin/bash
# Side-quest ETA re-probe after the DIMM fan (planning/side_quest_runs.md §1 / §3a), run after
# queue.sh (ViT-S + A2 JAX). JAX: 12-min windows, step lines wall-stamped. Verified: the §3a smoke
# (ONCE, MAX_STEPS=600, PROBE_WARM=200 -> median + PROBE-SPREAD), binaries rebuilt first.
# DIMM/CPU/GPU temps every 2 s per run -> temps_<name>.tsv.
set -u
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../.." && pwd)
cd "$R"
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
# run in its own session; on deadline kill the whole group (shim producers included)
grouprun() {  # $1 = secs, $2 = log, rest = cmd
  local secs=$1 log=$2; shift 2
  local pf="$D/.sid"; rm -f "$pf"
  setsid bash -c 'echo $$ > "$0"; exec "$@" 2>&1' "$pf" "$@" | stamp > "$log" &
  local p=$! end=$(( $(date +%s) + secs ))
  while kill -0 $p 2>/dev/null && [ "$(date +%s)" -lt "$end" ]; do sleep 5; done
  local sid; sid=$(cat "$pf" 2>/dev/null)
  if [ -n "$sid" ]; then
    pkill -TERM -s "$sid" 2>/dev/null; sleep 15; pkill -KILL -s "$sid" 2>/dev/null
  fi
  wait $p 2>/dev/null
}
jaxrun() {  # $1 = name, $2 = trainer
  logtemps "$D/temps_$1.tsv" & local lp=$!
  sleep 20
  grouprun "$JSECS" "$D/$1.log" env -C jax JAX_COMPILATION_CACHE_DIR=/home/skoonce/.jax_cache \
    TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets ../.venv/bin/python -u "$2"
  sleep 60; kill $lp; sleep 60
}
verrun() {  # $1 = job
  logtemps "$D/temps_v_$1.tsv" & local lp=$!
  sleep 20
  grouprun "$VSECS" "$D/v_$1.log" env ONCE=1 LEAN_MLIR_MAX_STEPS=600 LEAN_MLIR_PROBE_WARM=200 \
    scripts/supervise.sh "$1"
  sleep 60; kill $lp; sleep 60
}
until grep -q '^done' "$D/queue.out" 2>/dev/null; do sleep 30; done
echo "start $(date)"
jaxrun a1   .lake/build/generated_resnet50_imagenet_a1.py
jaxrun vitb .lake/build/generated_vit_b_imagenet_accum.py
jaxrun cnxs .lake/build/generated_convnext_s_imagenet.py
jaxrun cnxb .lake/build/generated_convnext_b_imagenet.py
echo "lake build $(date)"
lake build vit-s-imagenet-verified vit-b-imagenet-verified resnet50-imagenet-verified \
  convnext-s-imagenet-verified convnext-b-imagenet-verified > "$D/lake_build.log" 2>&1 \
  || { echo "lake build FAILED"; exit 1; }
for j in vits-default-emabf16-4gpu vitb-default-emabf16-4gpu r50-a2-bf16-4gpu r50-a1-bf16-4gpu \
         cnxs-default-emabf16-4gpu cnxb-default-emabf16-4gpu; do
  echo "verified $j $(date)"; verrun "$j"
done
echo "done $(date)"
