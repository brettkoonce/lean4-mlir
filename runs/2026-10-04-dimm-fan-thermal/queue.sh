#!/bin/bash
# DIMM-fan re-test (2026-10-04): DIMMs re-slotted per the manual + two fans on the memory.
# Baseline 09-29: ViT-S JAX smoke, DIMM 1-001d hit 79.7 C at t~200 s, first stall there,
# 54% stall, 656 ms/step overall vs 300 clean. Same smoke here, then RSB-A2 JAX.
# Temps every 2 s (jc42 DIMMs at 1-0018/19/1c/1d after the re-slot; were 1c-1f, k10temp, GPUs) -> temps_<net>.tsv;
# trainer lines wall-timestamped -> <net>.log.
set -u
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../.." && pwd)
SECS=${SECS:-900}
logtemps() {  # $1 = out
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
run() {  # $1 = name, $2 = trainer
  logtemps "$D/temps_$1.tsv" & local lp=$!
  sleep 20   # idle baseline
  ( cd "$R/jax" && timeout "$SECS" env JAX_COMPILATION_CACHE_DIR=/home/skoonce/.jax_cache \
      TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets ../.venv/bin/python -u "$2" 2>&1 ) \
    | while IFS= read -r l; do printf '%s\t%s\n' "$(date +%s.%N)" "$l"; done > "$D/$1.log"
  sleep 60   # cool-down tail
  kill $lp
}
echo "start $(date)"
run vits .lake/build/generated_vit_s_imagenet.py
sleep 120
run a2 .lake/build/generated_resnet50_imagenet_a2accum.py
echo "done $(date)"
