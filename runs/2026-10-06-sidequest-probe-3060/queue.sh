#!/bin/bash
# The side-quest ETA probe on the 3060 box (mars: 4x RTX 3060 12 GB, CUDA 13.3, /home/skoonce/.venv-cuda),
# on the tree at fc320752 — the same 14 runs as ares' runs/2026-10-04-dimm-fan-thermal/ (queue2/4/5),
# by the same method, so the two tables compare row for row:
#   JAX      = the trainer run directly, 12-min window (MNv4 20), step lines wall-stamped -> winrate.py
#   verified = the §3a smoke: ONCE=1 LEAN_MLIR_MAX_STEPS=600 LEAN_MLIR_PROBE_WARM=200 scripts/supervise.sh <job>
# No sensors(1) here (no jc42 / k10temp), so the per-run trace is the GPUs (temp, util, power,
# SM clock, memory used) and the host load every 2 s -> gpu_<name>.tsv.
# JAX runs write no checkpoint (no LEAN_MLIR_PARAMS_OUT / CKPT_EVERY); a 600-step verified smoke
# stops before its first epoch boundary.
set -u
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../.." && pwd)
cd "$R"
. scripts/jobs/_box.sh
JSECS=${JSECS:-720}; VSECS=${VSECS:-1800}
logtrace() {  # $1 = out
  echo -e "t\tload1\tgpu_temp\tgpu_util\tgpu_w\tsm_mhz\tmem_mib" > "$1"
  local t0; t0=$(date +%s.%N)
  while :; do
    local g; g=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.sm,memory.used \
      --format=csv,noheader,nounits | awk -F', ' '{for(i=1;i<=5;i++) c[i]=c[i] (NR>1?",":"") $i}
        END{print c[1]"\t"c[2]"\t"c[3]"\t"c[4]"\t"c[5]}')
    printf "%.1f\t%s\t%s\n" "$(echo "$(date +%s.%N) - $t0" | bc)" "$(cut -d' ' -f1 /proc/loadavg)" "$g" >> "$1"
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
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>100' | wc -l)" = 0 ] && return 0; sleep 5; done
  echo "⚠ GPUs not idle after 5 min"; }
jaxrun() {  # $1 = name, $2 = trainer (relative to jax/)
  gpus_idle; logtrace "$D/gpu_$1.tsv" & local lp=$!
  sleep 20
  grouprun "$JSECS" "$D/$1.log" env -C jax JAX_COMPILATION_CACHE_DIR=/home/skoonce/.jax_cache \
    TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets "$BOX_PY" -u "$2"
  sleep 30; kill $lp; sleep 30
}
verrun() {  # $1 = job
  gpus_idle; logtrace "$D/gpu_v_$1.tsv" & local lp=$!
  sleep 20
  grouprun "$VSECS" "$D/v_$1.log" env ONCE=1 LEAN_MLIR_MAX_STEPS=600 LEAN_MLIR_PROBE_WARM=200 \
    scripts/supervise.sh "$1"
  sleep 30; kill $lp; sleep 30
}
echo "start $(date -u)"
for spec in "vits .lake/build/generated_vit_s_imagenet.py" \
            "a2 .lake/build/generated_resnet50_imagenet_a2accum.py" \
            "a1 .lake/build/generated_resnet50_imagenet_a1.py" \
            "vitb .lake/build/generated_vit_b_imagenet_accum.py" \
            "cnxs .lake/build/generated_convnext_s_imagenet.py" \
            "cnxb .lake/build/generated_convnext_b_imagenet.py"; do
  set -- $spec; echo "jax $1 $(date -u)"; jaxrun "$1" "$2"
done
echo "jax mnv4 $(date -u)"; JSECS=1200 jaxrun mnv4 .lake/build/generated_mobilenet_v4_imagenet_full.py
for j in vits-default-emabf16-4gpu vitb-default-emabf16-4gpu r50-a2-bf16-4gpu r50-a1-bf16-4gpu \
         cnxs-default-emabf16-4gpu cnxb-default-emabf16-4gpu mnv4-full-4gpu; do
  echo "verified $j $(date -u)"; verrun "$j"
done
echo "done $(date -u)"
