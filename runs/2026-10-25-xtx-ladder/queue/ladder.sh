#!/bin/bash
# The B50's sub-Imagenette work, replayed on minerva's 7900 XTX (ROCm 7.2.4, kernel 6.8):
# lake run mnist, lake run cifar, the bbn node list (gradcheck, five seeds, the width grid).
# Runs after imagenette.sh; blocks <= 55 min with 15 min idle between (the box's one-hour rule).
cd /home/skoonce/lean/lean4-mlir
while pgrep -f xtx-queue/imagenette.sh >/dev/null; do sleep 60; done
export PATH=/opt/rocm/bin:$PATH PLATFORM_BACKEND=rocm
. scripts/platform/env.sh
[ "$PLATFORM_BACKEND" = rocm ] || { echo "env resolved $PLATFORM_BACKEND"; exit 1; }
L=runs/$(date +%F)-xtx-ladder; mkdir -p $L
G=/sys/bus/pci/devices/0000:0d:00.0; H=$(ls -d $G/hwmon/hwmon* | head -1)
say() { echo "=== $(date '+%F %H:%M:%S') $*"; }
since=$(date '+%Y-%m-%d %H:%M:%S')
dropped() { journalctl -k --since "$since" --no-pager 2>/dev/null | grep -qE 'device lost from bus|amdgpu.*(ring .* timeout|GPU reset)|AER: .*(Uncorrectable|Fatal)'; }
hwlog() {
  echo "time W edge_C junc_C mem_C busy link" > "$1"
  while sleep 1; do
    echo "$(date +%T) $(( $(cat $H/power1_average)/1000000 )) $(( $(cat $H/temp1_input)/1000 )) $(( $(cat $H/temp2_input)/1000 )) $(( $(cat $H/temp3_input)/1000 )) $(cat $G/gpu_busy_percent) $(cut -d' ' -f1 /sys/bus/pci/devices/0000:00:03.1/current_link_speed)x$(cat /sys/bus/pci/devices/0000:00:03.1/current_link_width)" >> "$1"
  done
}
git log --oneline -1 > $L/commit.txt; git status --short | grep -v '^??' >> $L/commit.txt
block() {  # name, then the steps as "n name cmd..." strings
  local b=$1; shift
  say "break 15 min before block $b"; sleep 900
  hwlog $L/hw-sensors.$b.log & local hp=$!
  local t0=$(date +%s)
  for s in "$@"; do
    local n=${s%% *}; s=${s#* }; local name=${s%% *}; local cmd=${s#* }
    local left=$(( 3300 - ($(date +%s) - t0) ))
    [ $left -lt 120 ] && { say "block $b out of time before $n-$name"; break; }
    say "START $n-$name (≤ ${left}s)"
    timeout $left bash -c "$cmd" > $L/$n-$name.out 2>&1; local rc=$?
    say "END $n-$name rc=$rc $(( ($(date +%s)-t0)/60 )) min into the block"
    if dropped; then journalctl -k --since "$since" --no-pager > $L/kernel-drop.txt; kill $hp; say "GPU DROP — stopping"; exit 2; fi
  done
  kill $hp 2>/dev/null; wait $hp 2>/dev/null
}
block 1 \
  "1 build lake build mnist-linear-verified mnist-mlp-verified mnist-cnn-verified cifar8w-ablation cifar8w-bn-ablation cifar8wb-bbn-ablation cifar8-bbn-gradcheck fwd-tie cifar8wb-bbn-grid" \
  "2 mnist lake run mnist" \
  "3 gradcheck .lake/build/bin/cifar8-bbn-gradcheck" \
  "4 cifar lake run cifar"
block 2 "5 seed-sweep SUITE=bbn scripts/sweeps/seed_sweep.sh 0 '1 2 3 4 5'"
block 3 "6 grid-sweep GPUS=0 scripts/sweeps/run_cifar8wb_bbn_grid_sweep.sh"
# a block cut by its time limit resumes from per-epoch checkpoints: one more pass each
if grep -qE 'END 5-seed-sweep rc=124' ~/xtx-queue/ladder.log; then block 2b "5b seed-sweep-resume SUITE=bbn scripts/sweeps/seed_sweep.sh 0 '1 2 3 4 5'"; fi
say "ALLDONE"
