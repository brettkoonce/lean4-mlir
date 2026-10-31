#!/bin/bash
# Tail of ladder.sh, added mid-run: on the 7900 XTX a bbn seed takes ~29 min (6.5 on a 4060 Ti),
# so five seeds need ~3 blocks and ladder.sh only gives them two. Waits for ladder.sh's PID, then
# finishes the seeds and any grid widths a timed-out block 3 left. Launched with the argument
# "xtx-queue/ladder.sh" so miopen.sh / imagenette2.sh's pgrep keeps waiting on it.
cd /home/skoonce/lean/lean4-mlir
while kill -0 125930 2>/dev/null; do sleep 60; done
export PATH=/opt/rocm/bin:$PATH PLATFORM_BACKEND=rocm
. scripts/platform/env.sh
L=runs/2026-10-25-xtx-ladder
G=/sys/bus/pci/devices/0000:0d:00.0; H=$(ls -d $G/hwmon/hwmon* | head -1)
say() { echo "=== $(date '+%F %H:%M:%S') $*"; }
since=$(date '+%Y-%m-%d %H:%M:%S')
dropped() { journalctl -k --since "$since" --no-pager 2>/dev/null | grep -qE 'device lost from bus|amdgpu.*(ring .* timeout|GPU reset)|AER: .*(Uncorrectable|Fatal)'; }
grep -q 'GPU DROP' ~/xtx-queue/ladder.log && { say "ladder dropped — not continuing"; exit 2; }
hwlog() {
  echo "time W edge_C junc_C mem_C busy link" > "$1"
  while sleep 1; do
    echo "$(date +%T) $(( $(cat $H/power1_average)/1000000 )) $(( $(cat $H/temp1_input)/1000 )) $(( $(cat $H/temp2_input)/1000 )) $(( $(cat $H/temp3_input)/1000 )) $(cat $G/gpu_busy_percent) $(cut -d' ' -f1 /sys/bus/pci/devices/0000:00:03.1/current_link_speed)x$(cat /sys/bus/pci/devices/0000:00:03.1/current_link_width)" >> "$1"
  done
}
run_block() {  # tag, cmd
  say "break 15 min before block $1"; sleep 900
  hwlog $L/hw-sensors.$1.log & local hp=$!
  say "START $1"; timeout 3300 bash -c "$2" > $L/$1.out 2>&1; local rc=$?
  say "END $1 rc=$rc"
  kill $hp 2>/dev/null; wait $hp 2>/dev/null
  if dropped; then journalctl -k --since "$since" --no-pager > $L/kernel-drop.txt; say "GPU DROP — stopping"; exit 2; fi
}
S=runs/2026-10-24-bbn-seeds
seeds_done() { [ $(ls $S/done 2>/dev/null | wc -l) -ge 5 ]; }
for i in 1 2 3; do
  seeds_done && break
  run_block 5c$i "SWEEP_DIR=$S SUITE=bbn scripts/sweeps/seed_sweep.sh 0 '1 2 3 4 5'"
done
seeds_done || { say "seeds still not done after 3 blocks — giving up on them"; }
GR=$(ls -d runs/2026-10-2[56]-cifar8wb-bbn-grid-xla 2>/dev/null | tail -1)
missing() { for d in 8 4096 16 2048 32 1024 64 512 128 256; do grep -q 'epoch 25:' $GR/cifar8wbbbn_fc$d.log 2>/dev/null || echo $d; done; }
if [ -n "$GR" ]; then
  for i in 1 2; do
    m=$(missing | tr '\n' ' '); [ -z "$m" ] && break
    run_block 6g$i "for d in $m; do rm -f .lake/build/cifar8wb_bbn_\${d}_adam_ckpt*; echo \"=== fc\$d\"; .lake/build/bin/cifar8wb-bbn-grid \$d 25 data > $GR/cifar8wbbbn_fc\$d.log 2>&1; done"
  done
fi
say "ALLDONE"
