#!/bin/bash
# Imagenette, 80 epochs, all seven nets on minerva's 7900 XTX (ROCm 7.2.4, kernel 6.8, no riser).
# Brett's rule for this box: no GPU stretch over an hour. Each block runs <= 55 min, then 15 min
# idle; a net cut by the block limit resumes from its last epoch (checkpoint tag xtx80).
cd /home/skoonce/lean/lean4-mlir
export PATH=/opt/rocm/bin:$PATH LEAN_MLIR_CKPT_TAG=xtx80
G=/sys/bus/pci/devices/0000:0d:00.0
H=$(ls -d $G/hwmon/hwmon* | head -1)
BLOCK=3300 BREAK=900
say() { echo "=== $(date '+%F %H:%M:%S') $*"; }
since=$(date '+%Y-%m-%d %H:%M:%S')
dropped() { journalctl -k --since "$since" --no-pager 2>/dev/null | grep -qE 'device lost from bus|amdgpu.*(ring .* timeout|GPU reset)|AER: .*(Uncorrectable|Fatal)'; }
hwlog() {
  echo "time W edge_C junc_C mem_C busy link" > "$1"
  while sleep 1; do
    echo "$(date +%T) $(( $(cat $H/power1_average)/1000000 )) $(( $(cat $H/temp1_input)/1000 )) $(( $(cat $H/temp2_input)/1000 )) $(( $(cat $H/temp3_input)/1000 )) $(cat $G/gpu_busy_percent) $(cut -d' ' -f1 /sys/bus/pci/devices/0000:00:03.1/current_link_speed)x$(cat /sys/bus/pci/devices/0000:00:03.1/current_link_width)" >> "$1"
  done
}
first=1   # the queue before it (miopen.sh) leaves the card warm: break first
while pgrep -f "xtx-queue/(ladder|miopen).sh" >/dev/null; do sleep 60; done
for net in "$@"; do
  tr=$net-verified-adam dir=runs/$(date +%F)-xtx-$tr; mkdir -p $dir; part=0
  while :; do
    [ $first = 0 ] && { say "break $((BREAK/60)) min"; sleep $BREAK; }; first=0
    part=$((part+1))
    [ -e $dir/$tr.log ] && mv $dir/$tr.log $dir/$tr.part$((part-1)).log
    hwlog $dir/hw-sensors.part$part.log & hp=$!
    say "start $tr block $part → $dir"
    ( export RUN_LOG_DIR=$dir; exec timeout $BLOCK ./run.sh $tr 0 rocm > /dev/null 2>&1 ) & rp=$!
    while kill -0 $rp 2>/dev/null; do
      sleep 30
      age=$(( $(date +%s) - $(stat -c %Y $dir/$tr.log 2>/dev/null || date +%s) ))
      [ $age -gt 1200 ] && { say "STALL ($age s) — killing $tr"; pkill -f "bin/$tr"; sleep 10; pkill -9 -f "bin/$tr"; }
      dropped && break
    done
    wait $rp; rc=$?
    kill $hp 2>/dev/null; wait $hp 2>/dev/null
    say "end $tr block $part rc=$rc $(grep -E 'epoch [0-9]+: val_acc' $dir/$tr.log | tail -1 | cut -c1-90)"
    if dropped; then journalctl -k --since "$since" --no-pager > $dir/kernel-drop.txt; pkill -f "bin/$tr"; say "GPU DROP — stopping the queue"; exit 2; fi
    [ $rc = 0 ] && break
    [ $rc = 124 ] && continue          # block limit: resume after the break
    [ $part -ge 12 ] && { say "giving up on $tr"; break; }
    say "$tr exited rc=$rc — resuming after the break"
  done
done
say "all done"
