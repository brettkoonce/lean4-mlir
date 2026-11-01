#!/usr/bin/env bash
# 1 Hz XTX sensor log: power W, edge/junction/mem C, busy %, link. sync each line so a reset keeps it.
D=/sys/bus/pci/devices/0000:43:00.0; H=$(ls -d $D/hwmon/hwmon* | head -1); OUT=$1
echo "time,W,edge,junction,mem,busy,link" >> "$OUT"
while true; do
  p=$(cat $H/power1_average 2>/dev/null || cat $H/power1_input 2>/dev/null)
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$(date +%T)" $((p/1000000)) $(( $(cat $H/temp1_input)/1000 )) \
    $(( $(cat $H/temp2_input)/1000 )) $(( $(cat $H/temp3_input)/1000 )) "$(cat $D/gpu_busy_percent)" \
    "$(cat $D/current_link_speed | tr -d ' ')x$(cat $D/current_link_width)" >> "$OUT"
  sync "$OUT"; sleep 1
done
