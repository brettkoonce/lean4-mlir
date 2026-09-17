#!/usr/bin/env bash
# Durable detector for the R34 bf16 run. Writes WATCH_EVENT and exits on whichever comes first —
# a SUSTAINED loader-fault signature, or the run ending.
#
# ⚠⚠ REVISED 2026-09-16 22:0x after the first version fired on e57 (1055 s) and stopped watching.
# That was NOT a loader fault: it tracked host MemAvailable falling 126 -> 87 GiB for ~2.5 h
# (something external took ~35 GiB, evicting the TFRecord page cache), and it recovered on its own
# when the RAM came back. A single slow epoch is not evidence — B0's real fault was MONOTONE and
# only ever cleared with a restart. So the trigger is now THREE CONSECUTIVE epochs over 950 s
# *while host memory is healthy*, which is the combination a genuine loader defect produces.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-16-r34-bf16-90ep
EV=$R/WATCH_EVENT
rm -f "$EV"
# ⛔ Only look FORWARD. The historical e54-e63 host-memory bump satisfies any retrospective
# test and would fire the watch the instant it starts — it did, on the first try.
FROM=${FROM:-75}
loaders() {
  awk -F'\t' 'NR>1 && $8>10 {last[$3]=$0} END{for (k in last) print last[k]}' "$R/loader_rss.tsv" 2>/dev/null \
    | sort -t$'\t' -k3 -n \
    | awk -F'\t' '{printf "  pid %s  age %5.1f h  rss %5.2f GiB  arena %5.2f GiB  maps %s\n", $3, $4/3600, $5, $6, $7}'
}
while systemctl --user is-active -q r34-bf16; do
  # three consecutive > 950 s, and MemAvailable healthy (> 110 GiB) on the last of them
  hit=$(tail -n +2 "$R/epoch_clock.tsv" 2>/dev/null | awk -F'\t' -v from="$FROM" '
    NR>1 { d=$2-p; if ($1+0 <= from+0) { p=$2; next }
           if (d>950) run++; else run=0
           if (run>=3 && $5/1048576 > 110) { printf "e%s=%ds avail=%.1fGiB", $1, d, $5/1048576; exit } }
    { p=$2 }')
  if [ -n "$hit" ]; then
    { echo "⚠⚠ SUSTAINED LOADER FAULT at $(date -u '+%F %T') UTC"
      echo "3+ consecutive epochs over 950 s with host memory healthy: $hit"
      echo "loaders at detection:"; loaders; } > "$EV"
    exit 0
  fi
  sleep 300
done
{ echo "=== RUN ENDED at $(date -u '+%F %T') UTC ==="
  tail -6 "$R/master.log"
  echo "--- final evals ---"
  grep -ahE '^  epoch [0-9]+: ' "$R/full.log" 2>/dev/null | tail -3
  echo "--- loaders at end ---"; loaders; } > "$EV"
