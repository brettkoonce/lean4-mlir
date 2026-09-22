#!/usr/bin/env bash
# Memory-error watcher, added 2026-09-22 after the box went down at epoch 263.
#
# ⛔ WHY IT EXISTS. At 22:25 UTC on 09-21 the DIMM CPU_SrcID#0_MC#1_Chan#1_DIMM#0 (32 GB) began
# logging corrected ECC errors under this run's load: 8,835 in 14 min. Uncorrectable ones followed
# at 22:35:57 and 22:38:29, and the box reset at ~22:39 (rebooted 22:42). The e263 checkpoint
# (22:30:19) predates the uncorrectable errors and rescored exactly to its in-run eval,
# 40552/50000. The run resumed from it with the DIMM still installed. None of the other watchers
# can see RAM, so this one logs the EDAC counters: one row per change, polled every 30 s.
# A row that keeps growing is the relapse signature.
# The counters reset at each boot, so a count here is errors SINCE the 09-21 22:42 reboot.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-18-cnx-verified-300ep
OUT=$R/edac.tsv
[ -s "$OUT" ] || printf 'utc\tdimm\tlabel\tce\tue\n' > "$OUT"
declare -A last
while systemctl --user is-active -q cnx-verified; do
  for d in /sys/devices/system/edac/mc/mc*/dimm*; do
    k="$(basename "$(dirname "$d")")/$(basename "$d")"
    v="$(cat "$d/dimm_ce_count" 2>/dev/null)/$(cat "$d/dimm_ue_count" 2>/dev/null)"
    if [ "${last[$k]:-}" != "$v" ]; then
      printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u '+%F %T')" "$k" "$(cat "$d/dimm_label")" \
        "${v%/*}" "${v#*/}" >> "$OUT"
      last[$k]=$v
    fi
  done
  sleep 30
done
