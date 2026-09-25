#!/usr/bin/env bash
# ▶ R50 2018 sync-BN copy (2026-09-24) of the R34/A3 sync-BN watchers; only paths/names/patterns retargeted.
# Durable detector for the R34 sync-BN bf16 run (copied 2026-09-22 from the ConvNeXt run's). Writes
# WATCH_EVENT and exits on whichever comes first — a SUSTAINED loader-fault signature, or the run
# ending.
#
# ⚠⚠ SELF-CALIBRATING. The 09-16 R34 run's watcher hardcoded 950 s. Streaming val and the sharded
# eval move this run's epoch time, so it stays a forecast until the run produces it, and a
# hardcoded threshold derived from a probe fires on a forecast error. Baseline = MEDIAN of the epochs in [BASE_FROM, BASE_TO], taken
# once both exist; trigger = THREE CONSECUTIVE epochs over FACTOR x baseline while host memory is
# healthy.
#
# ⚠ Three consecutive, and the memory condition, are both from the R34 lesson: its first watcher
# fired on ONE slow epoch that was tracking host MemAvailable falling 126 -> 87 GiB (something
# external took ~35 GiB and evicted the TFRecord page cache). That recovered on its own. B0's real
# fault was MONOTONE and only ever cleared with a restart.
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-09-24-r50-2018-syncbn-bf16-90ep
EV=$R/WATCH_EVENT
rm -f "$EV"
BASE_FROM=${BASE_FROM:-4}      # skip e1-e3: compile, val preload and the resume test live there
BASE_TO=${BASE_TO:-12}
FACTOR=${FACTOR:-1.25}
AVAIL_GIB=${AVAIL_GIB:-110}
loaders() {
  awk -F'\t' 'NR>1 && $8>10 {last[$3]=$0} END{for (k in last) print last[k]}' "$R/loader_rss.tsv" 2>/dev/null \
    | sort -t$'\t' -k3 -n \
    | awk -F'\t' '{printf "  pid %s  age %5.1f h  rss %5.2f GiB  arena %5.2f GiB  maps %s\n", $3, $4/3600, $5, $6, $7}'
}
while systemctl --user is-active -q r50-2018; do
  hit=$(tail -n +2 "$R/epoch_clock.tsv" 2>/dev/null | awk -F'\t' \
      -v bf="$BASE_FROM" -v bt="$BASE_TO" -v fac="$FACTOR" -v ag="$AVAIL_GIB" '
    { if (p>0) { d=$2-p; ep=$1+0; dur[ep]=d; if (ep>=bf && ep<=bt) base[n++]=d } p=$2; le=ep }
    END{
      if (n < 5) exit 0                                  # not enough baseline yet
      asort(base); med = (n%2) ? base[(n+1)/2] : (base[n/2]+base[n/2+1])/2
      thr = med * fac; run = 0
      for (e = bt+1; e <= le; e++) {
        if (!(e in dur)) continue
        if (dur[e] > thr) run++; else run = 0
        if (run >= 3) { printf "e%d=%ds thr=%.0fs base=%.0fs", e, dur[e], thr, med; exit }
      }
    }')
  if [ -n "$hit" ]; then
    # the memory condition, checked at detection time rather than in the awk
    av=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
    if awk -v a="$av" -v g="$AVAIL_GIB" 'BEGIN{exit !(a>g)}'; then
      { echo "⚠⚠ SUSTAINED LOADER FAULT at $(date -u '+%F %T') UTC"
        echo "3+ consecutive epochs over threshold with host memory healthy: $hit avail=${av}GiB"
        echo "⛔ The loaders run BARE on this conf (no LEAN_MLIR_SHIM_RESPAWN_EPOCHS), and R34's"
        echo "   flip-only shim never degraded; A3's shim runs RandAugment m6 (B0-like). A sustained fault here is a new"
        echo "   finding. Record it in RESULTS.md verbatim, then kill right after a checkpoint and"
        echo "   resume with LEAN_MLIR_SHIM_RESPAWN_EPOCHS=10, or REST_EPOCHS every 40 (B0's cure)."
        echo "loaders at detection:"; loaders; } > "$EV"
      exit 0
    fi
  fi
  sleep 300
done
{ echo "=== RUN ENDED at $(date -u '+%F %T') UTC ==="
  tail -6 "$R/master.log" 2>/dev/null
  echo "--- final evals ---"
  grep -ahE '^  epoch [0-9]+: ' "$R/full.log" 2>/dev/null | tail -3
  echo "--- loaders at end ---"; loaders; } > "$EV"
