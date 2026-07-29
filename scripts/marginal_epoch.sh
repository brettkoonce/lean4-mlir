#!/usr/bin/env bash
# marginal_epoch.sh — time a verified trainer the ONLY way that is honest.
#
#   scripts/marginal_epoch.sh <logfile> -- <command...>
#
# Runs <command>, stamps each "  epoch N:" line with elapsed seconds, and reports the MARGINAL
# epoch (T₃ − T₁)/2 — which cancels compile, the one-time dataset load and process startup exactly.
#
# Why this exists: wall-clock-minus-compile produced a WRONG ratio twice in the XLA thread
# (planning/xla_pjrt_handoff.md §2e-ter — a 1.43× that was really 1.67×, and a "21%/34% loader
# overhead" that was really 4.3 s and constant). Imagenette is read from disk once and expanded to
# ~7.45 GiB of f32 in host RAM, ~7–12 s, and that lands entirely in epoch 1. Subtracting only the
# compile leaves it in the measurement.
#
# Needs at least 3 epochs. Run it SOLO on the box — a concurrent job on the other GPU still
# contends for the host loader, which is ~4.1–4.5 s of every epoch.

set -uo pipefail

log=${1:?usage: marginal_epoch.sh <logfile> -- <command...>}
shift
[ "${1:-}" = "--" ] && shift

start=$(date +%s)
"$@" 2>&1 | awk -v s="$start" '
  /^  epoch [0-9]+:/ { t = systime() - s; printf "[t=%6ds] %s\n", t, $0; fflush(); next }
  { print; fflush() }
' | tee "$log"

echo
echo "── marginal epoch, from $log ──"
awk '
  match($0, /^\[t= *([0-9]+)s\] +epoch ([0-9]+):/, m) { t[m[2]+0] = m[1]+0; n++ }
  END {
    if (!(1 in t) || !(3 in t)) { print "  need >=3 epochs; got " n; exit 1 }
    printf "  T1 = %ds, T3 = %ds\n", t[1], t[3]
    printf "  MARGINAL EPOCH (T3-T1)/2 = %.1f s   =>  80 epochs = %.2f h\n", \
           (t[3]-t[1])/2, (t[3]-t[1])/2*80/3600
    printf "  (epoch 1 = %ds carries compile + the one-time ~7.45 GiB dataset load; do NOT quote it)\n", t[1]
  }
' "$log"
