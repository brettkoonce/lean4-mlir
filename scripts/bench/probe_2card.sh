#!/usr/bin/env bash
# probe_2card.sh — the rented-box sequence, cheapest answer first: host facts, the rendered steps on
# the cards, the loader on the CPU, then real-data rows on both paths. Everything lands in one
# runs/<date>-probe-<host>/, ready to scp home before the box is released.
#
#     lake run bench probe            PROBE_NETS="a2 vits vitb"   PROBE_IMAGES=400000
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT="runs/$(date -u +%F)-probe-$(hostname -s)"; mkdir -p "$OUT"
NETS="${PROBE_NETS:-a2 vits vitb}"
IMG="${PROBE_IMAGES:-400000}"
{ date -u; nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|MHz'; free -g; df -h "${TFDS_DATA_DIR:-$HOME/tensorflow_datasets}"
} 2>&1 | tee "$OUT/host.txt"
DEVICE_DIR="$OUT" scripts/bench/device.sh $NETS
FEED_DIR="$OUT"   scripts/bench/feed.sh a2 vits
for n in $NETS; do
  BENCH_DIR="$OUT" lake run bench "$n" --2card --images "$IMG" || echo "⛔ $n: see $OUT"
done
column -t -s $'\t' "$OUT/bench.tsv" | cut -c1-300
