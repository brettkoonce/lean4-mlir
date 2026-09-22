#!/usr/bin/env bash
# measure_producers.sh — planning/streaming_val.md §2: val producer throughput at 1, 2 and 4
# batch-block producers, streaming `validation` to /dev/null. Wall time of the SLOWEST producer per
# arm (the trainer's round-robin waits for it), images/s = 50,000 / that. CPU only.
#   runs/2026-09-22-streamed-val/measure_producers.sh [shim.py]
set -uo pipefail
cd "$(dirname "$0")/../.."
SHIM=${1:-jax/.lake/build/generated_vit_tiny_imagenet_shim.py}
PY=.venv/bin/python3
OUT=runs/2026-09-22-streamed-val/producers
mkdir -p "$OUT"
echo "shim $SHIM"
for N in 1 2 4; do
  t0=$(date +%s.%N); pids=()
  for ((k=0; k<N; k++)); do
    if [ "$N" = 1 ]; then sh=(); else sh=(SHIM_SHARD="$k/$N"); fi
    ( env SHIM_SPLIT=validation SHIM_BATCH=256 "${sh[@]}" "$PY" "$SHIM" 2> "$OUT/N${N}_p${k}.err" | wc -c > "$OUT/N${N}_p${k}.bytes" ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  t1=$(date +%s.%N)
  wall=$(python3 -c "print(round($t1 - $t0, 1))")
  bytes=$(cat "$OUT"/N${N}_p*.bytes | paste -sd+ | bc)
  # wire v3: per batch 4 (rows) + 4·rows (labels) + 4·rows·flat (images); flat = 150,528 → 602,116 B/img
  imgs=$(python3 -c "print(($bytes - 16) // (4 + 4*150528) if $bytes > 16 else 0)")
  printf "N=%d  wall %6.1f s  bytes %d  ≈ %d images  → %.0f img/s (incl. TF startup)\n" "$N" "$wall" "$bytes" "$imgs" "$(python3 -c "print($imgs / $wall)")"
done
