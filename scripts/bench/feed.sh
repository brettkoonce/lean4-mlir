#!/usr/bin/env bash
# feed.sh — the loader alone, CPU only: images/s against worker count, in the trainer's strict-turn
# read (rr) and with each worker draining freely (free). Which of the two scales says whether more
# vCPUs help or the loader's design is the limit (runs/2026-10-05-shim-feed-bench/README.md).
#
#     lake run bench feed <a2 | a1 | vits | vitb> …       FEED_WORKERS="4 8 12 16"   FEED_DIR=…
set -u
cd "$(dirname "$0")/../.." || exit 1
[ $# -gt 0 ] || { echo "usage: lake run bench feed <a2 | a1 | vits | vitb> …"; exit 2; }
OUT="${FEED_DIR:-runs/$(date -u +%F)-feed-$(hostname -s)}"; mkdir -p "$OUT"
{ lscpu | grep -E 'Model name|^CPU\(s\)|Thread|MHz'; free -g | sed -n 2p; } > "$OUT/host.txt"
for n in "$@"; do
  case "$n" in
    a2)   shim=generated_resnet50_imagenet_a2accum_shim.py; extra=(--env SHIM_MIX_CHUNK=256) ;;
    a1)   shim=generated_resnet50_imagenet_a1_shim.py;      extra=(--env SHIM_MIX_CHUNK=256) ;;
    vits|vitb) shim=generated_vit_tiny_imagenet_shim.py;  extra=() ;;   # the shim every verified ViT streams
    *) echo "⛔ no feed row '$n'"; exit 2 ;;
  esac
  for w in ${FEED_WORKERS:-4 8 12 16}; do
    for mode in rr free; do
      python3 scripts/bench/feed_bench.py --shim "$shim" --batch 512 --n "$w" --nclasses 1000 \
        --mode "$mode" --label "$n w$w $mode" "${extra[@]}" | tee -a "$OUT/feed.txt"
    done
  done
done
