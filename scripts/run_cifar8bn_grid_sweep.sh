#!/bin/bash
# FC-head width sweep of the verified cifar8-BN CNN (AdamW) on the XLA/PJRT path, split across
# GPUs. Rewritten 2026-09-13 for the 4x 4060 Ti box: the previous version ran on the retired
# ROCm pair and scored the pre-fix 9,984-image denominator; this one scores all 10,000 and
# records the Wilson interval the trainer prints.
#
#   scripts/run_cifar8bn_grid_sweep.sh [EPOCHS=25] [OUT=runs/<date>-cifar8bn-grid-xla]
#   GPUS="1 2 3" (env) — which cards to use; widths are dealt round-robin, small/large interleaved.
#
# Needs `lake build cifar8-bn-grid` and the width-slugged renders in .lake/build/
# (tests/TestCifar8AdamTrain.lean writes them for d in 8..4096).
set -u
cd "$(dirname "$0")/.."
EPOCHS="${1:-25}"
OUT="${2:-runs/$(date -u +%Y-%m-%d)-cifar8bn-grid-xla}"
GPUS="${GPUS:-1 2 3}"
BIN=.lake/build/bin/cifar8-bn-grid
[ -x "$BIN" ] || { echo "no $BIN — lake build cifar8-bn-grid first"; exit 1; }
mkdir -p "$OUT"

run_gpu () {
  local gpu="$1"; shift
  local res="$OUT/results_gpu${gpu}.tsv"
  : > "$res"
  for d in "$@"; do
    local log="$OUT/cifar8bn_fc${d}.log"
    rm -f .lake/build/cifar8_bn_${d}_adam_ckpt*                 # never resume an old width
    echo "=== $(date -u +%H:%M:%S) [gpu${gpu}] cifar8bn fc${d} (${EPOCHS} ep) ==="
    CUDA_VISIBLE_DEVICES="$gpu" "$BIN" "$d" "$EPOCHS" data > "$log" 2>&1
    # awk, not grep: the repo's grep wrapper honours .gitignore and skips *.log
    local last acc n lo hi floats msep
    last=$(awk '/test_acc = /{l=$0} END{print l}' "$log")
    acc=$(echo "$last"   | awk 'match($0,/= ([0-9.]+)%/,m){print m[1]}')
    n=$(echo "$last"     | awk 'match($0,/test_acc = [0-9]+\/([0-9]+)/,m){print m[1]}')
    lo=$(echo "$last"    | awk 'match($0,/95% CI ([0-9.]+)/,m){print m[1]}')
    hi=$(echo "$last"    | awk 'match($0,/95% CI [0-9.]+.([0-9.]+)\]/,m){print m[1]}')
    floats=$(awk 'match($0,/params, ([0-9]+) floats/,m){print m[1]; exit}' "$log")
    msep=$(awk 'match($0,/\(([0-9]+)ms\)/,m){l=m[1]} END{print l}' "$log")
    echo -e "${d}\t${acc:-NA}\t${n:-NA}\t${lo:-NA}\t${hi:-NA}\t${floats:-NA}\t${msep:-NA}" >> "$res"
    echo "    [gpu${gpu}] -> fc${d} acc=${acc:-NA}% n=${n:-NA} CI=[${lo:-NA},${hi:-NA}]"
  done
}

# deal widths round-robin over the GPUs, small and large interleaved so no card gets all the fat heads
WIDTHS=(8 4096 16 2048 32 1024 64 512 128 256)
read -ra G <<< "$GPUS"
declare -A LOTS; i=0
for d in "${WIDTHS[@]}"; do g=${G[$((i % ${#G[@]}))]}; LOTS[$g]+="$d "; i=$((i+1)); done
PIDS=()
for g in "${G[@]}"; do run_gpu "$g" ${LOTS[$g]} & PIDS+=($!); done
wait "${PIDS[@]}"

RESULTS="$OUT/results.tsv"
echo -e "d\tacc\tn_eval\tci_lo\tci_hi\tfloats\tms_per_ep" > "$RESULTS"
cat "$OUT"/results_gpu*.tsv | sort -n >> "$RESULTS"
echo "=== sweep done $(date -u +%H:%M:%S) ==="; cat "$RESULTS"
