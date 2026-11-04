#!/usr/bin/env bash
# Phase 5's sampling (planning/lidc_ct_denoising_demo.md §10.5), the GPUs in $GPUS as that many
# shards of each job, DDIM η = 1, 50 steps (≈ 1 s a 512² slice at batch 4 on an idle card; the
# smoke measured 2.1 s on a shared one):
#   1. LIDC 25 %: k = 0 on records_val.txt (the 1,205 nodule slices + 400 others)
#   2. LIDC 10 %: the same
#   3. LIDC 25 % spread study: k = 1, 2, 3 on records_spread.txt (200 slices)
#   4. LIDC 10 % spread study: the same
#   5. Mayo chest at 10 %: the 10 % model, k = 0 on records_mayo.txt (20 a patient)
# Each job's shards write disjoint records of one dump (+ a `.records` sidecar), so a killed job
# resumes. Jobs run in sequence; `JOBS="1 3"` picks a subset, `GPUS="0 3"` the cards.
#
#   GPUS="0 3" JOBS="1 3" runs/2026-11-02-lidc-diffusion/launch_sample.sh   # foreground; nohup it
#
# Needs: the diffusion checkpoints (launch_train.sh), the record lists
# (scripts/demos/lidc_select_records.py), data/mayo/val.bin (scripts/datasets/preprocess_mayo.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-02-lidc-diffusion
JOBS=${JOBS:-"1 2 3 4 5"}
GPUS=${GPUS:-"0 1 2 3"}
NSH=$(echo $GPUS | wc -w)
BATCH=${BATCH:-4}
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
lake build lidc-denoise-sample 2>&1 | tail -1
for g in $GPUS; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 500 ] || { echo "⛔ GPU $g busy ($used MiB)"; exit 1; }
done

sample() {  # sample PAIRS MODEL K RECORDS DUMP LOGTAG — one shard per GPU in $GPUS, waits for all
  local pairs=$1 model=$2 k=$3 recs=$4 dump=$5 tag=$6
  [ -e "$pairs" ] && [ -e "$recs" ] || { echo "⛔ missing $pairs or $recs"; exit 1; }
  mkdir -p "$dump" $RUN/sample
  local pids=() i=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG \
      .lake/build/bin/lidc-denoise-sample "$pairs" --model=$model --seed=0 --k=$k --records="$recs" \
        --shard=$i/$NSH --batch=$BATCH --dump="$dump" > $RUN/sample/$tag.k$k.shard$g.log 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  echo "$(date +%H:%M) $tag k=$k: $NSH shards launched on GPUs $GPUS"
  for p in "${pids[@]}"; do wait $p || { echo "⛔ a shard of $tag k=$k failed; see $RUN/sample/$tag.k$k.shard*.log"; exit 1; }; done
  tail -1 $RUN/sample/$tag.k$k.shard${GPUS%% *}.log
}
for j in $JOBS; do case $j in
  1) sample data/lidc/val.bin   lidc   0 $RUN/records_val.txt    data/lidc/out/diff   lidc25 ;;
  2) sample data/lidc10/val.bin lidc10 0 $RUN/records_val.txt    data/lidc10/out/diff lidc10 ;;
  3) for k in 1 2 3; do sample data/lidc/val.bin   lidc   $k $RUN/records_spread.txt data/lidc/out/diff   lidc25; done ;;
  4) for k in 1 2 3; do sample data/lidc10/val.bin lidc10 $k $RUN/records_spread.txt data/lidc10/out/diff lidc10; done ;;
  5) sample data/mayo/val.bin   lidc10 0 $RUN/records_mayo.txt   data/mayo/out/diff    mayo ;;
esac; done
echo "sampling done"
