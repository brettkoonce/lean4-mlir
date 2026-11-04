#!/usr/bin/env bash
# Plan §12: the samples and the eval passes, the GPUs in $GPUS as that many shards (DDIM η = 1, 50
# steps, batch 4, no clamp — §11's finding):
#   1. diff10b50 on Mayo (records_mayo.txt, 20 a patient)            → data/mayo/out/diff_b50
#   2. diff10b50 on its own val (records_val.txt)                     → data/lidc10b50/out/diff
#   3. diff10b50 on the Shepp-Logan 10 % val — the cross-kernel row   → data/lidc10/out/diff_b50
#   4. §9's diff10 (Shepp-Logan) on the b50f val — the other cross    → data/lidc10b50/out/diff_sl
#   5. spread seeds k = 1–3, diff10b50 on its own val (records_spread.txt)
#   6. the MSE nets' eval passes (the trainer's --eval-only on the first GPU of $GPUS, minutes each):
#      mse10b50 on Mayo → data/mayo/out/mse_b50; on the Shepp-Logan val → data/lidc10/out/mse_b50;
#      §9's mse10 on the b50f val → data/lidc10b50/out/mse_sl
# Each sampled job's shards write disjoint records of one dump (+ `.records` sidecar), so a killed
# job resumes. `JOBS="1 6"` picks a subset, `GPUS="0 3"` the cards.
#
#   GPUS="0 1 2 3" JOBS="1 2 3 4 5 6" runs/2026-11-03-lidc-kernel/launch_sample.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-03-lidc-kernel
R9=runs/2026-11-02-lidc-diffusion
JOBS=${JOBS:-"1 2 3 4 5 6"}
GPUS=${GPUS:-"0 1 2 3"}
NSH=$(echo $GPUS | wc -w)
BATCH=${BATCH:-4}
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
lake build lidc-denoise-sample lidc-denoise-train 2>&1 | tail -1
for g in $GPUS; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 500 ] || { echo "⛔ GPU $g busy ($used MiB)"; exit 1; }
done
mkdir -p $RUN/sample
sample() {  # sample PAIRS MODEL K RECORDS DUMP LOGTAG — one shard per GPU in $GPUS, waits for all
  local pairs=$1 model=$2 k=$3 recs=$4 dump=$5 tag=$6
  [ -e "$pairs" ] && [ -e "$recs" ] || { echo "⛔ missing $pairs or $recs"; exit 1; }
  mkdir -p "$dump"
  local pids=() i=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG \
      .lake/build/bin/lidc-denoise-sample "$pairs" --model=$model --seed=0 --k=$k --records="$recs" \
        --shard=$i/$NSH --batch=$BATCH --dump="$dump" > $RUN/sample/$tag.k$k.shard$g.log 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  echo "$(date +%H:%M) $tag k=$k: $NSH shards on GPUs $GPUS"
  for p in "${pids[@]}"; do wait $p || { echo "⛔ a shard of $tag k=$k failed; see $RUN/sample/$tag.k$k.shard*.log"; exit 1; }; done
  tail -1 $RUN/sample/$tag.k$k.shard${GPUS%% *}.log
}
evalmse() {  # evalmse TRAIN_BIN EVAL_BIN CKPT_DIR DUMP OUT LOGTAG — the trainer's forward pass with a finished net
  local train=$1 eval=$2 ckpt=$3 dump=$4 out=$5 tag=$6
  mkdir -p "$dump" "$out"
  CUDA_VISIBLE_DEVICES=${GPUS%% *} LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG .lake/build/bin/lidc-denoise-train \
    "$train" "$eval" --arm=mse --seed=0 --eval-only --ckpt="$ckpt" --dump="$dump" --out="$out" > $RUN/sample/$tag.log 2>&1 \
    || { echo "⛔ $tag failed; see $RUN/sample/$tag.log"; exit 1; }
  echo "$(date +%H:%M) $tag: $(tail -1 $RUN/sample/$tag.log)"
}
for j in $JOBS; do case $j in
  1) sample data/mayo/val.bin      lidc10b50 0 $R9/records_mayo.txt data/mayo/out/diff_b50      mayo-diff10b50 ;;
  2) sample data/lidc10b50/val.bin lidc10b50 0 $R9/records_val.txt  data/lidc10b50/out/diff     lidc10b50-diff10b50 ;;
  3) sample data/lidc10/val.bin    lidc10b50 0 $R9/records_val.txt  data/lidc10/out/diff_b50    lidc10-diff10b50 ;;
  4) sample data/lidc10b50/val.bin lidc10    0 $R9/records_val.txt  data/lidc10b50/out/diff_sl  lidc10b50-diff10 ;;
  5) for k in 1 2 3; do sample data/lidc10b50/val.bin lidc10b50 $k $R9/records_spread.txt data/lidc10b50/out/diff lidc10b50-diff10b50; done ;;
  6) evalmse data/lidc10b50/train.bin data/mayo/val.bin      data/lidc10b50/out/mse data/mayo/out/mse_b50      $RUN/mse10b50_eval eval-mse10b50-mayo
     evalmse data/lidc10b50/train.bin data/lidc10/val.bin    data/lidc10b50/out/mse data/lidc10/out/mse_b50    $RUN/mse10b50_eval eval-mse10b50-lidc10
     evalmse data/lidc10/train.bin    data/lidc10b50/val.bin data/lidc10/out/mse    data/lidc10b50/out/mse_sl $RUN/mse10_eval    eval-mse10-lidc10b50 ;;
  *) echo "⛔ unknown job $j"; exit 1 ;;
esac; done
echo "sampling done"
