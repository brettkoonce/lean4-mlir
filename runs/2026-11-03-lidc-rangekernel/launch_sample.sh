#!/usr/bin/env bash
# Plan §13: the samples and eval passes of the range × kernel arms, the GPUs in $GPUS as that many
# shards (DDIM η = 1, 50 steps, batch 4, no clamp; the generator reads each file's σ̂ sidecar):
#   1. diffmixsigb50 on Mayo (records_mayo.txt)                         → data/mayo/out/diffmixsig_b50
#   2. diffmixsigb50 on the b50f 10 % val (records_val.txt)             → data/lidc10b50/out/diffmixsig
#   3. diffmixsigb50 on its own val (records_mixb50.txt; gate R3)       → data/lidcmixb50/out/diffmixsig
#   4. msemixb50's forward pass (--eval-only) on Mayo and on the b50f 10 % val
#                                                                       → data/mayo/out/msemix_b50, data/lidc10b50/out/msemix
#   GPUS="0 1 2 3" JOBS="1 2 3 4" runs/2026-11-03-lidc-rangekernel/launch_sample.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-03-lidc-rangekernel
R9=runs/2026-11-02-lidc-diffusion
D=data/lidcmixb50
JOBS=${JOBS:-"1 2 3 4"}
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
sample() {  # sample PAIRS K RECORDS DUMP LOGTAG — model lidcmixb50 with --sigma
  local pairs=$1 k=$2 recs=$3 dump=$4 tag=$5
  [ -e "$pairs" ] && [ -e "$recs" ] && [ -e "$pairs.sigma.f32" ] || { echo "⛔ missing $pairs, $recs or $pairs.sigma.f32"; exit 1; }
  mkdir -p "$dump"
  local pids=() i=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG \
      .lake/build/bin/lidc-denoise-sample "$pairs" --model=lidcmixb50 --sigma --seed=0 --k=$k --records="$recs" \
        --shard=$i/$NSH --batch=$BATCH --dump="$dump" > $RUN/sample/$tag.k$k.shard$g.log 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  echo "$(date +%H:%M) $tag k=$k: $NSH shards on GPUs $GPUS"
  for p in "${pids[@]}"; do wait $p || { echo "⛔ a shard of $tag k=$k failed; see $RUN/sample/$tag.k$k.shard*.log"; exit 1; }; done
  tail -1 $RUN/sample/$tag.k$k.shard${GPUS%% *}.log
}
evalmse() {  # evalmse EVAL_BIN DUMP LOGTAG — msemixb50's forward pass with the finished net
  local eval=$1 dump=$2 tag=$3
  mkdir -p "$dump" $RUN/msemixb50_eval
  CUDA_VISIBLE_DEVICES=${GPUS%% *} LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG .lake/build/bin/lidc-denoise-train \
    $D/train.bin "$eval" --arm=mse --seed=0 --eval-only --ckpt=$D/out/msemix --dump="$dump" --out=$RUN/msemixb50_eval \
    > $RUN/sample/$tag.log 2>&1 || { echo "⛔ $tag failed; see $RUN/sample/$tag.log"; exit 1; }
  echo "$(date +%H:%M) $tag: $(tail -1 $RUN/sample/$tag.log)"
}
for j in $JOBS; do case $j in
  1) sample data/mayo/val.bin      0 $R9/records_mayo.txt    data/mayo/out/diffmixsig_b50   mayo-diffmixsigb50 ;;
  2) sample data/lidc10b50/val.bin 0 $R9/records_val.txt     data/lidc10b50/out/diffmixsig  lidc10b50-diffmixsigb50 ;;
  3) sample $D/val.bin             0 $RUN/records_mixb50.txt $D/out/diffmixsig              mixb50-diffmixsigb50 ;;
  4) evalmse data/mayo/val.bin      data/mayo/out/msemix_b50   eval-msemixb50-mayo
     evalmse data/lidc10b50/val.bin data/lidc10b50/out/msemix  eval-msemixb50-lidc10b50 ;;
  *) echo "⛔ unknown job $j"; exit 1 ;;
esac; done
echo "sampling done"
