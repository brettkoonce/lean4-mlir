#!/usr/bin/env bash
# Plan §11, R3: the samples, the GPUs in $GPUS as that many shards (DDIM η = 1, 50 steps, batch 4,
# --clamp on unless CLAMP=0):
#   0. the §9 generator, 10 % build, on Mayo with --clamp only (the sampler-side guard alone)
#   1. diffmix     on the 25 % val, the 10 % val (records_val.txt) and Mayo (records_mayo.txt), k = 0
#   2. diffmixsig  the same (its --sigma needs each file's .sigma.f32)
#   3. diffmix     on the range build's own val, k = 0 (gate R3)
#   4. diffmixsig  the same
#   5. spread seeds k = 1–3 for ARM_SPREAD (default diffmixsig) on records_spread.txt, both LIDC vals
# Dumps: data/<root>/out/{diff_clamp,diffmix,diffmixsig}/, each with its .records sidecar.
#
#   GPUS="0 1 2 3" JOBS="0 1 2 3 4" runs/2026-11-03-lidc-range/launch_sample.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-03-lidc-range
R9=runs/2026-11-02-lidc-diffusion
JOBS=${JOBS:-"0 1 2 3 4"}
GPUS=${GPUS:-"0 1 2 3"}
NSH=$(echo $GPUS | wc -w)
BATCH=${BATCH:-4}
CLAMP=${CLAMP:-1}; CL=""; [ "$CLAMP" = 1 ] && CL="--clamp"
ARM_SPREAD=${ARM_SPREAD:-diffmixsig}
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
lake build lidc-denoise-sample 2>&1 | tail -1
for g in $GPUS; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 500 ] || { echo "⛔ GPU $g busy ($used MiB)"; exit 1; }
done
mkdir -p $RUN/sample
sample() {  # sample PAIRS MODEL EXTRA K RECORDS DUMP LOGTAG
  local pairs=$1 model=$2 extra=$3 k=$4 recs=$5 dump=$6 tag=$7
  [ -e "$pairs" ] && [ -e "$recs" ] || { echo "⛔ missing $pairs or $recs"; exit 1; }
  case " $extra " in *" --sigma "*) [ -e "$pairs.sigma.f32" ] || { echo "⛔ missing $pairs.sigma.f32"; exit 1; };; esac
  mkdir -p "$dump"
  local pids=() i=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG \
      .lake/build/bin/lidc-denoise-sample "$pairs" --model=$model $extra $CL --seed=0 --k=$k --records="$recs" \
        --shard=$i/$NSH --batch=$BATCH --dump="$dump" > $RUN/sample/$tag.k$k.shard$g.log 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  echo "$(date +%H:%M) $tag k=$k: $NSH shards on GPUs $GPUS"
  for p in "${pids[@]}"; do wait $p || { echo "⛔ a shard of $tag k=$k failed; see $RUN/sample/$tag.k$k.shard*.log"; exit 1; }; done
  tail -1 $RUN/sample/$tag.k$k.shard${GPUS%% *}.log
}
for j in $JOBS; do case $j in
  0) sample data/mayo/val.bin    lidc10  ""        0 $R9/records_mayo.txt data/mayo/out/diff_clamp     mayo-diff10-clamp ;;
  1) sample data/lidc/val.bin    lidcmix ""        0 $R9/records_val.txt  data/lidc/out/diffmix        lidc25-diffmix
     sample data/lidc10/val.bin  lidcmix ""        0 $R9/records_val.txt  data/lidc10/out/diffmix      lidc10-diffmix
     sample data/mayo/val.bin    lidcmix ""        0 $R9/records_mayo.txt data/mayo/out/diffmix        mayo-diffmix ;;
  2) sample data/lidc/val.bin    lidcmix "--sigma" 0 $R9/records_val.txt  data/lidc/out/diffmixsig     lidc25-diffmixsig
     sample data/lidc10/val.bin  lidcmix "--sigma" 0 $R9/records_val.txt  data/lidc10/out/diffmixsig   lidc10-diffmixsig
     sample data/mayo/val.bin    lidcmix "--sigma" 0 $R9/records_mayo.txt data/mayo/out/diffmixsig     mayo-diffmixsig ;;
  3) sample data/lidcmix/val.bin lidcmix ""        0 $RUN/records_mix.txt data/lidcmix/out/diffmix     mix-diffmix ;;
  4) sample data/lidcmix/val.bin lidcmix "--sigma" 0 $RUN/records_mix.txt data/lidcmix/out/diffmixsig  mix-diffmixsig ;;
  5) ex=""; [ $ARM_SPREAD = diffmixsig ] && ex="--sigma"
     for k in 1 2 3; do
       sample data/lidc/val.bin   lidcmix "$ex" $k $R9/records_spread.txt data/lidc/out/$ARM_SPREAD   lidc25-$ARM_SPREAD
       sample data/lidc10/val.bin lidcmix "$ex" $k $R9/records_spread.txt data/lidc10/out/$ARM_SPREAD lidc10-$ARM_SPREAD
     done ;;
esac; done
echo "sampling done"
