#!/usr/bin/env bash
# Plan §13.4: three probes of the §13 both-at-once generator's SAMPLER (checkpoint
# data/lidcmixb50/out/diffmixsig, nothing retrained). Each differs from the scored arm
# (DDIM η = 1, 50 steps, σ̂ as the sidecar says) in one knob:
#   eta0  η = 0, deterministic DDIM, 50 steps
#   s20   20 steps at η = 1
#   cap   the σ̂ the net is told capped at the training median, 101 HU (runs/2026-11-03-lidc-rangekernel/sigma.txt)
# on Mayo (§10.5's records_mayo.txt, 250 slices) and then on the b50f 10 % val (records_val.txt, 402),
# the GPUs in $GPUS as that many shards; then the scores of each set: fidelity by noise bin, per-slice
# (one csv with the §12 kernel generator and the §13 arm beside the probes), nodules (the LIDC set), texture.
#   GPUS="0 1 2 3" SETS="mayo lidc10b50" PROBES="eta0 s20 cap" runs/2026-11-04-lidc-sampler-probes/launch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-04-lidc-sampler-probes
R9=runs/2026-11-02-lidc-diffusion
SETS=${SETS:-"mayo lidc10b50"}
PROBES=${PROBES:-"eta0 s20 cap"}
GPUS=${GPUS:-"0 1 2 3"}
NSH=$(echo $GPUS | wc -w)
BATCH=${BATCH:-4}
TW=${TW:-6}
PY=.venv-lidc/bin/python
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
lake build lidc-denoise-sample 2>&1 | tail -1
for g in $GPUS; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 500 ] || { echo "⛔ GPU $g busy ($used MiB)"; exit 1; }
done
mkdir -p $RUN/sample $RUN/score
flags() { case $1 in eta0) echo "--eta=0";; s20) echo "--steps=20";; cap) echo "--sigma-cap=101";; *) echo "⛔ unknown probe $1" >&2; exit 1;; esac; }
sample() {  # sample PAIRS RECORDS DUMP LOGTAG PROBE — model lidcmixb50 with --sigma, k = 0
  local pairs=$1 recs=$2 dump=$3 tag=$4 probe=$5 extra
  extra=$(flags $probe)
  [ -e "$pairs" ] && [ -e "$recs" ] && [ -e "$pairs.sigma.f32" ] || { echo "⛔ missing $pairs, $recs or $pairs.sigma.f32"; exit 1; }
  mkdir -p "$dump"
  local pids=() i=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG \
      .lake/build/bin/lidc-denoise-sample "$pairs" --model=lidcmixb50 --sigma --seed=0 --k=0 --records="$recs" \
        $extra --shard=$i/$NSH --batch=$BATCH --dump="$dump" > $RUN/sample/$tag.shard$g.log 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  echo "$(date +%H:%M) $tag ($extra): $NSH shards on GPUs $GPUS"
  for p in "${pids[@]}"; do wait $p || { echo "⛔ a shard of $tag failed; see $RUN/sample/$tag.shard*.log"; exit 1; }; done
  tail -1 $RUN/sample/$tag.shard${GPUS%% *}.log
}
for root in $SETS; do
  case $root in
    mayo)      recs=$R9/records_mayo.txt; by="sigma_add_hu:120,180,250,350"; base=diffmixsig_b50; ref="diff_b50.k0,diffmixsig_b50.k0";;
    lidc10b50) recs=$R9/records_val.txt;  by="sigma_add_hu:60,90,120,180";   base=diffmixsig;     ref="diff.k0,diffmixsig.k0";;
    *) echo "⛔ unknown set $root"; exit 1;;
  esac
  arms=""
  for p in $PROBES; do
    sample data/$root/val.bin $recs data/$root/out/${base}_$p $root-$p $p
    arms="$arms,${base}_$p.k0"
  done
  arms=${arms#,}
  for p in $PROBES; do
    nice -n 19 $PY scripts/probes/lidc_denoise_score.py data/$root/val.bin data/$root/out/${base}_$p/${root}_val.k0.eps.f32 \
      --records=$recs --by=$by > $RUN/score/fidelity_${root}_${base}_$p.txt 2>&1 || true
  done
  nice -n 19 $PY scripts/probes/lidc_per_slice.py data/$root/val.bin --arms=$ref,$arms --records=$recs --by=$by \
    --csv=$RUN/score/perslice_$root.csv > $RUN/score/perslice_$root.txt 2>&1 || true
  if [ $root = lidc10b50 ]; then
    nice -n 19 $PY scripts/demos/lidc_nodules.py --data=data/$root --records=$recs --arms=$arms \
      --json=$RUN/score/nodules_$root.json > $RUN/score/nodules_$root.txt 2>&1 || true
  fi
  nice -n 19 $PY scripts/demos/lidc_texture.py --data=data/$root --records=$recs --workers=$TW --arms=$arms --by=$by \
    --json=$RUN/score/texture_$root.json --nps-csv=$RUN/score/nps_$root.csv > $RUN/score/texture_$root.txt 2>&1 || true
  echo "$(date +%H:%M) $root scored"
done
echo "probes done"
