#!/usr/bin/env bash
# Plan §13.5: the x₀-target generators — the net predicts the clean residual x₀ = (full − low)/100 instead
# of the noise ε (lidc-denoise-train --target=x0; the sampler reads such a checkpoint with the same flag),
# everything else the §10.1 recipe (8 × 256², Adam 5e-4 cosine, seed 0, 60k steps; DDIM η = 1, 50 steps,
# no clamp):
#   diffmixsigx0b50  on data/lidcmixb50 (doses 3–25 % under Mayo's kernel), told σ̂ — the §13 generator's twin, GPU 0
#   diffx0b50        on data/lidc10b50 (10 % under Mayo's kernel), no level channel — the §12 generator's twin, GPU 1
# then, once both have trained, their DDIM samples on Mayo (records_mayo.txt), the b50f 10 % val (records_val.txt)
# and, for the mix net, its own val (records_mixb50.txt), the GPUs in $GPUS as shards; then each set's scores
# (fidelity by bin, per slice beside the ε twins and the clipped sampler, nodules, texture) and the table rows.
#   STAGES="train sample score" runs/2026-11-04-lidc-x0pred/launch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-04-lidc-x0pred
R9=runs/2026-11-02-lidc-diffusion
R13=runs/2026-11-03-lidc-rangekernel
DM=data/lidcmixb50
DK=data/lidc10b50
STAGES=${STAGES:-"train sample score"}
GPUS=${GPUS:-"0 1 2 3"}
NSH=$(echo $GPUS | wc -w)
BATCH=${BATCH:-4}
TW=${TW:-6}
STEPS=${STEPS:-60000}
PY=.venv-lidc/bin/python
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
mkdir -p $RUN/sample $RUN/score
gpufree() { local g used; for g in "$@"; do used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -lt 500 ] || { echo "⛔ GPU $g busy ($used MiB)"; exit 1; }; done; }
train() {  # train NAME GPU TRAIN_BIN EXTRA DUMP — in the background; returns once the first "train:" line is in the log
  local name=$1 g=$2 bin=$3 extra=$4 dump=$5
  mkdir -p $RUN/$name "$dump"
  setsid -f nohup env CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG \
    .lake/build/bin/lidc-denoise-train $bin $STEPS --arm=diff --target=x0 $extra --seed=0 --cosine \
      --out=$RUN/$name --dump=$dump > $RUN/$name/train.log 2>&1
  echo "$(date +%H:%M) launched $name on GPU $g → $RUN/$name/train.log"
  until grep -q "^train: " $RUN/$name/train.log 2>/dev/null; do
    grep -q "uncaught exception\|compile failed\|no lowerer shim\|⛔" $RUN/$name/train.log 2>/dev/null \
      && { echo "⛔ $name failed before training; see $RUN/$name/train.log"; exit 1; }
    sleep 10
  done
}
waittrain() {  # waittrain NAME
  local name=$1
  until grep -q "trained to step" $RUN/$name/train.log 2>/dev/null; do
    grep -q "uncaught exception\|⛔\|Killed" $RUN/$name/train.log 2>/dev/null \
      && { echo "⛔ $name died; see $RUN/$name/train.log"; exit 1; }
    sleep 60
  done
  echo "$(date +%H:%M) $name: $(grep 'trained to step' $RUN/$name/train.log | head -1)"
}
sample() {  # sample PAIRS RECORDS DUMP LOGTAG MODEL EXTRA — k = 0, --target=x0
  local pairs=$1 recs=$2 dump=$3 tag=$4 model=$5 extra=$6
  [ -e "$pairs" ] && [ -e "$recs" ] || { echo "⛔ missing $pairs or $recs"; exit 1; }
  mkdir -p "$dump"
  local pids=() i=0 g
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG \
      .lake/build/bin/lidc-denoise-sample "$pairs" --model=$model --target=x0 $extra --seed=0 --k=0 --records="$recs" \
        --shard=$i/$NSH --batch=$BATCH --dump="$dump" > $RUN/sample/$tag.shard$g.log 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  echo "$(date +%H:%M) $tag: $NSH shards on GPUs $GPUS"
  local pid; for pid in "${pids[@]}"; do wait $pid || { echo "⛔ a shard of $tag failed; see $RUN/sample/$tag.shard*.log"; exit 1; }; done
  tail -1 $RUN/sample/$tag.shard${GPUS%% *}.log
}
for st in $STAGES; do case $st in
  train)
    for f in $DM/train.bin $DM/train.bin.sigma.f32 $DK/train.bin "$PLUG" ffi/libpjrt_ffi.so; do [ -e "$f" ] || { echo "⛔ missing $f"; exit 1; }; done
    lake build lidc-denoise-train lidc-denoise-sample test-lidc-loader 2>&1 | tail -1
    for b in $DM/train.bin $DK/train.bin; do
      $PY scripts/probes/lidc_loader_check.py $b | tail -1 | grep -q PASS || { echo "⛔ loader gate failed on $b"; exit 1; }
    done
    gpufree 0 1
    train diffmixsigx0b50 0 $DM/train.bin "--sigma" $DM/out/diffmixsigx0
    train diffx0b50 1 $DK/train.bin "" $DK/out/diffx0
    waittrain diffmixsigx0b50
    waittrain diffx0b50 ;;
  sample)
    gpufree $GPUS
    sample data/mayo/val.bin $R9/records_mayo.txt data/mayo/out/diffmixsigx0_b50 mayo-diffmixsigx0b50 lidcmixb50 "--sigma"
    sample data/mayo/val.bin $R9/records_mayo.txt data/mayo/out/diffx0_b50 mayo-diffx0b50 lidc10b50 ""
    sample $DK/val.bin $R9/records_val.txt $DK/out/diffmixsigx0 lidc10b50-diffmixsigx0b50 lidcmixb50 "--sigma"
    sample $DK/val.bin $R9/records_val.txt $DK/out/diffx0 lidc10b50-diffx0b50 lidc10b50 ""
    sample $DM/val.bin $R13/records_mixb50.txt $DM/out/diffmixsigx0 mixb50-diffmixsigx0b50 lidcmixb50 "--sigma" ;;
  score)
    for root in mayo lidc10b50 lidcmixb50; do
      case $root in
        mayo)       recs=$R9/records_mayo.txt;    by="sigma_add_hu:120,180,250,350"; arms="diffx0_b50.k0,diffmixsigx0_b50.k0"; ref="diff_b50.k0,diffmixsig_b50.k0,diffmixsig_b50_clampx0.k0";;
        lidc10b50)  recs=$R9/records_val.txt;     by="sigma_add_hu:60,90,120,180";   arms="diffx0.k0,diffmixsigx0.k0";        ref="diff.k0,diffmixsig.k0,diffmixsig_clampx0.k0";;
        lidcmixb50) recs=$R13/records_mixb50.txt; by="dose:0.05,0.1,0.15,0.2";       arms="diffmixsigx0.k0";                  ref="diffmixsig.k0";;
      esac
      for a in $(echo $arms | tr , ' '); do
        nice -n 19 $PY scripts/probes/lidc_denoise_score.py data/$root/val.bin data/$root/out/${a%%.k*}/${root}_val.k0.eps.f32 \
          --records=$recs --by=$by > $RUN/score/fidelity_${root}_${a%%.k*}.txt 2>&1 || true
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
    $RUN/table_rows.sh | tail -2 ;;
  *) echo "⛔ unknown stage $st"; exit 1 ;;
esac; done
echo "x0pred done"
