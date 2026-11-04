#!/usr/bin/env bash
# Plan §11, R4: the tables, per data root and per noise bin, every arm with a dump — §9's arms and
# the range arms side by side. Mayo bins by the inserted noise (sigma_add_hu, the measured body std
# where that is NaN); the LIDC vals by their sigma_add_hu; the range build's own val by dose.
#   ROOTS="lidc lidc10 mayo lidcmix" runs/2026-11-03-lidc-range/score.sh     # roots in parallel: one each
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-03-lidc-range
R9=runs/2026-11-02-lidc-diffusion
ROOTS=${ROOTS:-"lidc lidc10 mayo lidcmix"}
TW=${TW:-6}
PY=.venv-lidc/bin/python
mkdir -p $RUN/score
for root in $ROOTS; do
  case $root in
    mayo) recs=$R9/records_mayo.txt; by="sigma_add_hu:120,180,250,350";;
    lidcmix) recs=$RUN/records_mix.txt; by="dose:0.05,0.1,0.15,0.2";;
    *) recs=$R9/records_val.txt; by="sigma_add_hu:60,90,120,180";;
  esac
  for d in data/$root/out/*/; do
    for f in $d*val*.eps.f32; do
      [ -e "$f" ] || continue
      $PY scripts/probes/lidc_denoise_score.py data/$root/val.bin "$f" --records=$recs --by=$by \
        > $RUN/score/fidelity_${root}_$(basename $d).$(basename $f .eps.f32).txt 2>&1 || true
    done
  done
  arms=$(ls -d data/$root/out/*/ | xargs -n1 basename | sed 's/$/,/' | tr -d '\n' | sed 's/diff,/diff.k0,/; s/diff25,/diff25.k0,/; s/diffmix,/diffmix.k0,/; s/diffmixsig,/diffmixsig.k0,/; s/diff_clamp,/diff_clamp.k0,/; s/,$//')
  if [ $root != mayo ]; then
    $PY scripts/demos/lidc_nodules.py --data=data/$root --records=$recs --arms=$arms \
      --json=$RUN/score/nodules_$root.json > $RUN/score/nodules_$root.txt 2>&1 || true
  fi
  nice -n 10 $PY scripts/demos/lidc_texture.py --data=data/$root --records=$recs --workers=$TW --arms=$arms --by=$by \
    --json=$RUN/score/texture_$root.json --nps-csv=$RUN/score/nps_$root.csv > $RUN/score/texture_$root.txt 2>&1
  if [ $root = lidc ] || [ $root = lidc10 ]; then
    nice -n 10 $PY scripts/demos/lidc_texture.py --data=data/$root --records=$R9/records_spread.txt --workers=$TW \
      --json=$RUN/score/spread_$root.json > $RUN/score/spread_$root.txt 2>&1 || true
  fi
  echo "$root scored"
done
echo "scores in $RUN/score/"
