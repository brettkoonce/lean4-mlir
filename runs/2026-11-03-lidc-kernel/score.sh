#!/usr/bin/env bash
# Plan §12: the tables, per data root and per noise bin — the b50f arms beside §9's and §11's on the
# same records (records_val.txt / records_mayo.txt), the per-slice view (share of slices worse than
# the floor, median) that found the Mayo tail, nodules, texture, spread. The Gaussian is re-tuned on
# the b50f train split for that root's filter row (stage `filters`, CPU only, so it can run while the
# nets train).
#   STAGES="filters tables" ROOTS="lidc10b50 lidc10 mayo" runs/2026-11-03-lidc-kernel/score.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-03-lidc-kernel
R9=runs/2026-11-02-lidc-diffusion
STAGES=${STAGES:-"tables"}
ROOTS=${ROOTS:-"lidc10b50 lidc10 mayo"}
TW=${TW:-6}
FW=${FW:-16}
PY=.venv-lidc/bin/python
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
mkdir -p $RUN/score
if has filters && [ ! -e data/lidc10b50/out/gauss/val.eps.f32 ]; then
  mkdir -p data/lidc10b50/out/gauss
  nice -n 10 $PY scripts/probes/lidc_classical.py data/lidc10b50/train.bin data/lidc10b50/val.bin gauss \
    data/lidc10b50/out/gauss/val.eps.f32 --workers $FW > $RUN/score/gauss_lidc10b50.tune.txt 2>&1
  tail -1 $RUN/score/gauss_lidc10b50.tune.txt
fi
has tables || exit 0
for root in $ROOTS; do
  case $root in
    mayo)      recs=$R9/records_mayo.txt; by="sigma_add_hu:120,180,250,350"; arms="diff_b50.k0,mse_b50,diff.k0,mse,diff25.k0,diffmixsig.k0,msemix,gauss";;
    lidc10)    recs=$R9/records_val.txt;  by="sigma_add_hu:60,90,120,180";   arms="diff_b50.k0,mse_b50,diff.k0,mse,gauss";;
    lidc10b50) recs=$R9/records_val.txt;  by="sigma_add_hu:60,90,120,180";   arms="diff.k0,mse,diff_sl.k0,mse_sl";
               [ -e data/lidc10b50/out/gauss/val.eps.f32 ] && arms="$arms,gauss";;
    *) echo "⛔ unknown root $root"; exit 1;;
  esac
  # the arms that exist (a root scored before its cross rows land keeps the table)
  have=""; for a in $(echo $arms | tr , ' '); do d=data/$root/out/${a%%.k*}; ls $d/*val*.eps.f32 >/dev/null 2>&1 && have="$have,$a"; done
  arms=${have#,}
  for a in $(echo $arms | tr , ' '); do
    d=data/$root/out/${a%%.k*}
    for f in $d/*val*.eps.f32; do
      case $a in *.k*) case $f in *.k${a##*.k}.eps.f32) ;; *) continue;; esac;; esac
      $PY scripts/probes/lidc_denoise_score.py data/$root/val.bin "$f" --records=$recs --by=$by \
        > $RUN/score/fidelity_${root}_${a%%.k*}.$(basename $f .eps.f32).txt 2>&1 || true
    done
  done
  $PY scripts/probes/lidc_per_slice.py data/$root/val.bin --arms=$arms --records=$recs --by=$by \
    --csv=$RUN/score/perslice_$root.csv > $RUN/score/perslice_$root.txt 2>&1 || true
  if [ $root != mayo ]; then
    $PY scripts/demos/lidc_nodules.py --data=data/$root --records=$recs --arms=$arms \
      --json=$RUN/score/nodules_$root.json > $RUN/score/nodules_$root.txt 2>&1 || true
  fi
  nice -n 10 $PY scripts/demos/lidc_texture.py --data=data/$root --records=$recs --workers=$TW --arms=$arms --by=$by \
    --json=$RUN/score/texture_$root.json --nps-csv=$RUN/score/nps_$root.csv > $RUN/score/texture_$root.txt 2>&1
  if [ $root = lidc10b50 ]; then
    nice -n 10 $PY scripts/demos/lidc_texture.py --data=data/$root --records=$R9/records_spread.txt --workers=$TW \
      --json=$RUN/score/spread_$root.json > $RUN/score/spread_$root.txt 2>&1 || true
  fi
  echo "$root scored"
done
echo "scores in $RUN/score/"
