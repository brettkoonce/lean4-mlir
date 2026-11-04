#!/usr/bin/env bash
# Plan §13: the tables of the range × kernel arms. The earlier arms' numbers are in
# runs/2026-11-03-lidc-kernel/score/ and runs/2026-11-03-lidc-range/score/ (and their eval dumps were
# removed for disk), so only the new dumps are read here; Mayo's per-slice csv is merged with §12's so
# every arm sits in one table.
#   mayo        diffmixsig_b50.k0, msemix_b50  — fidelity per noise bin, per-slice (+ merged), texture
#   lidc10b50   diffmixsig.k0, msemix          — fidelity, per-slice, nodules, texture (the LIDC block, as the kernel rows)
#   lidcmixb50  diffmixsig.k0                  — fidelity and texture by dose bin (gate R3)
#   ROOTS="mayo lidc10b50 lidcmixb50" TW=6 runs/2026-11-03-lidc-rangekernel/score.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-03-lidc-rangekernel
R9=runs/2026-11-02-lidc-diffusion
K=runs/2026-11-03-lidc-kernel/score
ROOTS=${ROOTS:-"mayo lidc10b50 lidcmixb50"}
TW=${TW:-6}
PY=.venv-lidc/bin/python
mkdir -p $RUN/score
for root in $ROOTS; do
  case $root in
    mayo)       recs=$R9/records_mayo.txt;    by="sigma_add_hu:120,180,250,350"; arms="diffmixsig_b50.k0,msemix_b50";;
    lidc10b50)  recs=$R9/records_val.txt;     by="sigma_add_hu:60,90,120,180";   arms="diffmixsig.k0,msemix";;
    lidcmixb50) recs=$RUN/records_mixb50.txt; by="dose:0.05,0.1,0.15,0.2";       arms="diffmixsig.k0";;
    *) echo "⛔ unknown root $root"; exit 1;;
  esac
  have=""; for a in $(echo $arms | tr , ' '); do d=data/$root/out/${a%%.k*}; ls $d/*val*.eps.f32 >/dev/null 2>&1 && have="$have,$a"; done
  arms=${have#,}; [ -n "$arms" ] || { echo "⛔ $root: no dumps yet"; continue; }
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
  if [ $root = mayo ] && [ -e $K/perslice_mayo.csv ] && [ -e $RUN/score/perslice_mayo.csv ]; then
    $PY - <<PY
import csv
new = {r["record"]: r for r in csv.DictReader(open("$RUN/score/perslice_mayo.csv"))}
old = list(csv.DictReader(open("$K/perslice_mayo.csv")))
extra = [c for c in next(iter(new.values())) if c not in old[0]]
cols = list(old[0]) + extra
with open("$RUN/score/perslice_mayo_all.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in old:
        w.writerow({**r, **{c: new.get(r["record"], {}).get(c, "") for c in extra}})
print("merged", len(old), "records, arms", cols[3:], "→ $RUN/score/perslice_mayo_all.csv")
PY
  fi
  if [ $root = lidc10b50 ]; then
    $PY scripts/demos/lidc_nodules.py --data=data/$root --records=$recs --arms=$arms \
      --json=$RUN/score/nodules_$root.json > $RUN/score/nodules_$root.txt 2>&1 || true
  fi
  nice -n 10 $PY scripts/demos/lidc_texture.py --data=data/$root --records=$recs --workers=$TW --arms=$arms --by=$by \
    --json=$RUN/score/texture_$root.json --nps-csv=$RUN/score/nps_$root.csv > $RUN/score/texture_$root.txt 2>&1
  echo "$root scored"
done
echo "scores in $RUN/score/"
