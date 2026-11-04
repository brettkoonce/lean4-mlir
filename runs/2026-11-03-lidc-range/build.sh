#!/usr/bin/env bash
# Plan §11, R1: the dose-range build and the σ̂ sidecars.
#   data/lidcmix/: the 300 LIDC patients, the seed-0 split of the other builds, each slice at a dose
#   drawn log-uniform in [0.03, 0.25] and a filter drawn from scikit-image's five, one realisation.
#   Then `ct_noise_level.py` writes `<file>.bin.sigma.f32` (σ̂ per record from the low-dose image
#   alone) for the range build and for every file a generator is sampled on, and the gates run:
#   the split equals data/lidc's, the loader test passes at N = 1, the record sizes, and σ̂'s ratio
#   to the known noise is stable across the dose quartiles.
#
#   runs/2026-11-03-lidc-range/build.sh [workers=14]     # ≈ 1 h at 14 workers; resumable
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${1:-14}
PY=.venv-lidc/bin/python
[ -d data/lidc/dicom ] && [ -d data/lidc/xml ] || { echo "⛔ data/lidc/{dicom,xml} missing"; exit 1; }
mkdir -p data/lidcmix
if [ ! -e data/lidcmix/val.bin ]; then
  nice -n 19 $PY scripts/datasets/preprocess_lidc.py data/lidc/dicom data/lidc/xml data/lidcmix \
    --dose-range 0.03 0.25 --filter mix --realisations 1 --workers "$W" --resume 2>&1 | tee -a data/lidcmix/preprocess.log
fi
cmp <(python3 -c "import json;print(json.load(open('data/lidc/split.json')))") \
    <(python3 -c "import json;print(json.load(open('data/lidcmix/split.json')))") \
  && echo "split identical to data/lidc" || { echo "⛔ split differs from data/lidc"; exit 1; }
lake build test-lidc-loader test-dataset-record-sizes 2>&1 | tail -1
for b in data/lidcmix/val.bin data/lidcmix/train.bin; do $PY scripts/probes/lidc_loader_check.py $b | tail -1; done
nice -n 19 $PY scripts/datasets/ct_noise_level.py data/lidcmix/train.bin data/lidcmix/val.bin data/lidc/val.bin \
  data/lidc10/val.bin data/mayo/val.bin --workers 8
ls -la data/lidcmix/*.sigma.f32 data/lidc/val.bin.sigma.f32 data/lidc10/val.bin.sigma.f32 data/mayo/val.bin.sigma.f32
# the range val's scoring set, by the §10.5 rule (every nodule record + 400 others)
$PY scripts/demos/lidc_select_records.py --data=data/lidcmix --out=runs/2026-11-03-lidc-range/records_mix.txt | tail -1
