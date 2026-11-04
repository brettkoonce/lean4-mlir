#!/usr/bin/env bash
# Phase 4b (planning/lidc_ct_denoising_demo.md §6, §10.4): the 10 %-dose LIDC build, one
# realisation, the same 300 patients and the same seed-0 patient split as the 25 % build at
# data/lidc (the split is a seeded shuffle of the landed patients, so it is identical by
# construction; the gate below checks split.json anyway). Mayo chest is 10 % of routine dose, so
# the MSE net and the generator that meet it train here.
#
#   runs/2026-11-02-lidc-diffusion/build_lidc10.sh [workers]      # from the lidc worktree root
#
# ≈ 8 s a slice a core (one sinogram, one calibration back-projection, one realisation) over
# 13,882 slices: ≈ 1 h on 30 idle cores, ≈ 4–6 h at `nice -n 19` with 8 workers beside a live GPU
# job. Output ≈ 18 GB (train 12.6 + val 5.6). Then the loader gate and the record-size test.
# Resumable: a killed build keeps its finished patients (--resume), so it can be restarted with more
# workers once the box is free.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${1:-8}
[ -d data/lidc/dicom ] && [ -d data/lidc/xml ] || { echo "⛔ data/lidc/{dicom,xml} missing"; exit 1; }
[ -e data/lidc10/val.bin ] && { echo "data/lidc10/val.bin exists; delete data/lidc10 to rebuild"; exit 0; }
mkdir -p data/lidc10
nice -n 19 .venv-lidc/bin/python scripts/datasets/preprocess_lidc.py data/lidc/dicom data/lidc/xml data/lidc10 \
  --dose 0.10 --realisations 1 --workers "$W" --resume 2>&1 | tee -a data/lidc10/preprocess.log
cmp <(python3 -c "import json;print(json.load(open('data/lidc/split.json')))") \
    <(python3 -c "import json;print(json.load(open('data/lidc10/split.json')))") \
  && echo "split identical to data/lidc" || { echo "⛔ split differs from data/lidc"; exit 1; }
lake build test-lidc-loader test-dataset-record-sizes 2>&1 | tail -1
.venv-lidc/bin/python scripts/probes/lidc_loader_check.py data/lidc10/val.bin | tail -1
.venv-lidc/bin/python scripts/probes/lidc_loader_check.py data/lidc10/train.bin | tail -1
.lake/build/bin/test-dataset-record-sizes 2>&1 | grep -i "lidc10" || true
