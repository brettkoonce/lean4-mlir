#!/bin/bash
# Fetch a seeded patient subset of LIDC-IDRI (chest CT with four radiologists' nodule
# annotations) from TCIA and preprocess it to the full-dose / low-dose pairs the CT denoising
# demo reads — planning/demo_slate.md §3.
#
# Source:   https://www.cancerimagingarchive.net/collection/lidc-idri/ (CC BY 3.0), through the
#           public NBIA REST API: no account, no manifest tool. The CT collection is 1,018
#           series / 1,010 patients / 128 GB; the default draw is 300 patients (about 38 GB),
#           stratified by scanner maker — scripts/datasets/lidc_fetch.py says how.
#           The annotations are TCIA's LIDC-XML-only.zip (one XML per series, the four readers'
#           nodule outlines and characteristics) and its nodule-count sheet.
# Citation: Armato et al., "The Lung Image Database Consortium (LIDC) and Image Database
#           Resource Initiative (IDRI): a completed reference database of lung nodules on CT
#           scans", Medical Physics 38(2), 915–931 (2011).
#           Clark et al., "The Cancer Imaging Archive (TCIA): maintaining and operating a
#           public information repository", J Digit Imaging 26(6), 1045–1057 (2013).
#
# Usage: ./scripts/datasets/download_lidc.sh [--patients 300] [--seed 0] [--workers 6]
#        LIDC_PREPROCESS_ARGS="--workers 8" ./scripts/datasets/download_lidc.sh   (beside a live GPU run, with nice -n 19)
#        (idempotent over data/lidc/: fetched patients are skipped, the pairs are rebuilt
#        only when data/lidc/train.bin is missing)
# Requires: curl, unzip, .venv-lidc (uv venv .venv-lidc --python 3.12;
#           uv pip install --python .venv-lidc/bin/python -r requirements-lidc-lock.txt).
set -e

TCIA="https://www.cancerimagingarchive.net/wp-content/uploads"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/data/lidc"
PY="$REPO_ROOT/.venv-lidc/bin/python"
[ -x "$PY" ] || { echo "no .venv-lidc — uv venv .venv-lidc --python 3.12 && uv pip install --python .venv-lidc/bin/python -r requirements-lidc-lock.txt"; exit 1; }

mkdir -p "$OUT"

# The annotations first: 8.6 MB, and the preprocessor's patient filter needs them.
if [ ! -d "$OUT/xml" ]; then
  curl -sSL --retry 5 -o "$OUT/LIDC-XML-only.zip" "$TCIA/LIDC-XML-only.zip"
  mkdir -p "$OUT/xml" && unzip -q -o "$OUT/LIDC-XML-only.zip" -d "$OUT/xml"
fi
[ -s "$OUT/lidc-idri-nodule-counts-6-23-2015.xlsx" ] \
  || curl -sSL --retry 5 -o "$OUT/lidc-idri-nodule-counts-6-23-2015.xlsx" "$TCIA/lidc-idri-nodule-counts-6-23-2015.xlsx"

"$PY" "$REPO_ROOT/scripts/datasets/lidc_fetch.py" "$OUT" "$@"

if [ -f "$OUT/train.bin" ] && [ -f "$OUT/val.bin" ]; then
  echo "data/lidc/{train,val}.bin already present — nothing to do. (Delete them to force a rebuild.)"
  exit 0
fi
"$PY" "$REPO_ROOT/scripts/datasets/preprocess_lidc.py" "$OUT/dicom" "$OUT/xml" "$OUT" ${LIDC_PREPROCESS_ARGS:-}
