#!/bin/bash
# Fetch NEU-DET (hot-rolled steel strip surface defects, detection boxes) and
# preprocess it to the two flat binaries the VisDrone detector's loaders read.
#
# Source: the dataset maintainer's own Google Drive copy (Kechen Song, NEU),
# linked from the official page faculty.neu.edu.cn/songkc — the same file the
# IEEE DataPort and Kaggle mirrors redistribute. 26 MB, `NEU-DET.zip`, with
# IMAGES/ (1,800 × 200×200 JPEG) and ANNOTATIONS/ (1,800 Pascal-VOC XML). Drive
# fronts it with a "can't scan for viruses" interstitial; `confirm=t` on the
# usercontent endpoint is what gdown sends to get past it.
#
# License: none stated — the official page requests a citation, nothing more.
# This script downloads from the maintainer; the repo does not redistribute.
#
# Citation: Song & Yan, "A noise robust method based on completed local binary
#           patterns for hot-rolled steel strip surface defects", Appl. Surf.
#           Sci. 285 (2013); boxes: He, Song, Meng & Yan, "An End-to-end Steel
#           Surface Defect Detection Approach via Fusing Multiple Hierarchical
#           Features", IEEE TIM 69(4) (2020).
#
# Usage: ./scripts/datasets/download_neu.sh
# Requires: curl, unzip, python3 + Pillow + numpy.
set -e

DRIVE_ID="1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k"
URL="https://drive.usercontent.google.com/download?id=${DRIVE_ID}&export=download&confirm=t"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/data/neu_det"

mkdir -p "$OUT"
cd "$OUT"

if [ -f "$REPO_ROOT/data/neu_det_fpn/train.bin" ] && [ -f "$REPO_ROOT/data/neu_det448/val.bin" ]; then
  echo "data/neu_det_fpn + data/neu_det448 already present — nothing to do."
  echo "(Delete them to force a rebuild.)"
  exit 0
fi

if [ ! -d IMAGES ] || [ ! -d ANNOTATIONS ]; then
  if [ ! -f NEU-DET.zip ]; then
    echo "Downloading NEU-DET.zip (26 MB) from the maintainer's Drive ..."
    curl -L --retry 5 --retry-delay 2 -o NEU-DET.zip "$URL"
  fi
  if ! unzip -t NEU-DET.zip >/dev/null 2>&1; then
    echo "ERROR: NEU-DET.zip is not a valid zip (Drive interstitial instead of the file?)."
    echo "       Delete it and re-run, or fetch it by hand from"
    echo "       https://drive.google.com/open?id=${DRIVE_ID}"
    exit 1
  fi
  echo "Extracting ..."
  unzip -q -o NEU-DET.zip
  # the zip nests everything under NEU-DET/; flatten to IMAGES/ + ANNOTATIONS/
  if [ -d NEU-DET/IMAGES ]; then mv -n NEU-DET/IMAGES NEU-DET/ANNOTATIONS . && rmdir NEU-DET; fi
fi

n_img=$(ls IMAGES/*.jpg 2>/dev/null | wc -l)
n_xml=$(ls ANNOTATIONS/*.xml 2>/dev/null | wc -l)
if [ "$n_img" != 1800 ] || [ "$n_xml" != 1800 ]; then
  echo "ERROR: expected 1800 images + 1800 XMLs, found $n_img + $n_xml"; exit 1
fi
echo "  1800 images, 1800 annotations."

cd "$REPO_ROOT"
echo "Fitting per-scale anchor priors on the train split ..."
python3 scripts/neu_anchors.py data/neu_det --save data/neu_det

echo "Preprocessing (FPN records → data/neu_det_fpn, single-grid → data/neu_det448) ..."
python3 scripts/datasets/preprocess_neu_det.py data/neu_det data/neu_det_fpn --size 448 --grid 14 --fpn data/neu_det
python3 scripts/datasets/preprocess_neu_det.py data/neu_det data/neu_det448 --size 448 --grid 14

echo
echo "Done. Train the FPN detector on steel with:"
echo "  CUDA_VISIBLE_DEVICES=0 FPN_TAG=run1 FPN_AUG=1 FPN_AFFINE=50 FPN_EPOCHS=30 \\"
echo "    lake exe yolov1-neudet-fpn data/neu_det_fpn"
echo "and the single-grid arm with:"
echo "  CUDA_VISIBLE_DEVICES=1 lake exe yolov1-neudet448 data/neu_det448"
