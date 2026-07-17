#!/bin/bash
# Fetch VisDrone-DET2019 (object-detection track) and preprocess it to the flat
# binary the Lean YOLO loader reads.
#
# Download reality (WS-0): the official VisDrone source
# (github.com/VisDrone/VisDrone-Dataset) links to Google Drive + Baidu, and the
# Drive links are large-file-flaky under gdown / need an account. We instead use
# the Ultralytics YOLOv5-release mirror on GitHub release assets: plain HTTPS,
# no account, resumable, stable URLs. Same bytes, same split sizes as the
# official set (train 6471 / val 548 / test-dev 1610 images).
#
# Source:   https://github.com/ultralytics/yolov5/releases/tag/v1.0
# Citation: Zhu et al., "Detection and Tracking Meet Drones Challenge",
#           IEEE TPAMI (2021); the VisDrone2019 dataset paper.
#
# Usage: ./download_visdrone.sh
# Requires: curl, unzip, python3 + Pillow + numpy.
set -e

BASE="https://github.com/ultralytics/yolov5/releases/download/v1.0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$REPO_ROOT/data/visdrone"

mkdir -p "$OUT"
cd "$OUT"

if [ -f "train.bin" ] && [ -f "val.bin" ]; then
  echo "data/visdrone/{train,val}.bin already present — nothing to do."
  echo "(Delete them to force a rebuild.)"
  exit 0
fi

fetch() {
  local name="$1"
  if [ -d "$name" ]; then
    echo "  $name/ already extracted."
    return
  fi
  if [ ! -f "$name.zip" ]; then
    echo "Downloading $name.zip ..."
    # GitHub release assets 302 to a signed githubusercontent URL; -L follows it
    # and -C - resumes a partial file if the fetch is interrupted.
    curl -L --retry 5 --retry-delay 2 -C - -o "$name.zip" "$BASE/$name.zip"
  fi
  echo "Verifying + extracting $name.zip ..."
  if ! unzip -t "$name.zip" >/dev/null 2>&1; then
    echo "ERROR: $name.zip is not a valid zip. Delete it and re-run."
    exit 1
  fi
  unzip -q -o "$name.zip"
}

fetch VisDrone2019-DET-train
fetch VisDrone2019-DET-val

echo "Preprocessing to train.bin / val.bin ..."
python3 "$REPO_ROOT/preprocess_visdrone.py" "$OUT" "$OUT"

echo
echo "Done. Train (single-grid YOLOv1 baseline, WS-A) with:"
echo "  lake exe yolov1-pets-train-bootstrap data/visdrone"
echo
echo "The zips and extracted image trees are no longer needed once the .bin"
echo "files exist; reclaim ~2 GB with:"
echo "  rm -rf data/visdrone/VisDrone2019-DET-* data/visdrone/*.zip"
