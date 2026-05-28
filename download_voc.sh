#!/bin/bash
# Download Pascal VOC 2007 trainval + test.
# Original Oxford host is slow/flaky; we try the pjreddie mirror first.
set -e

OUT="${1:-data/voc2007}"
mkdir -p "$OUT"
cd "$OUT"

TRAINVAL_URL_PRIMARY="http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar"
TEST_URL_PRIMARY="http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar"
TRAINVAL_URL_FALLBACK="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
TEST_URL_FALLBACK="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"

fetch() {
  local label="$1" primary="$2" fallback="$3" out="$4"
  if [ -f "$out" ]; then
    echo "$label already present at $out — skipping download"
    return
  fi
  echo "Downloading $label from primary mirror..."
  if wget --tries=2 --timeout=30 -O "$out.part" "$primary"; then
    mv "$out.part" "$out"
  else
    echo "Primary failed; trying fallback..."
    wget --tries=2 --timeout=60 -O "$out.part" "$fallback"
    mv "$out.part" "$out"
  fi
}

fetch "VOC2007 trainval (~440 MB)" "$TRAINVAL_URL_PRIMARY" "$TRAINVAL_URL_FALLBACK" "VOCtrainval_06-Nov-2007.tar"
fetch "VOC2007 test (~430 MB)"     "$TEST_URL_PRIMARY"     "$TEST_URL_FALLBACK"     "VOCtest_06-Nov-2007.tar"

if [ ! -d "VOCdevkit/VOC2007" ]; then
  echo "Extracting trainval..."
  tar xf VOCtrainval_06-Nov-2007.tar
  echo "Extracting test (overlay on the same VOCdevkit tree)..."
  tar xf VOCtest_06-Nov-2007.tar
fi

echo "Done. Tree at: $(pwd)/VOCdevkit/VOC2007"
ls -la VOCdevkit/VOC2007/ImageSets/Main/ | head -5
