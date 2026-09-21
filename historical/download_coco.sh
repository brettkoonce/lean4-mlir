#!/bin/bash
# Fetch MS-COCO 2017 (object-detection track) and preprocess it to the flat
# binary the Lean YOLO loaders read.
#
# Download reality: COCO needs no account and no torrent, but the canonical
# host does NOT work over verified HTTPS. images.cocodataset.org is a CNAME
# onto S3, and the certificate it serves is CN=s3.amazonaws.com whose SAN list
# does not include images.cocodataset.org — so
#   curl https://images.cocodataset.org/zips/train2017.zip
# fails cert verification on ANY machine ("subjectAltName does not match"),
# which is why COCO's own instructions still hand out http:// URLs.
#
# Rather than downgrade to http or pass -k (both of which drop authentication
# of a 19 GB download), we use the S3 path-style URL, which serves the exact
# same bytes from the same bucket under a hostname the certificate actually
# covers. Verified TLS, no mirror, no trust downgrade.
#
# Source:   https://cocodataset.org/#download
# Citation: Lin et al., "Microsoft COCO: Common Objects in Context",
#           ECCV (2014); the 2017 train/val split.
#
# Sizes: train2017 19 GB (118,287 images), val2017 1 GB (5,000 images),
#        annotations 241 MB. Budget ~45 GB to build in one pass (zips +
#        extracted trees + the .bin output); the zips and image trees can be
#        deleted afterwards, see the note at the end.
#
# Usage: ./download_coco.sh [--val-only]
# Requires: curl, unzip, python3 + Pillow + numpy.
set -e

BASE="https://s3.amazonaws.com/images.cocodataset.org"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$REPO_ROOT/data/coco"

VAL_ONLY=""
if [ "$1" = "--val-only" ]; then
  VAL_ONLY="--val-only"
fi

mkdir -p "$OUT"
cd "$OUT"

if [ -f "train.bin" ] && [ -f "val.bin" ]; then
  echo "data/coco/{train,val}.bin already present — nothing to do."
  echo "(Delete them to force a rebuild.)"
  exit 0
fi

# S3 throttles hard per-connection but does not cap aggregate bandwidth — the
# same behaviour download_brats.sh found and the same fix. MEASURED on this box
# against images.cocodataset.org:
#     1 connection   0.65 MB/s   -> 19.3 GB in ~8.2 h
#     4 connections  2.36 MB/s   -> ~2.2 h
#    12 connections  6.00 MB/s   -> ~54 min
# So fetch by byte range in parallel and reassemble. Lifted from
# download_brats.sh's fetch_parallel, including both of its guards.
fetch_parallel() {
  local url="$1" out="$2" parts=12
  local size
  size=$(curl -sfI -L "$url" | tr -d '\r' | awk 'tolower($1) ~ /^content-length:/ {print $2}' | tail -1)
  if [ -z "$size" ]; then
    echo "  could not determine size; falling back to single-connection curl"
    curl -L --retry 3 -C - -o "$out" "$url"
    return
  fi
  local chunk=$(( (size + parts - 1) / parts ))
  echo "  $size bytes in $parts ranged connections..."
  local i start end
  for ((i = 0; i < parts; i++)); do
    start=$(( i * chunk ))
    end=$(( start + chunk - 1 ))
    (( end >= size )) && end=$(( size - 1 ))
    curl -sf --retry 5 --retry-delay 2 -r "${start}-${end}" -o "${out}.part${i}" "$url" &
  done
  wait

  # Verify each part's length before assembling — a short part would produce a
  # right-sized-looking but corrupt zip.
  for ((i = 0; i < parts; i++)); do
    start=$(( i * chunk ))
    end=$(( start + chunk - 1 ))
    (( end >= size )) && end=$(( size - 1 ))
    local want=$(( end - start + 1 ))
    local got
    got=$(stat -c%s "${out}.part${i}" 2>/dev/null || echo 0)
    if [ "$got" != "$want" ]; then
      echo "  ERROR: part $i is $got bytes, expected $want. Re-run to retry."
      rm -f "${out}.part"*
      exit 1
    fi
  done

  # Assemble in NUMERIC order. `cat "$out".part*` would be wrong — the glob
  # sorts lexicographically (part0 part1 part10 part11 part2 ...), silently
  # producing a correctly-sized but scrambled file.
  rm -f "$out"
  for ((i = 0; i < parts; i++)); do
    cat "${out}.part${i}" >> "$out"
  done
  rm -f "${out}.part"*

  local final
  final=$(stat -c%s "$out")
  if [ "$final" != "$size" ]; then
    echo "  ERROR: assembled $final bytes, expected $size"
    exit 1
  fi
}

fetch() {
  local url="$1"
  local name="$2"
  local dir="$3"
  if [ -d "$dir" ]; then
    echo "  $dir/ already extracted."
    return
  fi
  if [ ! -f "$name" ]; then
    echo "Downloading $name ..."
    fetch_parallel "$url/$name" "$name"
  fi
  echo "Verifying + extracting $name ..."
  if ! unzip -t "$name" >/dev/null 2>&1; then
    echo "ERROR: $name is not a valid zip. Delete it and re-run."
    exit 1
  fi
  unzip -q -o "$name"
}

fetch "$BASE/annotations" annotations_trainval2017.zip annotations
fetch "$BASE/zips" val2017.zip val2017
if [ -z "$VAL_ONLY" ]; then
  fetch "$BASE/zips" train2017.zip train2017
fi

echo "Preprocessing to train.bin / val.bin ..."
# Default build: the FPN multi-scale format at 448px with the full 80 classes.
# That is the only format whose C loader is class-count-agnostic
# (lean_f32_load_voc_fpn takes ntot); the single-grid and anchor loaders
# hardcode 20 and 10 classes respectively. See preprocess_coco.py's docstring.
#
# The FPN build needs per-scale anchor priors. Generate them first if absent.
if [ ! -f "$OUT/anchors_fpn_p3.txt" ]; then
  echo "Computing per-scale k-means anchor priors ..."
  python3 "$REPO_ROOT/scripts/coco_anchors.py" "$OUT" --save "$OUT" $VAL_ONLY
fi

python3 "$REPO_ROOT/preprocess_coco.py" "$OUT" "$OUT" \
  --size 448 --grid 14 --classes all --fpn "$OUT" $VAL_ONLY

echo
echo "Done. For the VisDrone-transferable build instead (COCO emitted into"
echo "VisDrone's own 10-class index space, so a pretrained head transfers):"
echo "  python3 preprocess_coco.py data/coco data/coco_vd \\"
echo "      --size 448 --grid 14 --classes vdmap --fpn data/visdrone"
echo
echo "The zips and extracted image trees are no longer needed once the .bin"
echo "files exist; reclaim ~40 GB with:"
echo "  rm -rf data/coco/train2017 data/coco/val2017 data/coco/*.zip"
