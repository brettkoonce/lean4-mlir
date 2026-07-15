#!/bin/bash
# Fetch the Medical Segmentation Decathlon brain-tumour task (BraTS-derived)
# and preprocess it to the flat binary the Lean loader reads.
#
# We use MSD Task01_BrainTumour rather than the BraTS 2021 challenge set
# because it is the same task from openly-downloadable storage: BraTS 2021
# is gated behind a Synapse account and a signed research-use agreement, so
# it cannot be fetched by a script the way this can. Task01 is built from
# the BraTS 2016/2017 cases — same 4 modalities, same glioma sub-regions.
#
# Source:   https://registry.opendata.aws/msd/
# Citation: Antonelli et al., "The Medical Segmentation Decathlon",
#           Nature Communications 13, 4128 (2022). The underlying cases are
#           BraTS — cite Menze et al. 2015 / Bakas et al. 2017 as well.
#
# Usage: ./download_brats.sh
# Requires: curl, tar, python3 + numpy (NIfTI reading is dependency-free).
set -e

URL="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$REPO_ROOT/data/brats"
cd "$REPO_ROOT/data/brats"

if [ -f "train.bin" ] && [ -f "val.bin" ]; then
  echo "data/brats/{train,val}.bin already present — nothing to do."
  echo "(Delete them to force a rebuild.)"
  exit 0
fi

# S3 throttles hard per-connection (~0.6 MB/s measured, i.e. ~3 h for the
# full tar) but does not cap aggregate bandwidth: 12 ranged connections
# measured ~10 MB/s, turning the download into ~12 min. So we fetch by byte
# range in parallel and reassemble.
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

  # Verify each part's length before assembling — a short part would produce
  # a right-sized-looking but corrupt tar.
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

  # Assemble in NUMERIC order. Note: `cat "$out".part*` would be wrong —
  # the glob sorts lexicographically (part0 part1 part10 part11 part2 ...),
  # which silently produces a correctly-sized but scrambled file.
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

if [ ! -d "Task01_BrainTumour" ]; then
  if [ ! -f "Task01_BrainTumour.tar" ]; then
    echo "Downloading MSD Task01_BrainTumour (~7.6 GB)..."
    fetch_parallel "$URL" Task01_BrainTumour.tar
  fi
  echo "Verifying archive..."
  if ! tar tf Task01_BrainTumour.tar >/dev/null 2>&1; then
    echo "ERROR: Task01_BrainTumour.tar is not a valid tar. Delete it and re-run."
    exit 1
  fi
  echo "Extracting (~7 GB unpacked)..."
  tar xf Task01_BrainTumour.tar
fi

echo "Preprocessing NIfTI volumes to train.bin / val.bin..."
echo "  (484 volumes to decode — this takes a while.)"
python3 "$REPO_ROOT/preprocess_brats.py" Task01_BrainTumour .

echo
echo "Done. Train with:"
echo "  lake exe unet-brats-train data/brats"
echo
echo "The tar and the extracted NIfTI tree are no longer needed once"
echo "train.bin / val.bin exist; reclaim ~18 GB with:"
echo "  rm -rf data/brats/Task01_BrainTumour data/brats/Task01_BrainTumour.tar"
