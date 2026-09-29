#!/bin/bash
# Fetch EuroSAT MS (27,000 Sentinel-2 L1C chips, 64×64×13 uint16, ten land-cover classes)
# and torchgeo's split lists, then preprocess them to the flat f32 records
# demos/MainRsBands.lean reads — planning/remote_sensing_wavelengths_demo.md §2, Gate 0.
#
# Source: Zenodo 10.5281/zenodo.7711810 (Helber et al. 2019), MIT licence plus the
# Copernicus Sentinel data terms ("contains modified Copernicus Sentinel data").
# torchgeo mirrors the same archive as EuroSATallBands.zip (sha256 751f070f…df59) beside
# the Neumann et al. 2019 split lists; the mirror is tried first because Zenodo serves
# at ~0.4 MB/s from here (2026-09-29), and whichever archive is present is used.
#
# Citation: Helber, Bischke, Dengel & Borth, "EuroSAT: A Novel Dataset and Deep Learning
#           Benchmark for Land Use and Land Cover Classification", IEEE JSTARS 12(7), 2019.
#           Neumann, Pinto, Zhai & Houlsby, "In-domain representation learning for remote
#           sensing", arXiv:1911.06721 (the split lists).
#
# Usage: ./scripts/datasets/download_rs.sh            (idempotent over data/rs/)
# Requires: curl, .venv-rs (uv venv .venv-rs --python 3.12; uv pip install -r requirements-rs-lock.txt).
set -e

HF="https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c"
ZENODO="https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"
SHA_HF="751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/data/rs"
SRC="$OUT/eurosat"
PY="$REPO_ROOT/.venv-rs/bin/python"
[ -x "$PY" ] || { echo "no .venv-rs — uv venv .venv-rs --python 3.12 && uv pip install --python .venv-rs/bin/python -r requirements-rs-lock.txt"; exit 1; }

mkdir -p "$SRC"
cd "$SRC"

if [ -f "$OUT/eurosat_train.bin" ] && [ -f "$OUT/manifest_rs.json" ]; then
  echo "data/rs already preprocessed — nothing to do. (Delete data/rs/eurosat_*.bin to force a rebuild.)"
  exit 0
fi

for s in train val test; do
  [ -s "eurosat-$s.txt" ] || curl -sSL --retry 5 -o "eurosat-$s.txt" "$HF/eurosat-$s.txt"
done

if [ "$(find . -name '*.tif' | head -1)" = "" ]; then
  if [ ! -s EuroSATallBands.zip ] && [ ! -s EuroSAT_MS.zip ]; then
    echo "Fetching EuroSATallBands.zip (~2 GB) from the torchgeo mirror ..."
    curl -L --retry 5 -o EuroSATallBands.zip "$HF/EuroSATallBands.zip" \
      || { echo "mirror failed; fetching from Zenodo ..."; curl -L --retry 5 -o EuroSAT_MS.zip "$ZENODO"; }
  fi
  if [ -s EuroSATallBands.zip ]; then
    echo "$SHA_HF  EuroSATallBands.zip" | sha256sum -c -
  fi
fi

"$PY" "$REPO_ROOT/scripts/datasets/preprocess_rs_eurosat.py" "$SRC" "$OUT" "$@"
