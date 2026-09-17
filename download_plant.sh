#!/bin/bash
# Fetch PlantVillage (54,305 lab photographs of single leaves, 38 classes) and PlantDoc
# (2,578 field photographs, 28 classes that map into PlantVillage's 38) and preprocess
# both to the Imagenette-format records demos/MainPlantLeaf.lean reads —
# planning/plant_lab_to_field_demo.md §2.
#
# Sources (plain git clones, no account):
#   github.com/spMohanty/PlantVillage-Dataset   4.8 GB on disk; raw/{color,grayscale,segmented},
#                                               leaf_grouping/leaf-map.json (the maintainers' same-leaf
#                                               grouping); licence per its dataset card: CC BY-SA 3.0
#   github.com/pratikkayal/PlantDoc-Dataset     1.9 GB; LICENSE.txt = CC BY 4.0
#   huggingface.co/datasets/mohanty/PlantVillage/splits/color_{train,test}.txt
#                                               the maintainers' leaf-grouped 80/20 split (43,596 / 10,709)
# ⛔ Not the Kaggle "New Plant Diseases Dataset": PlantVillage with augmented copies on both
#    sides of its own split.
#
# Citations: Mohanty, Hughes & Salathé, "Using deep learning for image-based plant disease
#            detection", Front. Plant Sci. 7:1419 (2016); Singh et al., "PlantDoc: a dataset for
#            visual plant disease detection", CODS-COMAD (2020).
#
# Usage: ./download_plant.sh            (idempotent over data/plant/)
# Requires: git, curl, python3 + Pillow + numpy + scipy (the repo .venv has them), and
#           data/imagenette/imagenette2-320/ (download_imagenette.sh) for the composites.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$REPO_ROOT/data/plant"
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3
HF="https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits"

mkdir -p "$OUT/splits"
cd "$OUT"

if [ -f pvg_train.bin ] && [ -f pd_all.bin ] && [ -f meta_plant.npz ]; then
  echo "data/plant already preprocessed — nothing to do. (Delete the .bin files to force a rebuild.)"
  exit 0
fi

if [ ! -d PlantVillage-Dataset/raw/color ]; then
  echo "Cloning PlantVillage-Dataset (4.8 GB) ..."
  git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset PlantVillage-Dataset
fi
if [ ! -d PlantDoc-Dataset/train ]; then
  echo "Cloning PlantDoc-Dataset (1.9 GB) ..."
  git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset PlantDoc-Dataset
fi
for f in color_train color_test; do
  if [ ! -s "splits/$f.txt" ]; then
    echo "Fetching the maintainers' split file $f.txt ..."
    curl -sL --retry 5 --retry-delay 2 -o "splits/$f.txt" "$HF/$f.txt"
  fi
done

n_pv=$(find PlantVillage-Dataset/raw/color -type f | wc -l)
n_pd=$(find PlantDoc-Dataset/train PlantDoc-Dataset/test -type f | wc -l)
n_te=$(wc -l < splits/color_test.txt)
if [ "$n_pv" != 54305 ] || [ "$n_pd" != 2578 ] || [ "$n_te" != 10709 ]; then
  echo "ERROR: expected 54,305 PlantVillage colour images, 2,578 PlantDoc images and a 10,709-line"
  echo "       official test list; found $n_pv / $n_pd / $n_te"; exit 1
fi
echo "  PlantVillage 54,305 colour images, PlantDoc 2,578, official test list 10,709."

if [ ! -d "$REPO_ROOT/data/imagenette/imagenette2-320/train" ]; then
  echo "ERROR: data/imagenette/imagenette2-320/ missing — run ./download_imagenette.sh first (the composites need it)"
  exit 1
fi

cd "$REPO_ROOT"
echo "Preprocessing (both splits + census + leaf map + leak audit + masks + composites + augmentations → data/plant) ..."
"$PY" preprocess_plant.py data/plant data/plant --stats --composites --aug

echo
echo "Done. The base arm under each split:"
echo "  CUDA_VISIBLE_DEVICES=0 lake exe plant-leaf arm=base split=grouped epochs=10"
echo "  CUDA_VISIBLE_DEVICES=1 lake exe plant-leaf arm=base split=random  epochs=10"
