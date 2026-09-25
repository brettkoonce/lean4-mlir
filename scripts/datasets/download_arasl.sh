#!/bin/bash
# Fetch ArASL (Arabic sign-language alphabet, 54,049 grey 64×64 hand crops, 32
# letters) and preprocess it to the two-split flat binaries demos/MainAraslSigns.lean
# reads — planning/arasl_people_watching_demo.md §2.
#
# Source: Mendeley Data, dataset y7pckrw6z2 version 1 (2018-11-05), CC BY 4.0.
# The public download URLs carry a per-file uuid, so they are resolved from the
# public API at run time rather than hardcoded; the sha256 of each file is
# checked against what the API reports.
#
# Citation: Latif, Mohammad, Alghazo, AlKhalaf & AlKhalaf, "ArASL: Arabic
#           Alphabets Sign Language Dataset", Data in Brief 23, 103777 (2019).
#           doi 10.17632/y7pckrw6z2.1
#
# Usage: ./scripts/datasets/download_arasl.sh            (idempotent over data/arasl/)
# Requires: curl, unzip, python3 + Pillow + numpy (the repo .venv has both).
set -e

DATASET="y7pckrw6z2"
API="https://data.mendeley.com/public-api/datasets/${DATASET}/files?folder_id=root&version=1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/data/arasl"
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

mkdir -p "$OUT"
cd "$OUT"

if [ -f blocked_train.bin ] && [ -f random_train.bin ] && [ -f meta_blocked.npz ]; then
  echo "data/arasl already preprocessed — nothing to do. (Delete the .bin files to force a rebuild.)"
  exit 0
fi

# Resolve filename → download_url + sha256 from the API (one JSON array).
if [ ! -f ArASL_Database_54K_Final.zip ] || [ ! -f ArSL_Data_Labels.csv ]; then
  echo "Resolving file URLs from the Mendeley API ..."
  curl -sL --retry 5 --retry-delay 2 "$API" > files.json
  "$PY" - <<'EOF'
import json, subprocess, hashlib, os
want = ("ArASL_Database_54K_Final.zip", "ArSL_Data_Labels.csv", "Signs_32_New.png")
files = {f["filename"]: f["content_details"] for f in json.load(open("files.json"))}
for name in want:
    if name not in files:
        raise SystemExit(f"{name} not listed by the API — check {open('files.json').read()[:500]}")
    cd = files[name]
    if os.path.exists(name):
        if hashlib.sha256(open(name, "rb").read()).hexdigest() == cd["sha256_hash"]:
            print(f"  {name}: present, sha256 ok"); continue
        print(f"  {name}: present but sha256 differs — refetching")
    print(f"  fetching {name} ({cd['size'] / 1e6:.1f} MB)")
    subprocess.check_call(["curl", "-L", "--retry", "5", "--retry-delay", "2", "-o", name, cd["download_url"]])
    got = hashlib.sha256(open(name, "rb").read()).hexdigest()
    if got != cd["sha256_hash"]:
        raise SystemExit(f"{name}: sha256 {got} != {cd['sha256_hash']} (API)")
    print(f"  {name}: sha256 ok")
EOF
fi

if [ ! -d ArASL_Database_54K_Final ]; then
  echo "Extracting ..."
  unzip -q -o ArASL_Database_54K_Final.zip
fi
n=$(find ArASL_Database_54K_Final -type f | wc -l)
if [ "$n" != 54049 ]; then
  echo "ERROR: expected 54,049 image files, found $n"; exit 1
fi
echo "  54,049 images in 32 class folders."

cd "$REPO_ROOT"
echo "Preprocessing (both splits + census + chain statistic + leak audit → data/arasl) ..."
"$PY" scripts/datasets/preprocess_arasl.py data/arasl data/arasl --stats

echo
echo "Done. Train the chapter-4 CNN under each split with:"
echo "  CUDA_VISIBLE_DEVICES=0 lake exe arasl-signs net=cifar8w split=random  epochs=30"
echo "  CUDA_VISIBLE_DEVICES=1 lake exe arasl-signs net=cifar8w split=blocked epochs=30"
