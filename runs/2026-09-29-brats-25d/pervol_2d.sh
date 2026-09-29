#!/usr/bin/env bash
# Per-volume Dice of every 2D BraTS checkpoint on the box, one after another on one GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/2026-09-29-brats-25d
export CUDA_VISIBLE_DEVICES=2
run() { echo "=== $*"; ./.lake/build/bin/brats-eval "$@" 2>&1 | grep -v 'ptxas\|^W0\|^I0\|^E0\|pjrt_ffi'; }
run net=r34 arm=r34 best out=$OUT/pervol_2d_r34.csv
run net=r34 arm=scratch best out=$OUT/pervol_2d_scratch.csv
run net=r34noskip arm=scratch_noskip best out=$OUT/pervol_2d_noskip.csv
run arm=dicece best out=$OUT/pervol_2d_unet240_dicece.csv
run arm=ce best out=$OUT/pervol_2d_unet240_ce.csv
echo ALL-DONE
