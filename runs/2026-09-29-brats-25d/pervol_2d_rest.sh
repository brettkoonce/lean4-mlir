#!/usr/bin/env bash
# The remaining 2D checkpoints, after the scratch eval already running on GPU 2 exits.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/2026-09-29-brats-25d
export CUDA_VISIBLE_DEVICES=2
run() { echo "=== $*"; ./.lake/build/bin/brats-eval "$@" 2>&1 | grep -v 'ptxas\|^W0\|^I0\|^E0\|pjrt_ffi'; }
run net=r34noskip arm=scratch_noskip best out=$OUT/pervol_2d_noskip.csv
run arm=dicece best out=$OUT/pervol_2d_unet240_dicece.csv
run arm=ce best out=$OUT/pervol_2d_unet240_ce.csv
echo ALL-DONE
