#!/usr/bin/env bash
# The 2D R34 anchor's second seed (plan §4c item 6): the 09-25 r34 arm with LEAN_MLIR_SEED=1 moving the decoder's
# He-init, the shuffle and the augmentation; tag=s2 keeps its artifacts apart from the seed-0 checkpoint. ~50 min.
# Then its per-patient CSV. usage: seed2_2d.sh <gpu>
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
R=runs/2026-10-03-brats-3d
export CUDA_VISIBLE_DEVICES=${1:?gpu}
echo "[$(date +%H:%M)] r34 seed 1 on GPU $1"
LEAN_MLIR_SEED=1 ./.lake/build/bin/unet-brats-r34 data/brats224 10 r34 tag=s2 > $R/brats_skip_r34_s2.log 2>&1
echo "[$(date +%H:%M)] trained: $(tail -n 2 $R/brats_skip_r34_s2.log | tr '\n' ' ')"
sleep 5
lake exe brats-eval net=r34 arm=r34_s2 best out=$R/pervol_2d_r34_s2.csv > $R/pervol_2d_r34_s2.log 2>&1
tail -n 12 $R/pervol_2d_r34_s2.log
echo "[$(date +%H:%M)] seed2 done"
