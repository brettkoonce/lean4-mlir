#!/usr/bin/env bash
# The 3D UNet at 20k steps (plan §4c, the short form of the long session): 09-29's 4k-step run five times longer
# on the same recipe (128³ × B2, Adam 3e-4, 200-step warmup, cosine over 20k, Dice + CE), per-patient eval with the
# tail every 2k steps, checkpointed there. ~909 ms/step → ~5 h + ~5 min per eval. Re-launch with the resume line
# below if the box drops it. NAME= picks the output prefix (a second seed: NAME=unet3d_20k_s1 ... --seed 1).
# usage: [NAME=unet3d_20k] unet3d_20k.sh <gpu> [--seed k] [--resume runs/2026-10-03-brats-3d/<NAME>_ckpt.npz]
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
R=runs/2026-10-03-brats-3d
N=${NAME:-unet3d_20k}
export CUDA_VISIBLE_DEVICES=${1:?gpu}; shift
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
echo "[$(date +%H:%M)] 3D UNet 20k ($N) on GPU $CUDA_VISIBLE_DEVICES $*"
.venv/bin/python -u jax/scripts/unet3d_brats.py --steps 20000 --eval-every 2000 --out $R/$N "$@" \
  2>&1 | grep --line-buffered -vE '^[WI][0-9]{4} |ptxas' > $R/$N.out
echo "[$(date +%H:%M)] 3D done"
