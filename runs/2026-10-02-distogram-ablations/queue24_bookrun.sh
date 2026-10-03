#!/usr/bin/env bash
# Queue 24 (10:30) — THE BOOK RUN: queue22 (0.877 / 0.628 / 0.645 at 30 epochs) with epochs=100: 3B × 128 ch × crop 96 with
# pair=1 orient=1, then finish and the ω/φ fold. ~755 s/epoch → ~21 h of training, then ~45 min finish and ~20 min
# ω/φ fold. Written 2026-10-03 10:30, NOT launched — Brett schedules it. usage (the argument is the card):
#   setsid -f nohup runs/2026-10-02-distogram-ablations/queue24_bookrun.sh <gpu> > runs/2026-10-02-distogram-ablations/queue24.log 2>&1 < /dev/null
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
gpu=${1:?usage: queue24_bookrun.sh <gpu>}
export CUDA_VISIBLE_DEVICES=$gpu
echo "[$(date +%H:%M)] 3B × 128 ch × crop 96 × 100 ep, pair=1 orient=1, on GPU $gpu"
d=runs/$(date +%F)-distogram-r16x128-e100-esm3b-crop96-pair1-orient; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm3b dim=2569 epochs=100 batch=16 ch=128 units=16 crop=96 pair=1 orient=1 seed=1 tag=e100-esm3b-crop96-pair1-orient > $d/train.log 2>&1
echo "[$(date +%H:%M)] trained: $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh $gpu list=train_full fs=esm3b dim=2569 crop=96 ch=128 units=16 pair=1 orient=1 tag=e100-esm3b-crop96-pair1-orient > $A/finish_bookrun.log 2>&1
tail -n 5 $A/finish_bookrun.log
O=$(sed -n 's/^\[..:..\] done \(.*\)$/\1/p' $A/finish_bookrun.log | tail -n 1)
if [ -d "$O" ]; then
  echo "[$(date +%H:%M)] ω/φ fold $O"
  $P -u scripts/demos/casp16_fold.py $O --orient --restarts 0 --device cuda --max-len 512 --threads 4 > $O/fold_orient_all.log 2>&1
  $P -u scripts/demos/casp16_fold_score.py $O --suffix orient > $O/fold_scores_orient.log 2>&1; tail -n 5 $O/fold_scores_orient.log
fi
echo "[$(date +%H:%M)] queue24 done"
