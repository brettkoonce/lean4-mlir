#!/usr/bin/env bash
# Queue 10 (11:05): GPU 0 after crop 96 × 650M — the orientation heads on that config
# (650M, 64 ch, crop 96, batch 16, 30 ep), then predict / assemble / plain fold / ω-φ fold / score.
# Answers whether the crop-96 gain (+0.012 EUs, +0.014 / +0.017 fold) and the ω/φ fold gain stack.
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
export CUDA_VISIBLE_DEVICES=0
echo "[$(date +%H:%M)] crop 96 × 650M × orient, 64 ch"
d=runs/2026-10-02-distogram-r16x64-e30-esm650-crop96-orient; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=30 batch=16 ch=64 units=16 crop=96 orient=1 seed=1 tag=e30-esm650-crop96-orient > $d/train.log 2>&1
echo "[$(date +%H:%M)] $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 0 list=train_full fs=esm650 dim=1289 crop=96 ch=64 units=16 orient=1 tag=e30-esm650-crop96-orient > $A/finish_esm650_crop96_orient.log 2>&1
tail -n 5 $A/finish_esm650_crop96_orient.log
O=$(sed -n 's/^\[..:..\] done \(.*\)$/\1/p' $A/finish_esm650_crop96_orient.log | tail -n 1)
if [ -d "$O" ]; then
  echo "[$(date +%H:%M)] ω/φ fold $O"
  $P -u scripts/demos/casp16_fold.py $O --orient --restarts 0 --device cuda --max-len 512 --threads 4 > $O/fold_orient_all.log 2>&1
  $P -u scripts/demos/casp16_fold_score.py $O --suffix orient > $O/fold_scores_orient.log 2>&1; tail -n 5 $O/fold_scores_orient.log
fi
echo "[$(date +%H:%M)] queue10 done"
