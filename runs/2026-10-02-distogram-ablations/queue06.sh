#!/usr/bin/env bash
# Night queue 6 (06:10), GPU 0 (free after the esm150 × 128 ch finish): the orientation heads on
# 650M features — do the angular heads sharpen with a stronger LM, and does the ω/φ fold gain grow? —
# then crop 96 on 650M features. Each run is finished (predict / assemble / fold / score) in turn.
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
export CUDA_VISIBLE_DEVICES=0
echo "[$(date +%H:%M)] orient × 650M, 64 ch"
d=runs/2026-10-02-distogram-r16x64-e30-esm650-orient; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 orient=1 epochs=30 batch=32 ch=64 units=16 seed=1 tag=e30-esm650-orient > $d/train.log 2>&1
echo "[$(date +%H:%M)] $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 0 list=train_full fs=esm650 dim=1289 orient=1 ch=64 units=16 tag=e30-esm650-orient > $A/finish_esm650_orient.log 2>&1
out=$(sed -n 's/^\[..:..\] done \(.*\)$/\1/p' $A/finish_esm650_orient.log | tail -n 1)
tail -n 5 $A/finish_esm650_orient.log
if [ -d "$out" ]; then
  echo "[$(date +%H:%M)] ω/φ fold: $out"
  $P -u scripts/demos/casp16_fold.py $out --orient --restarts 0 --device cuda --max-len 512 --threads 4 > $out/fold_orient_all.log 2>&1
  $P -u scripts/demos/casp16_fold_score.py $out --suffix orient > $out/fold_scores_orient.log 2>&1
  tail -n 5 $out/fold_scores_orient.log
fi
echo "[$(date +%H:%M)] crop 96 × 650M, 64 ch"
d=runs/2026-10-02-distogram-r16x64-e30-esm650-crop96; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=30 batch=16 ch=64 units=16 crop=96 seed=1 tag=e30-esm650-crop96 > $d/train.log 2>&1
echo "[$(date +%H:%M)] $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 0 list=train_full fs=esm650 dim=1289 crop=96 ch=64 units=16 tag=e30-esm650-crop96 > $A/finish_esm650_crop96.log 2>&1
tail -n 5 $A/finish_esm650_crop96.log
echo "[$(date +%H:%M)] queue06 done"
