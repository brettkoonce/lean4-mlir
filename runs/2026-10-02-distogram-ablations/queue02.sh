#!/usr/bin/env bash
# Night queue 2 (01:30): GPU 2 — when the esm150 trainer exits, the orientation-heads run;
# GPU 0 — when the one-hot trainer exits, finish (predict/fold/score) esm150 and one-hot, then
# the esm150 × 128 ch run (both levers at once).
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
(
  while pgrep -f "[f]s=esm150 dim=649 epochs=30 batch=32 ch=64" > /dev/null; do sleep 60; done
  echo "[$(date +%H:%M)] esm150 done; orient run on GPU 2"
  d=runs/2026-10-02-distogram-r16x64-e30-orient; mkdir -p $d
  CUDA_VISIBLE_DEVICES=2 lake exe distogram-casp train list=train_full orient=1 epochs=30 batch=32 ch=64 units=16 seed=1 tag=e30-orient > $d/train.log 2>&1
  echo "[$(date +%H:%M)] orient run done: $(grep trained $d/train.log)"
) &
(
  while pgrep -f "[f]s=onehot dim=30" > /dev/null; do sleep 60; done
  echo "[$(date +%H:%M)] one-hot done; finishing esm150 and one-hot on GPU 0"
  MEMFRAC=0.5 $A/finish_run.sh 0 list=train_full fs=esm150 dim=649 ch=64 units=16 tag=e30-esm150 > $A/finish_esm150.log 2>&1
  MEMFRAC=0.5 $A/finish_run.sh 0 list=train_full fs=onehot dim=30 ch=64 units=16 tag=e30-onehot > $A/finish_onehot.log 2>&1
  echo "[$(date +%H:%M)] esm150 × 128 ch run on GPU 0"
  d=runs/2026-10-02-distogram-r16x128-e30-esm150; mkdir -p $d
  CUDA_VISIBLE_DEVICES=0 lake exe distogram-casp train list=train_full fs=esm150 dim=649 epochs=30 batch=32 ch=128 units=16 seed=1 tag=e30-esm150 > $d/train.log 2>&1
  echo "[$(date +%H:%M)] esm150 × 128 ch done: $(grep trained $d/train.log)"
) &
wait
