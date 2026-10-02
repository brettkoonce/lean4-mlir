#!/usr/bin/env bash
# GPUs 0 and 1 once the 84-EU fold sweeps release them: one-hot (no language model) and crop 96.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
while pgrep -f "[c]asp16_fold.py .lake/build" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] folds done; launching"
d0=runs/2026-10-02-distogram-r16x64-e30-onehot; mkdir -p $d0
d1=runs/2026-10-02-distogram-r16x64-e30-crop96; mkdir -p $d1
CUDA_VISIBLE_DEVICES=0 nohup lake exe distogram-casp train list=train_full fs=onehot dim=30 epochs=30 batch=32 ch=64 units=16 seed=1 tag=e30-onehot > $d0/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup lake exe distogram-casp train list=train_full epochs=30 batch=16 ch=64 units=16 crop=96 seed=1 tag=e30-crop96 > $d1/train.log 2>&1 &
wait
echo "[$(date +%H:%M)] both done"
