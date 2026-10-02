#!/usr/bin/env bash
# Queue 11 (11:40): GPU 2 after 650M × 128 ch — both levers at once: 650M, 128 ch, crop 96 (batch 16),
# 30 ep, + finish. The config a 650M book run would use; does +0.010 (width) stack with +0.012 (crop)?
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
export CUDA_VISIBLE_DEVICES=2
echo "[$(date +%H:%M)] 650M × 128 ch × crop 96 × 30 ep on GPU 2"
d=runs/2026-10-02-distogram-r16x128-e30-esm650-crop96; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=30 batch=16 ch=128 units=16 crop=96 seed=1 tag=e30-esm650-crop96 > $d/train.log 2>&1
echo "[$(date +%H:%M)] $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 2 list=train_full fs=esm650 dim=1289 crop=96 ch=128 units=16 tag=e30-esm650-crop96 > $A/finish_esm650x128_crop96.log 2>&1
tail -n 5 $A/finish_esm650x128_crop96.log
echo "[$(date +%H:%M)] queue11 done"
