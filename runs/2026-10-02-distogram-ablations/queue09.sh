#!/usr/bin/env bash
# Queue 9 (09:45): GPU 3 after the book run's finish and the ensemble fold — 650M × 64 ch × 30 ep at
# seed 2: the seed-noise bar at the headline LM (35M had one: 0.585 vs 0.599) and a third ensemble member.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
until grep -q "done \.lake" $A/finish_book_e100.log 2>/dev/null && grep -q "done \.lake" $A/ens_esm650x2.log 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=3
echo "[$(date +%H:%M)] 650M × 64 ch × 30 ep seed 2 on GPU 3"
d=runs/2026-10-02-distogram-r16x64-e30-esm650-s2; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=30 batch=32 ch=64 units=16 seed=2 tag=e30-esm650-s2 > $d/train.log 2>&1
echo "[$(date +%H:%M)] $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 3 list=train_full fs=esm650 dim=1289 ch=64 units=16 tag=e30-esm650-s2 > $A/finish_esm650_s2.log 2>&1
tail -n 5 $A/finish_esm650_s2.log
echo "[$(date +%H:%M)] queue09 done"
