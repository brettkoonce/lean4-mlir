#!/usr/bin/env bash
# Queue 17 (23:50; first written 22:10 for 650M) — the book-run candidate at 30 epochs: 3B × 128 ch × crop 96
# with pair=1 (the 3B contact head's plane). queue16 put 3B + plane at 64 ch at 0.866 (hard 0.724), above
# the 128 ch × crop 96 arm's 0.858 and the plane alone's 0.856; this is the same stack at the wide config,
# against queue11 (0.858 / 0.600 / 0.598) and the 3B × 128 ch × crop 96 arm without the plane (queue14,
# GPU 1). GPU 0 (free). ~6.5 h of training + 45 min finish.
# usage: setsid -f nohup runs/2026-10-02-distogram-ablations/queue17_pair_wide.sh > runs/2026-10-02-distogram-ablations/queue17.log 2>&1 < /dev/null
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
export CUDA_VISIBLE_DEVICES=0
echo "[$(date +%H:%M)] 3B × 128 ch × crop 96 × 30 ep, pair=1, on GPU 0"
d=runs/$(date +%F)-distogram-r16x128-e30-esm3b-crop96-pair1; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm3b dim=2569 epochs=30 batch=16 ch=128 units=16 crop=96 pair=1 seed=1 tag=e30-esm3b-crop96-pair1 > $d/train.log 2>&1
echo "[$(date +%H:%M)] trained: $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 0 list=train_full fs=esm3b dim=2569 crop=96 ch=128 units=16 pair=1 tag=e30-esm3b-crop96-pair1 > $A/finish_esm3b_crop96_pair1.log 2>&1
tail -n 5 $A/finish_esm3b_crop96_pair1.log
echo "[$(date +%H:%M)] queue17 done"
