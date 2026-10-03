#!/usr/bin/env bash
# Queue 23 (06:45) — schedule length with the plane: 3B × 64 ch × 100 ep pair=1, against the 30-epoch arm
# (queue16: 0.866 / 0.603 / 0.612; seed 2 0.866 / 0.605 / 0.614). At 650M × 64 ch the 100-epoch schedule was
# worth +0.009 / +0.018 / +0.016 — what the book run's 100 epochs add on top of the plane. GPU 0 (free since
# queue17 finished). ~4.3 h train + 45 min finish.
# usage: setsid -f nohup runs/2026-10-02-distogram-ablations/queue23_esm3b_pair_e100.sh > runs/2026-10-02-distogram-ablations/queue23.log 2>&1 < /dev/null
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
export CUDA_VISIBLE_DEVICES=0
echo "[$(date +%H:%M)] 3B × 64 ch × 100 ep, pair=1, on GPU 0"
d=runs/$(date +%F)-distogram-r16x64-e100-esm3b-pair1; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm3b dim=2569 epochs=100 batch=32 ch=64 units=16 pair=1 seed=1 tag=e100-esm3b-pair1 > $d/train.log 2>&1
echo "[$(date +%H:%M)] trained: $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 0 list=train_full fs=esm3b dim=2569 ch=64 units=16 pair=1 tag=e100-esm3b-pair1 > $A/finish_esm3b_pair1_e100.log 2>&1
tail -n 5 $A/finish_esm3b_pair1_e100.log
echo "[$(date +%H:%M)] queue23 done"
