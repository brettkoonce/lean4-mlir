#!/usr/bin/env bash
# Queue 20 (00:30) — seed 2 of the best arm, 3B × 64 ch × 30 ep pair=1 (queue16: 0.866 / 0.603 / 0.612): the
# error bar for the row the book run is being built on (the 650M seed gap was 0.002 / 0.001). GPU 2 (free
# since queue18 finished). ~77 min train + 45 min finish.
# 00:52: the first launch died at cuInit (CUDA_ERROR_NOT_INITIALIZED) — three 3B-pool trainers (33 GB each,
# 36–44 GB RSS) plus 108 GB of page cache left 23 GB free and the driver would not initialise; at most three
# 3B trainers at once. Relaunched waiting for queue19's trainer to exit.
# usage: setsid -f nohup runs/2026-10-02-distogram-ablations/queue20_esm3b_pair_s2.sh > runs/2026-10-02-distogram-ablations/queue20.log 2>&1 < /dev/null
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
while ! grep -q "^\[..:..\] trained: trained" $A/queue19.log 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=2
echo "[$(date +%H:%M)] 3B × 64 ch × 30 ep, pair=1, seed 2, on GPU 2"
d=runs/$(date +%F)-distogram-r16x64-e30-esm3b-pair1-s2; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm3b dim=2569 epochs=30 batch=32 ch=64 units=16 pair=1 seed=2 tag=e30-esm3b-pair1-s2 > $d/train.log 2>&1
echo "[$(date +%H:%M)] trained: $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 2 list=train_full fs=esm3b dim=2569 ch=64 units=16 pair=1 tag=e30-esm3b-pair1-s2 > $A/finish_esm3b_pair1_s2.log 2>&1
tail -n 5 $A/finish_esm3b_pair1_s2.log
echo "[$(date +%H:%M)] queue20 done"
