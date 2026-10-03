#!/usr/bin/env bash
# Queue 21 (01:55) — the 650M twin of queue17: 650M × 128 ch × crop 96 × 30 ep with pair=1, against queue11
# (the same without the plane, 0.858 / 0.600 / 0.598) and queue17 (the 3B version). Whether the book run
# needs 3B features at the wide config, or the plane alone carries it. GPU 3 (free since queue19 finished),
# started once queue20's trainer has exited (three 3B-pool trainers fill the host; a fourth dies at cuInit).
# ~6.1 h of training + 45 min finish.
# usage: setsid -f nohup runs/2026-10-02-distogram-ablations/queue21_650m_pair_wide.sh > runs/2026-10-02-distogram-ablations/queue21.log 2>&1 < /dev/null
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
while ! grep -q "^\[..:..\] trained: trained" $A/queue20.log 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=3
echo "[$(date +%H:%M)] 650M × 128 ch × crop 96 × 30 ep, pair=1, on GPU 3"
d=runs/$(date +%F)-distogram-r16x128-e30-esm650-crop96-pair1; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=30 batch=16 ch=128 units=16 crop=96 pair=1 seed=1 tag=e30-esm650-crop96-pair1 > $d/train.log 2>&1
echo "[$(date +%H:%M)] trained: $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 3 list=train_full fs=esm650 dim=1289 crop=96 ch=128 units=16 pair=1 tag=e30-esm650-crop96-pair1 > $A/finish_esm650_crop96_pair1.log 2>&1
tail -n 5 $A/finish_esm650_crop96_pair1.log
echo "[$(date +%H:%M)] queue21 done"
