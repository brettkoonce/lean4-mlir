#!/usr/bin/env bash
# Queue 12 (12:15): GPU 3 — fold + score the five-member 650M ensemble (casp16_ensemble.py esm650x5), then the
# no-template ablation on the right LM: list=train (the 82 template chains purged), 650M, 64 ch, 30 ep, + finish.
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
$A/ens_fold.sh 3 .lake/build/distogram_ens-esm650x5_targets
export CUDA_VISIBLE_DEVICES=3
echo "[$(date +%H:%M)] 650M × 64 ch × 30 ep on the purged list (train.csv) on GPU 3"
d=runs/2026-10-02-distogram-r16x64-e30-esm650-purged; mkdir -p $d
lake exe distogram-casp train list=train fs=esm650 dim=1289 epochs=30 batch=32 ch=64 units=16 seed=1 tag=e30-esm650-purged > $d/train.log 2>&1
echo "[$(date +%H:%M)] $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 3 list=train fs=esm650 dim=1289 ch=64 units=16 tag=e30-esm650-purged > $A/finish_esm650_purged.log 2>&1
tail -n 5 $A/finish_esm650_purged.log
echo "[$(date +%H:%M)] queue12 done"
