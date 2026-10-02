#!/usr/bin/env bash
# Night queue 7 (07:25): redo of queue06's tail after the disk filled mid-assembly (the orient × 650M
# accumulators are 132 channels wide — 7.8 GB — and the assembly died at 50 of 84 with ENOSPC).
# GPU 0: re-assemble, drop the accumulators, plain fold + score, ω/φ fold + score, then crop 96 on 650M.
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
export CUDA_VISIBLE_DEVICES=0
O=.lake/build/distogram_r16x64_esm650_orient_train_full_e30-esm650-orient_targets
echo "[$(date +%H:%M)] re-assemble $O"
$P scripts/demos/casp16_predict.py $O | tail -n 5
n=$(ls "${O:?}"/*.pred.npz | wc -l); echo "pred.npz: $n"
[ "$n" -ge 84 ] && rm -f "${O:?}"/*.acc.bin "${O:?}"/*.cnt.bin && echo "accumulators dropped; $(df -h . | tail -n 1 | awk '{print $4}') free"
echo "[$(date +%H:%M)] plain fold"
$P -u scripts/demos/casp16_fold.py $O --device cuda --max-len 512 --threads 4 > $O/fold_all.log 2>&1
$P -u scripts/demos/casp16_fold_score.py $O > $O/fold_scores.log 2>&1; tail -n 5 $O/fold_scores.log
echo "[$(date +%H:%M)] ω/φ fold"
$P -u scripts/demos/casp16_fold.py $O --orient --restarts 0 --device cuda --max-len 512 --threads 4 > $O/fold_orient_all.log 2>&1
$P -u scripts/demos/casp16_fold_score.py $O --suffix orient > $O/fold_scores_orient.log 2>&1; tail -n 5 $O/fold_scores_orient.log
echo "[$(date +%H:%M)] crop 96 × 650M, 64 ch"
d=runs/2026-10-02-distogram-r16x64-e30-esm650-crop96; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=30 batch=16 ch=64 units=16 crop=96 seed=1 tag=e30-esm650-crop96 > $d/train.log 2>&1
echo "[$(date +%H:%M)] $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 0 list=train_full fs=esm650 dim=1289 crop=96 ch=64 units=16 tag=e30-esm650-crop96 > $A/finish_esm650_crop96.log 2>&1
tail -n 5 $A/finish_esm650_crop96.log
echo "[$(date +%H:%M)] queue07 done"
