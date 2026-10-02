#!/usr/bin/env bash
# The 84-EU fold sweeps for the two 30-epoch arms (relaunched 2026-10-02 00:55: the 00:11 launch
# died with its session before writing a line). One per GPU beside the trainers — the fold's
# per-pair tensors are ~30 MB at L = 482, so --max-len 512 (78 EUs) fits in the ~4 GB the PJRT
# arena leaves; the six longer EUs wait for a free card. Then one scoring pass, the arms in turn.
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
P=.venv-casp/bin/python
A=.lake/build/distogram_r16x64_train_full_e30_targets
B=.lake/build/distogram_r16x128_train_full_e30-c128_targets
echo "[$(date +%H:%M)] folding"
CUDA_VISIBLE_DEVICES=0 $P -u scripts/demos/casp16_fold.py $A --device cuda --max-len 512 --threads 4 > $A/fold_all.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $P -u scripts/demos/casp16_fold.py $B --device cuda --max-len 512 --threads 4 > $B/fold_all.log 2>&1 &
wait
echo "[$(date +%H:%M)] scoring"
for d in $A $B; do $P -u scripts/demos/casp16_fold_score.py $d > $d/fold_scores.log 2>&1; done
echo "[$(date +%H:%M)] done"
