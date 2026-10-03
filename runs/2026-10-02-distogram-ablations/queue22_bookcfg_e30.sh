#!/usr/bin/env bash
# Queue 22 (02:40) — the book-run config at 30 epochs, the dress rehearsal: 3B × 128 ch × crop 96 with
# pair=1 orient=1, then finish and the ω/φ fold. Every lever the night settled, together: the 3B contact
# head's plane (+0.018 precision at 64 ch, stacks with 3B: 0.866 / 0.603 / 0.612), width + crop (+0.020 /
# +0.031 / +0.026 at 650M), the orientation heads (+0.016 TM through the ω/φ fold at 3B + plane). Against
# queue17 (the same without the heads, GPU 0). GPU 1, once queue14 (the 3B × 128 ch × crop 96 arm and its
# finish) is done — three trainers at once is the host's limit. ~6.5 h train + 45 min finish + 40 min ω/φ fold.
# usage: setsid -f nohup runs/2026-10-02-distogram-ablations/queue22_bookcfg_e30.sh > runs/2026-10-02-distogram-ablations/queue22.log 2>&1 < /dev/null
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
while ! grep -q "queue14 done" $A/queue14.log 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=1
echo "[$(date +%H:%M)] 3B × 128 ch × crop 96 × 30 ep, pair=1 orient=1, on GPU 1"
d=runs/$(date +%F)-distogram-r16x128-e30-esm3b-crop96-pair1-orient; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm3b dim=2569 epochs=30 batch=16 ch=128 units=16 crop=96 pair=1 orient=1 seed=1 tag=e30-esm3b-crop96-pair1-orient > $d/train.log 2>&1
echo "[$(date +%H:%M)] trained: $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 1 list=train_full fs=esm3b dim=2569 crop=96 ch=128 units=16 pair=1 orient=1 tag=e30-esm3b-crop96-pair1-orient > $A/finish_esm3b_crop96_pair1_orient.log 2>&1
tail -n 5 $A/finish_esm3b_crop96_pair1_orient.log
O=$(sed -n 's/^\[..:..\] done \(.*\)$/\1/p' $A/finish_esm3b_crop96_pair1_orient.log | tail -n 1)
if [ -d "$O" ]; then
  echo "[$(date +%H:%M)] ω/φ fold $O"
  $P -u scripts/demos/casp16_fold.py $O --orient --restarts 0 --device cuda --max-len 512 --threads 4 > $O/fold_orient_all.log 2>&1
  $P -u scripts/demos/casp16_fold_score.py $O --suffix orient > $O/fold_scores_orient.log 2>&1; tail -n 5 $O/fold_scores_orient.log
fi
echo "[$(date +%H:%M)] queue22 done"
