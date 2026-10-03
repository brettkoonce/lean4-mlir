#!/usr/bin/env bash
# Queue 18 (22:35) — the plane and the orientation heads together at 64 ch: 650M × 64 ch × 30 ep with
# pair=1 orient=1, then finish and the ω/φ fold. The book run carries orient=1 (queue10: the ω/φ fold
# stacks on crop 96), queue15 put the plane at 0.856 / 0.584 / 0.582 — do the two stack? GPU 2 (free
# since queue15 finished). ~70 min train + 45 min finish + 40 min ω/φ fold.
# usage: setsid -f nohup runs/2026-10-02-distogram-ablations/queue18_pair_orient.sh > runs/2026-10-02-distogram-ablations/queue18.log 2>&1 < /dev/null
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
export CUDA_VISIBLE_DEVICES=2
echo "[$(date +%H:%M)] 650M × 64 ch × 30 ep, pair=1 orient=1, on GPU 2"
d=runs/$(date +%F)-distogram-r16x64-e30-esm650-pair1-orient; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=30 batch=32 ch=64 units=16 pair=1 orient=1 seed=1 tag=e30-esm650-pair1-orient > $d/train.log 2>&1
echo "[$(date +%H:%M)] trained: $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 2 list=train_full fs=esm650 dim=1289 ch=64 units=16 pair=1 orient=1 tag=e30-esm650-pair1-orient > $A/finish_esm650_pair1_orient.log 2>&1
tail -n 5 $A/finish_esm650_pair1_orient.log
O=$(sed -n 's/^\[..:..\] done \(.*\)$/\1/p' $A/finish_esm650_pair1_orient.log | tail -n 1)
if [ -d "$O" ]; then
  echo "[$(date +%H:%M)] ω/φ fold $O"
  $P -u scripts/demos/casp16_fold.py $O --orient --restarts 0 --device cuda --max-len 512 --threads 4 > $O/fold_orient_all.log 2>&1
  $P -u scripts/demos/casp16_fold_score.py $O --suffix orient > $O/fold_scores_orient.log 2>&1; tail -n 5 $O/fold_scores_orient.log
fi
echo "[$(date +%H:%M)] queue18 done"
