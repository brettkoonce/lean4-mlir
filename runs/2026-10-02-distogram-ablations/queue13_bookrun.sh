#!/usr/bin/env bash
# Queue 13 — the 650M book run (plan §11): 128 ch × crop 96 × 100 epochs with the orientation heads,
# the config queue10 and queue11 settled (width and crop add; the ω/φ fold gain stacks on crop 96).
# ~727 s/epoch on a 4060 Ti → ~20 h of training, then predict / assemble / plain fold / score
# (~45 min, finish_run.sh) and the ω/φ fold + score (~40 min). Written 2026-10-02 18:40, NOT launched
# — Brett schedules it. usage (the argument is the card):
#   setsid -f nohup runs/2026-10-02-distogram-ablations/queue13_bookrun.sh 1 > runs/2026-10-02-distogram-ablations/queue13.log 2>&1 < /dev/null
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
gpu=${1:?usage: queue13_bookrun.sh <gpu>}
export CUDA_VISIBLE_DEVICES=$gpu
echo "[$(date +%H:%M)] 650M × 128 ch × crop 96 × 100 ep × orientation heads on GPU $gpu"
d=runs/$(date +%F)-distogram-r16x128-e100-esm650-crop96-orient; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=100 batch=16 ch=128 units=16 crop=96 orient=1 seed=1 tag=e100-esm650-crop96-orient > $d/train.log 2>&1
echo "[$(date +%H:%M)] $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh $gpu list=train_full fs=esm650 dim=1289 crop=96 ch=128 units=16 orient=1 tag=e100-esm650-crop96-orient > $A/finish_bookrun.log 2>&1
tail -n 5 $A/finish_bookrun.log
O=$(sed -n 's/^\[..:..\] done \(.*\)$/\1/p' $A/finish_bookrun.log | tail -n 1)
if [ -d "$O" ]; then
  echo "[$(date +%H:%M)] ω/φ fold $O"
  $P -u scripts/demos/casp16_fold.py $O --orient --restarts 0 --device cuda --max-len 512 --threads 4 > $O/fold_orient_all.log 2>&1
  $P -u scripts/demos/casp16_fold_score.py $O --suffix orient > $O/fold_scores_orient.log 2>&1; tail -n 5 $O/fold_scores_orient.log
fi
echo "[$(date +%H:%M)] queue13 done"
