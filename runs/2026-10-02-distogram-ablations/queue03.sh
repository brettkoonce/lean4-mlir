#!/usr/bin/env bash
# Night queue 3 (02:05). GPU 2 — when the orient trainer exits: finish it (predict/assemble/plain
# fold/score), predict the val subset and bench the orientation restraint weight on val, fold the
# EUs with the ω/φ restraints and score that too; then a seed-2 esm150 run for the row's error bar.
# GPU 1 — when the crop-96 trainer exits: finish it, then the `crops=3` (data passes) row.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
(
  while pgrep -f "[o]rient=1 epochs=30 batch=32 ch=64" > /dev/null; do sleep 60; done
  echo "[$(date +%H:%M)] orient trainer done; finishing on GPU 2"
  MEMFRAC=0.5 $A/finish_run.sh 2 list=train_full orient=1 ch=64 units=16 tag=e30-orient > $A/finish_orient.log 2>&1
  out=$(sed -n 's/^\[..:..\] done \(.*\)$/\1/p' $A/finish_orient.log | tail -n 1)
  echo "[$(date +%H:%M)] orient fold + val bench: $out"
  CUDA_VISIBLE_DEVICES=2 LEAN_MLIR_MEM_FRACTION=0.5 lake exe distogram-casp predict list=train_full orient=1 ch=64 units=16 tag=e30-orient pool=valsub > $A/predict_orient_valsub.log 2>&1
  vdir=$(sed -n 's/.* -> \(.*\)\/$/\1/p' $A/predict_orient_valsub.log | tail -n 1)
  CUDA_VISIBLE_DEVICES=2 $P -u scripts/demos/casp16_valfold.py $vdir --orient --restarts 0 --threads 4 --grid "ref:1;chain:1;clash:3;sigma:1;orient:1,0.3,3,10" > $A/valfold_orient.log 2>&1
  CUDA_VISIBLE_DEVICES=2 $P -u scripts/demos/casp16_valfold.py $vdir --restarts 0 --threads 4 --grid "ref:1;chain:1;clash:3;sigma:1" > $A/valfold_orient_plain.log 2>&1
  if [ -d "$out" ]; then
    CUDA_VISIBLE_DEVICES=2 $P -u scripts/demos/casp16_fold.py $out --orient --device cuda --max-len 512 --threads 4 > $out/fold_orient_all.log 2>&1
    $P -u scripts/demos/casp16_fold_score.py $out --suffix orient > $out/fold_scores_orient.log 2>&1
    tail -n 5 $out/fold_scores_orient.log
  fi
  echo "[$(date +%H:%M)] esm150 seed-2 run on GPU 2"
  d=runs/2026-10-02-distogram-r16x64-e30-esm150-s2; mkdir -p $d
  CUDA_VISIBLE_DEVICES=2 lake exe distogram-casp train list=train_full fs=esm150 dim=649 epochs=30 batch=32 ch=64 units=16 seed=2 tag=e30-esm150-s2 > $d/train.log 2>&1
  echo "[$(date +%H:%M)] esm150 seed-2 done: $(grep trained $d/train.log)"
) &
(
  while pgrep -f "[c]rop=96 seed=1" > /dev/null; do sleep 60; done
  echo "[$(date +%H:%M)] crop-96 trainer done; finishing on GPU 1"
  MEMFRAC=0.5 $A/finish_run.sh 1 list=train_full crop=96 ch=64 units=16 tag=e30-crop96 > $A/finish_crop96.log 2>&1
  echo "[$(date +%H:%M)] crops=3 run on GPU 1"
  d=runs/2026-10-02-distogram-r16x64-e30-crops3; mkdir -p $d
  CUDA_VISIBLE_DEVICES=1 lake exe distogram-casp train list=train_full crops=3 epochs=30 batch=32 ch=64 units=16 seed=1 tag=e30-crops3 > $d/train.log 2>&1
  echo "[$(date +%H:%M)] crops=3 done: $(grep trained $d/train.log)"
) &
wait
