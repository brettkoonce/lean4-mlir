#!/bin/bash
# Phase 3b: the Brazil-trained ceiling — cifar8w on brazil_all (7 classes), five folds by chip id, 80 epochs, arms all and rgb
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
P3=runs/2026-09-29-rs-phase3; GPU=${GPU:-2}
for arm in ${ARMS:-all rgb}; do for k in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$GPU .lake/build/bin/rs-bands arm=$arm train=brazil_all val=brazil_all score=brazil_all classes=7 fold=$k epochs=80 seed=1 tag=ceil out=$P3 > $P3/ceil_${arm}_fold${k}.log 2>&1 || { echo "FAILED ceiling $arm fold $k"; continue; }
  .venv-rs/bin/python scripts/demos/rs_score.py $P3/rs_cifar8w_${arm}_ceil_fold${k}_logits_brazil_all.bin --part brazil_all --fold $k --json > $P3/score_ceil_${arm}_fold${k}.json 2>&1 || echo "FAILED score ceiling $arm fold $k"
  echo "ceiling $arm fold $k: $(grep '^brazil_all' $P3/score_ceil_${arm}_fold${k}.json | sed 's/ (7-way.*acc / /; s/ \[.*macro-F1/ F1/')"
done; done
echo CEILING_DONE
