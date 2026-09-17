#!/bin/bash
# the field fine-tunes: from a saved checkpoint, 5 folds × {250 labels, all labels}, fixed 5 epochs at lr 1e-4
gpu=$1; init=$2; name=$3; shift 3
d=runs/2026-09-17-plant-$name; mkdir -p $d
for n in 250 0; do for k in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$gpu .lake/build/bin/plant-leaf split=grouped train=base init=$init field=$k fieldn=$n epochs=5 lr=0.0001 seed=1 tag=s1 out=$d \
    > $d/train_field${k}_n${n}_s1.log 2>&1
  echo "done field$k n$n: $(grep -E 'pd_fold' $d/train_field${k}_n${n}_s1.log | tail -1 | sed 's/  -> .*//')"
done; done
