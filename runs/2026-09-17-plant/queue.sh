#!/bin/bash
# one GPU, a list of runs in sequence; each into runs/2026-09-17-plant-<split>-<train>-<init>/
gpu=$1; shift
for spec in "$@"; do
  set -- $spec; split=$1; train=$2; init=$3; seed=$4; extra=$5
  d=runs/2026-09-17-plant-$split-$train-$init; mkdir -p $d
  CUDA_VISIBLE_DEVICES=$gpu .lake/build/bin/plant-leaf split=$split train=$train init=$init seed=$seed tag=s$seed epochs=10 $extra out=$d \
    > $d/train_s$seed.log 2>&1
  echo "done $split $train $init s$seed: $(grep -E 'pd_all|pd_fold' $d/train_s$seed.log | tail -1 | sed 's/  -> .*//')"
done
