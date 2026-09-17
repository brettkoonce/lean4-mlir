#!/bin/bash
# one GPU, a list of "net split seed" runs in sequence; each into runs/2026-09-17-arasl-<net>-<split>/
gpu=$1; shift
for spec in "$@"; do
  set -- $spec; net=$1; split=$2; seed=$3
  d=runs/2026-09-17-arasl-$net-$split; mkdir -p $d
  CUDA_VISIBLE_DEVICES=$gpu .lake/build/bin/arasl-signs net=$net split=$split epochs=30 seed=$seed tag=s$seed out=$d \
    > $d/train_s$seed.log 2>&1
  echo "done $net $split s$seed: $(tail -1 $d/train_s$seed.log)"
done
