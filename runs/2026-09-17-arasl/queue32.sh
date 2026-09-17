#!/bin/bash
# the size=32 arm on one GPU: the chapter net at its own native input, same splits downsampled
gpu=$1; shift
for split in "$@"; do
  d=runs/2026-09-17-arasl-cifar8w-$split; mkdir -p $d
  CUDA_VISIBLE_DEVICES=$gpu .lake/build/bin/arasl-signs net=cifar8w split=$split epochs=30 seed=1 size=32 tag=s1 out=$d \
    > $d/train_size32_s1.log 2>&1
  echo "done cifar8w-32 $split s1: $(tail -1 $d/train_size32_s1.log)"
done
