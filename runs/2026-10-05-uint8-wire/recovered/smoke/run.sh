#!/bin/bash
# 15-min smokes of the three batch-1 JAX trainers, sequential, no checkpoints.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir/jax || exit 1
OUT=/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax-mlir/09ff3375-5b36-40c8-8aeb-bad437be7c1d/scratchpad/smoke
for t in resnet50_imagenet_a2accum mobilenet_v4_imagenet_full vit_s_imagenet; do
  echo "start $t $(date +%s)" >> $OUT/status
  CUDA_VISIBLE_DEVICES=0,1,2,3 JAX_COMPILATION_CACHE_DIR=/home/skoonce/.jax_cache \
    TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets \
    timeout -s INT 900 ../.venv/bin/python -u .lake/build/generated_$t.py 2>&1 \
    | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > $OUT/$t.log
  echo "end $t $(date +%s)" >> $OUT/status
  sleep 20
done
echo alldone >> $OUT/status
