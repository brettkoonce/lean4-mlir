#!/bin/bash
# 15-min perf probes of the ViT-S JAX trainer with fewer host bytes. Scratch copies; no checkpoints.
P=/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax-mlir/09ff3375-5b36-40c8-8aeb-bad437be7c1d/scratchpad/probe
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir/jax || exit 1
for v in vits_u8 vits_u8_pilrot; do
  echo "start $v $(date +%s)" >> $P/status
  CUDA_VISIBLE_DEVICES=0,1,2,3 JAX_COMPILATION_CACHE_DIR=/home/skoonce/.jax_cache \
    TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets \
    timeout -s INT 900 ../.venv/bin/python -u $P/$v.py 2>&1 \
    | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > $P/$v.log
  echo "end $v $(date +%s)" >> $P/status
  sleep 20
done
echo alldone >> $P/status
