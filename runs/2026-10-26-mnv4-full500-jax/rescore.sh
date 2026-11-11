#!/bin/bash
# McNemar for the 500-epoch MNv4 pair (2026-11-11, ares). The verified half wrote its e500 bitmap in training
# (LEAN_MLIR_DUMP_CORRECT, runs/2026-10-26-mnv4-full500-verified/bitmaps/). The JAX e500 state is scored through
# eval_full50k.py at EVAL_DTYPE=f32 with a bitmap (the 100-epoch pair's recipe, runs/2026-10-24-mnv4-full100-verified/rescore.sh).
set -u
cd "$(dirname "$0")/../.." || exit 1
R=runs/2026-10-26-mnv4-full500-jax; V=runs/2026-10-26-mnv4-full500-verified
mkdir -p $R/bitmaps
(cd jax && JAX_COMPILATION_CACHE_DIR=/home/skoonce/.jax_cache TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets \
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GEN=generated/generated_mobilenet_v4_imagenet_full.py \
  CKPT=/home/skoonce/mnv4_full_r256_2612/mobilenet_v4_imagenet_full_e500.state.npz BATCH=50 PROTOCOL=train EVAL_DTYPE=f32 \
  DUMP_CORRECT=../$R/bitmaps/mnv4_full500_jax_f32 LABEL="mnv4 full500 jax e500 f32" \
  ../.venv/bin/python -u scripts/eval_full50k.py) 2>&1 | tee $R/rescore_jax_f32.log
.venv/bin/python scripts/demos/mcnemar.py $R/bitmaps/mnv4_full500_jax_f32.bin $V/bitmaps/*_e500.bin --label-a jax-f32 --label-b verified 2>&1 | tee $R/mcnemar.txt
