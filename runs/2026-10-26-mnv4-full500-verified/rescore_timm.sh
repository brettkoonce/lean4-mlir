#!/bin/bash
# The 500-epoch MNv4 pair at timm's test protocol (320 / crop 1.0 — the size timm's e500_r256 weights report 80.86 at),
# 2026-11-11, ares: the verified e500 checkpoint through scripts/parity/score_timm.sh (eval graph mnv4in256_fwd_eval_s320),
# the JAX e500 state through eval_full50k.py PROTOCOL=timm at EVAL_DTYPE=f32, both with a bitmap, then McNemar.
set -u
cd "$(dirname "$0")/../.." || exit 1
R=runs/2026-10-26-mnv4-full500-verified; J=runs/2026-10-26-mnv4-full500-jax
CKPT=.lake/build/mnv4in256_accdp8x128wxdropdowd01bf16_ckpt_xla.bin
[ "$(tr -cd 0-9 < $CKPT.epoch)" = 500 ] || { echo "⛔ $CKPT is not the e500 checkpoint"; exit 1; }
LEAN_MLIR_VARIANT=accdp8x128wxdropdowd01bf16 LEAN_MLIR_CKPT=$CKPT LEAN_MLIR_DUMP_CORRECT=$R/bitmaps/timm320_verified \
  CUDA_VISIBLE_DEVICES=0 scripts/parity/score_timm.sh mnv4-in256 data 2>&1 | tee $R/rescore_timm320_verified.log
(cd jax && JAX_COMPILATION_CACHE_DIR=/home/skoonce/.jax_cache TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets \
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GEN=generated/generated_mobilenet_v4_imagenet_full.py \
  CKPT=/home/skoonce/mnv4_full_r256_2612/mobilenet_v4_imagenet_full_e500.state.npz BATCH=50 PROTOCOL=timm EVAL_DTYPE=f32 \
  DUMP_CORRECT=../$R/bitmaps/timm320_jax_f32 LABEL="mnv4 full500 jax e500 f32 timm" \
  ../.venv/bin/python -u scripts/eval_full50k.py) 2>&1 | tee $R/rescore_timm320_jax_f32.log
.venv/bin/python scripts/demos/mcnemar.py $R/bitmaps/timm320_jax_f32.bin $R/bitmaps/timm320_verified*.bin --label-a jax-f32 --label-b verified 2>&1 | tee $R/mcnemar_timm320.txt
