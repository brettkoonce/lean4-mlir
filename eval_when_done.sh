#!/usr/bin/env bash
# Waits for the paced ViT run to finish (final EMA .bin on disk + trainer gone),
# then runs the canonical full-50k ImageNet val on the final EMA weights.
# Single-GPU (HIP_VISIBLE_DEVICES=0) so no RCCL collectives needed. Detached.
set -u
CKPT=/home/skoonce/vit_tiny_imagenet_bf16.bin
JAXDIR=/home/skoonce/lean/proof_verify_demo/verify-v2/jax
LOG=/home/skoonce/lean/proof_verify_demo/verify-v2/runs/vit_final_eval.log
VENV=/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python

echo "[eval-wait] $(date '+%F %T') waiting for final .bin ($CKPT) + trainer exit" > "$LOG"
# 1) wait for the final EMA checkpoint to land
while [ ! -f "$CKPT" ]; do sleep 30; done
# 2) wait for the trainer python to fully exit (frees the GPUs; pgrep excludes itself)
while pgrep -f "generated_vit_tiny_imagenet.py" >/dev/null; do sleep 30; done
sleep 20
echo "[eval-wait] $(date '+%F %T') GPUs free — running full-50k eval on $CKPT" >> "$LOG"
cd "$JAXDIR" || { echo "no jax dir"; exit 1; }
HIP_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets CKPT="$CKPT" \
  "$VENV" scripts/eval_vit_full50k.py >> "$LOG" 2>&1
echo "[eval-wait] $(date '+%F %T') eval complete" >> "$LOG"
