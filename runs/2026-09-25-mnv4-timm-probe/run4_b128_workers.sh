#!/usr/bin/env bash
# 2026-09-25: shim-worker sweep (6, 8) for the 4 × 128 bf16 fed arm. Scratch renders at B=128 (the current
# optimizer, no accum/EMA; accumulation adds one buffer per param, measured separately) swapped
# into verified_mlir/ for the duration, restored from HEAD afterwards.
set -u
cd "$(dirname "$0")/../.." || exit 1
R=runs/2026-09-25-mnv4-timm-probe
cp /tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax-mlir/6bb23e34-b896-486a-b7c1-aa96ccbfb443/scratchpad/mnv4b128/*.mlir verified_mlir/
ROW='mnv4w6|mobilenetv4-imagenet-verified|adamdp128|adamdp128bf16|128|SHIM_WORKERS=6
mnv4w8|mobilenetv4-imagenet-verified|adamdp128|adamdp128bf16|128|SHIM_WORKERS=8'
ROWS="$ROW" NETS='mnv4w6 mnv4w8' PRECS=bf16 ARMS=fed CKPT_TAG=probe-b128 scripts/bf16_probe_3060.sh $R/probe_b128_workers.tsv
rm -f verified_mlir/mnv4in_adamdp128bf16_train_step.mlir
git checkout HEAD -- verified_mlir/
git status --short verified_mlir/
echo RUN4 DONE
