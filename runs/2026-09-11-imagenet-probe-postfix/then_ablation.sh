#!/bin/bash
# Waits for the probe chain's ALL_DONE, then starts the ResNet-34 recipe ablation, bf16, seeds 1-5
# (planning/r34_ablation_seeds.md; bf16 first by the user's call, five fresh seeds because the
# 2026-09-01 bf16 logs did not survive). 40 runs over four cards.
set -u
cd "$(dirname "$0")/../.."
until grep -q "^ALL_DONE" runs/2026-09-11-imagenet-probe-postfix/fed.log; do sleep 60; done
sleep 20   # let the last trainer's CUDA context go away before wait_for_gpu looks
OUT=runs/2026-09-11-r34-ablation-bf16-seeds
echo "chain: probe done at $(date -u +%FT%TZ); starting ablation -> $OUT"
PREC=bf16 SEEDS="1 2 3 4 5" OUT="$OUT" \
  PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
  bash scripts/run_r34_ablation.sh
echo "ABLATION_DONE $(date -u +%FT%TZ)"
