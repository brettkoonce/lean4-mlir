#!/usr/bin/env bash
# 40-step smoke of mnv2-default-4gpu's variant (rmsdp64wxdols0eps0001bf16), then a resume into
# epoch 2. Checkpoint tag `smoke` keeps it off the real run's path.
set -u
cd "$(dirname "$0")/../.."
. scripts/jobs/mnv2-default-4gpu.conf
for leg in 1 2; do
  env "${ENV_EXTRA[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3 LEAN_MLIR_CKPT_TAG=smoke \
    LEAN_MLIR_G2_STEPS=40 LEAN_MLIR_MAX_EPOCHS=$leg "${CMD[@]}" \
    > runs/2026-09-25-mnv2-tf-recipe-smoke/leg$leg.log 2>&1
  echo "leg $leg exit $?"
done
