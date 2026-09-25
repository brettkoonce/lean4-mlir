#!/usr/bin/env bash
# 2026-09-25: end-to-end smoke of mnv4-half-4gpu's variant through the real trainer. Two short
# epochs (16 micro-batches = 2 optimizer steps each, eval after each), then a resume to epoch 3.
# Isolated checkpoint (LEAN_MLIR_CKPT_TAG=smoke-half) so the real run's is never touched.
set -u
cd "$(dirname "$0")/../.." || exit 1
. scripts/jobs/_box.sh
run() {
  env CUDA_VISIBLE_DEVICES=0,1,2,3 "${BOX_SHIMPY[@]}" PJRT_PLUGIN="$BOX_PLUG" \
    PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 LEAN_MLIR_VARIANT=emaaccdp8x128wxdowd005bf16 \
    LEAN_MLIR_BATCH=128 LEAN_MLIR_EPOCHS=50 LEAN_MLIR_BASE_LR_U=4000 LEAN_MLIR_G2_STEPS=16 \
    LEAN_MLIR_VAL_EVERY=1 PJRT_FFI_RESIDENT=1 SHIM_WORKERS=4 LEAN_MLIR_CKPT_TAG=smoke-half \
    LEAN_MLIR_MAX_EPOCHS="$1" timeout 3000 .lake/build/bin/mobilenetv4-imagenet-verified data
}
rm -f .lake/build/mnv4in_emaaccdp8x128wxdowd005bf16*smoke-half*
echo "=== leg 1: epochs 1-2 ==="; run 2
echo "=== leg 2: resume to epoch 3 ==="; run 3
ls -la .lake/build/ | grep smoke-half
echo SMOKE DONE
