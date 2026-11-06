#!/bin/bash
# The 2026-10-28 CPU protocol on a GPU: each named binary under LEAN_MLIR_LOWERER=iree IREE_BACKEND=cuda,
# ten steps of one epoch (LEAN_MLIR_G2_STEPS=10 LEAN_MLIR_MAX_EPOCHS=1; the MNIST trainers take the
# epoch cap only), the full eval, a fresh checkpoint tag (smoke312: the 3.13.0 pass used `smoke`, and a rerun under it RESUMES the finished epoch and trains nothing).
# Logs beside this file as run.sh writes them (RUN_LOG_DIR), wall and max RSS in <net>.time.
#   runs/2026-11-05-iree-cuda-smoke/smoke.sh <gpu> <net>...
cd "$(dirname "$0")/../.."
OUT=runs/2026-11-05-iree-cuda-smoke
gpu=$1; shift
export PATH="$PWD/.venv-iree/bin:$PATH"
export LEAN_MLIR_LOWERER=iree IREE_BACKEND=cuda LEAN_MLIR_G2_STEPS=10 LEAN_MLIR_MAX_EPOCHS=1 LEAN_MLIR_CKPT_TAG=smoke312
export RUN_LOG_DIR=$OUT
for n in "$@"; do
  echo "━━━ $n (gpu $gpu) $(date '+%T')"
  # The bbn net's bf16 arms need IREE's GPU vector-distribution pipeline off (lakefile.lean, `cifar-iree`).
  case $n in cifar8wb-bbn-ablation) export IREE_EXTRA_FLAGS=--iree-codegen-llvmgpu-use-vector-distribution=false ;; *) unset IREE_EXTRA_FLAGS ;; esac
  /usr/bin/time -f "$n wall %e s, maxrss %M KB" -o $OUT/$n.time nice -n 19 ./run.sh $n $gpu cuda > /dev/null 2> $OUT/$n.stderr \
    && echo "  ✓ $(cat $OUT/$n.time)" || echo "  ✗ $n exit $? — $OUT/$n.stderr"
done
