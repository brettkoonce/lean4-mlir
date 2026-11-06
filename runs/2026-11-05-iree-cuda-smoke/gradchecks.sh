#!/bin/bash
# The two gradient gates through IREE on cuda, 3.12.0: cifar8-bbn-gradcheck (the batch-statistics BN
# step's backward: tier-1 homogeneity identities, tier-2 adjoint probe) on its f32 adam artifact and,
# with the vector-distribution flag the bf16 arms compile under, on bf16adam; and r50-gradcheck (the
# ResNet-50 ImageNet batch-64 step, 49 invokes — 3050 s on a CPU on 10-28). The binaries run directly:
# `lake build cifar8-bbn-gradcheck r50-gradcheck` first.
#   runs/2026-11-05-iree-cuda-smoke/gradchecks.sh <gpu>
cd "$(dirname "$0")/../.."
OUT=runs/2026-11-05-iree-cuda-smoke/gradchecks; mkdir -p $OUT
gpu=$1
export PATH="$PWD/.venv-iree/bin:$PATH" CUDA_VISIBLE_DEVICES=$gpu LEAN_MLIR_LOWERER=iree IREE_BACKEND=cuda
run () { tag=$1; shift; echo "━━━ $tag (gpu $gpu) $(date '+%T')"; /usr/bin/time -f "$tag wall %e s, maxrss %M KB" -o $OUT/$tag.time nice -n 19 "$@" > $OUT/$tag.log 2>&1 && echo "  ✓ $(grep -cE 'PASS' $OUT/$tag.log) PASS lines, $(grep -cE 'FAIL' $OUT/$tag.log) FAIL lines — $(cat $OUT/$tag.time)" || echo "  ✗ $tag exit $? — $OUT/$tag.log"; }
run bbn-adam .lake/build/bin/cifar8-bbn-gradcheck
IREE_EXTRA_FLAGS=--iree-codegen-llvmgpu-use-vector-distribution=false C8BBN_GC_VARIANT=bf16adam run bbn-bf16adam .lake/build/bin/cifar8-bbn-gradcheck
run r50 .lake/build/bin/r50-gradcheck
