#!/usr/bin/env bash
# Validation gate for the mnist-linear verified trainer's committed render.
# Companion to LeanMlir/Proofs/Foundation/LinearFaithfulPoC.lean and
# planning/verified_faithful_sweep.md. Runs on the GPU box (needs iree-compile;
# CI's ubuntu runner can't, so this is local, not in proofs.yml).
#
#   (a) drift: the committed verified_mlir/linear_*.mlir == the proven renderer
#       (linearTrainStepModuleV / linearFwdModuleV in StableHLO.lean), and
#   (b) validity: those bytes iree-compile cleanly for the target backend.
#
# (a)+(b) + the LinearFaithfulPoC `den = certified` capstones = the chain
# "trainer bytes == proven renderer == certified loss-descent step, and iree
# accepts them". Usage: IREE_COMPILE=/path/to/iree-compile ./scripts/validate_linear_faithful.sh
set -euo pipefail
cd "$(dirname "$0")/.."

IREEC="${IREE_COMPILE:-iree-compile}"
BACKEND="${IREE_BACKEND:-rocm}"
CHIP="${IREE_CHIP:-gfx1100}"

echo "== (a) drift: committed verified_mlir/linear_* == proven renderer =="
lake env lean LeanMlir/Proofs/StableHLO.lean >/dev/null
git diff --exit-code -- verified_mlir/linear_train_step.mlir verified_mlir/linear_fwd.mlir \
  && echo "   OK: committed == renderer" \
  || { echo "   DRIFT: regenerate with 'lake env lean LeanMlir/Proofs/StableHLO.lean'"; exit 1; }

echo "== (b) validity: iree-compile the committed bytes ($BACKEND/$CHIP) =="
for f in linear_fwd linear_train_step; do
  "$IREEC" "verified_mlir/${f}.mlir" \
    --iree-hal-target-backends="$BACKEND" --iree-"$BACKEND"-target="$CHIP" \
    --iree-codegen-llvmgpu-use-reduction-vector-distribution=false \
    -o "/tmp/${f}_validate.vmfb"
  echo "   OK: ${f}.mlir -> $(stat -c%s "/tmp/${f}_validate.vmfb") byte vmfb"
done
echo "PASS: mnist-linear verified render is drift-free and iree-valid."
