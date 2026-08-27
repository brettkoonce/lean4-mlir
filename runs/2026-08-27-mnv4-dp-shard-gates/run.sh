#!/usr/bin/env bash
# The four gate runs and the two controls behind runs/2026-08-27-mnv4-dp-shard-gates/.
# ⚠ FOUR GPUs. MNv4 renders `adamdp64` at 4 replicas only — there is no 2-replica peer, so
# PJRT_REPLICAS=2 hits the shim's replica-count guard rather than degrading to a 2-way run.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset HIP_VISIBLE_DEVICES
export PJRT_PLUGIN=".venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so"
export CUDA_VISIBLE_DEVICES=0,2,3,4       # the AER-clean four (reference_ares_pcie_aer)

lake build mnv4-dp-check shard-check

PJRT_REPLICAS=4 .lake/build/bin/mnv4-dp-check
PJRT_REPLICAS=4 DP_VARIANT=adam64bf16 DP_VARIANT_DP=adamdp64bf16 .lake/build/bin/mnv4-dp-check
PJRT_REPLICAS=4 SHARD_REPLICAS=4 SHARD_VARIANT=adam64 SHARD_VARIANT_DP=adamdp64 \
  .lake/build/bin/shard-check mnv4in

# ── THE CONTROLS. A gate nobody has seen go red is an assertion. ──────────────
# Sum instead of mean: every all_reduce divisor 4.0 -> 1.0, so every gradient is 4x.
BROKEN=$(mktemp /tmp/mnv4in_adamdp64_SUMNOTMEAN.XXXX.mlir)
sed 's/stablehlo.constant dense<4\.0> : tensor/stablehlo.constant dense<1.0> : tensor/g' \
  verified_mlir/mnv4in_adamdp64_train_step.mlir > "$BROKEN"
PJRT_REPLICAS=4 .lake/build/bin/mnv4-dp-check "$BROKEN" && echo "⛔ CONTROL DID NOT FIRE" && exit 1
PJRT_REPLICAS=4 SHARD_REPLICAS=4 SHARD_VARIANT=adam64 SHARD_VARIANT_DP=adamdp64 \
  .lake/build/bin/shard-check mnv4in "$BROKEN" && echo "⛔ CONTROL DID NOT FIRE" && exit 1
rm -f "$BROKEN"
echo "✓ both gates green, both controls red"
