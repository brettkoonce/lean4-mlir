#!/usr/bin/env bash
# EfficientNet-B0 bf16 SHIM_WORKERS sweep, 4x RTX 3060 box, 2026-09-12.
# Exe rebuilt 22:37 UTC (post 13d90e68 / 46dfa714); jax/.lake/build in sync (regen box: 78 ok).
# 800-step window (WARM=200 STEPS=1000) -- twice the 2026-09-11 table's, so each arm spans ~4
# of the ~190-step-period stall episodes that README describes. Per-step dump per arm.
set -u
cd "$(dirname "$0")/../.." || exit 1
D=runs/2026-09-12-enet-bf16-sweep
export WARM=200 STEPS=1000 PRECS=bf16 NETS=enetema
for w in 4 6 8 10 12; do
  ARMS=fed WORKERS=$w LEAN_MLIR_PROBE_DUMP="$D/steps_fed_w$w.tsv" \
    scripts/bf16_probe_3060.sh "$D/sweep.tsv" >> "$D/sweep.log" 2>&1
done
ARMS=synth WORKERS=8 LEAN_MLIR_PROBE_DUMP="$D/steps_synth.tsv" \
  scripts/bf16_probe_3060.sh "$D/sweep.tsv" >> "$D/sweep.log" 2>&1
echo "SWEEP-DONE $(date -u '+%F %T')" >> "$D/sweep.log"
