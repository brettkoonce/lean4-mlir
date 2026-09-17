#!/usr/bin/env bash
# R34 bf16 SHIM_WORKERS re-probe on this box (4x RTX 3060), 2026-09-16.
#
# Why: r34-default-4gpu.conf carries SHIM_WORKERS=8, set before BOTH feed fixes — 13d90e68
# (mimalloc) and 4a0a2781 (tf.data determinism OFF). EfficientNet's identical re-probe moved its
# count 8 -> 4 and halved its ETA error. R34's shim is FLIP-ONLY (no AutoAugment, no RandAugment),
# the lightest in the fleet, so its plateau may sit at a different place than B0's.
#
# The synth arm is the compute-only floor; fed-minus-synth IS the shim starvation, in ms.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
OUT=runs/2026-09-16-r34-bf16-sweep/sweep.tsv

for w in 4 6 8 10 12; do
  WORKERS=$w ARMS=fed PRECS=bf16 NETS=r34 WARM=200 STEPS=1000 \
    CKPT_TAG="r34bf16sweep-w$w" scripts/bf16_probe_3060.sh "$OUT"
done

# The floor. Worker count is irrelevant with the shim read removed; run it once.
WORKERS=4 ARMS=synth PRECS=bf16 NETS=r34 WARM=200 STEPS=1000 \
  CKPT_TAG="r34bf16sweep-synth" scripts/bf16_probe_3060.sh "$OUT"

echo "SWEEP DONE"
column -t -s $'\t' "$OUT"
