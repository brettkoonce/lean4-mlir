#!/usr/bin/env bash
# BM3D, the classical arm's slow filter: ≈ 10 s per 512² slice on one core, so 48 × 8 tuning
# calls + 4,282 val slices on 30 workers ≈ 30–40 min, then the scorer (≈ 10 min, one core).
set -euo pipefail
cd "$(dirname "$0")/../.."
R=runs/2026-10-11-lidc-classical/bm3d
mkdir -p "$R"
nice -n 10 .venv-lidc/bin/python scripts/probes/lidc_classical.py data/lidc/train.bin data/lidc/val.bin bm3d \
  data/lidc/out/bm3d/val.eps.f32 --workers 30 > "$R/tune.txt" 2>&1
.venv-lidc/bin/python scripts/probes/lidc_denoise_score.py data/lidc/val.bin data/lidc/out/bm3d/val.eps.f32 \
  "$R/figure.png" > "$R/score.txt" 2>&1
