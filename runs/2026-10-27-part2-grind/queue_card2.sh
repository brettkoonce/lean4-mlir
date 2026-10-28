#!/bin/bash
# card 2: VisDrone aff30 seed 2, then the three BraTS arms at seed 2 (README.md here).
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
runs/2026-10-27-visdrone-aff30-seeds/run.sh 2 2
runs/2026-10-27-brats-r34-seeds/run.sh 2 r34:2 scratch:2 noskip:2
