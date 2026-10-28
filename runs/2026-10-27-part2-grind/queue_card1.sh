#!/bin/bash
# card 1: VisDrone aff30 seed 1, then the three BraTS arms at seed 1 (README.md here).
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
runs/2026-10-27-visdrone-aff30-seeds/run.sh 1 1
runs/2026-10-27-brats-r34-seeds/run.sh 1 r34:1 scratch:1 noskip:1
