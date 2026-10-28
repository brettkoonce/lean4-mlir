#!/bin/bash
# card 0: the TinyGPT ladder rows, then the Boltzmann flow seeds (README.md here).
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
runs/2026-10-27-tinygpt-ladder-xla/run.sh 0
runs/2026-10-27-boltzmann-seeds/run.sh 0
