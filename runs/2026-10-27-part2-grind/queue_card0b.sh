#!/bin/bash
# card 0, after queue_card0.sh (the ladder, then the Boltzmann seeds): the TinyGPT dropout arm,
# T6 — launched once the emitter had residual dropout (10-27 18:10 UTC).
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
while pgrep -f 'bash .*queue_card0\.sh' > /dev/null; do sleep 60; done
runs/2026-10-27-tinygpt-dropout-xla/run.sh 0
