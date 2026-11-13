#!/bin/bash
# deploy/vast/onstart.sh — the vast instance's on-start command: start bench.sh once, in the
# background (on-start has to return), its output to the instance log file.
#
#     vast template, on-start:  bash /opt/lean4-mlir/deploy/vast/onstart.sh
mkdir -p /var/log
nohup bash "${LEAN4MLIR:-/opt/lean4-mlir}/deploy/vast/bench.sh" >> /var/log/lean4-mlir-bench.log 2>&1 &
