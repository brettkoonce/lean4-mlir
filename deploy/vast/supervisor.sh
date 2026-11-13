#!/bin/bash
# The supervisor program's command: bench.sh, to the instance log and the portal's log directory.
mkdir -p /var/log/portal
exec > >(tee -a /var/log/portal/lean4-mlir-bench.log) 2>&1
exec bash "${LEAN4MLIR:-/opt/lean4-mlir}/deploy/vast/bench.sh"
