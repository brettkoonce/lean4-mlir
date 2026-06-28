#!/bin/bash
set -e
cd "$(dirname "$0")"
export PATH=$PWD/.venv/bin:$PATH
export IREE_BACKEND=rocm
echo "=== mnist-mlp-smooth $(date) ===" 
.lake/build/bin/mnist-mlp-smooth data > runs/smooth_mlp.log 2>&1
echo "=== mnist-cnn-smooth $(date) ==="
.lake/build/bin/mnist-cnn-smooth data > runs/smooth_cnn.log 2>&1
echo "=== cifar-smooth $(date) ==="
.lake/build/bin/cifar-smooth data > runs/smooth_cifar.log 2>&1
echo "=== ALL DONE $(date) ==="
