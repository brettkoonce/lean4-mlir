#!/bin/bash
# Launch the 8K-context TinyStories LM on the cloud A100 (CUDA).
#
# The long-context stack — all validated + committed this session:
#   * FlashAttention  (flashAttn) — attention memory O(T·blk) not O(T²).
#       At T=8192 the dense [B,H,T,T] scores would be ~4 GB/leg; flash keeps
#       it to tens of MB, so 8K fits an A100 at a healthy batch.
#   * RoPE            (rope)      — relative position, no length-fixed table.
#   * no absolute pos (posEmb off) — every weight is seqLen-independent, so
#       the same checkpoint also runs at other lengths (train-short/eval-long).
#
# Model: V=4096 BPE, T=8192, D=512, 8 heads, mlp=2048, 8 blocks, ~29M params.
#
# Prereqs on the box (see jax/probe/bootstrap.sh for the general setup):
#   * Lean toolchain + `lake build tinystories` (builds the FFI .so + exe).
#   * iree-compile on PATH (or .venv/bin) — the CUDA target.
#   * data/tinystories/{train,val}.bin  ← preprocess_tinystories.py
#     (byte-level BPE, 4096 vocab; ~50M train tokens — see planning/tinygpt_demo_v2.md).
#
# Memory (fp32): ~params 0.35 GB + ~2.5 GB per batch item at T=8192.
#   A100-40GB : batch 8-12         A100-80GB : batch 16-24
#   Use grad-accum (more steps, smaller batch) for a larger effective batch.
#
# Usage: ./run_tinystories_8k.sh [steps=12000] [batch=8] [lr_x10000=30]
set -u
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"
export IREE_BACKEND=cuda
export IREE_CHIP="${IREE_CHIP:-sm_80}"     # A100 = sm_80; H100 = sm_90
STEPS="${1:-12000}"
BATCH="${2:-8}"
LR="${3:-30}"

echo "=== TinyStories-8K launch $(date) ==="
echo "  backend=$IREE_BACKEND chip=$IREE_CHIP  steps=$STEPS batch=$BATCH lr=$(awk "BEGIN{print $LR/10000}")"
echo "  first step is slow: iree-compile of the 8K flash train step (fwd+bwd while-loops)."
mkdir -p runs
exec .lake/build/bin/tinystories train 8k "$STEPS" "$BATCH" "$LR" 2>&1 | tee "runs/tinystories_8k.log"
