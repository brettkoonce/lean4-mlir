#!/bin/bash
# Test the RoPE / long-context config on a fresh (RunPod) GPU pod.
#
# Runs the char-level RoPE-no-pos tinyGPT end to end and demonstrates the
# whole long-context story in a few minutes:
#   * trains nano-rope-nopos (RoPE, no absolute position table) on tinyshakespeare
#   * EXTRAPOLATES: the SAME T=64 checkpoint evaluated at T=128 / T=256 — it runs
#     and stays coherent, because every weight is seqLen-independent.
#   * contrast: the absolute-position `nano` model CAN'T even load at a new length.
#
# The heavy 8K TinyStories config (flash + rope + no-pos, ~29M params) is
# ./run_tinystories_8k.sh once data/tinystories/{train,val}.bin exist — same
# codegen, just bigger; this script proves the mechanism cheaply.
#
# First run bootstraps the full Lean→MLIR→IREE→CUDA pipeline (elan + iree +
# runtime build + mathlib oleans, ~10-20 min); later runs skip straight to it.
#
# Usage:  rope-test [steps=800] [batch=32]
set -euo pipefail
cd "$(dirname "$0")/.."           # repo root
STEPS="${1:-800}"
BATCH="${2:-32}"

# ── 1. Lean → IREE → CUDA pipeline (idempotent; builds on first run) ──
if [ ! -x .venv/bin/iree-compile ] || [ ! -f ffi/libiree_ffi.so ]; then
  echo "━━━ bootstrapping Lean/IREE pipeline (first run only) ━━━"
  bash jax/probe/bootstrap_lean_iree.sh
fi
export PATH="/root/.elan/bin:$PWD/.venv/bin:$PATH"
export IREE_BACKEND="${IREE_BACKEND:-cuda}"
export IREE_CHIP="${IREE_CHIP:-sm_80}"        # A100=sm_80, H100=sm_90

# ── 2. tinyshakespeare (tiny, ~1 MB; skip if baked/cached) ──
if [ ! -f data/shakespeare/train.bin ]; then
  echo "━━━ preparing tinyshakespeare ━━━"
  bash download_shakespeare.sh
  .venv/bin/python preprocess_shakespeare.py
fi

# ── 3. build the tinyGPT exe ──
echo "━━━ building tinygpt-shakespeare (first build compiles the proof cone; slow once) ━━━"
lake build tinygpt-shakespeare

# ── 4. train the RoPE-no-pos model ──
echo "━━━ training nano-rope-nopos ($STEPS steps) — RoPE + no absolute position ━━━"
.lake/build/bin/tinygpt-shakespeare train nano-rope-nopos "$STEPS" "$BATCH" 30

# ── 5. length extrapolation: SAME T=64 weights at longer context ──
echo ""
echo "━━━ EXTRAPOLATION — one T=64 checkpoint, evaluated at 3 context lengths ━━━"
for T in 64 128 256; do
  .lake/build/bin/tinygpt-shakespeare xeval nano-rope-nopos "$T"
done

# ── 6. contrast: the absolute-position model is length-locked ──
echo ""
echo "━━━ CONTRAST — the absolute-position 'nano' model can't run at a new length ━━━"
.lake/build/bin/tinygpt-shakespeare train nano 200 32 30 >/dev/null 2>&1 || true
.lake/build/bin/tinygpt-shakespeare xeval nano 128 || true
echo ""
echo "done — RoPE-no-pos trains AND extrapolates; absolute position is length-locked."
