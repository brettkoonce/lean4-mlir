#!/usr/bin/env bash
# VJP oracle runner. For each test case, runs the IREE side (Lean → MLIR → IREE, the hand-derived
# VJPs) with init dump + NO_SHUFFLE, runs the JAX side (autodiff) with matching init load +
# NO_SHUFFLE, and
# diffs step-2 loss. Step-2 is the first step whose loss depends on
# the backward pass, so a small Δ there means the Lean hand-derived
# VJP agrees with JAX's value_and_grad.
#
# Usage: tests/vjp_oracle/run.sh [case1 case2 ...]
# Runs the JAX side on whatever JAX_PLATFORMS resolves to, and the IREE side
# on $IREE_BACKEND (LeanMlir/Train.lean's default: cuda).
# Data: MNIST IDX files in $VJP_ORACLE_DATA (default data/). The diff reads steps 1-2 only, so
# CI points it at jax/tests/vjp_oracle/make_tiny_mnist.py's synthetic set.
set -u
cd "$(dirname "$0")/../.."  # → repo root
DATA=$(realpath "${VJP_ORACLE_DATA:-data}")

# On NVIDIA hosts, pin to a single GPU so the JAX side's auto-sharding
# doesn't inflate the effective batch size and diverge from the IREE side.
# Without this, the JAX Mesh spans all visible GPUs and feeds
# N samples per step instead of the NetSpec's batchSize.
CUDA_HOST=0
if command -v nvidia-smi >/dev/null 2>&1 || [ -e /dev/nvidia0 ]; then
  CUDA_HOST=1
  [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES=0
fi

if [ "$#" -gt 0 ]; then
  CASES=("$@")
else
  CASES=(dense dense-relu conv convbn conv-pool residual depthwise attention mbconv \
         global-avg-pool bottleneck mbconv-v3 fused-mbconv uib)
fi
FAIL=0

for name in "${CASES[@]}"; do
  init_bin=/tmp/vo_${name}.bin
  iree_trace=/tmp/vo_iree_${name}.jsonl
  jax_trace=/tmp/vo_jax_${name}.jsonl
  iree_log=/tmp/vo_iree_${name}.log
  jax_log=/tmp/vo_jax_${name}.log

  # IREE (the executables built from tests/vjp_oracle/phase3/)
  LEAN_MLIR_INIT_DUMP="$init_bin" \
  LEAN_MLIR_NO_SHUFFLE=1 \
  LEAN_MLIR_TRACE_OUT="$iree_trace" \
    ./.lake/build/bin/vjp-oracle-${name} "$DATA" > "$iree_log" 2>&1 \
    || { echo "FAIL  ${name}  IREE side crashed (see $iree_log)"; FAIL=1; continue; }

  # JAX — invoke the binary once to emit the generated Python
  # script (we don't care whether its own Python invocation succeeds;
  # we'll run Python ourselves against the right venv). Then call
  # Python directly from repo root with env vars so `.venv/` resolves.
  # Pass an absolute path so the generated Python still finds data
  # when invoked from repo root.
  ( cd jax && ./.lake/build/bin/vjp-oracle-${name} "$DATA" > /dev/null 2>&1 ) || true
  # Generated Python filename uses underscores; case names use hyphens.
  script=jax/.lake/build/generated_vjp_oracle_${name//-/_}.py
  [ -f "$script" ] || { echo "FAIL  ${name}  JAX side did not emit $script"; FAIL=1; continue; }
  LEAN_MLIR_INIT_LOAD="$(realpath "$init_bin")" \
  LEAN_MLIR_NO_SHUFFLE=1 \
  LEAN_MLIR_TRACE_OUT="$(realpath -m "$jax_trace")" \
    .venv/bin/python3 "$script" > "$jax_log" 2>&1 \
    || { echo "FAIL  ${name}  JAX side crashed (see $jax_log)"; FAIL=1; continue; }

  # Diff. Per-case tolerance: maxPool's argmax tiebreaks disagree
  # between XLA and IREE for tied input elements (MNIST has many
  # zero pixels → many ties), pushing the step-2 Δ above the 1e-4
  # default by ~5×. Not a correctness bug; just a looser noise floor.
  #
  # CUDA tightens the tolerance budget: cuBLAS GEMM and cuDNN conv
  # use different reduction orders than ROCm's rocBLAS/MIOpen, so
  # dense/dense-relu/conv land at 1.1e-5 / 1.1e-4 / 1.4e-4 instead
  # of the ≲1e-5 they hit on ROCm. Loosen to 2e-4 on NVIDIA hosts.
  case "$name" in
    conv-pool) tol=1e-3 ;;
    convbn)    tol=1e-4 ;;  # BN variance reductions ≲ 1e-4
    # Stacked-BN blocks: each convBn pass contributes ≲ 1e-6 of rounding
    # noise on ROCm; a 3-5 convBn composition lands at 1e-5 to 1e-4 step-2 Δ.
    # CUDA cuDNN kernels push some of these ~2-5× higher — same pattern
    # as dense/conv/depthwise. Loosen per case on NVIDIA.
    bottleneck)   tol=1e-4 ;;  # 3 convBn per block (reduce/3x3/expand)
    uib)          tol=1e-4 ;;  # preDW + expand + postDW + project (4 convBn)
    mbconv)
      # Expand + depthwise + project + SE. CUDA lands step-2 Δ ~2.2e-4.
      if [ "$CUDA_HOST" = "1" ]; then tol=3e-4; else tol=1e-4; fi
      ;;
    mbconv-v3)
      # 4 convBn + SE + h-swish/h-sigmoid. Piecewise-linear activations on
      # top of 4 convBn push CUDA step-2 Δ to ~5e-4 (mars: ~6e-6).
      if [ "$CUDA_HOST" = "1" ]; then tol=7e-4; else tol=1e-4; fi
      ;;
    fused-mbconv)
      # Stem + fused k×k convBn + project. CUDA step-2 Δ ~1.9e-4.
      if [ "$CUDA_HOST" = "1" ]; then tol=3e-4; else tol=1e-4; fi
      ;;
    depthwise)
      # 4 stacked BN passes (stem+expand+dw+proj). On CUDA, cuDNN's
      # depthwise conv kernel lands step-2 Δ around 5e-4 — same 5×
      # pattern as dense/conv. Loosen to 7e-4 on NVIDIA.
      if [ "$CUDA_HOST" = "1" ]; then
        tol=7e-4
      else
        tol=1e-4
      fi
      ;;
    *)
      if [ "$CUDA_HOST" = "1" ]; then
        tol=2e-4
      else
        tol=1e-5
      fi
      ;;
  esac
  .venv/bin/python3 tests/vjp_oracle/diff_step.py "$iree_trace" "$jax_trace" "$name" "$tol" || FAIL=1
done

exit $FAIL
