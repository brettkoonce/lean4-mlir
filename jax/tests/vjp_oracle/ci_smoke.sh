#!/usr/bin/env bash
# CI-runnable slice of the VJP oracle.
#
# The full oracle (tests/vjp_oracle/run.sh, at the repo root) diffs phase 3
# (Lean -> MLIR -> IREE, the hand-derived VJPs) against phase 2 (Lean -> JAX,
# autodiff) and is the thing that actually validates the VJPs. It needs IREE
# and a GPU, so it cannot run on a stock GitHub runner.
#
# This runs the half that can. Two tiers, both over all 14 layer families:
#
#   EMIT  (always)      each phase-2 oracle exe emits its JAX script, and that
#                       script must parse. Catches emitter breakage — e.g. the
#                       mbConvV3 signature drift that sat red for three months.
#   RUN   (if jax)      execute each emitted script against a tiny synthetic
#                       MNIST. Catches emitted code that parses but is wrong:
#                       shape mismatches, bad axes, NaN losses.
#
# RUN does NOT check the hand-derived VJPs — phase 2 gets its gradients from
# JAX's own autodiff, so there is no Lean-derived gradient in the loop. It
# checks the emitter. Do not let a green run here be read as "the VJPs agree";
# that claim needs the GPU oracle.
#
# Usage: jax/tests/vjp_oracle/ci_smoke.sh [case ...]
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> jax/

CASES=("$@")
if [ "${#CASES[@]}" -eq 0 ]; then
  CASES=(dense dense-relu conv convbn conv-pool residual depthwise attention mbconv \
         global-avg-pool bottleneck mbconv-v3 fused-mbconv uib)
fi

PY=${PYTHON:-python3}
DATA=$(mktemp -d)
trap 'rm -rf "$DATA"' EXIT

if $PY -c 'import jax' 2>/dev/null; then
  RUN=1
  $PY tests/vjp_oracle/make_tiny_mnist.py "$DATA" 64 512
  echo "jax $($PY -c 'import jax; print(jax.__version__)') — EMIT + RUN"
else
  RUN=0
  echo "no jax importable — EMIT only (install jax to enable the RUN tier)"
fi

FAIL=0
for name in "${CASES[@]}"; do
  bin=./.lake/build/bin/vjp-oracle-${name}
  script=.lake/build/generated_vjp_oracle_${name//-/_}.py
  rm -f "$script"

  # The exe emits the script and then tries to run it itself. We ignore that
  # inner attempt (it uses its own python discovery and exits 0 regardless);
  # what matters is that the script lands.
  "$bin" "$DATA" >/dev/null 2>&1
  if [ ! -f "$script" ]; then
    echo "FAIL  ${name}  did not emit ${script}"; FAIL=1; continue
  fi
  if ! $PY -m py_compile "$script" 2>/dev/null; then
    echo "FAIL  ${name}  emitted script does not parse"; FAIL=1; continue
  fi

  if [ "$RUN" = "0" ]; then
    echo "EMIT  ${name}  ok ($(wc -c < "$script") bytes)"
    continue
  fi

  log=$(mktemp)
  if LEAN_MLIR_NO_SHUFFLE=1 JAX_PLATFORMS=cpu $PY "$script" >"$log" 2>&1; then
    # Exit 0 is not enough: the script must have printed an epoch line whose
    # Loss parses as a FINITE float. Exit 0 with no such line means the run
    # degenerated (e.g. an eval loop that never executed), and a NaN loss is
    # a silent emitter bug. Parse the value rather than grepping for the
    # substrings "nan"/"inf", which match ordinary words like "info".
    loss=$(sed -n 's/.*Loss: \([0-9.eE+-]*\).*/\1/p' "$log" | tail -1)
    if [ -z "$loss" ]; then
      echo "FAIL  ${name}  ran but printed no epoch Loss"; tail -10 "$log"; FAIL=1
    elif ! $PY -c "import sys,math; v=float(sys.argv[1]); sys.exit(0 if math.isfinite(v) else 1)" "$loss" 2>/dev/null; then
      echo "FAIL  ${name}  non-finite loss: ${loss}"; FAIL=1
    else
      echo "RUN   ${name}  ok — loss ${loss}"
    fi
  else
    echo "FAIL  ${name}  python exited nonzero"; tail -15 "$log"; FAIL=1
  fi
  rm -f "$log"
done

[ "$FAIL" = "0" ] && echo "all ${#CASES[@]} cases ok" || echo "failures present"
exit $FAIL
