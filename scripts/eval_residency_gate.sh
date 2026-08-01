#!/usr/bin/env bash
# eval_residency_gate.sh — the gate for HOLD-mode residency on the eval forward.
#
#   scripts/eval_residency_gate.sh [<binary>...]     # default: the demo loops
#
# ▶ WHY THIS IS A SEPARATE GATE. `residency_gate.sh` compares the trained
# [θ|m|v] and that is the right check for the TRAIN step, where the retained
# buffers feed back into the next step and any defect propagates. It is
# structurally blind to the eval forward: eval does not feed training, so a held
# parameter set that went STALE would score the previous epoch's weights and the
# final [θ|m|v] would still be bit-identical. The gate would go green on exactly
# the failure hold mode can have.
#
# What catches it is the reported accuracy, epoch by epoch. A stale held set
# repeats an earlier epoch's number; a correct one reproduces the copying path's
# whole sequence. The check has power only because those numbers MOVE between
# epochs, which the script verifies rather than assumes.
#
# ⚠ Run against the deterministic shim, for `residency_gate.sh`'s reason: with
# the committed compile options autotuning makes two runs of the SAME path differ
# (measured on the MNIST CNN, three copying runs: 9833 / 9846 / 9847 of 10000), so
# an equality check would be noise. det_shim.sh is built automatically below.
set -uo pipefail

DET=${DET_SHIM:-/tmp/residency_detshim}
OUT=${GATE_OUT:-$(mktemp -d)}
EPOCHS=${EPOCHS:-3}
DEV=${GATE_DEV:-0}
mkdir -p "$OUT"

BINS=("$@")
if [ ${#BINS[@]} -eq 0 ]; then
  BINS=(mnist-linear-verified-xla mnist-mlp-verified-xla mnist-cnn-verified-xla)
fi

[ -f scripts/det_shim.sh ] || { echo "run from the repo root"; exit 2; }
if [ ! -f "$DET/libpjrt_ffi.so" ] || [ ffi/pjrt_ffi.c -nt "$DET/libpjrt_ffi.so" ]; then
  echo "building the deterministic shim in $DET ..."
  scripts/det_shim.sh "$DET" > "$OUT/det_shim.log" 2>&1 || {
    echo "✗ det_shim.sh failed:"; cat "$OUT/det_shim.log"; exit 2; }
fi

echo "── eval-forward residency gate (hold mode, §2d.3) ──"
echo "   shim   $DET/libpjrt_ffi.so   epochs $EPOCHS   device $DEV"
echo

FAILED=0
for bin in "${BINS[@]}"; do
  [ -x ".lake/build/bin/$bin" ] || { printf "  %-28s ⚠ SKIP — not built\n" "$bin"; continue; }
  for tag in copy res; do
    extra=""; [ "$tag" = res ] && extra="PJRT_FFI_RESIDENT=1"
    # shellcheck disable=SC2086
    env $extra LD_LIBRARY_PATH="$DET" CUDA_VISIBLE_DEVICES="$DEV" HIP_VISIBLE_DEVICES="$DEV" \
      LEAN_MLIR_MAX_EPOCHS="$EPOCHS" \
      ".lake/build/bin/$bin" data > "$OUT/${bin}_$tag.log" 2>&1
  done
  a=$(grep -oE "test_acc = [0-9]+/[0-9]+" "$OUT/${bin}_copy.log" | tr '\n' ' ')
  b=$(grep -oE "test_acc = [0-9]+/[0-9]+" "$OUT/${bin}_res.log"  | tr '\n' ' ')
  uniq_a=$(echo "$a" | tr ' ' '\n' | grep -c "test_acc" )
  distinct=$(echo "$a" | tr ' ' '\n' | grep "/" | sort -u | wc -l)

  if [ -z "$a" ] || [ -z "$b" ]; then
    printf "  %-28s ✗ FAIL — a run produced no accuracy line\n" "$bin"; FAILED=1; continue
  fi
  # A gate that cannot fail is not a gate: if every epoch reports the SAME
  # accuracy then a stale held set is indistinguishable from a correct one.
  if [ "$distinct" -lt 2 ]; then
    printf "  %-28s ✗ VACUOUS — accuracy never moves across %s epochs, so a stale\n" "$bin" "$EPOCHS"
    printf "  %-28s   held set would look identical. Use more epochs.\n" ""
    FAILED=1; continue
  fi
  if [ "$a" = "$b" ]; then
    printf "  %-28s ✓ PASS — %s epochs identical (%s distinct values)\n" "$bin" "$uniq_a" "$distinct"
  else
    printf "  %-28s ✗ FAIL\n      copying  %s\n      resident %s\n" "$bin" "$a" "$b"
    FAILED=1
  fi
done

echo
if [ $FAILED -eq 0 ]; then
  echo "✓ the held parameter set tracks the host's across epochs — eval scores the"
  echo "  weights it just trained, not an earlier set. Logs in $OUT."
  exit 0
fi
echo "✗ eval residency changed a reported accuracy. The likely cause is the"
echo "  generation token not advancing when the parameters do — see forwardF32's"
echo "  \`gen\` argument, which must change every time \`params\` does. Logs in $OUT."
exit 1
