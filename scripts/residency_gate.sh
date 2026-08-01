#!/usr/bin/env bash
# residency_gate.sh — the §2d.3 phase-3 gate: an ALTERNATIVE parameter-transport
# path must produce bit-identical training to the copying path.
#
#   scripts/residency_gate.sh [<binary>] [<slug>] [<steps>]
#
# Written BEFORE device-resident parameters exist, on purpose. A gate authored
# after the thing it gates tends to be written until it passes; this one was
# authored against a path whose answer nobody had a stake in, and it immediately
# earned its keep by settling whether "bit-identical" is even an achievable bar
# on this backend (handoff §3 says XLA is not bit-stable ACROSS PROCESSES, which
# would make §2d.3's stated gate design unbuildable as written).
#
# Four runs, each a fresh process from a deleted checkpoint:
#
#   A1, A2  FLOOR    the default path against ITSELF. Establishes what
#                    "identical" can mean here at all. Every verdict below is
#                    read against this, never against an assumption.
#   B       TEST     the alternative transport. Must match A1 as well as A2 does.
#   C       CONTROL  a perturbed initialisation. Must DIFFER, or the comparison
#                    is not sensitive to anything and the gate is VACUOUS —
#                    the failure mode §2d.1 hit with a reversed-batch control
#                    that produced no difference at all.
#   D       FAULT    a deliberate 1-ULP corruption of ONE returned float
#                    (PJRT_FFI_FAULT=1). Must DIFFER. This is the control that
#                    matters: C proves the harness can see *a* difference, D
#                    proves it can see a *transport* difference, which is the
#                    only thing this gate is for.
#
# To point it at device residency once that lands, change nothing but GATE_ALT:
#   GATE_ALT=PJRT_FFI_RESIDENT=1 scripts/residency_gate.sh
set -uo pipefail

BIN=${1:-resnet34-verified-adam-xla}
SLUG=${2:-resnet34}
STEPS=${3:-10}
VARIANT=${LEAN_MLIR_VARIANT:-adam}
# The alternative transport under test. Default is the pinned-d2h path, which is
# the only second path that exists today; it is REFUTED as an optimisation
# (§2d.3) but is a perfectly good stand-in here, because the property being
# gated is "changes no bits", not "is faster".
GATE_ALT=${GATE_ALT:-PJRT_FFI_PINNED=1}
OUT=${GATE_OUT:-$(mktemp -d)}
mkdir -p "$OUT"

BINPATH=.lake/build/bin/$BIN
[ -x "$BINPATH" ] || { echo "no such binary: $BINPATH (lake build $BIN)"; exit 2; }

CKPT=.lake/build/${SLUG}_${VARIANT}_ckpt_xla.bin

echo "── §2d.3 phase-3 residency gate ──"
echo "   binary   $BIN   (slug $SLUG, variant $VARIANT)"
echo "   steps    $STEPS, 1 epoch, synthetic inputs, eval skipped"
echo "   alt path $GATE_ALT"
echo "   scratch  $OUT"

# One run. A stale checkpoint would make the run a silent no-op or resume it from
# another run's state, which is handoff §4's most expensive trap — so every run
# starts from a deleted one rather than trusting that none is there.
run () {
  local tag=$1; shift
  rm -f "$CKPT" "$CKPT.epoch"
  # Both vendors' pinning vars, because each is inert on the other's runtime. With only
  # the HIP one set, a CUDA box silently ignored the pin and took whatever device 0 was —
  # which happens to be right by accident on an idle box and wrong the moment it is not.
  env "$@" \
    HIP_VISIBLE_DEVICES=0 \
    CUDA_VISIBLE_DEVICES=0 \
    LEAN_MLIR_VARIANT="$VARIANT" \
    LEAN_MLIR_BENCH_SYNTH=1 \
    LEAN_MLIR_SKIP_EVAL=1 \
    LEAN_MLIR_MAX_EPOCHS=1 \
    LEAN_MLIR_G2_STEPS="$STEPS" \
    LEAN_MLIR_DUMP_PARAMS="$OUT/$tag.bin" \
    "$BINPATH" data > "$OUT/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$OUT/$tag.bin" ]; then
    echo "   ✗ run $tag failed (rc=$rc); tail:"; tail -5 "$OUT/$tag.log"; exit 2
  fi
  printf "   %-3s %s bytes\n" "$tag" "$(stat -c%s "$OUT/$tag.bin")"
}

# `cmp -l` counts differing BYTES; the gate's stated bar is bit-identity, so the
# verdict is identical-or-not and the count is only there to say how badly.
diffbytes () {
  if cmp -s "$1" "$2"; then echo 0; else cmp -l "$1" "$2" 2>/dev/null | wc -l; fi
}

echo
echo "── runs ──"
run A1
run A2
run B  $GATE_ALT
run C  LEAN_MLIR_PERTURB_R=15990
run D  PJRT_FFI_FAULT=1

TOTAL=$(stat -c%s "$OUT/A1.bin")
FLOOR=$(diffbytes "$OUT/A1.bin" "$OUT/A2.bin")
TEST=$(diffbytes  "$OUT/A1.bin" "$OUT/B.bin")
CTRL=$(diffbytes  "$OUT/A1.bin" "$OUT/C.bin")
FAULT=$(diffbytes "$OUT/A1.bin" "$OUT/D.bin")

echo
echo "── verdict, over $TOTAL bytes of [θ|m|v] ──"
printf "   FLOOR   default vs default    %12s bytes differ\n" "$FLOOR"
printf "   TEST    default vs alt        %12s bytes differ\n" "$TEST"
printf "   CONTROL default vs perturbed  %12s bytes differ  ← must be non-zero\n" "$CTRL"
printf "   FAULT   default vs 1-ULP hit  %12s bytes differ  ← must be non-zero\n" "$FAULT"
echo

if [ "$CTRL" -eq 0 ]; then
  echo "✗ VACUOUS: a perturbed initialisation produced an IDENTICAL result."
  echo "  The comparison is not sensitive to anything — this gate cannot go red,"
  echo "  so its green means nothing. Fix the harness before reading any verdict."
  exit 1
fi

if [ "$FAULT" -eq 0 ]; then
  echo "✗ VACUOUS: a deliberate 1-ULP corruption of a returned float was NOT SEEN."
  echo "  The harness can see an init difference (control fired at $CTRL) but not a"
  echo "  TRANSPORT difference, which is the only thing this gate exists to catch."
  echo "  A green TEST here would be worthless. Fix the harness."
  exit 1
fi

if [ "$FLOOR" -ne 0 ]; then
  echo "⚠ THE FLOOR IS NOT BIT-EXACT: two runs of the SAME path differ in $FLOOR bytes."
  echo "  §2d.3's stated gate — 'residency must be BIT-IDENTICAL to the copying path'"
  echo "  — is therefore NOT buildable as written on this backend, and the finding is"
  echo "  the gate's, not the implementation's. Phase 3 needs a tolerance read against"
  echo "  this floor (the §4 A-vs-A rule), not an equality check."
  if [ "$TEST" -le "$FLOOR" ]; then
    echo "✓ TEST is within the floor ($TEST ≤ $FLOOR) — the alt path is indistinguishable"
    echo "  from run-to-run nondeterminism, which is the strongest available statement."
    exit 0
  fi
  echo "✗ TEST ($TEST) EXCEEDS the floor ($FLOOR) — the alt path changes the result."
  exit 1
fi

if [ "$TEST" -eq 0 ]; then
  echo "✓ PASS — bit-identical on all $TOTAL bytes, against a bit-exact A-vs-A floor,"
  echo "  with BOTH controls firing (init $CTRL, 1-ULP transport fault $FAULT)."
  echo "  The alternative transport changes no bit of the trained parameters."
  exit 0
fi

echo "✗ FAIL — the alt path differs in $TEST bytes while the floor is bit-exact."
echo "  That is graph-attributable, not noise: the transport changed the result."
exit 1
