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
#   D       FAULT    the ALT path with a deliberate transport fault injected
#                    ($GATE_FAULT). Must DIFFER. This is the control that
#                    matters: C proves the harness can see *a* difference, D
#                    proves it can see a *transport* difference in the path under
#                    test, which is the only thing this gate is for. ⚠ Which
#                    fault is usable is NET-DEPENDENT — see $GATE_FAULT below.
#
# To point it at device residency once that lands, change nothing but GATE_ALT:
#   GATE_ALT=PJRT_FFI_RESIDENT=1 scripts/residency_gate.sh
#
# ⚠⚠ ON CUDA YOU MUST RUN THIS AGAINST A DETERMINISTIC SHIM, or every verdict
# below is noise. Measured on ares 2026-08-01: with the committed compile
# options the FLOOR is 191,094,739 of 255,477,624 bytes at 10 steps — and even
# at ONE step it is 2,647,005 — because XLA autotuning picks different
# convolution algorithms per process and AdamW amplifies one differing bit into
# most of the blob within a few steps. §2d.3's Finding 1 ("the floor IS
# bit-exact across processes") was measured on ROCm and does NOT carry.
#
# It is suppressible, and then the gate reads exactly as designed (floor 0):
#
#   scripts/det_shim.sh /tmp/detshim              # autotune off, deterministic ops
#   LD_LIBRARY_PATH=/tmp/detshim \
#     GATE_ALT=PJRT_FFI_RESIDENT=1 scripts/residency_gate.sh
#
# The SATURATION check below is what stops a noisy floor being read as a
# verdict: on the committed shim the FAULT run — a deliberate 1-ULP corruption —
# scored 190,642,729, i.e. BELOW the floor. At that point the byte count cannot
# tell a real transport bug from run-to-run noise in either direction, so
# "TEST <= FLOOR" is not evidence of anything and the gate must say so.
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
# The transport fault used for run D, applied TO THE ALT PATH — the one under
# test, which is what the control is supposed to be about.
#
# ⚠ THE DEFAULT IS NOT UNIVERSAL. `PJRT_FFI_FAULT=1` is 1 ULP on one float, and
# on a net whose updates CONTRACT that is absorbed rather than amplified: on the
# MNIST MLP under plain SGD it moves 1 byte at 3 steps and **0 at 10**, on real
# data as well as synthetic (measured 2026-08-01). §2d.3's Finding 2 — "the
# system is chaotic, a small transport error does not stay small" — was measured
# on R34 + AdamW and does not carry. Where it does not, use the macroscopic one,
# which is also the failure mode residency actually has:
#
#   GATE_FAULT=PJRT_FFI_FAULT=2 ...    # drop one step's retained parameters
#
# The SATURATION check below is what stops you reading a verdict either way.
GATE_FAULT=${GATE_FAULT:-PJRT_FFI_FAULT=1}
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
# D faults the ALT path, not the default one: the question this control answers
# is "could this harness see a defect in the transport under test", and the
# transport under test is the alt. (It was the default path until 2026-08-01;
# faulting the alt is strictly more relevant and cost nothing.)
run D  $GATE_ALT $GATE_FAULT

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
printf "   FAULT   default vs alt+fault   %12s bytes differ  ← must be non-zero\n" "$FAULT"
echo

if [ "$CTRL" -eq 0 ]; then
  echo "✗ VACUOUS: a perturbed initialisation produced an IDENTICAL result."
  echo "  The comparison is not sensitive to anything — this gate cannot go red,"
  echo "  so its green means nothing. Fix the harness before reading any verdict."
  exit 1
fi

if [ "$FAULT" -eq 0 ]; then
  echo "✗ VACUOUS: the transport fault [$GATE_FAULT] was NOT SEEN."
  echo "  The harness can see an init difference (control fired at $CTRL) but not a"
  echo "  TRANSPORT difference, which is the only thing this gate exists to catch."
  echo "  A green TEST here would be worthless. ⚠ On a net whose updates CONTRACT,"
  echo "  PJRT_FFI_FAULT=1 is absorbed rather than amplified and reads as 0 — try"
  echo "  the macroscopic one: GATE_FAULT=PJRT_FFI_FAULT=2 (drop a step's params)."
  exit 1
fi

# ⚠ SATURATION — the check that decides whether ANY verdict below is readable.
#
# The two checks above ask whether the controls MOVED. This one asks whether
# they moved further than the noise, which is a different question and the one
# that matters: a 1-ULP fault is the smallest real defect this gate must catch,
# so if it does not clearly out-score two runs of the SAME path, nothing smaller
# can be separated either, and both branches below would be reporting a coin
# flip. Measured on ares with the committed compile options, two consecutive
# invocations of this script:
#
#   run 1   FLOOR 191,094,739   FAULT 190,642,729   ← the fault scored CLEANER
#   run 2   FLOOR 189,531,906   FAULT 191,037,525   ← 0.8% apart
#
# ⚠ The ORDERING is not the bar, and a first version of this check that used it
# passed run 2 — where TEST sat 0.9% above the floor and was reported as a real
# regression. It was not; nothing there is distinguishable from anything else.
# The bar is a MARGIN, which is what every other gate in this repo demands: the
# ties quote orders of magnitude of separation, and 2x is already generous.
if [ "$FLOOR" -ne 0 ] && [ "$FAULT" -lt $((2 * FLOOR)) ]; then
  echo "✗ SATURATED: a deliberate 1-ULP fault ($FAULT bytes) is not separated from the"
  echo "  A-vs-A floor ($FLOOR bytes) — it needs to be at least 2x it, and it is"
  echo "  $(( 100 * FAULT / FLOOR ))% of it. The comparison has no resolution left: after"
  echo "  this many steps a real transport defect and run-to-run nondeterminism are the"
  echo "  same number, so neither a PASS nor a FAIL would mean anything."
  echo
  echo "  This is the expected result on CUDA with the committed compile options — the"
  echo "  floor is autotuning, not the transport. Fix the FLOOR, do not read the TEST:"
  echo "    scripts/det_shim.sh /tmp/detshim"
  echo "    LD_LIBRARY_PATH=/tmp/detshim GATE_ALT=$GATE_ALT $0 $BIN $SLUG $STEPS"
  exit 1
fi

if [ "$FLOOR" -ne 0 ]; then
  echo "⚠ THE FLOOR IS NOT BIT-EXACT: two runs of the SAME path differ in $FLOOR bytes."
  echo "  §2d.3's stated gate — 'residency must be BIT-IDENTICAL to the copying path'"
  echo "  — is therefore NOT buildable as written here, and the finding is the gate's,"
  echo "  not the implementation's. The fault DID separate ($FAULT, >=2x the floor), so"
  echo "  the comparison still has power and is read against the floor (the §4 A-vs-A"
  echo "  rule) rather than against equality."
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
  echo "  with BOTH controls firing (init $CTRL, transport fault [$GATE_FAULT] $FAULT)."
  echo "  The alternative transport changes no bit of the trained parameters."
  exit 0
fi

echo "✗ FAIL — the alt path differs in $TEST bytes while the floor is bit-exact."
echo "  That is graph-attributable, not noise: the transport changed the result."
exit 1
