#!/usr/bin/env bash
# tests/prefetch_tie.sh — the depth-1 shim prefetch must not move a single bit.
#
#     tests/prefetch_tie.sh                    # R34/ImageNet 4x bs64, 2 epochs x 12 steps
#     STEPS=24 EPOCHS=1 tests/prefetch_tie.sh  # longer, single epoch
#
# WHAT IT GATES. `LEAN_MLIR_PREFETCH=1` (the default) issues step i+1's shim read before step i's
# invoke, so the producer fills the pipe during compute instead of sleeping through it. It changes
# **when** a batch is read and nothing else: same handle, same order, same bytes. So the trained
# state after N steps must be BIT-IDENTICAL with it on and off. Anything else means the depth-1
# discipline broke and two reads interleaved on one pipe — which does not fail loudly, it silently
# scrambles a batch (a pipe is a stream, not a message queue).
#
# ⚠⚠ WHY THREE RUNS AND NOT TWO. The verdict is A1-vs-B, but A1-vs-A2 is the CONTROL and it is not
# optional: this gate reads the REAL shim (there is nothing to prefetch under
# `LEAN_MLIR_BENCH_SYNTH`, which now skips the stream entirely), so it inherits the tfds pipeline's
# run-to-run determinism. If that does not hold, A1 == B is meaningless and A1 != B is unreadable.
# The control establishes which world we are in BEFORE the verdict is allowed to mean anything —
# 2026-08-05's lesson, where two monitors and a probe each failed silently and each was caught only
# by its own control.
#
# ⚠ It crosses an EPOCH BOUNDARY by default (2 epochs), because that is the one place the prefetch
# index arithmetic can be wrong on its own: the read issued at the last step of epoch e is
# `ep*nb + bi + 1`, which must land exactly on epoch e+1's first index. A single-epoch run cannot
# see that.
#
# ⚠⚠ IT NEEDS THE DETERMINISTIC SHIM, and builds one if absent. XLA autotuning picks convolution
# algorithms per PROCESS on CUDA, so two runs of the identical path already differ — measured here
# 2026-08-05 at 145,301,829 of 261,572,064 bytes over 24 steps, the same signature `det_shim.sh`
# records (191,094,739 of 255,477,624 at 10). Without it the control fails and nothing downstream
# is readable. ⚠ And therefore: DO NOT take a timing off this script — autotuning is off, so it is
# deliberately running a different, slower program.
#
# ⚠ NOT a throughput check. It says the bytes are the same, never that the overlap happened. The
# speed claim is the real-vs-synth split (planning/next_session_pipeline_then_r50.md §2.2).
set -u

STEPS=${STEPS:-12}
EPOCHS=${EPOCHS:-2}
DEVS=${DEVS:-0,2,3,4}
BIN=${BIN:-.lake/build/bin/resnet34-imagenet-verified}
VARIANT=${VARIANT:-momdp64}
BATCH=${BATCH:-64}
REPLICAS=${REPLICAS:-4}
OUT=${OUT:-$(mktemp -d)}

[ -f lakefile.lean ] || { echo "run from the repo root"; exit 2; }
[ -x "$BIN" ] || { echo "missing $BIN — lake build $(basename "$BIN")"; exit 2; }

CKPT=".lake/build/resnet34in_${VARIANT}_ckpt_xla.bin"

# ⚠⚠ THE GATE DELETES THE CHECKPOINT BETWEEN RUNS — it has to, because a stale one makes a run
# either a silent no-op or a resume from someone else's state (handoff §4's most expensive trap).
# On this slug that file may be a MULTI-DAY RESULT, so it is moved aside and put back on any exit
# path, including Ctrl-C. The archived copy under runs/ is the authority; this is belt and braces.
SAVED="$(mktemp -d)"
restore () {
  for f in "$CKPT" "$CKPT.epoch"; do
    [ -f "$SAVED/$(basename "$f")" ] && mv -f "$SAVED/$(basename "$f")" "$f"
  done
  return 0
}
trap restore EXIT INT TERM
for f in "$CKPT" "$CKPT.epoch"; do [ -f "$f" ] && cp -p "$f" "$SAVED/"; done
[ -f "$SAVED/$(basename "$CKPT")" ] && echo "   (stashed an existing $CKPT — restored on exit)"

echo "── depth-1 shim prefetch: bit-identity gate ──"
echo "   net      resnet34in/$VARIANT, ${REPLICAS} x bs${BATCH} on devices $DEVS"
echo "   window   $EPOCHS epoch(s) x $STEPS steps = $((EPOCHS * STEPS)) steps"
echo "   scratch  $OUT"

# Shared with `residency_gate_all.sh` by default — same shim, same staleness rule, so the two
# gates cannot drift onto different compile options.
DET=${DET_SHIM:-/tmp/residency_detshim}
if [ ! -f "$DET/libpjrt_ffi.so" ] || [ ffi/pjrt_ffi.c -nt "$DET/libpjrt_ffi.so" ]; then
  echo "   building the deterministic shim in $DET ..."
  scripts/det_shim.sh "$DET" > "$OUT/det_shim.log" 2>&1 || {
    echo "   ✗ det_shim.sh failed:"; cat "$OUT/det_shim.log"; exit 2; }
fi
echo "   shim     $DET/libpjrt_ffi.so (autotuning OFF — correctness instrument, not a timing one)"

run () {
  local tag=$1 pf=$2
  rm -f "$CKPT" "$CKPT.epoch"
  # ⚠ NO `LEAN_MLIR_BENCH_SYNTH` here, deliberately: it skips the shim spawn, so there would be no
  # read to prefetch and the gate would compare a path against itself and pass forever.
  # ⚠ `LEAN_MLIR_SEED` pinned — it seeds both the shim's shuffle and its augmentation, and the
  # control run is what proves that pinning is enough.
  env \
    LD_LIBRARY_PATH="$DET" \
    CUDA_VISIBLE_DEVICES="$DEVS" \
    HIP_VISIBLE_DEVICES="$DEVS" \
    PJRT_PLUGIN=".venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
    PJRT_REPLICAS="$REPLICAS" \
    LEAN_MLIR_REPLICAS="$REPLICAS" \
    LEAN_MLIR_VARIANT="$VARIANT" \
    LEAN_MLIR_BATCH="$BATCH" \
    LEAN_MLIR_BASE_LR_U=100000 \
    PJRT_FFI_RESIDENT=1 \
    LEAN_MLIR_SEED=1 \
    LEAN_MLIR_PREFETCH="$pf" \
    LEAN_MLIR_SKIP_EVAL=1 \
    LEAN_MLIR_MAX_EPOCHS="$EPOCHS" \
    LEAN_MLIR_G2_STEPS="$STEPS" \
    LEAN_MLIR_DUMP_PARAMS="$OUT/$tag.bin" \
    "$BIN" data > "$OUT/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$OUT/$tag.bin" ]; then
    echo "   ✗ run $tag failed (rc=$rc); tail:"; tail -8 "$OUT/$tag.log"; exit 2
  fi
  printf "   %-3s prefetch=%s  %s bytes  %s\n" "$tag" "$pf" \
    "$(stat -c%s "$OUT/$tag.bin")" "$(grep -o 'SHIM PREFETCH: [A-Z]*' "$OUT/$tag.log" | head -1)"
}

# `cmp -l` counts differing BYTES; the bar is bit-identity, so the verdict is identical-or-not and
# the count only says how badly.
diffbytes () { if cmp -s "$1" "$2"; then echo 0; else cmp -l "$1" "$2" 2>/dev/null | wc -l; fi; }

echo
echo "── runs ──"
run A1 0
run A2 0
run B  1

CTRL=$(diffbytes "$OUT/A1.bin" "$OUT/A2.bin")
VERD=$(diffbytes "$OUT/A1.bin" "$OUT/B.bin")

echo
echo "── verdict ──"
echo "   control  A1 vs A2 (both prefetch OFF): $CTRL differing bytes"
echo "   verdict  A1 vs B  (OFF vs ON)        : $VERD differing bytes"
echo

if [ "$CTRL" -ne 0 ]; then
  echo "⚠⚠ CONTROL FAILED — two identical OFF runs disagree, so there is no bit-exact floor and"
  echo "   this gate cannot say anything about the prefetch. A PASS below would have been"
  echo "   meaningless and a FAIL unreadable. In order of likelihood:"
  echo "     1. the deterministic shim did not take — check $DET/libpjrt_ffi.so is on"
  echo "        LD_LIBRARY_PATH and that det_shim.sh did not no-op (it self-checks)"
  echo "     2. the tfds stream is not reproducible at a pinned seed — for a MIXING net the"
  echo "        λ comes from numpy's Generator, so try SHIM_SOFT=0"
  exit 1
fi

if [ "$VERD" -ne 0 ]; then
  echo "✗ FAIL — the prefetch changed the trained state. The read order moved, which at depth 1"
  echo "  means two reads overlapped on one handle and interleaved a batch. $OUT kept."
  exit 1
fi

echo "✓ PASS — control clean, and the prefetch is bit-identical over $((EPOCHS * STEPS)) steps"
echo "  across an epoch boundary. Only *when* the read happens moved."
rm -rf "$OUT"
