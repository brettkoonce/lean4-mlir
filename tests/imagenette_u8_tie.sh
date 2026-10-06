#!/usr/bin/env bash
# tests/imagenette_u8_tie.sh — the raw-uint8 and streamed Imagenette loaders must not move a bit.
#
#     tests/imagenette_u8_tie.sh                 # ResNet-34 `adam`, 2 epochs x 40 steps, eval on
#     STEPS=295 tests/imagenette_u8_tie.sh       # whole epochs (9,469 / 32)
#     BIN=.lake/build/bin/vit-verified-adam SLUG=vit tests/imagenette_u8_tie.sh
#
# WHAT IT GATES. `loadData` has three ways to hold Imagenette for `trainAdamSched`: the f32 loader
# (every chapter transcript), `LEAN_MLIR_IMAGENETTE_U8=1` (pixels resident as the uint8 they are on
# disk, normalised one batch at a time by `sliceU8NormPad`) and `LEAN_MLIR_IMAGENETTE_STREAM=1`
# (only the labels resident, a shuffled u32 index array, each batch `pread` from train.bin by
# `readU8NormPad`). The last two are what let an 8 GB Jetson train Imagenette (deploy/ORIN.md), and
# the claim they make is exactness: the same batches, in the same order, normalised by the same
# float32 expression, with the same zero padding past the end of val. So the trained [θ|m|v] after
# N steps and every per-epoch eval line must be BIT-IDENTICAL across the three.
#
# Built on tests/u8_wire_tie.sh, and for the same reasons: a second f32 run is the control that
# proves the platform is bit-reproducible at all (the deterministic shim, autotuning off), and each
# opt-in run's log must SAY which loader it took, because a run that silently fell back to f32 would
# pass this gate forever. Eval stays ON: it is the uint8 val path (`evalScore`'s held branch), and
# 3,925 is not a multiple of the batch, so its last batch is the padded one.
set -u
STEPS=${STEPS:-40}
EPOCHS=${EPOCHS:-2}                     # two, so the per-epoch shuffle accumulates once
DEV=${DEV:-0}
BIN=${BIN:-.lake/build/bin/resnet34-verified-adam}
SLUG=${SLUG:-resnet34}
VARIANT=${VARIANT:-adam}
OUT=${OUT:-$(mktemp -d)}; mkdir -p "$OUT"
[ -f lakefile.lean ] || { echo "run from the repo root"; exit 2; }
[ -x "$BIN" ] || { echo "missing $BIN — lake build $(basename "$BIN")"; exit 2; }
[ -f data/imagenette/train.bin ] || { echo "missing data/imagenette/train.bin — scripts/datasets/download_imagenette.sh"; exit 2; }

. scripts/jobs/_box.sh   # BOX_PLUG

# Checkpoints are tag-scoped so no run resumes from another's (or from a real run's), and the
# scoped files (the blob, its .epoch marker, the BN running stats) are removed on exit.
TAG_BASE="u8tie$$"
cleanup () { rm -f .lake/build/${SLUG}_${VARIANT}_ckpt_xla_${TAG_BASE}_*; return 0; }
trap cleanup EXIT INT TERM

echo "── Imagenette uint8 / streamed loaders: bit-identity gate ──"
echo "   net      $SLUG/$VARIANT on device $DEV, $EPOCHS epoch(s) x $STEPS steps, eval on"
echo "   scratch  $OUT"
DET=${DET_SHIM:-/tmp/residency_detshim}
if [ ! -f "$DET/libpjrt_ffi.so" ] || [ ffi/pjrt_ffi.c -nt "$DET/libpjrt_ffi.so" ]; then
  echo "   building the deterministic shim in $DET ..."
  scripts/det_shim.sh "$DET" > "$OUT/det_shim.log" 2>&1 || {
    echo "   ✗ det_shim.sh failed:"; cat "$OUT/det_shim.log"; exit 2; }
fi

# run <tag> <U8> <STREAM>: the two knobs are passed as "" (unset) or "1".
run () {
  local tag=$1 u8=$2 stream=$3
  local -a knobs=()
  [ -n "$u8" ] && knobs+=(LEAN_MLIR_IMAGENETTE_U8="$u8")
  [ -n "$stream" ] && knobs+=(LEAN_MLIR_IMAGENETTE_STREAM="$stream")
  env \
    LD_LIBRARY_PATH="$DET" \
    CUDA_VISIBLE_DEVICES="$DEV" \
    PJRT_PLUGIN="$BOX_PLUG" \
    LEAN_MLIR_LOWERER=xla \
    LEAN_MLIR_VARIANT="$VARIANT" \
    LEAN_MLIR_SEED=1 \
    LEAN_MLIR_CKPT_TAG="${TAG_BASE}_$tag" \
    LEAN_MLIR_MAX_EPOCHS="$EPOCHS" \
    LEAN_MLIR_G2_STEPS="$STEPS" \
    LEAN_MLIR_DUMP_PARAMS="$OUT/$tag.bin" \
    "${knobs[@]}" \
    "$BIN" data > "$OUT/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$OUT/$tag.bin" ]; then
    echo "   ✗ run $tag failed (rc=$rc); tail:"; tail -8 "$OUT/$tag.log"; exit 2
  fi
  # The per-epoch eval lines, verbatim: counts, not percentages, so a one-image change shows.
  grep -o 'epoch [0-9]*: val_acc = [0-9]*/[0-9]* .*top5 = [0-9]*/[0-9]*' "$OUT/$tag.log" > "$OUT/$tag.eval"
  printf "   %-3s U8=%-1s STREAM=%-1s  %s bytes  loader: %s  eval lines: %s\n" "$tag" "${u8:-0}" "${stream:-0}" \
    "$(stat -c%s "$OUT/$tag.bin")" "$(loader_of "$OUT/$tag.log")" "$(wc -l < "$OUT/$tag.eval")"
}
loader_of () {
  if grep -q 'train STREAMED from train.bin' "$1"; then echo stream
  elif grep -q 'raw-uint8 resident' "$1"; then echo u8
  else echo f32; fi
}
diffbytes () { if cmp -s "$1" "$2"; then echo 0; else cmp -l "$1" "$2" 2>/dev/null | wc -l; fi; }

echo; echo "── runs ──"
run A1 "" ""
run A2 "" ""
run B  1  ""
run C  "" 1

CTRL=$(diffbytes "$OUT/A1.bin" "$OUT/A2.bin")
VB=$(diffbytes "$OUT/A1.bin" "$OUT/B.bin")
VC=$(diffbytes "$OUT/A1.bin" "$OUT/C.bin")
EB=$(diff -q "$OUT/A1.eval" "$OUT/B.eval" > /dev/null && echo same || echo DIFFER)
EC=$(diff -q "$OUT/A1.eval" "$OUT/C.eval" > /dev/null && echo same || echo DIFFER)
ECTRL=$(diff -q "$OUT/A1.eval" "$OUT/A2.eval" > /dev/null && echo same || echo DIFFER)
LA=$(loader_of "$OUT/A1.log"); LB=$(loader_of "$OUT/B.log"); LC=$(loader_of "$OUT/C.log")
echo; echo "── verdict ──"
echo "   control  A1 vs A2 (f32, twice)      : $CTRL differing bytes, eval lines $ECTRL"
echo "   verdict  A1 vs B  (f32 vs uint8)    : $VB differing bytes, eval lines $EB"
echo "   verdict  A1 vs C  (f32 vs streamed) : $VC differing bytes, eval lines $EC"
echo "   loaders  A1=$LA  B=$LB  C=$LC"
echo
[ "$CTRL" -ne 0 ] || [ "$ECTRL" != same ] && { echo "⚠⚠ CONTROL FAILED — two f32 runs disagree; no bit-exact floor, the verdict means nothing. Check $DET is the deterministic shim. $OUT kept."; exit 1; }
[ "$LA" != f32 ] && { echo "✗ run A1 did not take the f32 loader ($LA). $OUT kept."; exit 1; }
[ "$LB" != u8 ] && { echo "✗ VACUOUS — run B took the $LB loader, not the raw-uint8 one. $OUT kept."; exit 1; }
[ "$LC" != stream ] && { echo "✗ VACUOUS — run C took the $LC loader, not the streamed one. $OUT kept."; exit 1; }
[ "$(wc -l < "$OUT/A1.eval")" -ne "$EPOCHS" ] && { echo "✗ expected $EPOCHS eval lines in A1, got $(wc -l < "$OUT/A1.eval") — eval must be ON for the uint8 val path to be tested. $OUT kept."; exit 1; }
[ "$VB" -ne 0 ] || [ "$EB" != same ] && { echo "✗ FAIL — the raw-uint8 loader changed the trained state or an eval line. $OUT kept."; exit 1; }
[ "$VC" -ne 0 ] || [ "$EC" != same ] && { echo "✗ FAIL — the streamed loader changed the trained state or an eval line. $OUT kept."; exit 1; }
echo "✓ PASS — control clean; raw-uint8 and streamed loaders bit-identical to f32 over $EPOCHS x $STEPS steps, eval lines equal."
rm -rf "$OUT"
