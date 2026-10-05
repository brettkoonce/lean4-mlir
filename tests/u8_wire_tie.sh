#!/usr/bin/env bash
# tests/u8_wire_tie.sh — the shim's uint8 wire must not move a single bit.
#
#     tests/u8_wire_tie.sh                      # MobileNetV4 `full`, 4 x 128, 1 epoch x 16 micro-steps
#     STEPS=32 tests/u8_wire_tie.sh
#
# WHAT IT GATES. With the uint8 wire (wire v5/v6, `LEAN_MLIR_SHIM_U8`, on by default) a shim whose
# train images are uint8 before the normalize sends them as uint8 HWC, and the trainer applies the
# normalize and the transpose in C (`u8Norm` / `lean_mlir_u8_norm`) with the same float32 ops. So the
# trained state after N steps must be BIT-IDENTICAL to the float wire's (`LEAN_MLIR_SHIM_U8=0`).
#
# Built on tests/prefetch_tie.sh, and for the same reasons: THREE runs, because A1-vs-A2 (both on the
# float wire) is the control that proves the platform is bit-reproducible at all — the deterministic
# PJRT shim (autotuning off) and `SHIM_DETERMINISM=1` are what make it so. And a fourth check the
# prefetch gate does not need: run B's log must SAY the uint8 wire was granted, because a shim that
# silently answered v3/v4 would pass this gate forever.
set -u
STEPS=${STEPS:-16}                      # micro-steps: whole accumulation cycles of 8
DEVS=${DEVS:-0,1,2,3}
BIN=${BIN:-.lake/build/bin/mobilenetv4-imagenet-verified}
VARIANT=${VARIANT:-accdp8x128wxdropdowd01bf16}
SLUG=${SLUG:-mnv4in}
OUT=${OUT:-$(mktemp -d)}
[ -f lakefile.lean ] || { echo "run from the repo root"; exit 2; }
[ -x "$BIN" ] || { echo "missing $BIN — lake build $(basename "$BIN")"; exit 2; }
CKPT=".lake/build/${SLUG}_${VARIANT}_ckpt_xla.bin"
SAVED="$(mktemp -d)"
restore () {
  for f in "$CKPT" "$CKPT.epoch"; do
    [ -f "$SAVED/$(basename "$f")" ] && mv -f "$SAVED/$(basename "$f")" "$f"
  done
  return 0
}
trap restore EXIT INT TERM
for f in "$CKPT" "$CKPT.epoch"; do [ -f "$f" ] && cp -p "$f" "$SAVED/"; done

. scripts/jobs/_box.sh   # BOX_PLUG, BOX_SHIMPY

echo "── uint8 shim wire: bit-identity gate ──"
echo "   net      $SLUG/$VARIANT, 4 x 128 on devices $DEVS, $STEPS micro-steps"
echo "   scratch  $OUT"
DET=${DET_SHIM:-/tmp/residency_detshim}
if [ ! -f "$DET/libpjrt_ffi.so" ] || [ ffi/pjrt_ffi.c -nt "$DET/libpjrt_ffi.so" ]; then
  echo "   building the deterministic shim in $DET ..."
  scripts/det_shim.sh "$DET" > "$OUT/det_shim.log" 2>&1 || {
    echo "   ✗ det_shim.sh failed:"; cat "$OUT/det_shim.log"; exit 2; }
fi

run () {
  local tag=$1 u8=$2
  rm -f "$CKPT" "$CKPT.epoch"
  env \
    LD_LIBRARY_PATH="$DET" \
    CUDA_VISIBLE_DEVICES="$DEVS" \
    PJRT_PLUGIN="$BOX_PLUG" \
    "${BOX_SHIMPY[@]}" \
    PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
    LEAN_MLIR_RECIPE=full \
    LEAN_MLIR_VARIANT="$VARIANT" \
    LEAN_MLIR_BATCH=128 \
    LEAN_MLIR_EPOCHS=500 \
    LEAN_MLIR_BASE_LR_U=4000 \
    SHIM_WORKERS=4 \
    LEAN_MLIR_SEED=1 \
    SHIM_DETERMINISM=1 \
    LEAN_MLIR_SHIM_U8="$u8" \
    LEAN_MLIR_SKIP_EVAL=1 \
    LEAN_MLIR_MAX_EPOCHS=1 \
    LEAN_MLIR_G2_STEPS="$STEPS" \
    LEAN_MLIR_DUMP_PARAMS="$OUT/$tag.bin" \
    "$BIN" data > "$OUT/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$OUT/$tag.bin" ]; then
    echo "   ✗ run $tag failed (rc=$rc); tail:"; tail -8 "$OUT/$tag.log"; exit 2
  fi
  printf "   %-3s u8=%s  %s bytes  uint8 producers: %s of %s\n" "$tag" "$u8" \
    "$(stat -c%s "$OUT/$tag.bin")" "$(grep -c 'UINT8 wire' "$OUT/$tag.log")" \
    "$(grep -c 'imagenet shim: .*train split' "$OUT/$tag.log")"
}
diffbytes () { if cmp -s "$1" "$2"; then echo 0; else cmp -l "$1" "$2" 2>/dev/null | wc -l; fi; }

echo; echo "── runs ──"
run A1 0
run A2 0
run B  1

CTRL=$(diffbytes "$OUT/A1.bin" "$OUT/A2.bin")
VERD=$(diffbytes "$OUT/A1.bin" "$OUT/B.bin")
NU8=$(grep -c 'UINT8 wire' "$OUT/B.log")
NA=$(grep -c 'UINT8 wire' "$OUT/A1.log")
echo; echo "── verdict ──"
echo "   control  A1 vs A2 (float wire, twice): $CTRL differing bytes"
echo "   verdict  A1 vs B  (float vs uint8)    : $VERD differing bytes"
echo "   wire     B granted uint8 on $NU8 producer(s), A1 on $NA"
echo
[ "$CTRL" -ne 0 ] && { echo "⚠⚠ CONTROL FAILED — two float-wire runs disagree; no bit-exact floor, the verdict means nothing. $OUT kept."; exit 1; }
[ "$NU8" -eq 0 ] && { echo "✗ VACUOUS — run B never got the uint8 wire, so it compared the float wire with itself. $OUT kept."; exit 1; }
[ "$NA" -ne 0 ] && { echo "✗ run A1 got the uint8 wire with LEAN_MLIR_SHIM_U8=0. $OUT kept."; exit 1; }
[ "$VERD" -ne 0 ] && { echo "✗ FAIL — the uint8 wire changed the trained state. $OUT kept."; exit 1; }
echo "✓ PASS — control clean, B on the uint8 wire, trained state bit-identical over $STEPS micro-steps."
rm -rf "$OUT"
