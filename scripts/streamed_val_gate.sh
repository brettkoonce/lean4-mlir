#!/usr/bin/env bash
# streamed_val_gate.sh — the gate for STREAMED ImageNet val: the val split streamed per pass from
# TWO batch-block producers must score the SAME images in the SAME order as the old one-time
# 30 GB drain did — same count, same top-5, same per-image bitmap — and the gate must go red on
# the two ways a streamed reader can be wrong without crashing.
#
#   scripts/streamed_val_gate.sh golden <net> <variant> <ckpt> <outdir>   # TODAY's binary (drain)
#   scripts/streamed_val_gate.sh test   <net> <variant> <ckpt> <outdir>   # the streamed binary
#
#   e.g. scripts/streamed_val_gate.sh golden resnet50-in160 lambaccdp8x64bce \
#            .lake/build/resnet50in160_lambaccdp8x64bce_ckpt_xla.bin runs/2026-09-22-streamed-val/r50
#
# ▶ WHAT IT GATES (2026-09-22, planning/streaming_val.md §4). `loadData` drained the whole val split
# into RAM (50,000 × 150,528 × 4 B = 30 GB held for the life of the run) and `evalScore` sliced it.
# Now `evalScore` reads a ROW SOURCE: the held buffer for the small datasets, and for ImageNet a
# per-pass stream from N shim producers, each emitting the batch BLOCKS `k, k+N, k+2N, …` of the
# val split (the split spec `validation[0:256]+validation[512:768]+…`), read round-robin so the
# global order is the single-producer order exactly, tail included.
#
# ⛔ WHY IT IS AN EQUALITY ON THE BITMAP AND NOT ON THE COUNT. A reader that starts its round-robin
# on the wrong producer scores every image once — same count, same top-5 to a rounding — with the
# labels of a different image. That is a mis-pairing a count cannot see; the bitmap sees every bit.
# A reader that drops the 80-row tail (the C4 bug class, 49,920 for months) shows in the count.
#
#   golden  G1, GN  today's binary at R=1 and R=N: the DRAINED eval's line + bitmap, kept on disk
#   test    T1, TN  the streamed binary at R=1 and R=N: must equal G1 (count, top-5, every bit),
#                   and TN must also PROVE it was sharded and streamed (the banner lines)
#           F_order CONTROL  LEAN_MLIR_VAL_FAULT=order — round-robin starts on producer 1.
#                   MUST DIFFER from G1 in the bitmap while the COUNT stays plausible.
#           F_tail  CONTROL  LEAN_MLIR_VAL_FAULT=tail — the partial last batch is dropped.
#                   MUST read 49,920. The fault knobs are gate-only, in the PJRT_FFI_FAULT style.
#
# ⚠⚠ DEFAULTS TO THE DETERMINISTIC SHIM (GATE_DET=1) and SHIM_DETERMINISM=1, for the reasons
# sharded_eval_gate.sh records: a 1-replica floor cannot see XLA's per-process autotuning, and a
# new byte-identity gate that does not pin the shim's op order has a control that is noise.
set -uo pipefail

MODE=${1:?usage: $0 golden|test <net> <variant> <ckpt> <outdir>}
NET=${2:?}; VARIANT=${3:?}; CKPT=${4:?}; OUT=${5:?}
DATA=${GATE_DATA:-data}
N=${GATE_REPLICAS:-4}
BIN=.lake/build/bin/score-checkpoint
mkdir -p "$OUT"

[ -f ffi/pjrt_ffi.c ] || { echo "run from the repo root"; exit 2; }
[ -x "$BIN" ] || { echo "✗ $BIN missing — lake build score-checkpoint"; exit 2; }
[ -f "$CKPT" ] || { echo "✗ no checkpoint at $CKPT"; exit 2; }
if [ ffi/pjrt_ffi.c -nt ffi/libpjrt_ffi.so ]; then
  echo "✗ ffi/libpjrt_ffi.so is OLDER than ffi/pjrt_ffi.c — gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so"; exit 2
fi
if [ -z "${PJRT_PLUGIN:-}" ]; then
  for p in .venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so \
           /home/skoonce/.venv-cuda/lib/python3.12/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so; do
    [ -f "$p" ] && { export PJRT_PLUGIN="$p"; break; }
  done
fi
SHIMENV=(SHIM_DETERMINISM=1)
if [ "${GATE_DET:-1}" = 1 ]; then
  DET=${DET_SHIM:-$OUT/detshim}
  . scripts/lib/gpu.sh
  det_shim_ensure "$DET" "$OUT/det_shim.log" || exit 2
  SHIMENV+=(LD_LIBRARY_PATH="$DET")
fi

echo "── streamed-val gate ($MODE): $NET $VARIANT, 1 vs $N replicas ──"
echo "   ckpt   $CKPT"
echo "   logs   $OUT"

arm() {  # arm <tag> <replicas> [extra env...]
  local tag=$1 reps=$2; shift 2
  local t0=$SECONDS
  env "${SHIMENV[@]}" PJRT_REPLICAS="$N" LEAN_MLIR_REPLICAS="$reps" PJRT_FFI_RESIDENT=1 \
      LEAN_MLIR_VARIANT="$VARIANT" LEAN_MLIR_CKPT="$CKPT" LEAN_MLIR_DUMP_CORRECT="$OUT/$tag" "$@" \
      "$BIN" "$NET" "$DATA" > "$OUT/$tag.log" 2>&1
  local rc=$?
  local line; line=$(grep -oE 'checkpoint: acc = [0-9]+/[0-9]+ = [0-9.]+%  top5 = [0-9]+/[0-9]+' "$OUT/$tag.log")
  printf "  %-7s %-30s %s  (%ds)\n" "$tag" "R=$reps ${*:-}" "${line:-<no accuracy line, rc=$rc>}" $((SECONDS - t0))
  printf '%s\n' "$line" > "$OUT/$tag.line"
}
same() { cmp -s "$OUT/$1.bin" "$OUT/$2.bin"; }
ndiff() { cmp -l "$OUT/$1.bin" "$OUT/$2.bin" 2>/dev/null | wc -l; }
count() { sed -E 's/.*acc = ([0-9]+)\/([0-9]+).*/\2/' "$OUT/$1.line"; }

if [ "$MODE" = golden ]; then
  arm G1 1
  arm GN "$N"
  for t in G1 GN; do [ -s "$OUT/$t.line" ] && [ -f "$OUT/$t.bin" ] || { echo "✗ $t produced nothing — $OUT/$t.log"; exit 1; }; done
  if [ "$(cat "$OUT/G1.line")" = "$(cat "$OUT/GN.line")" ] && same G1 GN; then
    echo "✓ golden  G1 == GN ($(count G1) images, bitmaps bit-identical) — kept in $OUT"
    exit 0
  fi
  echo "✗ golden  G1 != GN on TODAY's binary ($(ndiff G1 GN) images) — the drained sharded eval itself disagrees; nothing to stream against"
  exit 1
fi

[ -s "$OUT/G1.line" ] && [ -f "$OUT/G1.bin" ] || { echo "✗ no golden in $OUT — run '$0 golden …' with the pre-change binary first"; exit 2; }
G1=$(cat "$OUT/G1.line")
arm T1 1
arm TN "$N"
arm F_order 1 LEAN_MLIR_VAL_FAULT=order
arm F_tail  1 LEAN_MLIR_VAL_FAULT=tail
echo
FAIL=0
for t in T1 TN F_order F_tail; do
  [ -s "$OUT/$t.line" ] && [ -f "$OUT/$t.bin" ] || { echo "✗ arm $t produced no accuracy line or bitmap — see $OUT/$t.log"; FAIL=1; }
done
[ $FAIL -eq 0 ] || exit 1
for t in T1 TN; do
  grep -q 'val: STREAMED' "$OUT/$t.log" || { echo "✗ $t      did not stream (no 'val: STREAMED' banner) — a pass would be the old drain against itself"; FAIL=1; }
done
grep -q "$N replicas, SHARDED" "$OUT/TN.log" || { echo "✗ TN      was NOT sharded — no '$N replicas, SHARDED' line"; FAIL=1; }
for t in T1 TN; do
  if [ "$(cat "$OUT/$t.line")" = "$G1" ] && same G1 "$t"; then
    echo "✓ test    $t == G1 (count, top-5 and all $(stat -c %s "$OUT/G1.bin") per-image bits)"
  else
    echo "✗ test    $t != G1 — $(ndiff G1 "$t") images differ"; echo "          G1 $G1"; echo "          $t $(cat "$OUT/$t.line")"; FAIL=1
  fi
done
if ! grep -q 'LEAN_MLIR_VAL_FAULT=order' "$OUT/F_order.log"; then
  echo "✗ control F_order: the fault did not engage (no banner) — the gate is unproven"; FAIL=1
elif [ "$(count F_order)" != "$(count G1)" ]; then
  echo "✗ control F_order changed the COUNT ($(count F_order) vs $(count G1)) — this control is meant to be the count-invisible fault"; FAIL=1
elif same G1 F_order; then
  echo "✗ control F_order == G1 with the round-robin started on producer 1 — the bitmap CANNOT SEE a mis-ordered stream"; FAIL=1
else
  echo "✓ control F_order != G1 — same count, $(ndiff G1 F_order) images flipped: the mis-pairing a count cannot see"
fi
if ! grep -q 'LEAN_MLIR_VAL_FAULT=tail' "$OUT/F_tail.log"; then
  echo "✗ control F_tail: the fault did not engage (no banner)"; FAIL=1
elif [ "$(count F_tail)" = "$(count G1)" ]; then
  echo "✗ control F_tail still reads $(count G1) — dropping the tail was invisible"; FAIL=1
else
  echo "✓ control F_tail reads $(count F_tail), not $(count G1): the dropped tail shows in the denominator"
fi
echo
if [ $FAIL -eq 0 ]; then echo "✓ PASS — streamed val scores the drained eval's images, in its order, bit for bit; and the gate goes red on a mis-order and on a dropped tail"; exit 0; fi
echo "✗ FAIL — logs in $OUT"; exit 1
