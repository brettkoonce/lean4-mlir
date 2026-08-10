#!/usr/bin/env bash
# residency_gate_all.sh — run the §2d.3 residency gate across every converted
# training loop, as ONE command.
#
#   scripts/residency_gate_all.sh [<net>...]      # default: all of them
#   scripts/residency_gate_all.sh mnist-mlp r34   # a subset, by short name
#
# ▶ WHY THIS EXISTS. `residency_gate.sh` gates ONE binary, and getting a readable
# verdict out of it needs three things that are easy to get wrong and were being
# retyped every time: a deterministic shim on `LD_LIBRARY_PATH` (or the floor is
# noise and every verdict is meaningless — see `det_shim.sh`), the right
# `GATE_FAULT` for the net (a 1-ULP fault is ABSORBED by nets whose updates
# contract, so it is not a usable control on the SGD demo loops — §2d.3), and the
# per-net slug. Encoding that once, here, is the same move `TestShardCheck.lean`
# made for the three DP harnesses: one table, no copies.
#
# Everything per-net lives in the table below and nothing else in this script
# knows a net's name.
set -uo pipefail

# name | binary | slug | steps | fault mode | extra env
#
# ▶ FAULT MODE, and it is the entry most likely to be wrong for a NEW net:
#   1 = 1 ULP on one float. Correct for a CHAOTIC net — R34 under AdamW
#       amplifies it into ~70% of the blob within ten steps.
#   2 = drop one step's retained parameters (a stale handle). Required for a
#       CONTRACTIVE net: measured 2026-08-01, the MNIST MLP under plain SGD
#       absorbs a 1-ULP hit completely (1 byte at 3 steps, 0 at 10, on real data
#       as well as synthetic), so mode 1 there leaves the gate with NO transport
#       control and it correctly refuses as VACUOUS.
# If a new net's gate reports VACUOUS on the fault, try 2 before suspecting the
# implementation — and read §2d.3, because which one applies is a fact about the
# net's optimizer, not about residency.
#
# ⚠ AND "AdamW ⇒ chaotic ⇒ mode 1" IS NOT THE RULE. Measured 2026-08-01 when ViT
# was added: ViT-Tiny under the SAME AdamW schedule as R34 absorbs a 1-ULP fault
# down to **1 byte of 66,316,152** at 10 steps, single-device and at 4 replicas
# alike, where R34 amplifies one into ~184M of 255M. It passes on mode 1 only
# because 1 > 0 — one more step of contraction and it would report VACUOUS, which
# is a gate that could rot into a false green. So mode 2 (42,938,702 bytes) is the
# honest control for it. The transferable form of §2d.3's Finding 2: chaos is a
# property of the net, and the optimizer alone does not predict it.
#
# ⚠ THIRD DATA POINT, 2026-08-02: **RMSProp is CONTRACTIVE on both nets that use
# it**, which is a fact about the OPTIMIZER this time rather than the net — and it
# is the same optimizer whose ε placement the two nets sit on opposite sides of, so
# it was not a given. On mnv2 the 1-ULP fault lands at **2 bytes of 26,840,184** at
# 10 steps (measured uncontended; mode 2 gives 19,311,310), i.e. ViT's situation
# exactly: passing on mode 1 only because 2 > 0. Both rms rows below are mode 2.
# That is unsurprising in hindsight — the update normalises by √(mean-square + ε)
# and then damps through a momentum buffer — but "unsurprising in hindsight" is
# what §2d.3's Finding 2 also was.
NETS=(
  "mnist-linear|mnist-linear-verified|linear|10|2|"
  "mnist-mlp|mnist-mlp-verified|mlp|10|2|"
  "mnist-cnn|mnist-cnn-verified-xla|cnn|10|2|"
  "cifar8-bn-sgd|cifar8-bn-verified-xla|cifar8_bn|10|2|"
  "cifar8-bn-adam|cifar8-bn-verified-adam-xla|cifar8_bn|10|1|"
  "r34|resnet34-verified-adam-xla|resnet34|10|1|"
  "efficientnet|efficientnet-verified-adam|efficientnet|10|1|"
  "vit|vit-verified-adam|vit|10|2|"
  # ⚠ The EMA row gates a FOUR-region blob, `[θ|m|v|ema]`, where every other row here is three.
  # That is the whole reason it is worth a row of its own: `nResident` goes 3·P → 4·P, so this is
  # the only check that the shim's "n tensors in, n out, counts agree tensor for tensor" contract
  # actually holds at the wider layout rather than merely being argued to (`planning/ema.md` §4).
  # Mode 2 for the same reason the plain `vit` row is: ViT+AdamW is CONTRACTIVE and absorbs a 1-ULP
  # fault to ~1 byte of 66M, so mode 1 would pass only because 1 > 0 and could rot into a false
  # green. ⚠ It also cannot see the SHADOW's own correctness — eval-only state is structurally
  # invisible to this gate, exactly as hold-mode is. That is what the tracks-then-exceeds
  # trajectory gate is for.
  "vit-ema|vit-verified-adam|vit|10|2|LEAN_MLIR_VARIANT=ema"
  "mnv2-rms|mobilenetv2-verified-adam|mobilenetv2|10|2|LEAN_MLIR_VARIANT=rms"
  "enet-rms|efficientnet-verified-adam|efficientnet|10|2|LEAN_MLIR_VARIANT=rms"
)

# ⚠ A COLLISION THAT VOIDED THIS GATE ONCE (2026-08-02), and it costs a re-run to
# notice: `residency_gate.sh` deletes `<slug>_<variant>_ckpt_xla.bin` between each
# of its four passes, so running it while a TRAINER is live on the same slug AND
# variant lets that trainer's per-epoch checkpoint write land mid-gate. The
# symptom is not a crash — mnv2 reported a clean PASS and enet reported a
# saturated floor, and only the second was obviously wrong. Run this on an idle
# box, or at least `ps -eo comm | grep verified` first.

# The DATA-PARALLEL rows are NOT in the table above, because it is a single-device
# harness by construction (`residency_gate.sh` pins to $GATE_DEVICES, default "0")
# and a DP render bakes its replica count into `replica_groups`. Run them by hand
# on a box with the devices; both passed 2026-08-01:
#
#   GATE_DEVICES=0,1,2,3 LD_LIBRARY_PATH=$DET GATE_ALT=PJRT_FFI_RESIDENT=1 \
#     GATE_FAULT=PJRT_FFI_FAULT=2 LEAN_MLIR_VARIANT=adamdp32x4 LEAN_MLIR_BATCH=32 \
#     LEAN_MLIR_REPLICAS=4 PJRT_REPLICAS=4 \
#     scripts/residency_gate.sh vit-verified-adam vit 10

DET=${DET_SHIM:-/tmp/residency_detshim}
OUT=${GATE_OUT:-$(mktemp -d)}
mkdir -p "$OUT"

[ -f scripts/residency_gate.sh ] || { echo "run from the repo root"; exit 2; }

echo "── §2d.3 residency gate — all converted training loops ──"
echo "   scratch $OUT"

# The deterministic shim is a PREREQUISITE, not an option: without it the A-vs-A
# floor is autotuning noise and no verdict below can be read. Built once, reused.
if [ ! -f "$DET/libpjrt_ffi.so" ] || [ ffi/pjrt_ffi.c -nt "$DET/libpjrt_ffi.so" ]; then
  echo "   building the deterministic shim in $DET ..."
  scripts/det_shim.sh "$DET" > "$OUT/det_shim.log" 2>&1 || {
    echo "   ✗ det_shim.sh failed:"; cat "$OUT/det_shim.log"; exit 2; }
fi
echo "   shim    $DET/libpjrt_ffi.so"
echo

WANT=("$@")
declare -a NAMES VERDICTS
FAILED=0

for row in "${NETS[@]}"; do
  IFS='|' read -r name bin slug steps fault extra <<< "$row"
  if [ ${#WANT[@]} -gt 0 ]; then
    match=0; for w in "${WANT[@]}"; do [ "$w" = "$name" ] && match=1; done
    [ $match -eq 1 ] || continue
  fi
  if [ ! -x ".lake/build/bin/$bin" ]; then
    printf "  %-15s ⚠ SKIP — not built (lake build %s)\n" "$name" "$bin"
    NAMES+=("$name"); VERDICTS+=("SKIP (not built)"); continue
  fi
  printf "  %-15s gating (%s, fault mode %s) ... " "$name" "$bin" "$fault"
  # shellcheck disable=SC2086
  env $extra LD_LIBRARY_PATH="$DET" \
      GATE_OUT="$OUT/$name" \
      GATE_ALT=PJRT_FFI_RESIDENT=1 \
      GATE_FAULT="PJRT_FFI_FAULT=$fault" \
      scripts/residency_gate.sh "$bin" "$slug" "$steps" > "$OUT/$name.log" 2>&1
  rc=$?
  line=$(grep -E "^(✓|✗|⚠)" "$OUT/$name.log" | head -1)
  bytes=$(grep -oE "over [0-9]+ bytes" "$OUT/$name.log" | head -1 | tr -d 'a-z ')
  if [ $rc -eq 0 ]; then
    printf "✓ PASS (%s bytes bit-identical)\n" "${bytes:-?}"
    NAMES+=("$name"); VERDICTS+=("PASS  ${bytes:-?} bytes")
  else
    printf "✗ %s\n" "${line:-rc=$rc}"
    NAMES+=("$name"); VERDICTS+=("FAIL  ${line:-rc=$rc}")
    FAILED=1
  fi
done

echo
echo "── scoreboard ──"
for i in "${!NAMES[@]}"; do printf "   %-15s %s\n" "${NAMES[$i]}" "${VERDICTS[$i]}"; done
echo
if [ $FAILED -eq 0 ]; then
  echo "✓ every gated loop is bit-identical to the copying path, against a bit-exact"
  echo "  A-vs-A floor, with both controls firing. Logs in $OUT."
  exit 0
fi
echo "✗ at least one loop did not pass — read its log in $OUT before anything else."
echo "  A VACUOUS verdict is a HARNESS problem, not a residency one: check the"
echo "  fault mode in this script's table (see the comment above it)."
exit 1
