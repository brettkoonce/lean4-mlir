#!/usr/bin/env bash
# sharded_eval_gate.sh — the gate for the SHARDED eval: one checkpoint scored on 1 device and on
# N must give the SAME correct count and the SAME per-image bitmap.
#
#   scripts/sharded_eval_gate.sh <net> <variant> <ckpt> [dataDir]      # nets: score-checkpoint's
#
#   e.g. scripts/sharded_eval_gate.sh vit adam64 .lake/build/vit_adam64_ckpt_xla.bin
#        SHIM_PYTHON=... scripts/sharded_eval_gate.sh vit-in emadp128x4wxclipdropbf16 \
#            .lake/build/vitin_emadp128x4wxclipdropbf16_ckpt_xla.bin
#
# ▶ WHAT IT GATES (2026-09-18). The per-epoch eval ran on replica 0 while the train step ran on
# all N: ~100 s/epoch at 91% util on GPU 0, GPUs 1-3 at 0% (the killed ConvNeXt run). The eval
# forward has no cross-replica op, so the SAME module runs sharded with no new render:
# `LowererSession.createDp` + `forwardF32Dp`, x split by rows, logits GATHERED back in row order
# (`d2h_gather` in ffi/pjrt_ffi.c).
#
# ⛔ WHY IT IS AN EQUALITY AND NOT A TOLERANCE. A sharded eval that scores the wrong subset — drops
# the ragged tail, double-counts a shard, returns four copies of shard 0 — does not crash. It
# prints a plausible accuracy. It is the one defect in this path that shows up as an ACCURACY
# CHANGE (`argmax10` capped ImageNet predictions at labels 0..9 for months). Same weights, same
# images, same compiled per-device program: only the sharding moves, so nothing may move with it.
#
# Five fresh processes through `score-checkpoint`, whose loop IS the trainer's (`evalScore`):
#
#   A1, A2  FLOOR    1 replica against ITSELF. If A1 != A2 the gate stops: an equality below
#                    would mean nothing. ⚠ It measures 1-REPLICA noise only — see GATE_DET below.
#   B       TEST     N replicas, resident HOLD mode — the path a PJRT_FFI_RESIDENT=1 run takes.
#   C       TEST     N replicas, COPYING path (`pjrt_ffi_invoke_f32_dp`) — the other read-back site.
#   D       CONTROL  B with PJRT_FFI_FAULT=3: every replica's slot filled from replica 0 — the
#                    replica-0-only read-back this replaces. MUST DIFFER. A gate nobody has seen
#                    fail is not a gate. First run: Imagenette ViT 51.77% → 46.27%, 1,438 of 3,925
#                    images flipped, a perfectly plausible number.
#
# B and C must also PROVE they were sharded (the "SHARDED" compile line). Otherwise a
# LEAN_MLIR_REPLICAS that was silently ignored would pass trivially, as 1 = 1.
#
# ⚠⚠ DEFAULTS TO THE DETERMINISTIC SHIM (GATE_DET=1), like residency_gate.sh and
# eval_residency_gate.sh, and for their reason. Measured 2026-09-18, ViT/ImageNet epoch-300 checkpoint
# on the COMMITTED shim: arms A1, A2, C, C2, B2 and B3 were all bit-exact against each other over
# 50,000 images, and B differed by 9 images (36172 vs 36175; top-5 identical). The 9 images were
# scattered across all four replicas and many invokes, flipped both ways, and did NOT reproduce
# when the identical path ran twice more. The shard/gather code is deterministic host code, so a
# difference that does not reproduce comes from XLA's per-process compile (autotuning), not the
# code under test. The 1-replica floor cannot see it, because the 4-replica executable is a
# separate compile. GATE_DET=0 runs the committed shim instead. That measures what production
# runs, but it can go red at the few-image level for no sharding reason.
set -uo pipefail

NET=${1:?usage: $0 <net> <variant> <ckpt> [dataDir]}
VARIANT=${2:?usage: $0 <net> <variant> <ckpt> [dataDir]}
CKPT=${3:?usage: $0 <net> <variant> <ckpt> [dataDir]}
DATA=${4:-data}
N=${GATE_REPLICAS:-4}
OUT=${GATE_OUT:-$(mktemp -d)}
BIN=.lake/build/bin/score-checkpoint
mkdir -p "$OUT"

[ -f ffi/pjrt_ffi.c ] || { echo "run from the repo root"; exit 2; }
[ -x "$BIN" ] || { echo "✗ $BIN missing — lake build score-checkpoint"; exit 2; }
[ -f "$CKPT" ] || { echo "✗ no checkpoint at $CKPT"; exit 2; }
# ⚠ The shim is dlopen'd and is NOT a lake target, so a stale one is silent. A shim without
# `pjrt_ffi_session_create_dp` fails B loudly, but one that has it and predates a fix would not.
if [ ffi/pjrt_ffi.c -nt ffi/libpjrt_ffi.so ]; then
  echo "✗ ffi/libpjrt_ffi.so is OLDER than ffi/pjrt_ffi.c — rebuild it:"
  echo "    gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so"
  exit 2
fi
if [ -z "${PJRT_PLUGIN:-}" ]; then
  for p in .venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so \
           /home/skoonce/.venv-cuda/lib/python3.12/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so; do
    [ -f "$p" ] && { export PJRT_PLUGIN="$p"; break; }
  done
fi
SHIMENV=()
if [ "${GATE_DET:-1}" = 1 ]; then
  DET=${DET_SHIM:-$OUT/detshim}
  . scripts/lib/gpu.sh
  det_shim_ensure "$DET" "$OUT/det_shim.log" || exit 2
  SHIMENV=(LD_LIBRARY_PATH="$DET")
fi

echo "── sharded-eval gate: $NET $VARIANT, 1 vs $N replicas ──"
echo "   ckpt   $CKPT"
echo "   shim   ${SHIMENV[*]:-ffi/libpjrt_ffi.so}   plugin ${PJRT_PLUGIN:-<compiled-in default>}"
echo "   logs   $OUT"
echo

arm() {  # arm <tag> <replicas> [extra env...]
  local tag=$1 reps=$2; shift 2
  local t0=$SECONDS
  env "${SHIMENV[@]}" PJRT_REPLICAS="$N" LEAN_MLIR_REPLICAS="$reps" \
      LEAN_MLIR_VARIANT="$VARIANT" LEAN_MLIR_CKPT="$CKPT" LEAN_MLIR_DUMP_CORRECT="$OUT/$tag" "$@" \
      "$BIN" "$NET" "$DATA" > "$OUT/$tag.log" 2>&1
  local rc=$?
  local line; line=$(grep -oE 'checkpoint: acc = [0-9]+/[0-9]+ = [0-9.]+%  top5 = [0-9]+/[0-9]+' "$OUT/$tag.log")
  printf "  %-4s %-34s %s  (%ds)\n" "$tag" "R=$reps ${*:-}" "${line:-<no accuracy line, rc=$rc>}" $((SECONDS - t0))
  printf -v "LINE_$tag" '%s' "$line"
}
same() { cmp -s "$OUT/$1.bin" "$OUT/$2.bin"; }
ndiff() { cmp -l "$OUT/$1.bin" "$OUT/$2.bin" 2>/dev/null | wc -l; }

arm A1 1 PJRT_FFI_RESIDENT=1
arm A2 1 PJRT_FFI_RESIDENT=1
arm B  "$N" PJRT_FFI_RESIDENT=1
arm C  "$N"
arm D  "$N" PJRT_FFI_RESIDENT=1 PJRT_FFI_FAULT=3
echo

FAIL=0
for t in A1 A2 B C D; do
  v="LINE_$t"
  [ -n "${!v}" ] && [ -f "$OUT/$t.bin" ] || { echo "✗ arm $t produced no accuracy line or bitmap — see $OUT/$t.log"; FAIL=1; }
done
[ $FAIL -eq 0 ] || exit 1

# FLOOR first: without it an equality below means nothing.
if [ "$LINE_A1" != "$LINE_A2" ] || ! same A1 A2; then
  echo "✗ FLOOR — two 1-replica runs of the same checkpoint differ ($(ndiff A1 A2) images)."
  echo "  Cross-process nondeterminism${SHIMENV:+ even on the deterministic shim}. NOT a sharding result."
  exit 1
fi
echo "✓ floor   A1 == A2 (bit-identical bitmaps across processes)"

for t in B C; do
  if ! grep -q "$N replicas, SHARDED" "$OUT/$t.log"; then
    echo "✗ $t      was NOT sharded — no '$N replicas, SHARDED' compile line. A pass would be 1 = 1."
    FAIL=1; continue
  fi
  v="LINE_$t"   # ⚠ indirect, not `eval echo` — that collapses the line's double spaces
  if [ "${!v}" = "$LINE_A1" ] && same A1 "$t"; then
    echo "✓ test    $t == A1 (count, top-5 and all $(stat -c %s "$OUT/A1.bin") per-image bits)"
  else
    echo "✗ test    $t != A1 — $(ndiff A1 "$t") images differ"
    echo "          A1 $LINE_A1"
    echo "          $t  ${!v}"
    FAIL=1
  fi
done

if ! grep -q 'PJRT_FFI_FAULT=3' "$OUT/D.log"; then
  echo "✗ control the fault did not engage (no PJRT_FFI_FAULT=3 banner) — the gate is unproven."
  FAIL=1
elif same A1 D; then
  echo "✗ control D == A1 even with every replica reading shard 0 — the comparison CANNOT SEE a"
  echo "          mis-gather here, so the passes above prove nothing. (Is the val set < 2 invokes?)"
  FAIL=1
else
  echo "✓ control D != A1 — the mis-gather flips $(ndiff A1 D) images and reads as a plausible"
  echo "          ${LINE_D#checkpoint: }"
fi

echo
if [ $FAIL -eq 0 ]; then
  echo "✓ PASS — the $N-replica eval scores the same images with the same per-image top-1 outcome"
  echo "  and the same top-5 count as the 1-replica eval; and the gate goes red on a mis-gather."
  exit 0
fi
echo "✗ FAIL — logs in $OUT"
exit 1
