#!/usr/bin/env bash
# allreduce_bytes_probe.sh — does the multi-card step's cross-card cost follow the NUMBER of
# all_reduces or the BYTES they exchange? ConvNeXt-S production render, synthetic batches, 4 cards.
#
#     scripts/probes/allreduce_bytes_probe.sh [outdir]      # cards idle; ~3 min a row
#
# Four rows, same binary and env: the render as committed (344 all_reduces, 200 MB), one
# concatenated all_reduce (same bytes), half the gradient exchanged, almost none exchanged.
# The last two are numerically wrong on purpose (scripts/probes/coalesce_allreduce.py).
# 4x RTX 3060, 2026-11-15: probe 354 / 354 / 331 / 314 ms, compute 331 / 333 / 311 / 291 —
# the count is free, the bytes cost ~0.2 ms per MB.
#
# ⚠ It OVERWRITES the committed render in place for each row (the driver opens it by name) and
# restores it with `git checkout` on exit, including on ^C. Refuses to start if that file is dirty.
# 12 GB cards need LEAN_MLIR_MEM_FRACTION=0.95 in the environment; 16 GB cards do not.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OUT="${1:-/tmp/allreduce_bytes_probe}"; mkdir -p "$OUT"
VARIANT=emaaccdp16xwxdroperfbf16
R="verified_mlir/convnextsin_${VARIANT}_train_step.mlir"
BIN=.lake/build/bin/convnext-s-imagenet-verified
git diff --quiet -- "$R" || { echo "⛔ $R has local changes — not overwriting it."; exit 1; }
[ -x "$BIN" ] || { echo "⛔ $BIN not built: lake build convnext-s-imagenet-verified"; exit 1; }
. scripts/platform/env.sh
trap 'git checkout -- "$R"' EXIT
row() {  # name ngroups skip_frac
  if [ "$2" != "-" ]; then
    SKIP_FRAC="$3" python3 scripts/probes/coalesce_allreduce.py "$R" "$OUT/$1.mlir" "$2" 2>/dev/null || return 1
    cat "$OUT/$1.mlir" > "$R"
  fi
  env "${PLATFORM_SHIMPY[@]}" PJRT_PLUGIN="$PLATFORM_PLUGIN" CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 LEAN_MLIR_VARIANT=$VARIANT LEAN_MLIR_BATCH=64 \
    LEAN_MLIR_EPOCHS=300 LEAN_MLIR_BASE_LR_U=4000 LEAN_MLIR_G2_STEPS=4992 SHIM_WORKERS=4 \
    LEAN_MLIR_BENCH_SYNTH=1 PJRT_FFI_TIMING=40 LEAN_MLIR_MAX_STEPS="${MAX_STEPS:-240}" \
    LEAN_MLIR_PROBE_WARM="${PROBE_WARM:-40}" "$BIN" data > "$OUT/$1.out" 2>&1
  local rc=$?
  git checkout -- "$R"
  printf '%-10s exit=%s  %s\n' "$1" "$rc" "$(grep -m1 'PROBE:' "$OUT/$1.out" | sed 's/ (median.*//')"
  grep 'COMPUTE' "$OUT/$1.out" | tail -2 | head -1
  grep -E '^  step [0-9]+/' "$OUT/$1.out" | tail -1
  grep -m1 -E 'Out of memory|uncaught' "$OUT/$1.out" | cut -c1-200
}
row rendered - 0
row one      1 0
row half     1 0.5
row none     1 0.9999
