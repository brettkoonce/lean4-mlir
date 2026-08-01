#!/usr/bin/env bash
# det_shim.sh — build a THROWAWAY PJRT shim whose compile options disable XLA
# autotuning, so that two runs of the same binary in two processes produce
# bit-identical parameters.
#
#   scripts/det_shim.sh [<outdir>]        # default /tmp/detshim
#   LD_LIBRARY_PATH=<outdir> .lake/build/bin/<binary> data
#
# ▶ WHY. `scripts/residency_gate.sh` and every other cross-PROCESS bit-identity
# check need a bit-exact A-vs-A floor to be readable at all. Handoff §3 records
# that XLA is bit-identical WITHIN a process but "not quite bit-stable" across
# them, and §2d.3's Finding 1 then measured the floor as bit-exact anyway — on
# ROCm. That does not carry to CUDA. Measured on ares 2026-08-01, R34, the
# committed shim: two runs of the SAME path differ in 191,094,739 of 255,477,624
# bytes at 10 steps, and in 2,647,005 at ONE step. The cause is autotuning
# choosing different convolution algorithms per process; AdamW then amplifies a
# single differing bit into most of the blob within a few steps (§2d.3's
# Finding 2, from the other direction).
#
# With this shim the same comparison is 0 bytes, and the gate reads as designed.
#
# ▶ WHY A SEPARATE SHIM RATHER THAN AN ENV VAR. $XLA_FLAGS is silently INERT on
# this path (handoff §4): the generated `pjrt_compile_options.h` embeds a fully
# populated DebugOptions that overrides the environment, with no warning. The
# flags therefore have to be captured AT GENERATION TIME, which is what this
# does — leaving the committed header and the committed shim untouched, because
# they are what ships and what every performance number is measured on.
#
# ▶ DO NOT MEASURE SPEED WITH THIS. Autotuning is off, so the executable is a
# different (and generally slower) program. It is an instrument for correctness
# comparisons only — the same caveat §4 attaches to the dump-enabled shim.
set -euo pipefail

OUT=${1:-/tmp/detshim}
VENV=${SHIM_PYTHON:-.venv/bin/python3}
FLAGS=${DET_FLAGS:---xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true}

[ -x "$VENV" ] || { echo "no python at $VENV (set \$SHIM_PYTHON)"; exit 2; }
[ -f ffi/pjrt_ffi.c ] || { echo "run from the repo root"; exit 2; }
mkdir -p "$OUT"

XLA_FLAGS="$FLAGS" "$VENV" scripts/gen_pjrt_compile_options.py > "$OUT/pjrt_compile_options.h"

# The flags landing is not something to assume: a name XLA no longer recognises
# would leave the header byte-identical to the committed one and this whole
# script would be an expensive no-op that still reported success.
if cmp -s ffi/pjrt_compile_options.h "$OUT/pjrt_compile_options.h"; then
  echo "✗ the generated options are IDENTICAL to the committed ones — the flags"
  echo "  did not land. Check the names against this XLA version:"
  echo "    $FLAGS"
  exit 1
fi

# The quoted #include in pjrt_ffi.c must resolve to $OUT first, hence the copy.
cp ffi/pjrt_ffi.c "$OUT/"
gcc -fPIC -O2 -shared "$OUT/pjrt_ffi.c" -I"$OUT" -Iffi -ldl -o "$OUT/libpjrt_ffi.so"

echo "✓ deterministic shim: $OUT/libpjrt_ffi.so"
echo "  flags: $FLAGS"
echo "  use:   LD_LIBRARY_PATH=$OUT <binary>      (the shim's rpath is RUNPATH, so this wins)"
