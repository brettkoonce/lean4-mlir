#!/usr/bin/env bash
# scripts/platform/neuron_aot.sh — compile verified_mlir/ renders for Trainium with neuronx-cc, on
# any Linux x86_64 box: no Neuron device, driver or runtime needed. deploy/TRAINIUM.md §2.
#
#   scripts/platform/neuron_aot.sh                       # the suite's tier-2 graphs (tier2_artifacts.tsv)
#   scripts/platform/neuron_aot.sh mnist_cnn_train_step resnet34_adam_train_step   # these renders
#   NEURON_AOT_TARGET=trn2 NEURON_AOT_JOBS=8 scripts/platform/neuron_aot.sh
#
# Neuron's PJRT plugin compiles by handing the program to `neuronx-cc compile --framework=XLA
# model.mlir` (libneuronxla/neuron_cc_wrapper.py), so a render that compiles here has passed the
# compiler's op coverage and lowering for that target before any paid hour. It does not show
# that the plugin's own pre-passes accept the module, or that the result computes the right
# numbers: the box's `smoke` and `suite` rungs do that.
#
# Each render gets the entry rename ffi/pjrt_ffi.c's rename_entry_to_main does (one func.func,
# renamed to @main) and the flags the trainers will run with: --auto-cast=none, so an f32 graph
# stays f32 (Neuron's default casts f32 matmuls to bf16), plus $NEURON_CC_FLAGS.
#
# Needs neuronx-cc: .venv-neuron (deploy/trainium_setup.sh venv) or $NEURON_AOT_CC. Writes
# runs/platform/<date>-<host>-neuron-aot-<target>/results.tsv (render, status, seconds, NEFF
# bytes, first error line; -2, -3 … for a second run that day), manifest.tsv (compiler, target,
# flags) and one log per render under logs/; exit 1 if any render failed.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

TARGET=${NEURON_AOT_TARGET:-trn1}
JOBS=${NEURON_AOT_JOBS:-$(( $(nproc) / 8 > 0 ? $(nproc) / 8 : 1 ))}
CC=${NEURON_AOT_CC:-}
if [ -z "$CC" ]; then
  for c in .venv-neuron/bin/neuronx-cc "$(command -v neuronx-cc 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] && { CC=$c; break; }
  done
fi
[ -n "$CC" ] || { echo "no neuronx-cc: run deploy/trainium_setup.sh venv, or set NEURON_AOT_CC" >&2; exit 2; }
CC=$(cd "$(dirname "$CC")" && pwd)/$(basename "$CC")

if [ $# -gt 0 ]; then RENDERS=("$@")
else mapfile -t RENDERS < <(grep -v '^#' scripts/platform/tier2_artifacts.tsv | cut -f1)
fi

OUT=${NEURON_AOT_OUT:-}
if [ -z "$OUT" ]; then
  # A fresh directory per run, as check.sh does, so a second run cannot overwrite a first's logs.
  base=runs/platform/$(date +%F)-$(hostname -s)-neuron-aot-$TARGET; OUT=$base; k=2
  while [ -e "$OUT" ]; do OUT=$base-$k; k=$((k+1)); done
fi
mkdir -p "$OUT/logs"
OUT=$(cd "$OUT" && pwd)
# The compiler's scratch and the NEFFs (tens of MB a graph) go in a temporary directory, removed
# on exit; NEURON_AOT_KEEP=1 keeps them under $OUT/work instead.
if [ "${NEURON_AOT_KEEP:-0}" = 1 ]; then WORK=$OUT/work; mkdir -p "$WORK"
else WORK=$(mktemp -d "${TMPDIR:-/tmp}/neuron-aot.XXXXXX"); trap 'rm -rf "$WORK"' EXIT
fi
CCV=$("$CC" --version 2>&1 | sed -n 's/.*version //p' | head -1)
printf 'neuronx-cc\t%s\ntarget\t%s\nflags\t--auto-cast=none %s\nhost\t%s (%s cores)\nrepo\t%s\n' "$CCV" "$TARGET" \
  "${NEURON_CC_FLAGS:-}" "$(hostname -s)" "$(nproc)" "$(git rev-parse --short HEAD 2>/dev/null)" > "$OUT/manifest.tsv"
echo "neuronx-cc $CCV, target $TARGET, $JOBS at a time → $OUT"

# One render: rename the entry, compile in its own work directory, print one results row.
compile_one () {
  local name=$1 src=verified_mlir/$1.mlir w=$WORK/$1 log=$OUT/logs/$1.log t0 rc secs bytes err
  [ -f "$src" ] || { printf '%s\tMISSING\t-\t-\tno %s\n' "$name" "$src"; return; }
  if [ "$(grep -c 'func\.func @' "$src")" -ne 1 ]; then
    printf '%s\tSKIP\t-\t-\tnot exactly one func.func (the shim refuses it too)\n' "$name"; return
  fi
  rm -rf "$w"; mkdir -p "$w"
  sed -E '0,/func\.func @[A-Za-z0-9_.]+\(/s//func.func @main(/' "$src" > "$w/model.mlir"
  t0=$(date +%s.%N)
  # shellcheck disable=SC2086
  (cd "$w" && "$CC" compile --framework=XLA model.mlir --target="$TARGET" --auto-cast=none \
      ${NEURON_CC_FLAGS:-} --output model.neff) > "$log" 2>&1
  rc=$?
  secs=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
  if [ $rc -eq 0 ] && [ -s "$w/model.neff" ]; then
    bytes=$(stat -c %s "$w/model.neff")
    printf '%s\tPASS\t%s\t%s\t\n' "$name" "$secs" "$bytes"
  else
    err=$(grep -a -m1 -E 'ERROR|Error|error:|Check failed|Unsupported|not supported' "$log" | cut -c1-200 | tr '\t' ' ')
    printf '%s\tFAIL\t%s\t-\t%s\n' "$name" "$secs" "${err:-exit $rc, see logs/$name.log}"
  fi
}
export -f compile_one
export OUT WORK CC TARGET

{
  printf 'render\tstatus\tseconds\tneff_bytes\terror\n'
  printf '%s\n' "${RENDERS[@]}" | xargs -P "$JOBS" -I{} bash -c 'compile_one "$1"' _ {} | sort
} | tee "$OUT/results.tsv" | column -t -s $'\t'

pass=$(grep -c $'\tPASS\t' "$OUT/results.tsv")
total=$(( $(wc -l < "$OUT/results.tsv") - 1 ))
echo "$pass of $total compiled for $TARGET"
[ "$pass" -eq "$total" ]
