#!/bin/bash
# Build and run one trainer with the right env vars set — what `lake run train <net>` and
# `lake run mnist|cifar|imagenette` call for each net.
#
# Usage:
#   ./run.sh <trainer> [gpu] [backend] [-- args…]
#
#   trainer  - the lean_exe name (e.g. "resnet34-verified-adam"); a name without "-train" also
#              resolves to "<name>-train" if that target exists
#   gpu      - GPU index to expose (default: 0)
#   backend  - "cuda", "rocm", "xpu", "neuron" or "metal" (default: what scripts/platform/env.sh
#              detects on this box; metal has one GPU, so the index is ignored there; on neuron
#              the index is a NeuronCore); "llvm-cpu" or "vmvx" is the IREE lowerer on the CPU
#              (LEAN_MLIR_LOWERER=iree on a box without a GPU), with no device to select
#   args     - after `--`, passed to the trainer as is (`./run.sh pong-dqn 0 cuda -- mode=pixels`);
#              `lake run demo` uses this
#
# Examples:
#   ./run.sh mnist-cnn-verified
#   ./run.sh resnet34-verified-adam 1
#   ./run.sh resnet34-verified-adam 0 rocm
#   ./run.sh alphazero-ttt 0 cuda -- n=4 iters=40
#
# Output is teed to runs/<YYYY-MM-DD>-<trainer>/<trainer>.log (RUN_LOG_DIR overrides the
# directory). It used to land in the repo root as <trainer>.log, which .gitignore hid and
# every run re-created — 57 of them by 2026-09-08.

set -e -o pipefail

if [ -z "$1" ]; then
  echo "Usage: $0 <trainer> [gpu] [backend]" >&2
  echo "  trainer: a lean_exe name, e.g. mnist-cnn-verified, resnet34-verified-adam" >&2
  exit 1
fi

# Resolve the target: an exact lean_exe name first, then the "<name>" → "<name>-train"
# convention (e.g. "unet-brats" → "unet-brats-train").
trainer="$1"
if grep -q "^lean_exe «${trainer}»" lakefile.lean; then
  bin="$trainer"
elif grep -q "^lean_exe «${trainer}-train»" lakefile.lean; then
  bin="$trainer-train"
else
  bin="$trainer"   # not in the lakefile: a binary built elsewhere, run as found
fi

# Build first, so an edited source or a fresh pull never runs a stale binary; a no-op build
# is about a second. A box with only the binaries copied in (no `lake`) runs what it has.
if command -v lake >/dev/null && grep -q "^lean_exe «${bin}»" lakefile.lean; then
  lake -q build "$bin"   # quiet: no replay of cached warnings; errors still print
fi

binpath=".lake/build/bin/$bin"
if [ ! -x "$binpath" ]; then
  echo "Binary not found at $binpath. Try: lake build $bin" >&2
  exit 1
fi

# Split off the trainer's own arguments, everything after `--`.
own=(); pass=(); seen=0
for a in "${@:2}"; do
  if [ $seen = 0 ] && [ "$a" = "--" ]; then seen=1
  elif [ $seen = 0 ]; then own+=("$a")
  else pass+=("$a"); fi
done

# A backend word alone (`./run.sh r34 rocm`) is the backend on GPU 0.
case "${own[0]:-}" in cuda|rocm|xpu|neuron|metal|llvm-cpu|vmvx) [ -z "${own[1]:-}" ] && own=(0 "${own[0]}") ;; esac
gpu="${own[0]:-0}"
# The platform profile: the backend when none is named, and the device variable it selects on.
. scripts/platform/env.sh
backend="${own[1]:-$PLATFORM_BACKEND}"

logdir="${RUN_LOG_DIR:-runs/$(date +%F)-$(echo "$trainer" | tr '/' '_')}"
mkdir -p "$logdir"
logfile="$logdir/$(echo "$trainer" | tr '/' '_').log"

case "$backend" in
  cuda) export CUDA_VISIBLE_DEVICES="$gpu" ;;
  rocm) export HIP_VISIBLE_DEVICES="$gpu" ;;
  xpu)  export ZE_AFFINITY_MASK="$gpu" ;;
  neuron) export NEURON_RT_VISIBLE_CORES="$gpu" ;;   # a NeuronCore; a trn1 chip has two
  metal) ;;
  llvm-cpu|vmvx)      # the IREE lowerer on the CPU: no device variable, the index is moot
    export LEAN_MLIR_LOWERER="${LEAN_MLIR_LOWERER:-iree}" ;;   # the trainers default to XLA
  *)    echo "Unknown backend: $backend (expected cuda, rocm, xpu, neuron, metal, llvm-cpu or vmvx)" >&2; exit 1 ;;
esac

export IREE_BACKEND="$backend"

echo "→ $bin (GPU $gpu, $backend) → $logfile"
# pipefail: the trainer's exit status, not tee's, is this script's — `lake run` reports it.
"$binpath" "${pass[@]}" 2>&1 | tee "$logfile"
