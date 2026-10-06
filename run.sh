#!/bin/bash
# Run one trainer with the right env vars set — what `lake run mnist|cifar|imagenette` calls
# for each net.
#
# Usage:
#   ./run.sh <trainer> [gpu] [backend]
#
#   trainer  - the lean_exe name (e.g. "resnet34-verified-adam"); a name without "-train" also
#              resolves to "<name>-train" if that binary exists
#   gpu      - GPU index to expose (default: 0)
#   backend  - "cuda", "rocm" or "xpu" (default: what scripts/platform/env.sh detects on this box)
#
# Examples:
#   ./run.sh mnist-cnn-verified
#   ./run.sh resnet34-verified-adam 1
#   ./run.sh resnet34-verified-adam 0 rocm
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

# Normalize the trainer name. Accept an exact binary name first (e.g. the
# verified trainers "cifar8-bn-verified-adam"), then fall back to the
# "<name>" → "<name>-train" convention (e.g. "resnet34" → "resnet34-train").
trainer="$1"
if [ -x ".lake/build/bin/$trainer" ]; then
  bin="$trainer"
else
  case "$trainer" in
    *-train) bin="$trainer" ;;
    *)       bin="$trainer-train" ;;
  esac
fi

binpath=".lake/build/bin/$bin"
if [ ! -x "$binpath" ]; then
  echo "Binary not found at $binpath. Try: lake build $bin" >&2
  exit 1
fi

gpu="${2:-0}"
# The platform profile: the backend when none is named, and the device variable it selects on.
. scripts/platform/env.sh
backend="${3:-$PLATFORM_BACKEND}"

logdir="${RUN_LOG_DIR:-runs/$(date +%F)-$(echo "$trainer" | tr '/' '_')}"
mkdir -p "$logdir"
logfile="$logdir/$(echo "$trainer" | tr '/' '_').log"

case "$backend" in
  cuda) export CUDA_VISIBLE_DEVICES="$gpu" ;;
  rocm) export HIP_VISIBLE_DEVICES="$gpu" ;;
  xpu)  export ZE_AFFINITY_MASK="$gpu" ;;
  *)    echo "Unknown backend: $backend (expected cuda, rocm or xpu)" >&2; exit 1 ;;
esac

export IREE_BACKEND="$backend"

echo "→ $bin (GPU $gpu, $backend) → $logfile"
# pipefail: the trainer's exit status, not tee's, is this script's — `lake run` reports it.
"$binpath" 2>&1 | tee "$logfile"
