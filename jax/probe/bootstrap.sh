#!/bin/bash
# One-command pod bootstrap for the runpod-probe kit. Idempotent — re-run freely.
# Encodes every lesson from the first live pod session (2026-07-08):
#   venv (image python/pip mismatch), iree pins stripped (live on iree.dev, not
#   PyPI), local disk not /workspace (MooseFS), tar --no-same-owner (root pods).
#
# Usage, from a fresh pod shell:
#   git clone -b runpod-probe --depth 1 https://github.com/brettkoonce/lean4-mlir.git lean4-jax
#   cd lean4-jax && bash jax/probe/bootstrap.sh
#   source /root/venv/bin/activate      # for subsequent shells
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

echo "━━━ [1/4] venv + locked deps (iree pins stripped — probe pods don't need them)"
if [ ! -f /root/venv/bin/activate ]; then python -m venv /root/venv; fi
source /root/venv/bin/activate
LOCK=jax/requirements-cuda-lock.txt
if ! command -v nvidia-smi >/dev/null 2>&1; then LOCK=jax/requirements-rocm-lock.txt; fi
grep -v "^iree" "$LOCK" > /tmp/req-probe.txt
python -m pip install -q -r /tmp/req-probe.txt Pillow

echo "━━━ [2/4] device check"
# Pod images set LD_LIBRARY_PATH to their system CUDA, which shadows the pip
# nvidia wheels (symptom: "The cuSPARSE library was not found" -> cpu fallback).
# The wheels self-resolve via RPATH; empty LD_LIBRARY_PATH is the correct state.
unset LD_LIBRARY_PATH
python - <<'EOF'
import jax
d = jax.devices()[0]
assert d.platform == "gpu", f"not on a GPU: {d}"
print(f"  {len(jax.devices())}x {d.device_kind}")
EOF
echo "  NB new shells need: source /root/venv/bin/activate && unset LD_LIBRARY_PATH"

echo "━━━ [3/4] imagenette (local disk; ~1.5 GB first time)"
./download_imagenette.sh

echo "━━━ [4/4] ready. Probes (epoch 1 = XLA compile, discard; Ctrl-C freely):"
echo "  python jax/probe/probe_resnet50_imagenette.py"
echo "  python jax/probe/probe_vit_tiny_imagenette.py"
echo "  python jax/probe/probe_resnet34_imagenette.py"
