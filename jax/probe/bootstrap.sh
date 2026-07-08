#!/bin/bash
# One-command pod bootstrap for the runpod-probe kit. Idempotent — re-run freely.
# Encodes every lesson from the first live pod session (2026-07-08):
#   venv (image python/pip mismatch); iree pins stripped (live on iree.dev, not
#   PyPI); local disk not /workspace (MooseFS); tar --no-same-owner (root pods);
#   unset LD_LIBRARY_PATH (image CUDA shadows pip wheels -> silent cpu fallback);
#   .bashrc helper (new Jupyter shells forget cwd/venv/env); gpukill helper
#   (Ctrl-C leaves CUDA zombies that hold VRAM -- kill by nvidia-smi PID).
#
# Usage, from a fresh pod shell:
#   git clone -b runpod-probe --depth 1 https://github.com/brettkoonce/lean4-mlir.git lean4-jax
#   cd lean4-jax && bash jax/probe/bootstrap.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
REPO_DIR="$(pwd)"

echo "━━━ [1/5] venv + locked deps (iree pins stripped — probe pods don't need them)"
if [ ! -f /root/venv/bin/activate ]; then python -m venv /root/venv; fi
source /root/venv/bin/activate
LOCK=jax/requirements-cuda-lock.txt
if ! command -v nvidia-smi >/dev/null 2>&1; then LOCK=jax/requirements-rocm-lock.txt; fi
grep -v "^iree" "$LOCK" > /tmp/req-probe.txt
python -m pip install -q -r /tmp/req-probe.txt Pillow

echo "━━━ [2/5] device check"
unset LD_LIBRARY_PATH
python - <<'EOF'
import jax
d = jax.devices()[0]
assert d.platform == "gpu", f"not on a GPU: {d}"
print(f"  {len(jax.devices())}x {d.device_kind}")
EOF

echo "━━━ [3/5] shell helpers (~/.bashrc: auto cd+venv+env, gpukill)"
if ! grep -q "runpod-probe kit" /root/.bashrc 2>/dev/null; then
  cat >> /root/.bashrc <<EOF
# runpod-probe kit — every new shell lands ready
cd $REPO_DIR && source /root/venv/bin/activate && unset LD_LIBRARY_PATH
gpukill() {  # kill CUDA zombies (Ctrl-C survivors holding VRAM)
  for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    kill -9 "\$p"; done; sleep 2; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
}
EOF
fi

echo "━━━ [4/5] imagenette (local disk; ~1.5 GB first time)"
./download_imagenette.sh

echo "━━━ [5/5] ready."
echo "  ⚠ THIS shell first (bootstrap's venv dies with its subshell; new shells auto-set):"
echo "      source /root/venv/bin/activate && unset LD_LIBRARY_PATH"
echo "  Hardware anchors (fresh subprocess per bs, allocator env handled internally):"
echo "    python jax/probe/step_timer.py jax/probe/probe_resnet50_imagenette_noaug.py --batches 192 512 1024"
echo "    python jax/probe/step_timer.py jax/probe/probe_vit_tiny_imagenette.py --batches 192 512 1024"
echo "  Then:  python jax/probe/estimate.py --r50-sec-epoch <9408/img_s> --vit-sec-epoch <9408/img_s>"
echo "  (Full trainers probe_*.py run too, but their epoch loops are HOST-bound — see README.)"
echo "  Stuck VRAM after Ctrl-C? -> gpukill (new shells have it)"
