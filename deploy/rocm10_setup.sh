#!/usr/bin/env bash
# ROCm 10.1 for the 7900 XTX (gfx1100) as pip wheels, in .venv-rocm10 — no root, no apt, no /opt/rocm.
#
#   bash deploy/rocm10_setup.sh          # from the repo root; makes .venv-rocm10, checks the card
#   . deploy/rocm10_env.sh               # then, in each shell that trains
#   ./run.sh mobilenetv4-verified-adam 0 rocm
#
# ROCm 10 is TheRock's build of the userspace (HIP, hipBLASLt, MIOpen, the compiler). It never touches
# the kernel driver: the distribution's in-tree amdgpu (any kernel >= ~6.2 has gfx1100) is all it needs.
# The one thing that takes root, once, on a box that is not yours: your user in the `render` (and
# `video`) group, for /dev/kfd and /dev/dri/renderD*. Containers that pass the GPU through have it.
#
# Measured on minerva against the pinned 7.2.4 (deploy/ROCM_7900XTX.md §1,
# runs/2026-10-25-rocm10-bench/): the wheels and the debs time the same to 1 ms; 10.1 is ~13 % slower
# on ResNet-34 and 3–4 % faster on the depthwise nets. rocm10_env.sh's MIOpen switches are required,
# not tuning: without them ViT fails to compile and bf16 grouped convolutions return garbage.
set -euo pipefail

VENV=.venv-rocm10
ROCM_PIN="rocm[libraries,device-gfx1100]==10.1.0"
PLUGIN_PIN="jax-rocm10-pjrt==0.11.1+rocm10.1.0"
INDEX=https://stable.repo.amd.com/rocm/whl-next/

[ -f lakefile.lean ] || { echo "run from the repo root"; exit 1; }

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "device access"
ok=1
for d in /dev/kfd /dev/dri/renderD*; do
  [ -e "$d" ] || { echo "  missing $d — is amdgpu loaded? (lsmod | grep amdgpu)"; ok=0; continue; }
  [ -r "$d" ] && [ -w "$d" ] && echo "  ok $d" || { echo "  no rw on $d — add your user to its group: $(stat -c %G "$d")"; ok=0; }
done
[ $ok = 1 ] || echo "  (continuing: the install needs no device; training does)"

say "python"
PY=$(command -v python3.12 || command -v python3)
"$PY" -c 'import sys; assert sys.version_info >= (3, 12), sys.version' \
  || { echo "need Python 3.12+ (the wheels and the plugin path are cp312)"; exit 1; }
echo "  $PY ($("$PY" --version))"

say "$VENV: $ROCM_PIN + $PLUGIN_PIN"
[ -x $VENV/bin/python3 ] || "$PY" -m venv --without-pip $VENV
"$PY" -m pip --python $VENV/bin/python3 install --index-url $INDEX --extra-index-url https://pypi.org/simple \
  "$ROCM_PIN" "$PLUGIN_PIN"

PLUGIN=$(ls $VENV/lib/python3*/site-packages/jax_plugins/xla_rocm10/xla_rocm_plugin.so 2>/dev/null | head -1)
[ -n "$PLUGIN" ] || { echo "plugin .so not found under $VENV"; exit 1; }
echo "  plugin: $PLUGIN"

say "the card, as the wheels' runtime sees it"
$VENV/bin/rocminfo 2>/dev/null | grep -m1 -E 'Name:[[:space:]]+gfx' || echo "  rocminfo sees no gfx agent (device access above)"
$VENV/bin/amd-smi static --asic 2>/dev/null | grep -m1 MARKET_NAME || true

cat <<EOF

Done. In each shell that trains:
  . deploy/rocm10_env.sh
  scripts/platform/check.sh                         # the suite first
  ./run.sh mobilenetv4-verified-adam 0 rocm
EOF
