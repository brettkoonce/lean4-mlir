# scripts/platform/env.sh — which GPU platform this is, and what it runs with. ONE place.
#
#     . scripts/platform/env.sh          # from the repo root; sets PLATFORM_* and nothing else
#
# Sourced by scripts/platform/check.sh (the suite), scripts/jobs/_box.sh (every ImageNet job conf
# and the two tie gates), run.sh (the demo tiers' device variable), deploy/orin_imagenette.sh (the
# Orin tier) and the lakefile's backend detection, so that adding a platform is one arm of the
# `case` below, one `scripts/platform/expected/<backend>-<gpu>.txt` baseline and one runbook page
# in deploy/. Before this file the same facts sat in five places and disagreed on the Jetson's
# allocator defaults.
#
# Safe under `set -eu -o pipefail` (run.sh sources it so). What it sets (every one overridable by
# exporting it first; $PJRT_PLUGIN and $IREE_BACKEND are
# honoured as the older spellings of the plugin and the backend):
#   PLATFORM_BACKEND  cuda | rocm | xpu — the vendor SMI on PATH decides, nvidia first
#   PLATFORM_KIND     desktop | tegra — a Jetson shares one DRAM between host and device
#   PLATFORM_HOST     ares | 3060 | orin | <hostname> — the box profile the job confs key on
#   PLATFORM_GPU      the slug the suite's baselines and PLATFORMS.md are keyed by (rtx-4060-ti,
#                     orin-nano, …), "" when no tool on the box can name the device
#   PLATFORM_DEVVAR   CUDA_VISIBLE_DEVICES | HIP_VISIBLE_DEVICES | ZE_AFFINITY_MASK
#   PLATFORM_PLUGIN   the PJRT plugin .so the shim dlopens ("" when the platform has none installed)
#   PLATFORM_PY       the python the JAX reference side runs with ("" on a box with no JAX)
#   PLATFORM_SHIMPY   array, SHIM_PYTHON=<py> where the trainers' shim python is not .venv's
#   PLATFORM_RUNBOOK  the deploy/ page for this platform ("" for the training boxes)
# and on a Jetson, when unset, the unified-memory allocator recipe the runbook arrived at:
#   LEAN_MLIR_PREALLOCATE=1  LEAN_MLIR_MEM_FRACTION=0.25        (deploy/ORIN.md §2; per net, §4)
#
# A box's own machine-local file (the Orin's ~/pjrt/orin_env.sh, written by deploy/orin_setup.sh:
# plugin path, the Tegra cuDNN on LD_LIBRARY_PATH) is sourced here when it exists and PJRT_PLUGIN
# is not already set, so sourcing this file is enough on every platform.

PLATFORM_KIND=desktop
[ -f /etc/nv_tegra_release ] && PLATFORM_KIND=tegra

# The backend: an explicit PLATFORM_BACKEND or IREE_BACKEND wins; else the SMI on PATH.
if [ -z "${PLATFORM_BACKEND:-}" ]; then
  if [ -n "${IREE_BACKEND:-}" ]; then PLATFORM_BACKEND=$IREE_BACKEND
  elif [ "$PLATFORM_KIND" = tegra ] || command -v nvidia-smi > /dev/null 2>&1; then PLATFORM_BACKEND=cuda
  elif command -v rocm-smi > /dev/null 2>&1; then PLATFORM_BACKEND=rocm
  elif command -v xpu-smi > /dev/null 2>&1; then PLATFORM_BACKEND=xpu
  else PLATFORM_BACKEND=cuda
  fi
fi

_platform_slug () { tr 'A-Z' 'a-z' | sed -E 's/nvidia geforce |nvidia |amd radeon |intel\(r\) |jetson | developer kit//g; s/[^a-z0-9]+/-/g; s/^-|-$//g'; }
PLATFORM_PY=""; PLATFORM_SHIMPY=(); PLATFORM_RUNBOOK=""; PLATFORM_GPU=${PLATFORM_GPU:-}
case "$PLATFORM_BACKEND:$PLATFORM_KIND" in
  cuda:tegra)
    # A Jetson. The plugin is the sm_87 build deploy/orin_setup.sh stages under ~/pjrt, with the
    # Tegra cuDNN it was compiled against; the board has no JAX.
    PLATFORM_HOST=orin
    PLATFORM_DEVVAR=CUDA_VISIBLE_DEVICES
    PLATFORM_RUNBOOK=deploy/ORIN.md
    if [ -z "${PJRT_PLUGIN:-}" ] && [ -f "$HOME/pjrt/orin_env.sh" ]; then . "$HOME/pjrt/orin_env.sh"; fi
    PLATFORM_PLUGIN=${PJRT_PLUGIN:-$HOME/pjrt/pjrt_c_api_gpu_plugin.so}
    [ -z "$PLATFORM_GPU" ] && [ -r /proc/device-tree/model ] && PLATFORM_GPU=$(tr -d '\0' < /proc/device-tree/model | _platform_slug) || true
    export LEAN_MLIR_PREALLOCATE=${LEAN_MLIR_PREALLOCATE:-1} LEAN_MLIR_MEM_FRACTION=${LEAN_MLIR_MEM_FRACTION:-0.25}
    ;;
  cuda:*)
    PLATFORM_DEVVAR=CUDA_VISIBLE_DEVICES
    # The two training boxes, by which plugin is installed: ares runs the repo's pinned .venv
    # (jax_cuda12_pjrt); the 3060 box runs /home/skoonce/.venv-cuda (xla_cuda13, driver 610 / CUDA
    # 13.3), and there SHIM_PYTHON is required because that box's .venv/bin/python3 execs a
    # retired venv. Any other CUDA box: the plugin the shim would find on its own.
    _ares=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
    _3060=/home/skoonce/.venv-cuda/lib/python3.12/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so
    if [ -f "$_ares" ]; then
      PLATFORM_HOST=ares; PLATFORM_PLUGIN=${PJRT_PLUGIN:-$_ares}; PLATFORM_PY=.venv/bin/python3
    elif [ -f "$_3060" ]; then
      PLATFORM_HOST=3060; PLATFORM_PLUGIN=${PJRT_PLUGIN:-$_3060}
      PLATFORM_PY=/home/skoonce/.venv-cuda/bin/python3; PLATFORM_SHIMPY=(SHIM_PYTHON="$PLATFORM_PY")
    else
      PLATFORM_HOST=$(hostname -s 2>/dev/null || echo cuda)
      PLATFORM_PLUGIN=${PJRT_PLUGIN:-}
      [ -z "$PLATFORM_PLUGIN" ] && for _p in "$_ares" "$_3060"; do [ -f "$_p" ] && { PLATFORM_PLUGIN=$_p; break; }; done
      [ -x .venv/bin/python3 ] && PLATFORM_PY=.venv/bin/python3
    fi
    [ -z "$PLATFORM_GPU" ] && command -v nvidia-smi > /dev/null 2>&1 && \
      PLATFORM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null | grep -v "^No devices" | head -1 | _platform_slug) || true
    ;;
  rocm:*)
    # The jax rocm plugin in the pinned .venv (jax-rocm7-pjrt); the code paths stay, the next run
    # is the MI300 cloud box (planning/platform_integration.md §5).
    PLATFORM_HOST=$(hostname -s 2>/dev/null || echo rocm)
    PLATFORM_DEVVAR=HIP_VISIBLE_DEVICES
    PLATFORM_PLUGIN=${PJRT_PLUGIN:-.venv/lib/python3.12/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so}
    [ -x .venv/bin/python3 ] && PLATFORM_PY=.venv/bin/python3
    [ -z "$PLATFORM_GPU" ] && command -v rocm-smi > /dev/null 2>&1 && \
      PLATFORM_GPU=$(rocm-smi --showproductname 2>/dev/null | sed -n 's/.*Card series:[[:space:]]*//p' | head -1 | _platform_slug) || true
    ;;
  xpu:*)
    # Intel's OpenXLA PJRT plugin, when the Arc Pro B60 arrives (planning/platform_integration.md §6):
    # the plugin lives in its own venv, never the pinned .venv, and PJRT_PLUGIN names it until the
    # path is settled here.
    PLATFORM_HOST=$(hostname -s 2>/dev/null || echo xpu)
    PLATFORM_DEVVAR=ZE_AFFINITY_MASK
    PLATFORM_PLUGIN=${PJRT_PLUGIN:-}
    ;;
  *)
    echo "scripts/platform/env.sh: unknown backend '$PLATFORM_BACKEND' (cuda | rocm | xpu)" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
unset _ares _3060 _p
export PLATFORM_BACKEND PLATFORM_KIND PLATFORM_HOST PLATFORM_GPU PLATFORM_DEVVAR PLATFORM_PLUGIN PLATFORM_PY PLATFORM_RUNBOOK
