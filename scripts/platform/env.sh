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
#   PLATFORM_BACKEND  cuda | rocm | xpu | neuron | metal — metal on macOS, else the vendor SMI on
#                     PATH decides, nvidia first (neuron: a /dev/neuron0 device); llvm-cpu | vmvx
#                     only when asked for (IREE_BACKEND), the IREE lowerer on a box without a GPU
#   PLATFORM_KIND     desktop | tegra — a Jetson shares one DRAM between host and device
#   PLATFORM_HOST     ares | 3060 | orin | <hostname> — the box profile the job confs key on
#   PLATFORM_GPU      the slug the suite's baselines and PLATFORMS.md are keyed by (rtx-4060-ti,
#                     orin-nano, …), "" when no tool on the box can name the device
#   PLATFORM_DEVVAR   CUDA_VISIBLE_DEVICES | HIP_VISIBLE_DEVICES | ZE_AFFINITY_MASK |
#                     NEURON_RT_VISIBLE_CORES | "" (metal: one GPU, nothing to select)
#   PLATFORM_PLUGIN   the PJRT plugin .so the shim dlopens ("" when the platform has none installed)
#   PLATFORM_PY       the python the JAX reference side runs with ("" on a box with no JAX)
#   PLATFORM_SHIMPY   array, SHIM_PYTHON=<py> where the trainers' shim python is not .venv's
#   PLATFORM_RUNBOOK  the deploy/ page for this platform ("" for the training boxes)
#   PLATFORM_TEST_TIMEOUT  seconds the suite gives one test (300; 900 on a Jetson, whose six A78AE
#                     cores take ~265 s to compile the bf16 MobileNetV2 step alone, clocks pinned or not)
# and on a Jetson, when unset, the unified-memory allocator recipe the runbook arrived at:
#   LEAN_MLIR_PREALLOCATE=1  LEAN_MLIR_MEM_FRACTION=0.25        (deploy/ORIN.md §2; per net, §4)
# and on a Mac, when unset, the client options jax-graft's own python passes (deploy/MAC.md):
#   LEAN_MLIR_PJRT_PLATFORM=METAL  LEAN_MLIR_ALLOCATOR=platform  LEAN_MLIR_PREALLOCATE=0
#   LEAN_MLIR_MEM_FRACTION=1.0  LEAN_MLIR_STAGE_H2D=0
#
# A box's own machine-local file (the Orin's ~/pjrt/orin_env.sh, written by deploy/orin_setup.sh:
# plugin path, the Tegra cuDNN on LD_LIBRARY_PATH) is sourced here when it exists and PJRT_PLUGIN
# is not already set, so sourcing this file is enough on every platform.

PLATFORM_KIND=desktop
[ -f /etc/nv_tegra_release ] && PLATFORM_KIND=tegra

# The backend: an explicit PLATFORM_BACKEND or IREE_BACKEND wins; else the SMI on PATH.
if [ -z "${PLATFORM_BACKEND:-}" ]; then
  if [ -n "${IREE_BACKEND:-}" ]; then PLATFORM_BACKEND=$IREE_BACKEND
  elif [ "$(uname -s)" = Darwin ]; then PLATFORM_BACKEND=metal
  elif [ "$PLATFORM_KIND" = tegra ] || command -v nvidia-smi > /dev/null 2>&1; then PLATFORM_BACKEND=cuda
  elif command -v rocm-smi > /dev/null 2>&1; then PLATFORM_BACKEND=rocm
  elif command -v xpu-smi > /dev/null 2>&1; then PLATFORM_BACKEND=xpu
  elif [ -e /dev/neuron0 ]; then PLATFORM_BACKEND=neuron
  else PLATFORM_BACKEND=cuda
  fi
fi

_platform_slug () { tr 'A-Z' 'a-z' | sed -E 's/nvidia geforce |nvidia |amd radeon |intel\(r\) |jetson | developer kit//g; s/[^a-z0-9]+/-/g; s/^-|-$//g'; }
PLATFORM_PY=""; PLATFORM_SHIMPY=(); PLATFORM_RUNBOOK=""; PLATFORM_GPU=${PLATFORM_GPU:-}
PLATFORM_TEST_TIMEOUT=${PLATFORM_TEST_TIMEOUT:-300}
case "$PLATFORM_BACKEND:$PLATFORM_KIND" in
  cuda:tegra)
    # A Jetson. The plugin is the sm_87 build deploy/orin_setup.sh stages under ~/pjrt, with the
    # Tegra cuDNN it was compiled against; the board has no JAX.
    PLATFORM_HOST=orin
    PLATFORM_DEVVAR=CUDA_VISIBLE_DEVICES
    PLATFORM_RUNBOOK=deploy/ORIN.md
    [ "$PLATFORM_TEST_TIMEOUT" = 300 ] && PLATFORM_TEST_TIMEOUT=900
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
    # The jax rocm plugin in the pinned .venv (jax-rocm7-pjrt) against the system ROCm in /opt/rocm;
    # deploy/ROCM_7900XTX.md is the runbook. Detection needs rocm-smi on PATH (/opt/rocm/bin is not
    # on it by default); PJRT_PLUGIN selects another ROCm, e.g. 10.1's (the runbook's §1).
    PLATFORM_HOST=$(hostname -s 2>/dev/null || echo rocm)
    PLATFORM_DEVVAR=HIP_VISIBLE_DEVICES
    PLATFORM_PLUGIN=${PJRT_PLUGIN:-.venv/lib/python3.12/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so}
    [ -x .venv/bin/python3 ] && PLATFORM_PY=.venv/bin/python3
    [ -z "$PLATFORM_GPU" ] && command -v rocm-smi > /dev/null 2>&1 && \
      PLATFORM_GPU=$(rocm-smi --showproductname 2>/dev/null | sed -n 's/.*Card series:[[:space:]]*//p' | head -1 | _platform_slug) || true
    ;;
  xpu:*)
    # An Intel Arc: Intel's OpenXLA PJRT plugin (jax_oneapi_pjrt, platform `oneapi`) in .venv-xpu,
    # which deploy/arc_setup.sh makes — never the pinned .venv. The plugin finds its oneAPI runtime
    # (the jax_oneapi_plugin[with-oneapi] wheels: SYCL, oneDNN, MKL) only through .venv-xpu/lib on
    # LD_LIBRARY_PATH, so that goes on here. deploy/ARC_B50.md is the runbook; the device index is
    # ZE_AFFINITY_MASK (run.sh); the GPU slug comes from xpu-smi (arc-pro-b50), which keys the
    # suite's baseline scripts/platform/expected/xpu-arc-pro-b50.txt.
    PLATFORM_HOST=$(hostname -s 2>/dev/null || echo xpu)
    PLATFORM_DEVVAR=ZE_AFFINITY_MASK
    PLATFORM_RUNBOOK=deploy/ARC_B50.md
    if [ -z "${PJRT_PLUGIN:-}" ]; then
      for _p in .venv-xpu/lib/python3.*/site-packages/jax_plugins/xla_oneapi/xla_oneapi_plugin.so; do
        [ -f "$_p" ] && { PJRT_PLUGIN=$_p; export PJRT_PLUGIN; break; }
      done
    fi
    PLATFORM_PLUGIN=${PJRT_PLUGIN:-}
    [ -x .venv-xpu/bin/python3 ] && PLATFORM_PY=.venv-xpu/bin/python3
    [ -d .venv-xpu/lib ] && export LD_LIBRARY_PATH="$PWD/.venv-xpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    [ -z "$PLATFORM_GPU" ] && command -v xpu-smi > /dev/null 2>&1 && \
      PLATFORM_GPU=$(xpu-smi discovery 2>/dev/null | sed -n 's/.*Device Name:[[:space:]]*//p' | head -1 | _platform_slug) || true
    ;;
  neuron:*)
    # AWS Trainium: Neuron's PJRT plugin (libneuronxla's libneuronpjrt.so) in .venv-neuron, which
    # deploy/trainium_setup.sh makes, else in a Neuron DLAMI's JAX venv. The plugin links the Neuron
    # runtime (libnrt.so.1, the aws-neuronx-runtime-lib apt package, in /opt/aws/neuron/lib) and
    # compiles by running `neuronx-cc` from PATH, so both go on here. It serves several PJRT C-API
    # versions and picks one from NEURON_INTERNAL_PJRT_C_API_VERSION: 0.90 is ffi/pjrt_c_api.h's.
    # NEURON_CC_FLAGS=--auto-cast=none keeps an f32 graph f32 — the compiler's default casts f32
    # matmuls to bf16, which the suite's f32 tolerances would catch as a wrong answer. The device
    # index is a NeuronCore, NEURON_RT_VISIBLE_CORES (run.sh): a trn1 chip is two of them. The slug
    # is the instance type (trn1-2xlarge), keying scripts/platform/expected/neuron-<slug>.txt.
    # deploy/TRAINIUM.md is the runbook.
    PLATFORM_HOST=$(hostname -s 2>/dev/null || echo neuron)
    PLATFORM_DEVVAR=NEURON_RT_VISIBLE_CORES
    PLATFORM_RUNBOOK=deploy/TRAINIUM.md
    _nvenv=""
    for _v in .venv-neuron /opt/aws_neuronx_venv_jax*; do
      for _p in "$_v"/lib/python3.*/site-packages/libneuronxla/libneuronpjrt.so; do
        [ -f "$_p" ] && { _nvenv=$_v; [ -z "${PJRT_PLUGIN:-}" ] && { PJRT_PLUGIN=$_p; export PJRT_PLUGIN; }; break 2; }
      done
    done
    PLATFORM_PLUGIN=${PJRT_PLUGIN:-}
    if [ -n "$_nvenv" ]; then
      [ -x "$_nvenv/bin/python3" ] && PLATFORM_PY=$_nvenv/bin/python3
      case "$_nvenv" in /*) ;; *) _nvenv=$PWD/$_nvenv ;; esac
      export PATH="$_nvenv/bin:$PATH"
    fi
    [ -d /opt/aws/neuron/lib ] && export LD_LIBRARY_PATH="/opt/aws/neuron/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    [ -d /opt/aws/neuron/bin ] && export PATH="$PATH:/opt/aws/neuron/bin"
    export NEURON_INTERNAL_PJRT_C_API_VERSION=${NEURON_INTERNAL_PJRT_C_API_VERSION:-0.90} \
           NEURON_CC_FLAGS=${NEURON_CC_FLAGS:---auto-cast=none}
    [ -z "$PLATFORM_GPU" ] && [ -r /sys/class/dmi/id/product_name ] && \
      PLATFORM_GPU=$(_platform_slug < /sys/class/dmi/id/product_name) || true
    ;;
  metal:*)
    # An Apple Silicon Mac: jax-graft's Metal PJRT plugin (the `metal-pjrt-core` wheel) in the
    # repo's .venv, which deploy/mac_setup.sh makes. One GPU, so no device variable. The plugin is
    # XLA's GPU client, created with the options jax_graft/__init__.py passes: the METAL
    # StreamExecutor platform, XLA's pass-through allocator (the plugin caches and budgets buffers
    # itself; METAL_PJRT_MEMORY_FRACTION scales that budget), no preallocation and no H2D staging
    # on unified memory.
    PLATFORM_HOST=$(hostname -s 2>/dev/null || echo mac)
    PLATFORM_DEVVAR=""
    PLATFORM_RUNBOOK=deploy/MAC.md
    if [ -z "${PJRT_PLUGIN:-}" ]; then
      for _p in .venv/lib/python3.*/site-packages/metal_pjrt_core/pjrt_c_api_mtl_plugin.dylib; do
        [ -f "$_p" ] && { PJRT_PLUGIN=$_p; export PJRT_PLUGIN; break; }
      done
    fi
    PLATFORM_PLUGIN=${PJRT_PLUGIN:-}
    [ -x .venv/bin/python3 ] && PLATFORM_PY=.venv/bin/python3
    [ -z "$PLATFORM_GPU" ] && PLATFORM_GPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | _platform_slug) || true
    export LEAN_MLIR_PJRT_PLATFORM=${LEAN_MLIR_PJRT_PLATFORM:-METAL} \
           LEAN_MLIR_ALLOCATOR=${LEAN_MLIR_ALLOCATOR:-platform} \
           LEAN_MLIR_PREALLOCATE=${LEAN_MLIR_PREALLOCATE:-0} \
           LEAN_MLIR_MEM_FRACTION=${LEAN_MLIR_MEM_FRACTION:-1.0} \
           LEAN_MLIR_STAGE_H2D=${LEAN_MLIR_STAGE_H2D:-0}
    ;;
  llvm-cpu:*|vmvx:*)
    # No GPU: the IREE lowerer's CPU targets (LEAN_MLIR_LOWERER=iree with IREE_BACKEND=llvm-cpu, or
    # vmvx for the slow reference lowering), the `-iree` twins' only backend on a box without a
    # card. Never detected — only an explicit IREE_BACKEND lands here. Nothing to dlopen and no
    # device to select; deploy/iree_cpu_setup.sh builds the shim, historical/IREE_BUILD.md is the
    # runbook. The slug is the CPU model, in case a suite baseline is ever keyed by it.
    PLATFORM_HOST=$(hostname -s 2>/dev/null || echo cpu)
    PLATFORM_DEVVAR=""
    PLATFORM_PLUGIN=""
    PLATFORM_RUNBOOK=historical/IREE_BUILD.md
    [ -x .venv/bin/python3 ] && PLATFORM_PY=.venv/bin/python3
    [ -z "$PLATFORM_GPU" ] && command -v lscpu > /dev/null 2>&1 && \
      PLATFORM_GPU=$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1 | _platform_slug) || true
    ;;
  *)
    echo "scripts/platform/env.sh: unknown backend '$PLATFORM_BACKEND' (cuda | rocm | xpu | neuron | metal | llvm-cpu | vmvx)" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
unset _ares _3060 _p _v _nvenv
export PLATFORM_BACKEND PLATFORM_KIND PLATFORM_HOST PLATFORM_GPU PLATFORM_DEVVAR PLATFORM_PLUGIN PLATFORM_PY PLATFORM_RUNBOOK PLATFORM_TEST_TIMEOUT
