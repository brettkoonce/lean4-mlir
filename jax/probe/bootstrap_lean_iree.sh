#!/bin/bash
# One-command pod bootstrap for the FULL Lean → MLIR → IREE → GPU pipeline
# (`lake run benchmark` / `mnist` / `imagenette`). Companion to bootstrap.sh,
# which only sets up the gitignored JAX probes; THIS builds the verified-IREE
# trainers, which need Lean + a from-source IREE runtime.
#
# Encodes every lesson from the first live A100 pod session (2026-07-08):
#   - iree-compile from pip pulls the LATEST nightly, not the lock's date — the
#     IREE runtime you build MUST match that wheel's commit or you get a HAL ABI
#     mismatch at module load. We read the commit off `iree-compile --version`
#     and check the runtime out to it, instead of trusting the lock's pin.
#   - runtime-only submodules (~235 MB) — a recursive clone balloons past 9 GB.
#   - static archives + --whole-archive so the CUDA HAL driver-register symbol
#     survives into libiree_ffi.so (verify: nm ... | grep driver_module_register).
#   - Mathlib is a hard dep: `lake exe cache get` (download prebuilt oleans);
#     building it from source is hours. Needs ~6 GB free disk.
#   - A100 is sm_80, but the trainers default to --iree-cuda-target=sm_86, and
#     sm_86 PTX will NOT JIT down to sm_80. Fix: export IREE_CHIP=sm_80. (No
#     source edit — IREE_CHIP is read from the env in LeanMlir/Types.lean.)
#   - DISK: the default 30 GB RunPod container overlay is MARGINAL. Budget:
#     elan 2.8G + mathlib oleans 6.9G + IREE build ~2G + imagenette 2.3G +
#     project build ~3G ≈ 17G, leaving ~8G. It fits, barely. Provision the pod
#     with a >=50 GB container disk (or mount a volume for .lake + $IREE_SRC).
#     The local NVMe is nvidia-driver bind-mounts only (read-only); /workspace
#     is MooseFS (network — fine for cold storage, too slow for `lake build`).
#
# Live A100-80GB result (2026-07-08), `lake run benchmark`, verified-IREE/CUDA
# vs the reference AMD 7900 XTX (IREE/ROCm):  dense 8.14x  conv 8.44x  attn 3.65x.
# I.e. the untuned IREE-CUDA path is ~8x behind a consumer AMD card on conv and
# ~3.6x on attention (tensor-core GEMM closes the gap). NOTE this is the SLOW
# verified path; the JAX/XLA probe (bootstrap.sh) hit 2110 img/s on R50 — that's
# the fast path. The benchmark is a codegen-tuning signal, not the A100's ceiling.
#
# Usage, from a fresh pod shell (after bootstrap.sh has fetched imagenette):
#   git clone -b runpod-probe --depth 1 https://github.com/brettkoonce/lean4-mlir.git lean4-jax
#   cd lean4-jax && bash jax/probe/bootstrap_lean_iree.sh
#   # then, in THIS shell:
#   source .venv/bin/activate && export IREE_CHIP=sm_80 && lake run benchmark
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
REPO_DIR="$(pwd)"
IREE_SRC=${IREE_SRC:-/root/src/iree}
IREE_BUILD=${IREE_BUILD:-/root/src/iree-build}

echo "━━━ [1/6] Lean toolchain (elan + the pinned lean-toolchain)"
if ! command -v lake >/dev/null 2>&1 && [ ! -x /root/.elan/bin/lake ]; then
  curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain "$(cat lean-toolchain)"
fi
export PATH="/root/.elan/bin:$PATH"
lake --version

echo "━━━ [2/6] iree-compile (pip, into ./.venv where the lakefile looks for it)"
if [ ! -x .venv/bin/iree-compile ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -f https://iree.dev/pip-release-links.html --pre \
    'iree-base-compiler>=3.12.0rc20260428'
fi
WHEEL_COMMIT=$(.venv/bin/iree-compile --version | grep -oE '@ [0-9a-f]{40}' | cut -d' ' -f2)
echo "  iree-compile wheel commit: $WHEEL_COMMIT"

echo "━━━ [3/6] IREE runtime source @ the wheel commit (ABI must match)"
if [ ! -d "$IREE_SRC/.git" ]; then
  mkdir -p "$(dirname "$IREE_SRC")"
  git init -q "$IREE_SRC" && git -C "$IREE_SRC" remote add origin https://github.com/iree-org/iree.git
fi
if [ "$(git -C "$IREE_SRC" rev-parse HEAD 2>/dev/null)" != "$WHEEL_COMMIT" ]; then
  git -C "$IREE_SRC" fetch -q --depth 1 origin "$WHEEL_COMMIT"
  git -C "$IREE_SRC" checkout -q FETCH_HEAD
  xargs -a "$IREE_SRC/build_tools/scripts/git/runtime_submodules.txt" \
    git -C "$IREE_SRC" submodule update --init --depth 1 -q
fi

echo "━━━ [4/6] IREE runtime build (static, CUDA driver) + link ffi/libiree_ffi.so"
if [ ! -f ffi/libiree_ffi.so ]; then
  mkdir -p "$IREE_BUILD" && ( cd "$IREE_BUILD" && cmake "$IREE_SRC" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_SAMPLES=OFF -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_HAL_DRIVER_CUDA=ON -DBUILD_SHARED_LIBS=OFF && ninja )
  ( cd ffi
    gcc -fPIC -O2 -c iree_ffi.c -I"$IREE_SRC/runtime/src" -I"$IREE_BUILD/runtime/src" \
      -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
    gcc -shared -o libiree_ffi.so iree_ffi.o \
      -Wl,--whole-archive "$IREE_BUILD/runtime/src/iree/runtime/libiree_runtime_unified.a" \
      -Wl,--no-whole-archive -Wl,--start-group \
        "$IREE_BUILD"/build_tools/third_party/flatcc/libflatcc_runtime.a \
        "$IREE_BUILD"/build_tools/third_party/flatcc/libflatcc_parsing.a \
      -Wl,--end-group -lm -lpthread -ldl )
fi
nm ffi/libiree_ffi.so | grep -q iree_hal_cuda_driver_module_register \
  || { echo "!! CUDA driver symbol missing — --whole-archive dropped"; exit 1; }
echo "  ✓ libiree_ffi.so ($(du -h ffi/libiree_ffi.so | cut -f1)), CUDA driver present"

echo "━━━ [5/6] Mathlib oleans (download; building from source is hours)"
lake exe cache get

echo "━━━ [6/6] ready."
echo "  ⚠ THIS shell first:  source .venv/bin/activate && export IREE_CHIP=sm_80"
echo "  Then (first build compiles the proof cone + vmfbs — 10-20 min):"
echo "      lake run benchmark      # probe this GPU, print per-chapter estimates"
echo "      lake run mnist          # verified MNIST linear/MLP/CNN end to end"
echo "  A100=sm_80: the trainers default to sm_86, which won't JIT down — hence IREE_CHIP."
