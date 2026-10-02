#!/bin/bash
# Stage the XLA stack on a freshly installed Jetson Orin: the jax 0.11.1 PJRT CUDA plugin
# (built on the training box by deploy/build_orin_pjrt_plugin.sh) and the Tegra cuDNN it
# was built against, then write an env file to source. Idempotent: anything already in
# place with the right checksum is kept. deploy/ORIN.md is the runbook around it.
#
# Usage:  bash deploy/orin_setup.sh [stage-dir]     (default ~/pjrt)
#   TRAININGBOX=<ssh host>  where the plugin comes from (default: trainingbox)
#   PLUGIN_SRC=<path>       the plugin's path on that host
set -euo pipefail
STAGE="${1:-$HOME/pjrt}"
TRAININGBOX="${TRAININGBOX:-trainingbox}"
PLUGIN_SRC="${PLUGIN_SRC:-lean/klawd_max_power/jax-orin-build/dist011/pjrt_c_api_gpu_plugin.so}"
PLUGIN_MD5=0881a0e28c3c63cd1919e3a44cf55ec6
# The cuDNN the plugin was built against, as rules_ml_toolchain fetched it: Tegra
# (linux-aarch64), not SBSA. JetPack's own cuDNN is older, and XLA refuses a runtime
# cuDNN older than the one it was compiled with.
CUDNN_URL=https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-aarch64/cudnn-linux-aarch64-9.12.0.46_cuda12-archive.tar.xz
CUDNN_SHA256=b4a8bbc760f44985f7f4c16784e61432147fc8d7d6135b1fc737aa45d3adaa31

die() { echo "FATAL: $*" >&2; exit 1; }
[ "$(uname -m)" = aarch64 ] || die "this is the device-side script; run it on the Orin"
[ -f /etc/nv_tegra_release ] || echo "warning: no /etc/nv_tegra_release — not a Jetson?"

echo "=== device ==="
head -1 /etc/nv_tegra_release 2>/dev/null || true
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "(nvidia-smi query unsupported)"
command -v nvpmodel >/dev/null && sudo -n nvpmodel -q 2>/dev/null | head -2 || true

mkdir -p "$STAGE"
cd "$STAGE"

echo "=== plugin ==="
if [ ! -f pjrt_c_api_gpu_plugin.so ] || [ "$(md5sum < pjrt_c_api_gpu_plugin.so | cut -d' ' -f1)" != "$PLUGIN_MD5" ]; then
  scp "$TRAININGBOX:$PLUGIN_SRC" pjrt_c_api_gpu_plugin.so
fi
got=$(md5sum < pjrt_c_api_gpu_plugin.so | cut -d' ' -f1)
[ "$got" = "$PLUGIN_MD5" ] || die "plugin md5 $got, expected $PLUGIN_MD5 — a different build; stop rather than guess"
echo "pjrt_c_api_gpu_plugin.so: md5 OK"

echo "=== cuDNN (Tegra) ==="
if [ ! -f cudnn912/lib/libcudnn.so.9 ]; then
  tarball=$(basename "$CUDNN_URL")
  [ -f "$tarball" ] || curl -fL -o "$tarball" "$CUDNN_URL"
  echo "$CUDNN_SHA256  $tarball" | sha256sum -c - || die "cuDNN archive checksum mismatch"
  mkdir -p cudnn912
  tar -xJf "$tarball" -C cudnn912 --strip-components=1
  rm -f "$tarball"
fi
ls cudnn912/lib/libcudnn.so.9* >/dev/null || die "cuDNN libraries missing under $STAGE/cudnn912/lib"
echo "cudnn912: $(ls cudnn912/lib/libcudnn.so.9.* | head -1)"

echo "=== ship gate (sm_87 cubins + PTX) ==="
CUOBJ=$(command -v cuobjdump || ls /usr/local/cuda/bin/cuobjdump 2>/dev/null || true)
if [ -n "$CUOBJ" ]; then
  arch=$("$CUOBJ" --list-elf pjrt_c_api_gpu_plugin.so | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')
  nptx=$("$CUOBJ" --list-ptx pjrt_c_api_gpu_plugin.so | grep -c 'PTX file' || true)
  echo "cubin archs: $arch  PTX modules: $nptx"
  [ "$arch" = "sm_87 " ] && [ "$nptx" -gt 0 ] || die "plugin is not an sm_87 build"
else
  echo "(no cuobjdump; md5 already pins the build)"
fi

echo "=== env file ==="
cat > "$STAGE/orin_env.sh" <<EOF
# Source before any XLA run on this Orin: . $STAGE/orin_env.sh
export PJRT_PLUGIN=$STAGE/pjrt_c_api_gpu_plugin.so
# cuDNN 9.12 first: the plugin was compiled against it, and JetPack's copy is older.
export LD_LIBRARY_PATH=$STAGE/cudnn912/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export LEAN_MLIR_LOWERER=xla
# Unified memory: the device pool and the pinned host staging share one DRAM.
# Preallocation off is required; the fraction is per model family (deploy/ORIN.md).
export LEAN_MLIR_PREALLOCATE=0
export LEAN_MLIR_MEM_FRACTION=\${LEAN_MLIR_MEM_FRACTION:-0.15}
EOF
echo "wrote $STAGE/orin_env.sh"
echo
echo "next:  . $STAGE/orin_env.sh && scripts/platform/check.sh --plan"
