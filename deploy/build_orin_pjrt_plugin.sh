#!/bin/bash
# One-shot: stand up the emulated aarch64 build environment on ANY x86 Linux box and
# start the jax 0.11.1 Orin PJRT CUDA plugin build.
#
# No GPU is used. No NVIDIA GPU or driver is required: the CUDA toolkit and cuDNN
# arrive as hermetic redist tarballs and the plugin links against CUDA *stubs*
# (--config=cuda_libraries_from_stubs), dlopening the real libraries only at run
# time on the Orin. An AMD-only box builds this fine.
#
# Needs: docker, ~80 GB free disk, network. Registers qemu-user binfmt if absent.
# Usage:  bash deploy/build_orin_pjrt_plugin.sh [workdir]   (default ~/orin-plugin-build)
set -euo pipefail
WORK="${1:-$HOME/orin-plugin-build}"
JOBS="$(nproc)"

echo "=== host ==="
lscpu | grep -E '^Model name|^CPU\(s\)'
free -g | sed -n 2p
df -h "$(dirname "$WORK")" | tail -1
command -v docker >/dev/null || { echo "FATAL: docker not installed"; exit 1; }

echo "=== qemu binfmt (aarch64 emulation) ==="
if [ ! -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]; then
  docker run --privileged --rm tonistiigi/binfmt --install arm64
fi
grep -q enabled /proc/sys/fs/binfmt_misc/qemu-aarch64 && echo "qemu-aarch64: enabled"

echo "=== workdir ==="
mkdir -p "$WORK"/{bin,home,dist011}
cd "$WORK"

echo "=== bazel 7.7.1 (aarch64 — jax 0.11.1 pins this exact version) ==="
[ -x bin/bazel-7.7.1 ] || {
  curl -fsSL -o bin/bazel-7.7.1 \
    https://github.com/bazelbuild/bazel/releases/download/7.7.1/bazel-7.7.1-linux-arm64
  chmod +x bin/bazel-7.7.1
}

echo "=== jax 0.11.1 source ==="
[ -d jax011 ] || git clone --depth 1 --branch jax-v0.11.1 https://github.com/jax-ml/jax.git jax011

echo "=== container (hostname MUST contain 'tegra') ==="
# rules_ml_toolchain picks the Tegra linux-aarch64 redists only when `uname -a`
# contains the substring "tegra" — that is what the hostname is for.
if ! docker inspect jaxbuild >/dev/null 2>&1; then
  docker pull nvcr.io/nvidia/l4t-jetpack:r36.4.0
  docker run -d --name jaxbuild --hostname tegra-build \
    -v "$WORK":/work nvcr.io/nvidia/l4t-jetpack:r36.4.0 sleep infinity
else
  docker start jaxbuild
fi
sleep 3
docker exec jaxbuild uname -a | grep -q tegra && echo "tegra detection: OK"

echo "=== toolchain inside the container (slow: every apt binary is emulated) ==="
docker exec jaxbuild bash -c '
set -eux
[ -x /usr/lib/llvm-18/bin/clang ] && exit 0
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q git curl wget ca-certificates python3 python3-dev python3-pip \
  build-essential lsb-release gnupg software-properties-common patch unzip zip pkg-config
wget -q -O /tmp/llvm.sh https://apt.llvm.org/llvm.sh
bash /tmp/llvm.sh 18
'
docker exec jaxbuild git config --global --add safe.directory /work/jax011

echo "=== build script ==="
cat > "$WORK/run_build_011.sh" <<INNER
#!/bin/bash
# planning/orin_plugin_rebuild.md §1, with the two version strings corrected:
#   12.6.77 is the NVCC inside toolkit 12.6.2 -- redistrib_12.6.77.json is a 404.
#   cuDNN redist keys are three-part; redistrib_9.12.0.json IS 9.12.0.46 inside.
# Both original values abort at fetch, because rules_ml_toolchain fails on an
# unsupported key rather than falling back.
#
# -- experimental_worker_for_repo_fetching=off is REQUIRED under QEMU. Bazel 7 hands
# each repository rule to a worker thread while the SkyFunction thread waits on a
# semaphore for it to signal back. Under aarch64 user-mode emulation the worker never
# signals, so the build deadlocks right after the redists are fetched: zero RUNNABLE
# JVM threads, ~3% CPU, no disk or network, evaluator parked forever at
# WorkerSkyKeyComputeState.startOrContinueWork. Bazel 6.5.0 had no such mode, which is
# why the older jax 0.4.38 build never hit it.
set -euxo pipefail
export HOME=/work/home
cd /work/jax011
python3 build/build.py build --wheels=jax-cuda-pjrt \\
  --cuda_version=12.6.2 --cudnn_version=9.12.0 \\
  --cuda_compute_capabilities=sm_87,compute_87 \\
  --clang_path=/usr/lib/llvm-18/bin/clang \\
  --python_version=3.12 \\
  --disable_nccl --disable_mkl_dnn \\
  --bazel_path=/work/bin/bazel-7.7.1 \\
  --bazel_startup_options=--host_jvm_args=-Xmx12g \\
  --bazel_options=--jobs=${JOBS} \\
  --bazel_options=--local_cpu_resources=${JOBS} \\
  --bazel_options=--spawn_strategy=local \\
  --bazel_options=--experimental_worker_for_repo_fetching=off \\
  --bazel_options=--verbose_failures \\
  --output_path=/work/dist011 --verbose
echo BUILD_OK
INNER
chmod +x "$WORK/run_build_011.sh"

echo "=== launch (detached) ==="
docker exec -d jaxbuild bash -c \
  'bash /work/run_build_011.sh > /work/build_011.log 2>&1; echo "EXIT=$?" >> /work/build_011.log'

cat <<EOM

STARTED. Log: $WORK/build_011.log

  tail -f $WORK/build_011.log
  grep -c linux-sbsa $WORK/build_011.log     # MUST be 0
  grep -c linux-aarch64 $WORK/build_011.log  # MUST be > 0

Expect roughly 9-11 hours emulated. Artifact lands in $WORK/dist011.
EOM
