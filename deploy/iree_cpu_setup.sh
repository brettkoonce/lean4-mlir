#!/bin/bash
# deploy/iree_cpu_setup.sh — the second lowerer on a box: `iree-compile` in .venv-iree and a
# `ffi/libiree_ffi.so` for one target, so that `LEAN_MLIR_LOWERER=iree` runs the `-iree` twins
# (`lake run mnist-iree`, `cifar-iree`, `imagenette-iree`) and the gradient gates here.
# historical/IREE_BUILD.md is the runbook (the GPU builds by hand); this is it as commands, the
# recipe .github/workflows/iree.yml's oracle job uses, with the version moved. The name is
# historical — the CPU build came first (atlas, 2026-10-28); `IREE_TARGET=cuda` is the same
# script for an NVIDIA box (ares, 2026-11-05).
#
#   bash deploy/iree_cpu_setup.sh                       # every stage, in order, CPU target
#   IREE_TARGET=cuda bash deploy/iree_cpu_setup.sh      # the same on a CUDA box: CUDA HAL in the shim
#   bash deploy/iree_cpu_setup.sh runtime               # start at a stage (check venv runtime shim smoke)
#   bash deploy/iree_cpu_setup.sh venv shim             # only the stages named
#
# Stages, each proving one layer before the next leans on it:
#   check    gcc, git, a python >= 3.12 or uv, disk; on cuda, nvidia-smi and libcuda.so.1
#   venv     .venv-iree with iree-base-compiler and iree-base-runtime at $IREE_VERSION, plus cmake
#            and ninja — its own venv, not .venv: .venv is the pinned JAX env
#            (jax/requirements-cuda-lock.txt), one env per lock. `lake run *-iree` puts
#            .venv-iree/bin on PATH itself (lakefile.lean, runDemoGroup); a bare binary wants it there
#   runtime  the IREE runtime source at THE WHEEL'S COMMIT (read off `iree-compile --version`; a
#            skew is a HAL ABI mismatch at module load), runtime submodules only (~240 MB, not the
#            LLVM chain), the static archives with the local-task and local-sync drivers, plus the
#            CUDA HAL driver on cuda
#   shim     ffi/libiree_ffi.so from ffi/iree_ffi.c and the archives, linked --no-undefined so a
#            missing archive fails here; -DUSE_CPU on cpu (no GPU driver, local-task default), the
#            CUDA driver compiled in on cuda (device `cuda` by default, `IREE_DEVICE` overrides).
#            An existing .so is moved beside $IREE_BUILD, not deleted
#   smoke    the committed MobileNetV2 eval forward compiled for the target and run through
#            iree-run-module — on the CPU the 224² case that IREE 3.11.0 segfaults on (below); on
#            cuda the sm_86 PTX the trainers compile (IREE_CHIP), plus a probe of whether this
#            release accepts sm_89 (3.11 did not: iree-org/iree#21122; 3.12 and 3.13 do), in the manifest
#
# IREE_VERSION defaults to 3.12.0 — the one release that clears both boxes' bars.
#   NOT 3.11.0, CI's old pin: 3.11.0's CPU task device (local-task) segfaults inside the first
#   invoke of every 224² artifact — MobileNetV2's eval forward and train step, reproduced with
#   iree-run-module alone, with one worker and with 8 MB worker stacks, while the same .vmfb runs
#   on local-sync — and 3.12.0 and 3.13.0 run them (atlas, 2026-10-28). The MNIST-sized
#   artifacts CI ran are fine on 3.11.0, which is why CI never saw it.
#   NOT 3.13.0, the CPU box's first pin: on cuda its NVPTX backend aborts in instruction selection
#   ("Cannot select: v2i16 = umul_lohi") on the ResNet-34 and ResNet-50 train steps — the
#   dispatch that flips a 512×512×3×3 kernel for the conv backward, delinearised to a 1-D index
#   with div/rem by 3 that LLVM narrows to a 16-bit vector the backend cannot multiply. No
#   compile flag of the twelve tried avoids it; 3.12.0 and the 3.14 nightly of 2026-11-04 compile
#   the same text (ares, 2026-11-05, runs/2026-11-05-iree-cuda-setup/). 3.12.0 compiles all 35
#   artifacts the thirteen `-iree` twins use (the bf16 bbn arms with
#   --iree-codegen-llvmgpu-use-vector-distribution=false, as 3.13.0 also needs).
# The compiler wheel, the runtime wheel and the runtime commit move together: a .vmfb from one
# release does not load on another's runtime ("required module 'hal' version mismatch").
#
# Everything is teed to runs/<date>-iree-<target>-setup/setup.log, with the box's facts in
# runs/<date>-iree-<target>-setup/manifest.txt. Stale `.lake/build/*_<backend>.vmfb` from an older
# release are removed at the shim stage: compileVmfb reuses any cache newer than its .mlir.
set -euo pipefail
cd "$(dirname "$0")/.."

ALL="check venv runtime shim smoke"
IREE_TARGET=${IREE_TARGET:-cpu}
IREE_VERSION=${IREE_VERSION:-3.12.0}
IREE_SRC=${IREE_SRC:-$HOME/src/iree}
IREE_BUILD=${IREE_BUILD:-$HOME/src/iree-build}
VENV=${IREE_VENV:-.venv-iree}
GPU=${LEAN_DEMO_GPU:-0}
case "$IREE_TARGET" in cpu|cuda) ;; *) echo "IREE_TARGET must be cpu or cuda (got '$IREE_TARGET')" >&2; exit 2 ;; esac

if [ $# -eq 0 ]; then STAGES=$ALL
elif [ $# -eq 1 ]; then
  case " $ALL " in *" $1 "*) ;; *) echo "unknown stage '$1' (one of: $ALL)" >&2; exit 2 ;; esac
  STAGES=${ALL#*"$1"}; STAGES="$1$STAGES"          # from that stage to the end
else STAGES="$*"
fi

STAGE=${STAGES%% *}
LOGDIR=runs/$(date +%F)-iree-$IREE_TARGET-setup
mkdir -p "$LOGDIR"
LOG=$LOGDIR/setup.log
exec > >(tee -a "$LOG") 2>&1
echo "════ iree_cpu_setup $(date '+%F %T') — target $IREE_TARGET — stages: $STAGES — IREE $IREE_VERSION — venv $VENV"

say ()  { printf '\n━━━ %s\n' "$*"; }
ok ()   { printf '  ✓ %s\n' "$*"; }
warn () { printf '  ⚠ %s\n' "$*"; }
die ()  { printf '\n  ✗ %s\n\n  The log is %s. Fix the above, then resume with:\n    IREE_TARGET=%s bash deploy/iree_cpu_setup.sh %s\n' "$1" "$LOG" "$IREE_TARGET" "$STAGE" >&2; exit 1; }
want () { case " $STAGES " in *" $1 "*) STAGE=$1; return 0 ;; *) return 1 ;; esac; }
# The HAL backend name the trainers pass to iree-compile and the .vmfb suffix they cache under.
if [ "$IREE_TARGET" = cuda ]; then BACKEND=cuda; DEVICE=cuda; else BACKEND=llvm-cpu; DEVICE=local-task; fi

# The newest python >= 3.12 on PATH, or "" if there is none (the IREE wheels ship for 3.10–3.13).
find_python () {
  for p in python3.13 python3.12 python3; do
    command -v "$p" > /dev/null 2>&1 || continue
    if "$p" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info < (3, 14) else 1)' 2> /dev/null; then
      command -v "$p"; return 0
    fi
  done
  return 0
}

# ─── check ───────────────────────────────────────────────────────────────────
if want check; then
  say "check: the box"
  [ "$(uname -s)" = Linux ] || die "not Linux ($(uname -s)) — this recipe is the Linux one; the Mac's IREE build is historical/IREE_BUILD.md by hand"
  command -v gcc > /dev/null || die "no gcc — sudo apt install build-essential"
  ok "C compiler: $(gcc --version | head -1)"
  command -v git > /dev/null || die "no git"
  PY=$(find_python)
  if [ -n "$PY" ]; then ok "python: $PY ($("$PY" -V 2>&1))"
  elif command -v uv > /dev/null 2>&1; then ok "no python 3.12/3.13 on PATH; uv is, and the venv stage will use it"
  else die "no python 3.12/3.13 on PATH and no uv — sudo apt install python3.12-venv, or https://docs.astral.sh/uv/"
  fi
  free_gb=$(df -BG --output=avail . | awk 'NR==2 {gsub("G",""); print $1}')
  [ "$free_gb" -ge 5 ] || die "only ${free_gb} GB free here — the runtime source and build want ~2 GB, the wheels ~1 GB"
  ok "disk: ${free_gb} GB free"
  ok "cpu: $(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1), $(nproc) threads"
  if [ "$IREE_TARGET" = cuda ]; then
    command -v nvidia-smi > /dev/null || die "no nvidia-smi — IREE_TARGET=cuda wants an NVIDIA driver (IREE_TARGET=cpu for the CPU build)"
    ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1' || die "libcuda.so.1 is not on the loader path — the CUDA HAL dlopens it at the first session"
    ok "gpu $GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$GPU" 2>/dev/null || echo '?'); libcuda.so.1 found"
  fi
fi

# ─── venv ────────────────────────────────────────────────────────────────────
if want venv; then
  say "venv: $VENV with iree-base-compiler + iree-base-runtime $IREE_VERSION, cmake, ninja"
  if [ -x $VENV/bin/iree-compile ] && $VENV/bin/iree-compile --version 2>/dev/null | grep -c "version $IREE_VERSION" > /dev/null; then
    ok "$VENV already has iree-compile $IREE_VERSION"
  else
    [ -d $VENV ] && warn "$VENV exists without iree-compile $IREE_VERSION — installing into it"
    if command -v uv > /dev/null 2>&1; then
      [ -x $VENV/bin/python3 ] || uv venv --python 3.12 $VENV || die "uv venv failed"
      uv pip install --python $VENV/bin/python iree-base-compiler==$IREE_VERSION iree-base-runtime==$IREE_VERSION cmake ninja numpy \
        || die "uv pip install failed — is $IREE_VERSION a PyPI release? (https://pypi.org/project/iree-base-compiler/#history)"
    else
      PY=$(find_python); [ -n "$PY" ] || die "no python 3.12/3.13 on PATH"
      [ -x $VENV/bin/python3 ] || "$PY" -m venv $VENV || die "python -m venv failed (sudo apt install python3.12-venv)"
      $VENV/bin/pip install --quiet iree-base-compiler==$IREE_VERSION iree-base-runtime==$IREE_VERSION cmake ninja numpy \
        || die "pip install failed — is $IREE_VERSION a PyPI release?"
    fi
  fi
  ok "$($VENV/bin/iree-compile --version | sed -n 2p | sed 's/^ *//')"
fi

# ─── runtime ─────────────────────────────────────────────────────────────────
if want runtime; then
  say "runtime: $IREE_SRC at the wheel's commit, static archives in $IREE_BUILD ($IREE_TARGET)"
  [ -x $VENV/bin/iree-compile ] || die "no $VENV/bin/iree-compile — run the venv stage"
  COMMIT=$($VENV/bin/iree-compile --version | grep -oE '@ [0-9a-f]{40}' | cut -d' ' -f2)
  [ -n "$COMMIT" ] || die "could not read the wheel's commit off iree-compile --version"
  ok "wheel commit $COMMIT"
  if [ ! -d "$IREE_SRC/.git" ]; then
    mkdir -p "$(dirname "$IREE_SRC")"
    git init -q "$IREE_SRC" && git -C "$IREE_SRC" remote add origin https://github.com/iree-org/iree.git
  fi
  if [ "$(git -C "$IREE_SRC" rev-parse HEAD 2>/dev/null)" != "$COMMIT" ]; then
    git -C "$IREE_SRC" fetch -q --depth 1 origin "$COMMIT" || die "git fetch of $COMMIT failed"
    git -C "$IREE_SRC" checkout -q FETCH_HEAD
    # Runtime-only submodules (~240 MB); a recursive clone pulls LLVM through torch-mlir/stablehlo.
    xargs -a "$IREE_SRC/build_tools/scripts/git/runtime_submodules.txt" \
      git -C "$IREE_SRC" submodule update -q --init --depth 1 || die "submodule update failed"
    rm -rf "$IREE_BUILD"   # archives from another commit must not be reused
  fi
  ok "source at $(git -C "$IREE_SRC" log -1 --format='%h %cs')"
  export PATH="$PWD/$VENV/bin:$PATH"
  CUDA_ON=OFF; [ "$IREE_TARGET" = cuda ] && CUDA_ON=ON
  cmake -S "$IREE_SRC" -B "$IREE_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_HAL_DRIVER_CUDA=$CUDA_ON \
    -DBUILD_SHARED_LIBS=OFF > "$LOGDIR/cmake.log" 2>&1 || die "cmake failed — $LOGDIR/cmake.log"
  # The archives the shim links. printf moved: its own archive in 3.11, inside the unified one by
  # 3.13 — build it when the target exists and link whatever is there.
  TARGETS="runtime/src/iree/runtime/libiree_runtime_unified.a build_tools/third_party/flatcc/libflatcc_runtime.a build_tools/third_party/flatcc/libflatcc_parsing.a"
  PRINTF=$(ninja -C "$IREE_BUILD" -t targets all 2>/dev/null | grep -oE '^build_tools/third_party/printf/libprintf[a-z_]*\.a' | head -1 || true)
  ninja -C "$IREE_BUILD" $TARGETS $PRINTF > "$LOGDIR/ninja.log" 2>&1 || die "ninja failed — $LOGDIR/ninja.log"
  ok "archives: $(find "$IREE_BUILD" -name 'libiree_runtime_unified.a' -o -name 'libflatcc_*.a' -o -name 'libprintf*.a' | wc -l) ($(du -sh "$IREE_BUILD" | cut -f1) build, CUDA HAL $CUDA_ON)"
fi

# ─── shim ────────────────────────────────────────────────────────────────────
if want shim; then
  say "shim: ffi/libiree_ffi.so, $IREE_TARGET"
  U=$IREE_BUILD/runtime/src/iree/runtime/libiree_runtime_unified.a
  [ -f "$U" ] || die "no $U — run the runtime stage"
  if [ "$IREE_TARGET" = cuda ]; then
    grep -q '^IREE_HAL_DRIVER_CUDA:BOOL=ON' "$IREE_BUILD/CMakeCache.txt" || die "$IREE_BUILD was configured without the CUDA HAL — run the runtime stage with IREE_TARGET=cuda"
  fi
  ARCHIVES="$IREE_BUILD/build_tools/third_party/flatcc/libflatcc_runtime.a $IREE_BUILD/build_tools/third_party/flatcc/libflatcc_parsing.a"
  for a in "$IREE_BUILD"/build_tools/third_party/printf/libprintf*.a; do [ -f "$a" ] && ARCHIVES="$ARCHIVES $a"; done
  CPUFLAG=-DUSE_CPU; [ "$IREE_TARGET" = cuda ] && CPUFLAG=
  gcc -fPIC -O2 -c ffi/iree_ffi.c -o "$LOGDIR/iree_ffi.o" -I"$IREE_SRC/runtime/src" -I"$IREE_BUILD/runtime/src" \
    -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl $CPUFLAG || die "compiling ffi/iree_ffi.c failed"
  if [ -f ffi/libiree_ffi.so ]; then
    PREV="$(dirname "$IREE_BUILD")/libiree_ffi.so.$(date -r ffi/libiree_ffi.so +%F).previous"
    mv ffi/libiree_ffi.so "$PREV" && ok "previous shim moved to $PREV"
  fi
  # --whole-archive keeps the driver registration symbols; --no-undefined makes a missing archive
  # fail now rather than at the first session create.
  gcc -shared -o ffi/libiree_ffi.so "$LOGDIR/iree_ffi.o" \
    -Wl,--whole-archive "$U" -Wl,--no-whole-archive \
    -Wl,--start-group $ARCHIVES -Wl,--end-group -lm -lpthread -ldl -Wl,--no-undefined \
    || die "linking ffi/libiree_ffi.so failed"
  rm -f "$LOGDIR/iree_ffi.o"
  # grep -c, not -q: under pipefail, -q's early exit SIGPIPEs nm and fails the pipeline.
  nm ffi/libiree_ffi.so | grep -c iree_hal_local_task_driver_module_register > /dev/null || die "local-task driver symbol missing — --whole-archive dropped?"
  if [ "$IREE_TARGET" = cuda ]; then
    nm ffi/libiree_ffi.so | grep -c iree_hal_cuda_driver_module_register > /dev/null || die "cuda driver symbol missing — the CUDA HAL did not make it into the shim"
    DRIVERS="cuda + local-task + local-sync"
  else DRIVERS="local-task + local-sync"; fi
  nm -D ffi/libiree_ffi.so | grep -c ' T iree_ffi_train_step_adam$' > /dev/null || die "iree_ffi_train_step_adam not exported"
  ok "ffi/libiree_ffi.so ($(du -h ffi/libiree_ffi.so | cut -f1)), $DRIVERS drivers, $(nm -D ffi/libiree_ffi.so | grep -c ' T iree_ffi_') entry points"
  if [ "$IREE_TARGET" = cuda ]; then STALE=".lake/build/*_cuda.vmfb"; else STALE=".lake/build/*_llvm-cpu.vmfb .lake/build/*_vmvx.vmfb"; fi
  # shellcheck disable=SC2086
  n=$(ls $STALE 2>/dev/null | wc -l)
  # shellcheck disable=SC2086
  if [ "$n" -gt 0 ]; then rm -f $STALE; ok "removed $n cached $BACKEND .vmfb (a cache from another release would not load; they recompile at first use)"; fi
fi

# ─── smoke ───────────────────────────────────────────────────────────────────
if want smoke; then
  say "smoke: the MobileNetV2 eval forward on $BACKEND, $DEVICE"
  [ -x $VENV/bin/iree-compile ] || die "no $VENV/bin/iree-compile — run the venv stage"
  M=verified_mlir/mobilenetv2_fwd_eval.mlir
  CHIPARGS=; [ "$IREE_TARGET" = cuda ] && CHIPARGS="--iree-cuda-target=${IREE_CHIP:-sm_86}"
  # shellcheck disable=SC2086
  $VENV/bin/iree-compile "$M" --iree-hal-target-backends=$BACKEND $CHIPARGS -o "$LOGDIR/smoke.vmfb" 2> "$LOGDIR/smoke_compile.log" \
    || die "iree-compile $M failed — $LOGDIR/smoke_compile.log"
  # One zero-filled --input per tensor argument of the function, read off its signature.
  INPUTS=$($VENV/bin/python -I - "$M" mobilenetv2_fwd_eval <<'PY'
import re, sys
src = open(sys.argv[1]).read()
m = re.search(r'func\.func @' + re.escape(sys.argv[2]) + r'\((.*?)\)\s*->', src, re.S)
print(' '.join(f'--input={a}f32=0' for a in re.findall(r'tensor<([0-9x]*)f32>', m.group(1))))
PY
)
  export CUDA_VISIBLE_DEVICES=$GPU
  if $VENV/bin/iree-run-module --module="$LOGDIR/smoke.vmfb" --device=$DEVICE --function=mobilenetv2_fwd_eval $INPUTS > "$LOGDIR/smoke_run.log" 2>&1 \
     && grep -q '^32x10xf32=' "$LOGDIR/smoke_run.log"; then
    ok "mobilenetv2_fwd_eval ran on $DEVICE: $(grep -m1 -oE '^32x10xf32=\[[^]]*\]' "$LOGDIR/smoke_run.log")..."
  else
    [ "$IREE_TARGET" = cuda ] && die "mobilenetv2_fwd_eval did not run on cuda (rc, or no 32x10 result) — $LOGDIR/smoke_run.log; is GPU $GPU free, and is the driver at least as new as the PTX?"
    die "mobilenetv2_fwd_eval did not run on local-task (rc, or no 32x10 result) — $LOGDIR/smoke_run.log. On IREE 3.11.0 this is the known segfault; IREE_VERSION=3.12.0 bash deploy/iree_cpu_setup.sh venv"
  fi
  rm -f "$LOGDIR/smoke.vmfb"
  SM89="n/a"
  if [ "$IREE_TARGET" = cuda ]; then
    # 3.11 had no target metadata for Ada (iree-org/iree#21122), so the trainers compile sm_86 PTX
    # and let the driver JIT it. Does this release take sm_89? Recorded, not acted on: the chip
    # the trainers use is ireeCompileArgs' default (IREE_CHIP overrides it per run).
    if $VENV/bin/iree-compile "$M" --iree-hal-target-backends=cuda --iree-cuda-target=sm_89 -o "$LOGDIR/smoke_sm89.vmfb" 2> "$LOGDIR/smoke_sm89_compile.log"; then
      if $VENV/bin/iree-run-module --module="$LOGDIR/smoke_sm89.vmfb" --device=cuda --function=mobilenetv2_fwd_eval $INPUTS > "$LOGDIR/smoke_sm89_run.log" 2>&1 \
         && grep -q '^32x10xf32=' "$LOGDIR/smoke_sm89_run.log"; then SM89="compiles and runs"; else SM89="compiles, does not run ($LOGDIR/smoke_sm89_run.log)"; fi
    else SM89="rejected: $(grep -m1 -oE 'error: .*' "$LOGDIR/smoke_sm89_compile.log" | cut -c1-120)"; fi
    ok "sm_89 on IREE $IREE_VERSION: $SM89"
    rm -f "$LOGDIR/smoke_sm89.vmfb"
  fi
  {
    echo "host	$(hostname -s)"
    echo "target	$IREE_TARGET ($BACKEND on $DEVICE)"
    echo "cpu	$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
    echo "threads	$(nproc)"
    if [ "$IREE_TARGET" = cuda ]; then
      echo "gpu	$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$GPU" 2>/dev/null | sed 's/, / · driver /') · chip ${IREE_CHIP:-sm_86}; sm_89: $SM89"
    fi
    echo "iree	$($VENV/bin/iree-compile --version | sed -n 2p | sed 's/^ *IREE compiler version //')"
    echo "runtime_commit	$(git -C "$IREE_SRC" rev-parse HEAD 2>/dev/null || echo '?')"
    echo "gcc	$(gcc --version | head -1)"
    echo "shim	$(stat -c %s ffi/libiree_ffi.so) bytes, ${CPUFLAG:-no -DUSE_CPU}, $DRIVERS"
    echo "venv	$VENV"
    echo "repo	$(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty)"
  } > "$LOGDIR/manifest.txt"
  ok "manifest: $LOGDIR/manifest.txt"
  echo
  echo "  Next, from this shell:"
  if [ "$IREE_TARGET" = cuda ]; then
    echo "    lake run mnist-iree                      # a few minutes; IREE_BACKEND defaults to cuda here"
    echo "    lake run cifar-iree                      # the six arms"
    echo "    LEAN_DEMO_GPU=1 lake run imagenette-iree # the seven, hours per net on one card"
  else
    echo "    LEAN_MLIR_LOWERER=iree IREE_BACKEND=llvm-cpu lake run mnist-iree          # a few minutes"
    echo "    LEAN_MLIR_LOWERER=iree IREE_BACKEND=llvm-cpu lake run imagenette-iree     # the seven, days on a CPU"
  fi
  echo "  The lakefile puts $VENV/bin on PATH for the -iree groups itself; a bare binary wants it on PATH."
fi
