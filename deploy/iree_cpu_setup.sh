#!/bin/bash
# deploy/iree_cpu_setup.sh — the second lowerer on a box WITHOUT a GPU: `iree-compile` in .venv and a
# CPU-only `ffi/libiree_ffi.so`, so that `LEAN_MLIR_LOWERER=iree IREE_BACKEND=llvm-cpu` runs the
# `-iree` twins (`lake run mnist-iree`, `cifar-iree`, `imagenette-iree`) and the CPU gradient gates
# here. historical/IREE_BUILD.md is the runbook (the GPU builds); this is its CPU variant as commands,
# the recipe .github/workflows/iree.yml's oracle job uses, with the version moved.
#
#   bash deploy/iree_cpu_setup.sh            # every stage, in order
#   bash deploy/iree_cpu_setup.sh runtime    # start at a stage (check venv runtime shim smoke)
#   bash deploy/iree_cpu_setup.sh venv shim  # only the stages named
#
# Stages, each proving one layer before the next leans on it:
#   check    gcc, git, a python >= 3.12 or uv, disk
#   venv     .venv with iree-base-compiler and iree-base-runtime at $IREE_VERSION, plus cmake and
#            ninja — .venv because LeanMlir/Types.lean runs `.venv/bin/iree-compile` when it exists
#   runtime  the IREE runtime source at THE WHEEL'S COMMIT (read off `iree-compile --version`; a
#            skew is a HAL ABI mismatch at module load), runtime submodules only (~240 MB, not the
#            LLVM chain), the static archives with the local-task and local-sync drivers
#   shim     ffi/libiree_ffi.so from ffi/iree_ffi.c (-DUSE_CPU: no GPU driver, local-task default)
#            and the archives, linked --no-undefined so a missing archive fails here
#   smoke    the committed MobileNetV2 eval forward compiled for llvm-cpu and run on local-task
#            through iree-run-module — the 224² case that IREE 3.11.0 segfaults on (below)
#
# IREE_VERSION defaults to 3.13.0. NOT 3.11.0, CI's pin: 3.11.0's CPU task device (local-task)
# segfaults inside the first invoke of every 224² artifact — MobileNetV2's eval forward and train
# step, reproduced with iree-run-module alone, with one worker and with 8 MB worker stacks, while
# the same .vmfb runs on local-sync — and 3.12.0 and 3.13.0 run them (atlas, 2026-10-28). The
# MNIST-sized artifacts CI runs are fine on 3.11.0, which is why CI never saw it. The compiler
# wheel, the runtime wheel and the runtime commit move together: a .vmfb from one release does not
# load on another's runtime ("required module 'hal' version mismatch").
#
# Everything is teed to runs/<date>-iree-cpu-setup/setup.log, with the box's facts in
# runs/<date>-iree-cpu-setup/manifest.txt. Stale `.lake/build/*_llvm-cpu.vmfb` from an older
# release are removed at the shim stage: compileVmfb reuses any cache newer than its .mlir.
set -euo pipefail
cd "$(dirname "$0")/.."

ALL="check venv runtime shim smoke"
IREE_VERSION=${IREE_VERSION:-3.13.0}
IREE_SRC=${IREE_SRC:-$HOME/src/iree}
IREE_BUILD=${IREE_BUILD:-$HOME/src/iree-build}
VENV=.venv

if [ $# -eq 0 ]; then STAGES=$ALL
elif [ $# -eq 1 ]; then
  case " $ALL " in *" $1 "*) ;; *) echo "unknown stage '$1' (one of: $ALL)" >&2; exit 2 ;; esac
  STAGES=${ALL#*"$1"}; STAGES="$1$STAGES"          # from that stage to the end
else STAGES="$*"
fi

STAGE=${STAGES%% *}
LOGDIR=runs/$(date +%F)-iree-cpu-setup
mkdir -p "$LOGDIR"
LOG=$LOGDIR/setup.log
exec > >(tee -a "$LOG") 2>&1
echo "════ iree_cpu_setup $(date '+%F %T') — stages: $STAGES — IREE $IREE_VERSION"

say ()  { printf '\n━━━ %s\n' "$*"; }
ok ()   { printf '  ✓ %s\n' "$*"; }
warn () { printf '  ⚠ %s\n' "$*"; }
die ()  { printf '\n  ✗ %s\n\n  The log is %s. Fix the above, then resume with:\n    bash deploy/iree_cpu_setup.sh %s\n' "$1" "$LOG" "$STAGE" >&2; exit 1; }
want () { case " $STAGES " in *" $1 "*) STAGE=$1; return 0 ;; *) return 1 ;; esac; }

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
  say "runtime: $IREE_SRC at the wheel's commit, static archives in $IREE_BUILD"
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
  cmake -S "$IREE_SRC" -B "$IREE_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DBUILD_SHARED_LIBS=OFF > "$LOGDIR/cmake.log" 2>&1 || die "cmake failed — $LOGDIR/cmake.log"
  # The archives the shim links. printf moved: its own archive in 3.11, inside the unified one by
  # 3.13 — build it when the target exists and link whatever is there.
  TARGETS="runtime/src/iree/runtime/libiree_runtime_unified.a build_tools/third_party/flatcc/libflatcc_runtime.a build_tools/third_party/flatcc/libflatcc_parsing.a"
  PRINTF=$(ninja -C "$IREE_BUILD" -t targets all 2>/dev/null | grep -oE '^build_tools/third_party/printf/libprintf[a-z_]*\.a' | head -1 || true)
  ninja -C "$IREE_BUILD" $TARGETS $PRINTF > "$LOGDIR/ninja.log" 2>&1 || die "ninja failed — $LOGDIR/ninja.log"
  ok "archives: $(find "$IREE_BUILD" -name 'libiree_runtime_unified.a' -o -name 'libflatcc_*.a' -o -name 'libprintf*.a' | wc -l) ($(du -sh "$IREE_BUILD" | cut -f1) build)"
fi

# ─── shim ────────────────────────────────────────────────────────────────────
if want shim; then
  say "shim: ffi/libiree_ffi.so, CPU only"
  U=$IREE_BUILD/runtime/src/iree/runtime/libiree_runtime_unified.a
  [ -f "$U" ] || die "no $U — run the runtime stage"
  ARCHIVES="$IREE_BUILD/build_tools/third_party/flatcc/libflatcc_runtime.a $IREE_BUILD/build_tools/third_party/flatcc/libflatcc_parsing.a"
  for a in "$IREE_BUILD"/build_tools/third_party/printf/libprintf*.a; do [ -f "$a" ] && ARCHIVES="$ARCHIVES $a"; done
  gcc -fPIC -O2 -c ffi/iree_ffi.c -o "$LOGDIR/iree_ffi.o" -I"$IREE_SRC/runtime/src" -I"$IREE_BUILD/runtime/src" \
    -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl -DUSE_CPU || die "compiling ffi/iree_ffi.c failed"
  # --whole-archive keeps the driver registration symbols; --no-undefined makes a missing archive
  # fail now rather than at the first session create.
  gcc -shared -o ffi/libiree_ffi.so "$LOGDIR/iree_ffi.o" \
    -Wl,--whole-archive "$U" -Wl,--no-whole-archive \
    -Wl,--start-group $ARCHIVES -Wl,--end-group -lm -lpthread -ldl -Wl,--no-undefined \
    || die "linking ffi/libiree_ffi.so failed"
  rm -f "$LOGDIR/iree_ffi.o"
  # grep -c, not -q: under pipefail, -q's early exit SIGPIPEs nm and fails the pipeline.
  nm ffi/libiree_ffi.so | grep -c iree_hal_local_task_driver_module_register > /dev/null || die "local-task driver symbol missing — --whole-archive dropped?"
  nm -D ffi/libiree_ffi.so | grep -c ' T iree_ffi_train_step_adam$' > /dev/null || die "iree_ffi_train_step_adam not exported"
  ok "ffi/libiree_ffi.so ($(du -h ffi/libiree_ffi.so | cut -f1)), local-task + local-sync drivers, $(nm -D ffi/libiree_ffi.so | grep -c ' T iree_ffi_') entry points"
  n=$(ls .lake/build/*_llvm-cpu.vmfb .lake/build/*_vmvx.vmfb 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then rm -f .lake/build/*_llvm-cpu.vmfb .lake/build/*_vmvx.vmfb; ok "removed $n cached CPU .vmfb (a cache from another release would not load; they recompile at first use)"; fi
fi

# ─── smoke ───────────────────────────────────────────────────────────────────
if want smoke; then
  say "smoke: the MobileNetV2 eval forward on llvm-cpu, local-task (what 3.11.0 segfaults on)"
  [ -x $VENV/bin/iree-compile ] || die "no $VENV/bin/iree-compile — run the venv stage"
  M=verified_mlir/mobilenetv2_fwd_eval.mlir
  $VENV/bin/iree-compile "$M" --iree-hal-target-backends=llvm-cpu -o "$LOGDIR/smoke.vmfb" 2> "$LOGDIR/smoke_compile.log" \
    || die "iree-compile $M failed — $LOGDIR/smoke_compile.log"
  # One zero-filled --input per tensor argument of the function, read off its signature.
  INPUTS=$($VENV/bin/python -I - "$M" mobilenetv2_fwd_eval <<'PY'
import re, sys
src = open(sys.argv[1]).read()
m = re.search(r'func\.func @' + re.escape(sys.argv[2]) + r'\((.*?)\)\s*->', src, re.S)
print(' '.join(f'--input={a}f32=0' for a in re.findall(r'tensor<([0-9x]*)f32>', m.group(1))))
PY
)
  if $VENV/bin/iree-run-module --module="$LOGDIR/smoke.vmfb" --device=local-task --function=mobilenetv2_fwd_eval $INPUTS > "$LOGDIR/smoke_run.log" 2>&1 \
     && grep -q '^32x10xf32=' "$LOGDIR/smoke_run.log"; then
    ok "mobilenetv2_fwd_eval ran on local-task: $(grep -m1 -oE '^32x10xf32=\[[^]]*\]' "$LOGDIR/smoke_run.log")..."
  else
    die "mobilenetv2_fwd_eval did not run on local-task (rc, or no 32x10 result) — $LOGDIR/smoke_run.log. On IREE 3.11.0 this is the known segfault; IREE_VERSION=3.13.0 bash deploy/iree_cpu_setup.sh venv"
  fi
  rm -f "$LOGDIR/smoke.vmfb"
  {
    echo "host	$(hostname -s)"
    echo "cpu	$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
    echo "threads	$(nproc)"
    echo "iree	$($VENV/bin/iree-compile --version | sed -n 2p | sed 's/^ *IREE compiler version //')"
    echo "runtime_commit	$(git -C "$IREE_SRC" rev-parse HEAD 2>/dev/null || echo '?')"
    echo "gcc	$(gcc --version | head -1)"
    echo "shim	$(stat -c %s ffi/libiree_ffi.so) bytes, -DUSE_CPU, local-task + local-sync"
    echo "repo	$(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty)"
  } > "$LOGDIR/manifest.txt"
  ok "manifest: $LOGDIR/manifest.txt"
  echo
  echo "  Next, from this shell:"
  echo "    LEAN_MLIR_LOWERER=iree IREE_BACKEND=llvm-cpu lake run mnist-iree          # a few minutes"
  echo "    LEAN_MLIR_LOWERER=iree IREE_BACKEND=llvm-cpu lake run imagenette-iree     # the seven, days on a CPU"
  echo "  The lakefile puts .venv/bin on PATH for the -iree groups itself; a bare binary wants it on PATH."
fi
