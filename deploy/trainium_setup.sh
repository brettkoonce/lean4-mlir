#!/bin/bash
# deploy/trainium_setup.sh — from a fresh clone on an AWS Trainium instance (trn1.2xlarge) to
# `lake run mnist` and `lake run cifar` on a NeuronCore, one rung at a time. deploy/TRAINIUM.md is
# the runbook; this is its commands.
#
#   bash deploy/trainium_setup.sh            # every stage, in order (check venv jax smoke suite data lean mnist cifar)
#   bash deploy/trainium_setup.sh smoke      # start at a stage
#   bash deploy/trainium_setup.sh venv aot   # only the stages named — these two need no Trainium
#
# `aot` is not in the default order: it is for an x86 box before the instance exists (TRAINIUM.md
# §2). On the instance the plugin compiles every graph itself, and the tier-2 sweep that takes an
# hour on 88 cores would take most of a day on 8 vCPU.
#
# Each stage proves one layer before the next one leans on it, and stops at the first failure
# with the layer named, so the log says where it broke:
#   check  the box: a Neuron device (/dev/neuron0) under the aws-neuronx-dkms driver, neuron-ls,
#          the Neuron runtime (libnrt.so.1), gcc, Python 3.10+, elan, disk
#   venv   .venv-neuron with jax-neuronx (Neuron's PJRT plugin libneuronxla, its compiler
#          neuronx-cc) and jax 0.10.0 — never the pinned .venv
#   aot    (by name only) scripts/platform/neuron_aot.sh: neuronx-cc compiles the suite's tier-2
#          graphs for the target on the CPU — op coverage and limits, no device (TRAINIUM.md §2)
#   jax    JAX itself computes on the "neuron" device — the plugin works before any of our code
#   smoke  ffi/libpjrt_ffi.so (our PJRT shim) and ffi/test_pjrt_smoke.c: load the plugin, create
#          a client, run verified_mlir/linear_fwd.mlir against the host's answer, compile the
#          other five MNIST renders — the shim works before any Lean
#   suite  scripts/platform/check.sh, tiers 0–2: the shim's guards and 16 train steps against
#          XLA:CPU goldens — what admits a platform to PLATFORMS.md
#   data   MNIST into data/mnist/, CIFAR-10 into data/cifar-10/
#   lean   the toolchain (elan installs it from lean-toolchain), Mathlib's prebuilt cache, and
#          the five trainers — only those: 8 vCPU is not a box for the proof tree
#   mnist  `lake run mnist`: linear, MLP and CNN on a NeuronCore
#   cifar  `lake run cifar`: the six optimizer arms on a NeuronCore
#
# Imagenette is not a stage: run it by hand once the suite is green (TRAINIUM.md, "Run it").
#
# Everything is teed to runs/<date>-trainium-setup/setup.log, with the box's facts in
# runs/<date>-trainium-setup/manifest.txt. When a stage fails, that log is what to send back.
set -euo pipefail
cd "$(dirname "$0")/.."

ALL="check venv jax smoke suite data lean mnist cifar"
# Neuron 2.33.0 (2026-10-19): the release this runbook was written against. Its newest jax is
# 0.10.0 (the repo's other platforms are on 0.11; jax-neuronx[stable] would give 0.9.2), and only
# the jax stage uses it. The plugin serves PJRT C-API 0.70–0.104: scripts/platform/env.sh pins the
# 0.90 our header has, and the jax stage unsets that so jaxlib 0.10.0 gets its own 0.104.
JAX_VERSION=${JAX_VERSION:-0.10.0}
JAX_NEURONX_VERSION=${JAX_NEURONX_VERSION:-0.10.0.1.0.14198}
LIBNEURONXLA_VERSION=${LIBNEURONXLA_VERSION:-3.0.10135.0}
NEURONX_CC_VERSION=${NEURONX_CC_VERSION:-2.28.7631.0}
NEURON_PIP=https://pip.repos.neuron.amazonaws.com
VENV=.venv-neuron

if [ $# -eq 0 ]; then STAGES=$ALL
elif [ "$1" = aot ] && [ $# -eq 1 ]; then STAGES=aot      # by name only: alone
elif [ $# -eq 1 ]; then
  case " $ALL " in *" $1 "*) ;; *) echo "unknown stage '$1' (one of: $ALL, or aot)" >&2; exit 2 ;; esac
  STAGES=${ALL#*"$1"}; STAGES="$1$STAGES"          # from that stage to the end
else STAGES="$*"
fi

STAGE=${STAGES%% *}   # the stage running, for die's resume line
LOGDIR=runs/$(date +%F)-trainium-setup
mkdir -p "$LOGDIR"
LOG=$LOGDIR/setup.log
exec > >(tee -a "$LOG") 2>&1
echo "════ trainium_setup $(date '+%F %T') — stages: $STAGES"

say ()  { printf '\n━━━ %s\n' "$*"; }
ok ()   { printf '  ✓ %s\n' "$*"; }
warn () { printf '  ⚠ %s\n' "$*"; }
die ()  { printf '\n  ✗ %s\n\n  The log is %s. Fix the above, then resume with:\n    bash deploy/trainium_setup.sh %s\n' "$1" "$LOG" "$STAGE" >&2; exit 1; }
want () { case " $STAGES " in *" $1 "*) STAGE=$1; return 0 ;; *) return 1 ;; esac; }

NEURON_APT="the Neuron apt packages (TRAINIUM.md §1, or a Neuron DLAMI, which has them):
    . /etc/os-release
    echo \"deb [signed-by=/etc/apt/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com \$VERSION_CODENAME main\" | sudo tee /etc/apt/sources.list.d/neuron.list
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/neuron.gpg
    sudo apt-get update && sudo apt-get install -y linux-headers-\$(uname -r) aws-neuronx-dkms aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools"

# The newest python >= 3.10 on PATH (neuronx-cc ships cp311–cp314 wheels), or "" if there is none.
find_python () {
  for p in python3.14 python3.13 python3.12 python3.11 python3; do
    command -v "$p" > /dev/null 2>&1 || continue
    if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2> /dev/null; then
      command -v "$p"; return 0
    fi
  done
  return 0
}

instance_type () { cat /sys/class/dmi/id/product_name 2>/dev/null || echo '?'; }

# ─── check ───────────────────────────────────────────────────────────────────
if want check; then
  say "check: the box"
  [ "$(uname -s)" = Linux ] || die "not Linux ($(uname -s)) — Neuron ships for Linux only"
  [ "$(uname -m)" = x86_64 ] || die "not x86_64 ($(uname -m)) — Trainium instances are x86_64"
  ok "instance: $(instance_type), $(nproc) vCPU, $(awk '/MemTotal/ { printf "%d GB", $2 / 1048576 }' /proc/meminfo)"
  [ -e /dev/neuron0 ] || die "no /dev/neuron0 — not a Trainium/Inferentia instance, or the driver is not loaded (lsmod | grep neuron). Needs $NEURON_APT"
  ok "Neuron devices: $(ls /dev/neuron* | wc -l | tr -d ' ') (/dev/neuron0 …)"
  export PATH="$PATH:/opt/aws/neuron/bin"
  command -v neuron-ls > /dev/null || die "no neuron-ls (aws-neuronx-tools) — $NEURON_APT"
  neuron-ls 2>&1 | sed 's/^/    /' || die "neuron-ls failed — the driver sees no device"
  [ -f /opt/aws/neuron/lib/libnrt.so.1 ] || ldconfig -p 2>/dev/null | grep -q 'libnrt\.so\.1' \
    || die "no Neuron runtime (libnrt.so.1, aws-neuronx-runtime-lib) — the plugin links it. Needs $NEURON_APT"
  ok "Neuron runtime: libnrt.so.1 ($(dpkg-query -W -f='${Version}' aws-neuronx-runtime-lib 2>/dev/null || echo 'not from dpkg'))"
  nneff=$(find "${NEURON_COMPILE_CACHE_URL:-/var/tmp/neuron-compile-cache}" -name model.neff 2>/dev/null | wc -l)
  if [ "$nneff" -gt 0 ]; then ok "compile cache: $nneff compiled graphs (warmed? TRAINIUM.md §3)"
  else warn "compile cache empty: every graph compiles here first — warm it on an x86 box (TRAINIUM.md §3)"
  fi
  command -v gcc > /dev/null || die "no gcc — sudo apt install build-essential"
  ok "C compiler: $(gcc --version | head -1)"
  PY=$(find_python)
  [ -n "$PY" ] || die "no python >= 3.11 on PATH — sudo apt install python3 python3-venv (24.04 has 3.12)"
  ok "python: $PY ($("$PY" -V 2>&1))"
  if ! command -v lake > /dev/null 2>&1 && [ ! -x "$HOME/.elan/bin/lake" ]; then
    die "no elan/lake — install it with:
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
  then open a new terminal (or: source ~/.elan/env)"
  fi
  ok "elan: $(command -v elan || echo "$HOME/.elan/bin/elan")"
  free_gb=$(df -BG --output=avail . | awk 'NR==2 {gsub("G",""); print $1}')
  [ "$free_gb" -ge 25 ] || die "only ${free_gb} GB free here — the Neuron venv (~3 GB), Mathlib's cache, the build and the compile cache want ~25 GB"
  ok "disk: ${free_gb} GB free"
fi

# ─── venv ────────────────────────────────────────────────────────────────────
if want venv; then
  say "venv: $VENV with jax-neuronx $JAX_NEURONX_VERSION"
  if [ ! -x $VENV/bin/python3 ]; then
    PY=$(find_python)
    [ -n "$PY" ] || die "no python >= 3.11 on PATH"
    "$PY" -m venv $VENV || die "$PY -m venv failed — sudo apt install python3-venv"
    ok "created $VENV ($($VENV/bin/python3 -V))"
  else
    ok "reusing $VENV ($($VENV/bin/python3 -V))"
  fi
  $VENV/bin/pip install -q -U pip || die "pip upgrade failed"
  pkgs=("jax==$JAX_VERSION" "jaxlib==$JAX_VERSION" "jax-neuronx==$JAX_NEURONX_VERSION"
        "libneuronxla==$LIBNEURONXLA_VERSION" "neuronx-cc==$NEURONX_CC_VERSION")
  $VENV/bin/pip install -q --extra-index-url=$NEURON_PIP "${pkgs[@]}" || die "pip install ${pkgs[*]} failed"
  vers=$($VENV/bin/python3 - << 'EOF'
import importlib.metadata as m
def v(p):
    try: return m.version(p)
    except m.PackageNotFoundError: return "MISSING"
print(", ".join(f"{p} {v(p)}" for p in ("jax", "jaxlib", "jax-neuronx", "libneuronxla", "neuronx-cc")))
EOF
)
  echo "  installed: $vers"
  case "$vers" in *MISSING*) die "a package is missing — see pip's output above" ;; esac
  ls $VENV/lib/python3.*/site-packages/libneuronxla/libneuronpjrt.so > /dev/null 2>&1 \
    || die "libneuronxla installed, but its libneuronpjrt.so is not where scripts/platform/env.sh looks"
  ok "plugin: $(ls $VENV/lib/python3.*/site-packages/libneuronxla/libneuronpjrt.so)"
  [ -x $VENV/bin/neuronx-cc ] || die "no $VENV/bin/neuronx-cc — the plugin compiles by running it"
  ok "compiler: $($VENV/bin/neuronx-cc --version 2>&1 | sed -n 's/.*version //p' | head -1)"
fi

# The platform profile: the plugin, neuronx-cc and libnrt on the paths, the PJRT C-API pin,
# --auto-cast=none.
export PLATFORM_BACKEND=neuron
. scripts/platform/env.sh
[ "$PLATFORM_BACKEND" = neuron ] || die "scripts/platform/env.sh resolved backend '$PLATFORM_BACKEND', not neuron"

# ─── aot ─────────────────────────────────────────────────────────────────────
if want aot; then
  say "aot: neuronx-cc compiles the suite's tier-2 graphs for ${NEURON_AOT_TARGET:-trn1}, on the CPU"
  bash scripts/platform/neuron_aot.sh \
    || die "a graph did not compile (results.tsv and logs/ under the runs/platform/ directory it named) — an op the compiler does not lower, before any device is involved"
fi

# ─── jax ─────────────────────────────────────────────────────────────────────
if want jax; then
  say "jax: JAX computes on the neuron device"
  # JAX_PLATFORMS=neuron makes a plugin that fails to start an error rather than a quiet CPU
  # fallback; without env.sh's C-API pin, libneuronxla's hook picks jaxlib's own version.
  env -u NEURON_INTERNAL_PJRT_C_API_VERSION JAX_PLATFORMS=neuron $VENV/bin/python3 - << 'EOF' 2>&1 | grep -v '^[IW][0-9]' || die "JAX could not compute on neuron — the plugin itself does not work here (see the Python error above)"
import jax, jax.numpy as jnp
import numpy as np
d = jax.devices()[0]
assert d.platform == "neuron", f"default device is {d} ({d.platform}), not neuron"
xs = np.arange(12.0, dtype=np.float32).reshape(3, 4)
want = float((xs @ xs.T).sum())                      # the host's answer, not a constant
got = float((jnp.asarray(xs) @ jnp.asarray(xs).T).sum())
assert got == want, f"neuron gave {got}, numpy on the CPU {want}"
print(f"  ✓ {d} ({d.device_kind}), {len(jax.devices())} devices: (x @ x.T).sum() = {got}, as numpy on the CPU")
EOF
fi

# ─── smoke ───────────────────────────────────────────────────────────────────
if want smoke; then
  say "smoke: our PJRT shim on Neuron's plugin, before any Lean"
  [ -n "${PJRT_PLUGIN:-}" ] || die "no plugin: $VENV has no libneuronxla/libneuronpjrt.so — run the venv stage"
  echo "  PJRT_PLUGIN=$PJRT_PLUGIN"
  echo "  NEURON_INTERNAL_PJRT_C_API_VERSION=$NEURON_INTERNAL_PJRT_C_API_VERSION NEURON_CC_FLAGS=$NEURON_CC_FLAGS"
  # The same one-liner the lakefile's ensurePjrtShim runs.
  gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so || die "the shim did not compile"
  ok "built ffi/libpjrt_ffi.so"
  gcc -O2 ffi/test_pjrt_smoke.c -ldl -lm -o ffi/test_pjrt_smoke || die "the smoke test did not compile"
  # One NeuronCore, as a trainer runs (run.sh sets it from the GPU index).
  NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0} PJRT_FFI_TRACE=1 ./ffi/test_pjrt_smoke 2>&1 | grep -v '^[IW][0-9]' \
    || die "the shim could not drive the plugin (its [pjrt_ffi] lines above name the call that failed)"
fi

# ─── suite ───────────────────────────────────────────────────────────────────
if want suite; then
  say "suite: scripts/platform/check.sh, tiers 0–2"
  # Both NeuronCores visible: tier 1's two-device tests (allreduce, dp) run instead of skipping.
  # Each test compiles its graph first, so the 300 s default cap becomes 4 h (TRAINIUM.md §2).
  CHECK_TIMEOUT=${CHECK_TIMEOUT:-14400} bash scripts/platform/check.sh --backend neuron || die "the platform suite failed (results.tsv and logs/ under the runs/platform/ directory it named)"
fi

# ─── data ────────────────────────────────────────────────────────────────────
if want data; then
  say "data: MNIST and CIFAR-10"
  bash scripts/datasets/download_mnist.sh > /dev/null || die "the MNIST download failed"
  ok "$(ls data/mnist/*-ubyte | wc -l | tr -d ' ') MNIST files in data/mnist/"
  bash scripts/datasets/download_cifar.sh > /dev/null || die "the CIFAR-10 download failed"
  ok "CIFAR-10: $(ls -d data/cifar* 2>/dev/null | head -1)"
fi

# ─── lean ────────────────────────────────────────────────────────────────────
TRAINERS="mnist-linear-verified mnist-mlp-verified mnist-cnn-verified cifar8w-ablation cifar8w-bn-ablation"
if want lean; then
  say "lean: toolchain, Mathlib's prebuilt cache, the five MNIST and CIFAR trainers"
  [ -x "$HOME/.elan/bin/lake" ] && export PATH="$HOME/.elan/bin:$PATH"
  lake --version || die "elan could not install the toolchain in lean-toolchain ($(cat lean-toolchain))"
  lake exe cache get || die "Mathlib's prebuilt cache did not download"
  for t in $TRAINERS; do
    echo "  ▸ lake build $t"
    lake build "$t" || die "lake build $t failed"
  done
  ok "built the five trainers"
fi

# ─── mnist ───────────────────────────────────────────────────────────────────
if want mnist; then
  say "mnist: lake run mnist on a NeuronCore"
  [ -x "$HOME/.elan/bin/lake" ] && export PATH="$HOME/.elan/bin:$PATH"
  lake run mnist || die "lake run mnist failed (each trainer's log is under runs/$(date +%F)-mnist-*/)"
fi

# ─── cifar ───────────────────────────────────────────────────────────────────
if want cifar; then
  say "cifar: lake run cifar on a NeuronCore"
  [ -x "$HOME/.elan/bin/lake" ] && export PATH="$HOME/.elan/bin:$PATH"
  lake run cifar || die "lake run cifar failed (each trainer's log is under runs/$(date +%F)-cifar8w-*/)"
fi

# ─── manifest ────────────────────────────────────────────────────────────────
{
  echo "date         $(date '+%F %T %Z')"
  echo "repo         $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
  echo "host         $(hostname -s)"
  echo "instance     $(instance_type)"
  echo "cpu          $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1) ($(nproc) vCPU)"
  echo "kernel       $(uname -r)"
  echo "os           $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}")"
  echo "devices      $(ls /dev/neuron* 2>/dev/null | tr '\n' ' ')"
  for p in aws-neuronx-dkms aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools; do
    printf '%-12s %s\n' "${p#aws-neuronx-}" "$(dpkg-query -W -f='${Version}' "$p" 2>/dev/null || echo '-')"
  done
  echo "lean         $(cat lean-toolchain)"
  [ -x $VENV/bin/python3 ] && $VENV/bin/python3 -c 'import importlib.metadata as m, platform
print("python      ", platform.python_version())
for p in ("jax", "jaxlib", "jax-neuronx", "libneuronxla", "neuronx-cc"):
    try: print(f"{p:<12} {m.version(p)}")
    except m.PackageNotFoundError: pass'
  [ -n "${PJRT_PLUGIN:-}" ] && echo "plugin md5   $(md5sum "$PJRT_PLUGIN" | cut -d' ' -f1)"
  echo "pjrt c-api   ${NEURON_INTERNAL_PJRT_C_API_VERSION:-?}   cc flags: ${NEURON_CC_FLAGS:-}"
  echo "stages run   $STAGES"
} > "$LOGDIR/manifest.txt" 2>/dev/null || true
say "done: $STAGES — log $LOG, box facts $LOGDIR/manifest.txt"
