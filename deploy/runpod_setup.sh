#!/bin/bash
# deploy/runpod_setup.sh — from a fresh clone on a rented 2-card CUDA pod (RunPod 2× A100 80 GB,
# ~120 GB of host RAM per card) to `lake run bench probe`, one rung at a time.
#
#   bash deploy/runpod_setup.sh             # every stage, in order
#   bash deploy/runpod_setup.sh data        # start at a stage (check venv jax smoke lean gen data warm)
#   bash deploy/runpod_setup.sh check warm  # only the stages named
#
# Then the probe itself (scripts/bench/probe_2card.sh — memory, bare-card step, loader scaling,
# a 400k-image window per path per net):
#
#   lake run bench probe
#
# Stages, each proving one layer before the next leans on it:
#   check  two NVIDIA cards, host RAM for the ImageNet tfds tree (~150 GB) in page cache, cores,
#          disk on the persistent volume, gcc, python3.12, git, curl
#   venv   .venv from jax/requirements-cuda-lock.txt — the pinned stack ares trains on; its
#          xla_cuda12 plugin is what scripts/platform/env.sh keys the `ares` profile on, so the
#          job confs, the JAX pin check and the shim's python all resolve as on ares
#   jax    JAX sees both cards and computes on them
#   smoke  ffi/libpjrt_ffi.so and ffi/test_pjrt_smoke against the plugin, before any Lean
#   lean   elan, Mathlib's prebuilt cache, the three verified trainers and the ViT layout gate
#   gen    scripts/regen_jax_generated.sh sync — .lake/build's trainers are the committed ones
#   data   imagenet2012 under $TFDS_DATA_DIR (default /workspace/tensorflow_datasets, the pod's
#          persistent volume); with TFDS_SRC=<host:/path/to/tensorflow_datasets> it rsyncs it
#          there first. ~/tensorflow_datasets is linked to it: both loaders' default
#   warm   reads the tree once so the first measured epoch is not a cold disk
#
# The persistent volume (/workspace) holds the data and the JAX compile cache (~/.jax_cache is
# linked into it), so a second pod on the same volume skips `data` and the cold compiles.
#
# Everything is teed to runs/<date>-runpod-setup/setup.log, with the box's facts in manifest.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

ALL="check venv jax smoke lean gen data warm"
VOL=${VOL:-/workspace}
export TFDS_DATA_DIR=${TFDS_DATA_DIR:-$VOL/tensorflow_datasets}
MIN_RAM_GB=${MIN_RAM_GB:-200}       # the tfds tree (~150 GB) plus the loaders' working set
TRAINERS="resnet50-imagenet-verified vit-s-imagenet-verified vit-b-imagenet-verified vit-ema-drop-render"

if [ $# -eq 0 ]; then STAGES=$ALL
elif [ $# -eq 1 ]; then
  case " $ALL " in *" $1 "*) ;; *) echo "unknown stage '$1' (one of: $ALL)" >&2; exit 2 ;; esac
  STAGES=${ALL#*"$1"}; STAGES="$1$STAGES"
else STAGES="$*"
fi

STAGE=${STAGES%% *}
LOGDIR=runs/$(date +%F)-runpod-setup
mkdir -p "$LOGDIR"
LOG=$LOGDIR/setup.log
exec > >(tee -a "$LOG") 2>&1
echo "════ runpod_setup $(date '+%F %T') — stages: $STAGES"

say ()  { printf '\n━━━ %s\n' "$*"; }
ok ()   { printf '  ✓ %s\n' "$*"; }
warn () { printf '  ⚠ %s\n' "$*"; }
die ()  { printf '\n  ✗ %s\n\n  The log is %s. Fix the above, then resume with:\n    bash deploy/runpod_setup.sh %s\n' "$1" "$LOG" "$STAGE" >&2; exit 1; }
want () { case " $STAGES " in *" $1 "*) STAGE=$1; return 0 ;; *) return 1 ;; esac; }
[ -x "$HOME/.elan/bin/lake" ] && export PATH="$HOME/.elan/bin:$PATH"

# ─── check ───────────────────────────────────────────────────────────────────
if want check; then
  say "check: the pod"
  command -v nvidia-smi > /dev/null || die "no nvidia-smi — not a CUDA pod"
  n=$(nvidia-smi -L | wc -l)
  [ "$n" -ge 2 ] || die "$n GPU(s); the 2-card jobs need two"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
  [ "$mem" -ge 70000 ] || warn "card 0 has ${mem} MiB; the probe's budgets assume 80 GB (40 GB fits by estimate, ViT-B tightest)"
  ram=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
  [ "$ram" -ge "$MIN_RAM_GB" ] && ok "host RAM ${ram} GB" \
    || warn "host RAM ${ram} GB < ${MIN_RAM_GB}: the tfds tree will not stay in page cache — reads go to disk every epoch"
  ok "$(nproc) cores visible ($(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1))"
  [ -d "$VOL" ] || die "no $VOL — attach a network volume (VOL=<path> to use another)"
  free_gb=$(df -BG --output=avail "$VOL" | tail -1 | tr -dc 0-9)
  if [ -d "$TFDS_DATA_DIR/imagenet2012" ]; then ok "imagenet2012 already on the volume; ${free_gb} GB free"
  else [ "$free_gb" -ge 200 ] && ok "${free_gb} GB free on $VOL" || die "${free_gb} GB free on $VOL; the tfds tree needs ~160"; fi
  for t in gcc git curl rsync python3.12; do command -v $t > /dev/null || die "missing $t (apt-get install -y build-essential git curl rsync python3.12 python3.12-venv)"; done
  ok "gcc, git, curl, rsync, python3.12"
fi

# ─── venv ────────────────────────────────────────────────────────────────────
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
if want venv; then
  say "venv: the pinned CUDA 12 stack from jax/requirements-cuda-lock.txt"
  [ -x .venv/bin/python3 ] || python3.12 -m venv .venv || die "python3.12 -m venv failed (python3.12-venv?)"
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r jax/requirements-cuda-lock.txt || die "the lock did not install"
  [ -f "$PLUG" ] || die "no $PLUG after install — env.sh would not resolve this box as ares"
  ok "jax $(.venv/bin/python3 -c 'import jax; print(jax.__version__)'), plugin at $PLUG"
fi

. scripts/platform/env.sh

# ─── jax ─────────────────────────────────────────────────────────────────────
if want jax; then
  say "jax: both cards compute"
  [ "$PLATFORM_HOST" = ares ] || die "env.sh resolved '$PLATFORM_HOST', not the ares profile — run the venv stage"
  JAX_PLATFORMS=cuda .venv/bin/python3 - << 'EOF' 2>&1 | grep -v '^[IW][0-9]' || die "JAX could not compute on both cards"
import jax, jax.numpy as jnp, numpy as np
ds = jax.devices()
assert len(ds) >= 2, f"JAX sees {len(ds)} device(s)"
xs = np.arange(12.0, dtype=np.float32).reshape(3, 4)
want = float((xs @ xs.T).sum())
for d in ds[:2]:
    got = float(jax.device_put(xs, d).__matmul__(jax.device_put(xs.T, d)).sum())
    assert got == want, f"{d} gave {got}, numpy {want}"
    print(f"  ✓ {d} ({d.device_kind}): (x @ x.T).sum() = {got}")
EOF
fi

# ─── smoke ───────────────────────────────────────────────────────────────────
if want smoke; then
  say "smoke: our PJRT shim on the CUDA plugin, before any Lean"
  gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so || die "the shim did not compile"
  ok "built ffi/libpjrt_ffi.so"
  gcc -O2 ffi/test_pjrt_smoke.c -ldl -lm -o ffi/test_pjrt_smoke || die "the smoke test did not compile"
  PJRT_PLUGIN=$PLATFORM_PLUGIN PJRT_FFI_TRACE=1 ./ffi/test_pjrt_smoke 2>&1 | grep -v '^[IW][0-9]' \
    || die "the shim could not drive the plugin (its [pjrt_ffi] lines above name the call that failed)"
fi

# ─── lean ────────────────────────────────────────────────────────────────────
if want lean; then
  say "lean: toolchain, Mathlib's prebuilt cache, the verified trainers"
  if ! command -v lake > /dev/null; then
    curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none \
      || die "elan did not install"
    export PATH="$HOME/.elan/bin:$PATH"
  fi
  lake --version || die "elan could not install the toolchain in lean-toolchain ($(cat lean-toolchain))"
  lake exe cache get || die "Mathlib's prebuilt cache did not download"
  for t in $TRAINERS; do
    echo "  ▸ lake build $t"
    lake build "$t" || die "lake build $t failed"
  done
  ok "built $TRAINERS"
fi

# ─── gen ─────────────────────────────────────────────────────────────────────
if want gen; then
  say "gen: the JAX trainers and shims in .lake/build match the committed copies"
  bash scripts/regen_jax_generated.sh sync || die "regen_jax_generated.sh sync failed"
  ok "synced"
fi

# ─── data ────────────────────────────────────────────────────────────────────
if want data; then
  say "data: imagenet2012 under $TFDS_DATA_DIR"
  if [ ! -d "$TFDS_DATA_DIR/imagenet2012" ]; then
    [ -n "${TFDS_SRC:-}" ] || die "no $TFDS_DATA_DIR/imagenet2012 — set TFDS_SRC=<host:/path/to/tensorflow_datasets> to copy it"
    mkdir -p "$TFDS_DATA_DIR"
    rsync -a --info=progress2 "${TFDS_SRC%/}/imagenet2012" "$TFDS_DATA_DIR/" || die "rsync from $TFDS_SRC failed"
  fi
  if [ "$TFDS_DATA_DIR" != "$HOME/tensorflow_datasets" ]; then
    [ -e "$HOME/tensorflow_datasets" ] && [ ! -L "$HOME/tensorflow_datasets" ] \
      && die "$HOME/tensorflow_datasets exists and is not a link; move it aside"
    ln -sfn "$TFDS_DATA_DIR" "$HOME/tensorflow_datasets"
  fi
  mkdir -p "$VOL/.jax_cache"
  [ -e "$HOME/.jax_cache" ] || ln -sfn "$VOL/.jax_cache" "$HOME/.jax_cache"
  ok "$(du -sh "$TFDS_DATA_DIR/imagenet2012" | cut -f1) in imagenet2012; ~/tensorflow_datasets and ~/.jax_cache point at $VOL"
fi

# ─── warm ────────────────────────────────────────────────────────────────────
if want warm; then
  say "warm: read the tree once into page cache"
  t0=$(date +%s)
  find -L "$TFDS_DATA_DIR/imagenet2012" -type f -print0 | xargs -0 -P "$(nproc)" -n 16 cat > /dev/null
  ok "read in $(( $(date +%s) - t0 )) s; $(free -g | awk '/Mem:/ {print $6}') GB now buff/cache"
fi

# ─── manifest ────────────────────────────────────────────────────────────────
{
  echo "date         $(date '+%F %T %Z')"
  echo "repo         $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
  echo "host         $(hostname -s)  profile ${PLATFORM_HOST:-?}"
  echo "cpu          $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1) × $(nproc)"
  echo "ram          $(awk '/MemTotal/ {printf "%d GB", $2/1048576}' /proc/meminfo)"
  echo "gpus         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';')"
  echo "driver       $(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2>/dev/null)"
  echo "kernel       $(uname -r)"
  echo "os           $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}")"
  echo "jax          $(.venv/bin/python3 -c 'import jax; print(jax.__version__)' 2>/dev/null || echo -)"
  echo "tfds         $TFDS_DATA_DIR"
} > "$LOGDIR/manifest.txt"
printf '\n════ done: %s. Next: lake run bench probe\n' "$STAGES"
