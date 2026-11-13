#!/bin/bash
# deploy/vast/bench.sh — what the image runs at boot (supervisor program lean4-mlir-bench), and by
# hand:  bash /opt/lean4-mlir/deploy/vast/bench.sh
#
#   1 box     cards, driver, cores, the cgroup CPU quota (the shell is pinned to it), RAM, disk
#   2 jax     every card computes on the pinned stack
#   3 smoke   our PJRT shim drives the plugin
#   4 device  per net: XLA's peak per card and the bare-card step of the job's own render, then the
#             hours a verified pass would take at that step with a loader that keeps up
#   5 fed     only with imagenet2012 under $TFDS_DATA_DIR: page-cache warm, the loader alone at
#             FEED_WORKERS, then `lake run bench` rows on both paths, BENCH_IMAGES images each
#
#   BENCH_AUTORUN=0     boot without running (set it in the vast template's environment)
#   BENCH_NETS="a1 vitb cnxs cnxb a2 vits"       device rows, in this order
#   BENCH_FED_NETS="a1 vitb cnxs cnxb"           fed rows (stage 5)
#   BENCH_IMAGES=150000  BENCH_PRICE=<$/hr, for the cost column>  BENCH_FORCE=1 (run again)
#   BENCH_OUT=/workspace/lean4-mlir-bench         results; one directory per run
#
# It runs once: a container restart finds $BENCH_OUT/.done and stops there. Nothing here ends the
# rental — the instance bills until it is destroyed.
set -u
cd "${LEAN4MLIR:-/opt/lean4-mlir}" || exit 1
export PATH="${ELAN_HOME:-/opt/elan}/bin:$PATH"
[ "${BENCH_AUTORUN:-1}" = 1 ] || { echo "BENCH_AUTORUN=${BENCH_AUTORUN}: not running. By hand: bash $PWD/deploy/vast/bench.sh"; exit 0; }
ROOT=${BENCH_OUT:-/workspace/lean4-mlir-bench}
[ ! -e "$ROOT/.done" ] || [ -n "${BENCH_FORCE:-}" ] || { echo "already ran: $(cat "$ROOT/.done") (BENCH_FORCE=1 to run again)"; exit 0; }
OUT=$ROOT/$(date -u +%F-%H%M)-$(hostname -s)
mkdir -p "$OUT" || exit 1
exec > >(tee -a "$OUT/bench.log") 2>&1
say ()  { printf '\n━━━ %s  [%s]\n' "$*" "$(date -u +%T)"; }
ok ()   { printf '  ✓ %s\n' "$*"; }
warn () { printf '  ⚠ %s\n' "$*"; WARNED=1; }
WARNED=0
echo "════ lean4-mlir bench, image $(cat .image_rev 2>/dev/null || echo '?'), $(date -u '+%F %T') UTC → $OUT"

# ─── 1 box ───────────────────────────────────────────────────────────────────
say "box"
command -v nvidia-smi > /dev/null || { echo "  ✗ no nvidia-smi — not a CUDA instance"; exit 1; }
NGPU=$(nvidia-smi -L | wc -l)
CPUS=$(nproc)
if read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ] && [ $((q / per)) -lt "$CPUS" ]; then
  CPUS=$((q / per))
  taskset -cp "0-$((CPUS - 1))" $$ > /dev/null && ok "CPU quota is $CPUS cores of $(nproc) visible; pinned to 0-$((CPUS - 1))"
fi
{
  echo "image        $(cat .image_rev 2>/dev/null)"
  echo "gpus         $NGPU × $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0)"
  echo "driver       $(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)"
  echo "cpu          $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)"
  echo "cores        $CPUS usable of $(nproc) visible (cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo -))"
  echo "ram          $(awk '/MemTotal/ {printf "%d GB", $2/1048576}' /proc/meminfo) (cgroup memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo -))"
  echo "disk         $(df -BG --output=avail "$ROOT" | tail -1 | tr -d ' ') free under $ROOT"
  echo "kernel       $(uname -r)"
} | tee "$OUT/box.txt" | sed 's/^/  /'
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader | sed 's/^/  /'

. scripts/platform/env.sh
PY=.venv/bin/python3

# ─── 2 jax ───────────────────────────────────────────────────────────────────
say "jax: every card computes"
JAX_PLATFORMS=cuda $PY - << 'PYEOF' 2>&1 | grep -v '^[IW][0-9]'
import jax, numpy as np
ds = jax.devices()
xs = np.arange(12.0, dtype=np.float32).reshape(3, 4)
want = float((xs @ xs.T).sum())
for d in ds:
    got = float(jax.device_put(xs, d).__matmul__(jax.device_put(xs.T, d)).sum())
    assert got == want, f"{d} gave {got}, numpy {want}"
    print(f"  ✓ {d} ({d.device_kind}): (x @ x.T).sum() = {got}")
print(f"  jax {jax.__version__}, {len(ds)} device(s)")
PYEOF
[ "${PIPESTATUS[0]}" = 0 ] || { echo "  ✗ JAX could not compute on the cards — the pinned stack does not run here; nothing below would"; exit 1; }

# ─── 3 smoke ─────────────────────────────────────────────────────────────────
say "smoke: our PJRT shim on the CUDA plugin"
PJRT_PLUGIN=$PLATFORM_PLUGIN PJRT_FFI_TRACE=1 ./ffi/test_pjrt_smoke 2>&1 | grep -v '^[IW][0-9]' | tail -25
[ "${PIPESTATUS[0]}" = 0 ] && ok "smoke passed" \
  || warn "smoke did not pass. On the A100s this was linear_fwd at 2.4e-3 against a 1e-3 bound (TF32 matmul), and the trainers ran; a failed CALL above is a real stop"

# ─── 4 device ────────────────────────────────────────────────────────────────
# net  render (the 4-card job's own)  images a step  epochs
ROWS4="a2   resnet50in_lambaccdp4x128wxclipdropbcebf16         512  300
a1   resnet50in_lambaccdp4x128wxclipdropbcewd001bf16    512  600
vits vitsin_emadp256x4wxdropeps0000001smxerfbf16        1024 300
vitb vitbin_emaaccdp2x128x4wxdropeps0000001smxerfbf16   512  300
cnxs convnextsin_emaaccdp16xwxdroperfbf16               256  300
cnxb convnextbin_emaaccdp16xwxdroperfbf16               256  300"
NETS=${BENCH_NETS:-a1 vitb cnxs cnxb a2 vits}
if [ "$NGPU" -ge 4 ]; then
  say "device: the 4-card renders, bare, on cards 0-3"
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  MIB=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits)
  BUDGET=$(awk -v m="$MIB" 'BEGIN{printf "%.2f", m*0.75/1024}')
  for n in $NETS; do
    read -r _ art imgs ep <<< "$(echo "$ROWS4" | awk -v n="$n" '$1==n')"
    [ -n "${art:-}" ] || { warn "no device row '$n'"; continue; }
    f=verified_mlir/${art}_train_step.mlir
    echo "== $n: $f" | tee "$OUT/device-$n.txt"
    { $PY scripts/probes/bf16_peak_memory.py --budget-gib "$BUDGET" --replicas 4 "$f"
      $PY scripts/probes/bf16_device_step.py --replicas 4 --reps 15 --warmup 5 "$f"
    } 2>&1 | grep -v 'cuda_timer.cc' | tee -a "$OUT/device-$n.txt"
    ms=$(awk '/ms\/step/ {print $1; exit}' "$OUT/device-$n.txt")
    [ -n "$ms" ] || { warn "$n: no step time — see $OUT/device-$n.txt"; continue; }
    printf '%s\t%s\t%s\t%s\t%s\n' "$n" "$ms" "$imgs" "$ep" \
      "$(awk '/_train_step.mlir/ && / G / {print $2; exit}' "$OUT/device-$n.txt")" >> "$OUT/device.tsv"
  done
  unset CUDA_VISIBLE_DEVICES
elif [ "$NGPU" -ge 2 ]; then
  say "device: $NGPU cards — the 2-card renders (scripts/bench/device.sh)"
  DEVICE_DIR=$OUT bash scripts/bench/device.sh $NETS
else
  warn "one card: every ImageNet render here is 2 or 4 replicas — no device rows"
fi

summary () {
  echo "lean4-mlir bench — $(sed -n 's/^gpus *//p' "$OUT/box.txt"), $(sed -n 's/^cores *//p' "$OUT/box.txt")"
  [ -s "$OUT/device.tsv" ] || { echo "(no device rows)"; return; }
  echo
  echo "Bare-card step and the verified pass it implies, loader keeping up (2× A100: a2 130.6, vits 234.2, vitb 294.2 ms on the 2-card renders):"
  awk -F'\t' -v price="${BENCH_PRICE:-0}" '
    BEGIN { printf "  %-5s %10s %8s %10s %9s %7s%s\n", "net", "ms/step", "img/s", "peak/card", "epochs", "hours", (price > 0 ? "    cost" : "") }
    { h = $4 * 1281167 / $3 * $2 / 3.6e6; tot += h
      printf "  %-5s %10.1f %8.0f %9sG %9d %7.0f", $1, $2, $3 / $2 * 1000, $5, $4, h
      if (price > 0) printf "  $%6.0f", h * price
      printf "\n" }
    END { printf "  %-5s %10s %8s %10s %9s %7.0f", "total", "", "", "", "", tot
          if (price > 0) printf "  $%6.0f", tot * price
          printf "\n" }' "$OUT/device.tsv"
  echo "  (training steps only; eval and checkpoints add a few percent. img/s is what the loader must supply.)"
}

# ─── 5 fed ───────────────────────────────────────────────────────────────────
export TFDS_DATA_DIR=${TFDS_DATA_DIR:-/workspace/tensorflow_datasets}
if [ -d "$TFDS_DATA_DIR/imagenet2012" ]; then
  say "fed: imagenet2012 under $TFDS_DATA_DIR"
  [ -e "$HOME/tensorflow_datasets" ] || ln -sfn "$TFDS_DATA_DIR" "$HOME/tensorflow_datasets"
  t0=$(date +%s)
  find -L "$TFDS_DATA_DIR/imagenet2012" -type f -print0 | xargs -0 -P "$CPUS" -n 16 cat > /dev/null
  ok "read once in $(( $(date +%s) - t0 )) s; $(free -g | awk '/Mem:/ {print $6}') GB buff/cache"
  FEED_DIR=$OUT FEED_WORKERS="${FEED_WORKERS:-8 16 32 64}" lake run bench feed a1 vitb \
    || warn "the feed rows failed"
  # shellcheck disable=SC2086
  BENCH_DIR=$OUT lake run bench ${BENCH_FED_NETS:-a1 vitb cnxs cnxb} --images "${BENCH_IMAGES:-150000}" \
    || warn "a fed row failed — $OUT/<net>-<path>/ has its log"
else
  say "fed: skipped — no imagenet2012 under $TFDS_DATA_DIR"
fi

say "summary"
summary | tee "$OUT/summary.txt"
[ "$WARNED" = 0 ] || echo "  ⚠ there were warnings above"
echo "$OUT" > "$ROOT/.done"
printf '\n════ BENCH DONE %s — results in %s (bench.log, summary.txt, device-*.txt). The instance is still billing.\n' "$(date -u +%T)" "$OUT"
