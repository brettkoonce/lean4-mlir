#!/usr/bin/env bash
# bench.sh — real-data throughput rows for the ImageNet trainers, both paths, on whatever box this is.
#
#     lake run bench [plan] <net … | all> [--jax | --verified]   # the way in: names → rows (lakefile's
#                                                                  # imagenetPairs), then this
#     scripts/bench/bench.sh [plan] <net:path:job> …             # rows spelled out; path = jax | verified
#
#   BENCH_EPOCHS=2      epochs per row; epoch 1 carries compile, epoch 2 is the number
#   BENCH_DIR=…         output (default runs/<date>-bench-<host>/: bench.tsv, one dir per row)
#
# Each row is a real job conf (scripts/jobs/<job>.conf), sourced the way scripts/supervise.sh sources
# it — the same command, devices and environment the book's run used — on the real ImageNet stream,
# for BENCH_EPOCHS epochs, then stopped. scripts/bench/bench_run.py timestamps the output and writes
# one TSV row per epoch (wall, train / val split, steady step time, img/s, GPU util and memory, host
# CPU busy), plus the live process's environment as env.json. `plan` sources each conf and runs its
# PRECHECK dry, launching nothing.
#
# ⛔ A bench row must never touch a real checkpoint lineage. The JAX rows point CKPT_BASE at
# $HOME/bench_ckpt/<stamp>/ (BENCH_CKPT_DIR, read by scripts/lib/jax_job.sh); the verified rows
# append LEAN_MLIR_CKPT_TAG=bench<stamp>, which wins over the conf's own tag. bench_run.py reads
# both back from /proc and kills the run on a mismatch, and every bench checkpoint is deleted after
# its row.
set -u
cd "$(dirname "$0")/../.." || exit 1

PLAN=0; [ "${1:-}" = plan ] && { PLAN=1; shift; }
ROWS=()
for r in "$@"; do
  IFS=: read -r net path job <<< "$r"
  case "$path" in jax|verified) ;; *) echo "⛔ row '$r' is not net:jax|verified:job"; exit 2 ;; esac
  [ -f "scripts/jobs/$job.conf" ] || { echo "⛔ row '$r': no scripts/jobs/$job.conf"; exit 2; }
  ROWS+=("$net $path $job")
done
[ ${#ROWS[@]} -gt 0 ] || { echo "usage: $0 [plan] <net:jax|verified:job> …   (or: lake run bench)"; exit 2; }
E="${BENCH_EPOCHS:-2}"
STAMP="$(date -u +%Y%m%d%H%M%S)"
HOST="$(hostname -s)"
OUT="${BENCH_DIR:-runs/$(date -u +%F)-bench-$HOST}"
CKDIR="$HOME/bench_ckpt/$STAMP"
TSV="$OUT/bench.tsv"

gpus_idle() {   # up to 3 min for the last row's processes to leave the cards
  local i; for i in $(seq 1 36); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0; sleep 5
  done; return 1
}

# one row, in a subshell so one conf's variables never leak into the next
row() (
  local net="$1" path="$2" job="$3" rd="$OUT/$1-$2"
  REST_EPOCHS=""; REST_SECS=1800; TEMP_MAX=0; TEMP_RESUME=""; STALL_SECS=1800
  MAX_ATTEMPTS=60; ENV_EXTRA=(); PRECHECK=""; RECIPE=""; POLL_SECS=10; EPOCH_SECS=0
  export BENCH_CKPT_DIR="$CKDIR/$job"
  [ "$PLAN" = 1 ] && export DRY_RUN=1
  # shellcheck disable=SC1090
  . "scripts/jobs/$job.conf" || { echo "⛔ $job: the conf did not source"; exit 1; }
  local expect=()
  if [ "$path" = jax ]; then
    [ "${CKPT_BASE#"$CKDIR"/}" != "$CKPT_BASE" ] || {
      echo "⛔ $job: CKPT_BASE=$CKPT_BASE is not under $CKDIR — jax_job.sh did not take BENCH_CKPT_DIR"; exit 1; }
    expect=(--expect-env "LEAN_MLIR_PARAMS_OUT=$CKPT_BASE")
  else
    expect=(--expect-env "LEAN_MLIR_CKPT_TAG=bench$STAMP")
  fi
  echo "━━━ $net / $path: $job — ${E} epoch(s), devices $DEVS"
  echo "    cmd: ${CMD[*]}"
  echo "    checkpoints: $([ "$path" = jax ] && echo "$CKPT_BASE" || echo ".lake/build/*_ckpt_xla_bench$STAMP.bin")"
  # lake's exe cache: build the exe this row runs, not just the lib — before the PRECHECK, which
  # refuses a binary older than its sources
  if [ "$path" = verified ] && [ "$PLAN" = 0 ]; then
    lake build "$(basename "${CMD[0]}")" >/dev/null || { echo "⛔ $job: the exe did not build"; exit 1; }
  fi
  # the conf's PRECHECK, on the conf as written; its checkpoint warnings describe the conf's own
  # lineage, which this row does not use. It reads RUNDIR, as under supervise.sh (some confs put
  # their per-epoch bitmaps there); a plan's goes to /tmp, as supervise.sh's DRY_RUN does.
  RUNDIR="$rd"; [ "$PLAN" = 1 ] && RUNDIR="/tmp/bench_dry_$job"
  mkdir -p "$RUNDIR"
  if [ -n "$PRECHECK" ] && ! $PRECHECK; then echo "⛔ $job: PRECHECK failed"; exit 1; fi
  # after the PRECHECK: the A3 conf refuses MAX_EPOCHS below EPOCHS, rightly, for a real run
  [ "$path" = verified ] && ENV_EXTRA+=(LEAN_MLIR_CKPT_TAG="bench$STAMP" LEAN_MLIR_MAX_EPOCHS="$E")
  [ "$PLAN" = 1 ] && exit 0
  env CUDA_VISIBLE_DEVICES="$DEVS" "${ENV_EXTRA[@]}" RUNDIR="$rd" \
    python3 scripts/bench/bench_run.py --row-dir "$rd" --tsv "$TSV" --job "$job" --path "$path" \
      --net "$net" --epochs "$E" "${expect[@]}" -- "${CMD[@]}"
  local rc=$?
  if [ "$path" = jax ]; then rm -rf "$CKDIR/$job"
  else rm -f .lake/build/*_ckpt_xla_bench"$STAMP".bin*; fi
  exit $rc
)

[ "$PLAN" = 1 ] || mkdir -p "$OUT"
failed=()
# The verified rows dlopen ffi/libpjrt_ffi.so. Rebuilt only when it is not what the source builds to,
# and only once the cards are idle; the rename is atomic, so a process that has it mapped is unharmed.
if [ "$PLAN" = 0 ]; then
  gpus_idle || { echo "⛔ GPUs busy — not starting"; exit 1; }
  so_tmp="$(mktemp ffi/.libpjrt_ffi.XXXXXX.so)"
  gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o "$so_tmp" || { echo "⛔ ffi/pjrt_ffi.c does not compile"; exit 1; }
  if cmp -s "$so_tmp" ffi/libpjrt_ffi.so; then rm -f "$so_tmp"
  else chmod 755 "$so_tmp" && mv "$so_tmp" ffi/libpjrt_ffi.so && echo "▸ rebuilt ffi/libpjrt_ffi.so from ffi/pjrt_ffi.c"; fi
fi
for r in "${ROWS[@]}"; do
  read -r net path job <<< "$r"
  if [ "$PLAN" = 0 ] && ! gpus_idle; then echo "⛔ GPUs still busy before $job — stopping"; failed+=("$job"); break; fi
  row "$net" "$path" "$job" || failed+=("$job")
done
[ "$PLAN" = 1 ] || rmdir "$CKDIR" "$HOME/bench_ckpt" 2>/dev/null
[ "$PLAN" = 1 ] || { echo; column -t -s $'\t' "$TSV" 2>/dev/null | cut -c1-400; }
[ ${#failed[@]} -eq 0 ] || { echo "⛔ failed: ${failed[*]}"; exit 1; }
