#!/bin/bash
# A/B: does the trainer's host-memory growth track the depth-n SHIM prefetch?
# Arm A = default (depth = SHIM_WORKERS = 8). Arm B = LEAN_MLIR_PREFETCH=0, the
# documented control in VerifiedTrain.lean (read blocks the step, same bytes, same order).
# Both arms sample the trainer's RssAnon against its step counter.
set -u
cd "$(dirname "$0")/../.."
D=runs/2026-09-11-vit-leak-ab
run_arm () {
  local name="$1"; shift
  local log="$D/$name.log"
  env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
    SHIM_PYTHON="$PWD/.venv/bin/python3" \
    PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
    LEAN_MLIR_VARIANT=adamdp128x4wxclipdrop LEAN_MLIR_BATCH=128 \
    LEAN_MLIR_PROBE_WARM=50 LEAN_MLIR_MAX_STEPS=400 \
    LEAN_MLIR_CKPT_TAG="leakab-$name" "$@" \
    timeout 2400 .lake/build/bin/vit-imagenet-verified data > "$log" 2>&1 &
  local job=$!
  : > "$D/$name.rss.tsv"
  echo -e "epoch_s\tstep\trss_anon_gb" >> "$D/$name.rss.tsv"
  local t0=$(date +%s)
  while kill -0 $job 2>/dev/null; do
    local p; p=$(ps -eo pid,rss,comm --no-headers --sort=-rss | awk '$3 ~ /vit-imagenet/ {print $1; exit}')
    if [ -n "${p:-}" ]; then
      local ra; ra=$(awk '/RssAnon/{print $2}' /proc/$p/status 2>/dev/null)
      local st; st=$(grep -oE 'step [0-9]+/' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
      [ -n "${ra:-}" ] && printf "%s\t%s\t%.2f\n" "$(( $(date +%s) - t0 ))" "${st:-0}" "$(python3 -c "print($ra/1048576)")" >> "$D/$name.rss.tsv"
    fi
    sleep 5
  done
  wait $job
  echo "$name done: $(grep -E 'PROBE-SPREAD' "$log" | tail -1)"
}
run_arm A_prefetch_depth8
run_arm B_prefetch_off LEAN_MLIR_PREFETCH=0
echo ALL_DONE
