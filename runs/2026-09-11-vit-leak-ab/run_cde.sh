#!/bin/bash
# Arms C, D, E run back to back. No pgrep gating: `pgrep -f` matches the spawning
# shell's own command line, which is what deadlocked run2/run3.
set -u
cd "$(dirname "$0")/../.."
D=runs/2026-09-11-vit-leak-ab
run_arm () {
  local name="$1"; local workers="$2"; shift 2
  local log="$D/$name.log"
  env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
    SHIM_PYTHON="$PWD/.venv/bin/python3" \
    PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 PJRT_FFI_RESIDENT=1 SHIM_WORKERS="$workers" \
    LEAN_MLIR_VARIANT=adamdp128x4wxclipdrop LEAN_MLIR_BATCH=128 \
    LEAN_MLIR_PROBE_WARM=50 LEAN_MLIR_MAX_STEPS=400 \
    LEAN_MLIR_CKPT_TAG="leakab-$name" "$@" \
    timeout 2400 .lake/build/bin/vit-imagenet-verified data > "$log" 2>&1 &
  local job=$!
  echo -e "t_s\tstep\trss_anon_gb\tfree_gb" > "$D/$name.rss.tsv"
  local t0=$(date +%s)
  while kill -0 $job 2>/dev/null; do
    local p; p=$(ps -eo pid,rss,comm --no-headers --sort=-rss | awk '$3 ~ /vit-imagenet/ {print $1; exit}')
    if [ -n "${p:-}" ]; then
      local ra; ra=$(awk '/RssAnon/{print $2}' /proc/$p/status 2>/dev/null)
      local st; st=$(grep -oE 'step [0-9]+/' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
      [ -n "${ra:-}" ] && printf "%s\t%s\t%.2f\t%s\n" "$(( $(date +%s) - t0 ))" "${st:-0}" "$(python3 -c "print($ra/1048576)")" "$(free -g | awk 'NR==2{print $4}')" >> "$D/$name.rss.tsv"
    fi
    sleep 5
  done
  wait $job
  echo "$name done: $(grep -E 'PROBE-SPREAD' "$log" | tail -1)"
}
run_arm C_w8_depth4      8 LEAN_MLIR_PREFETCH_DEPTH=4
run_arm D_w4             4
run_arm E_w8_mmap_pinned 8 MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576
echo ALL_DONE_CDE
