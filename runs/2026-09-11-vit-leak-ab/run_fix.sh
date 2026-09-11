#!/bin/bash
# Post-fix re-run of arm A (the production config) with the owner-thread batch buffers
# (VerifiedTrain.readInto). Success = the mean falling to the median (~220 ms/step), RssAnon flat,
# LazyFree ~0. Samples RssAnon / LazyFree / minflt / free RAM against the STEP counter every 5 s.
set -u
cd "$(dirname "$0")/../.."
D=runs/2026-09-11-vit-leak-ab
name=${1:-A_fix_owner_thread}; shift || true
log="$D/$name.log"
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
  SHIM_PYTHON="$PWD/.venv/bin/python3" \
  PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
  LEAN_MLIR_VARIANT=adamdp128x4wxclipdrop LEAN_MLIR_BATCH=128 \
  LEAN_MLIR_PROBE_WARM=50 LEAN_MLIR_MAX_STEPS=${STEPS:-400} \
  LEAN_MLIR_CKPT_TAG="leakab-$name" "$@" \
  timeout 2400 .lake/build/bin/vit-imagenet-verified data > "$log" 2>&1 &
job=$!
tsv="$D/$name.rss.tsv"
echo -e "t_s\tstep\trss_anon_gb\tlazyfree_gb\tminflt\tfree_gb" > "$tsv"
echo -e "t_s\tstep\tpid\tcpu_s\tstat\trss_mb" > "$D/$name.shims.tsv"
t0=$(date +%s)
while kill -0 $job 2>/dev/null; do
  p=$(ps -eo pid,rss,comm --no-headers --sort=-rss | awk '$3 ~ /vit-imagenet/ {print $1; exit}')
  if [ -n "${p:-}" ]; then
    ra=$(awk '/RssAnon/{print $2}' /proc/$p/status 2>/dev/null)
    lf=$(awk '/LazyFree/{print $2}' /proc/$p/smaps_rollup 2>/dev/null)
    mf=$(awk '{print $10}' /proc/$p/stat 2>/dev/null)
    st=$(grep -oE 'step [0-9]+/' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
    fr=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
    [ -n "${ra:-}" ] && printf "%s\t%s\t%.2f\t%.2f\t%s\t%s\n" "$(( $(date +%s) - t0 ))" "${st:-0}" \
      "$(python3 -c "print($ra/1048576)")" "$(python3 -c "print(${lf:-0}/1048576)")" "${mf:-0}" "$fr" >> "$tsv"
    # per-producer: pid, cumulative cpu seconds, state, rss MB — one line per shim process
    ps -eo pid,cputimes,stat,rss,args --no-headers 2>/dev/null | awk -v t="$(( $(date +%s) - t0 ))" -v st="${st:-0}" \
      '/generated_.*_shim\.py/ && !/awk/ {printf "%s\t%s\t%s\t%s\t%s\t%d\n", t, st, $1, $2, $3, $4/1024}' >> "$D/$name.shims.tsv"
  fi
  sleep 5
done
wait $job
echo "$name done: $(grep -E 'PROBE-SPREAD' "$log" | tail -1)"
