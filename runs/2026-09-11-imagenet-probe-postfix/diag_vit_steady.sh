#!/bin/bash
# Reproduce the probe's ViT f32 row (WARM=200, 600 steps, 8 producers) with the per-step dump,
# and sample vmstat (disk blocks in, idle %) + page cache + producer CPU every 5 s beside it.
set -u
cd "$(dirname "$0")/../.."
D=runs/2026-09-11-imagenet-probe-postfix; name=${1:-diag_vit_w8}; shift || true
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
  SHIM_PYTHON="$PWD/.venv/bin/python3" PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 PJRT_FFI_RESIDENT=1 \
  SHIM_WORKERS=8 LEAN_MLIR_VARIANT=adamdp128x4wxclipdrop LEAN_MLIR_BATCH=128 \
  LEAN_MLIR_PROBE_WARM=200 LEAN_MLIR_MAX_STEPS=600 LEAN_MLIR_CKPT_TAG="diag-$name" \
  LEAN_MLIR_PROBE_DUMP=$D/$name.steps.tsv "$@" \
  timeout 2400 .lake/build/bin/vit-imagenet-verified data > $D/$name.log 2>&1 &
job=$!
vmstat -t 5 > $D/$name.vmstat.txt 2>&1 &
vm=$!
echo -e "t_s\tstep\tcached_gb\tavail_gb\tprod_cpu_s_total\tdisk_rd_sectors\ttrainer_cpu_ticks\tmain_state\tmain_wchan\ttrainer_threads_R\ttrainer_vmlck_kb\ttrainer_rssanon_kb\tmlocked_kb" > $D/$name.sys.tsv
t0=$(date +%s)
while kill -0 $job 2>/dev/null; do
  st=$(grep -oE 'step [0-9]+/' $D/$name.log 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  ca=$(awk '/^Cached:/{printf "%.0f", $2/1048576}' /proc/meminfo); av=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
  pc=$(ps -eo cputimes,args --no-headers | awk '/generated_.*_shim\.py/ && !/awk/ {s+=$1} END {print s+0}')
  rd=$(awk '$3 ~ /^nvme[0-9]+n[0-9]+$/ {s+=$6} END {print s+0}' /proc/diskstats)
  tp=$(pgrep -x vit-imagenet-ve | head -1)
  if [ -n "$tp" ]; then
    tc=$(awk '{print $14+$15}' /proc/$tp/stat 2>/dev/null); ms=$(awk '{print $3}' /proc/$tp/task/$tp/stat 2>/dev/null); mw=$(cat /proc/$tp/task/$tp/wchan 2>/dev/null)
    nr=$(for t in /proc/$tp/task/*; do awk '{print $3}' $t/stat 2>/dev/null; done | grep -c R)
    lk=$(awk '/VmLck/{print $2}' /proc/$tp/status); ra=$(awk '/RssAnon/{print $2}' /proc/$tp/status)
  else tc=0; ms=-; mw=-; nr=0; lk=0; ra=0; fi
  ml=$(awk '/^Mlocked/{print $2}' /proc/meminfo)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(( $(date +%s) - t0 ))" "${st:-0}" "$ca" "$av" "$pc" "$rd" "$tc" "$ms" "$mw" "$nr" "${lk:-0}" "${ra:-0}" "${ml:-0}" >> $D/$name.sys.tsv
  sleep 5
done
kill $vm 2>/dev/null; wait $job
grep -E "PROBE" $D/$name.log
