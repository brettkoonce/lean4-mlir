#!/bin/bash
# The trainer runs UNDER gdb (gdb must be an ancestor under ptrace_scope=1). A watcher sends the
# inferior SIGINT when the box enters a storm; gdb stops, dumps every thread's stack, continues.
set -u
cd "$(dirname "$0")/../.."
D=runs/2026-09-11-imagenet-probe-postfix; name=diag_vit_w8_gdb2
export CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
  SHIM_PYTHON="$PWD/.venv/bin/python3" PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 PJRT_FFI_RESIDENT=1 \
  SHIM_WORKERS=8 LEAN_MLIR_VARIANT=adamdp128x4wxclipdrop LEAN_MLIR_BATCH=128 \
  LEAN_MLIR_PROBE_WARM=200 LEAN_MLIR_MAX_STEPS=600 LEAN_MLIR_CKPT_TAG="diag-$name" \
  LEAN_MLIR_PROBE_DUMP=$D/$name.steps.tsv
gdb -batch -q -ex 'set pagination off' -ex 'handle SIGINT stop print nopass' -ex 'handle SIGPIPE nostop noprint pass' \
    -ex run \
    -ex 'echo \n=== STORM 1 ===\n' -ex 'thread apply all bt 16' -ex continue \
    -ex 'echo \n=== STORM 2 ===\n' -ex 'thread apply all bt 16' -ex continue \
    -ex 'echo \n=== STORM 3 ===\n' -ex 'thread apply all bt 16' -ex continue \
    --args .lake/build/bin/vit-imagenet-verified data > $D/$name.gdb.txt 2>&1 &
G=$!
sleep 20; n=0
while kill -0 $G 2>/dev/null && [ $n -lt 3 ]; do
  T=$(pgrep -x vit-imagenet-ve | head -1); r=$(awk '/procs_running/{print $2}' /proc/stat)
  if [ -n "$T" ] && [ "$r" -ge 40 ] && grep -q "step 2/" $D/$name.gdb.txt; then
    n=$((n+1)); echo "watcher: storm $n at $(date +%T) procs_running=$r -> SIGINT $T" >> $D/$name.watch.txt
    kill -INT $T; sleep 45
  fi
  sleep 0.5
done
wait $G; grep -E "PROBE-SPREAD" $D/$name.gdb.txt
