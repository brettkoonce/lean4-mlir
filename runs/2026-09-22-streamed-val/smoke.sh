#!/usr/bin/env bash
# smoke.sh — planning/streaming_val.md §4.4: one capped epoch of the ViT/ImageNet verified trainer at
# R=1 with a tagged checkpoint; the in-run bitmap must equal score-checkpoint's on that checkpoint.
# Also measures the trainer's peak RSS and the eval window. Deterministic shim on both sides.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=runs/2026-09-22-streamed-val; OUT=$D/smoke; DET=$D/vit/detshim
export PJRT_PLUGIN=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
COMMON=(SHIM_DETERMINISM=1 LD_LIBRARY_PATH="$DET" PJRT_FFI_RESIDENT=1 CUDA_VISIBLE_DEVICES=0
        LEAN_MLIR_VARIANT=adam128 LEAN_MLIR_BATCH=128 LEAN_MLIR_REPLICAS=1 PJRT_REPLICAS=1)
TAG=streamsmoke
rm -f .lake/build/vitin_adam128_ckpt_xla_${TAG}.bin*
echo "=== trainer $(date -Is)"
env "${COMMON[@]}" LEAN_MLIR_CKPT_TAG=$TAG LEAN_MLIR_MAX_EPOCHS=1 LEAN_MLIR_G2_STEPS=40 \
    LEAN_MLIR_DUMP_CORRECT=$OUT/inrun \
    .lake/build/bin/vit-imagenet-verified data 2>&1 | while IFS= read -r l; do printf '%s %s\n' "$(date +%T)" "$l"; done > $OUT/train.log &
TP=$!
sleep 2; PID=$(pgrep -n -f 'bin/vit-imagenet-verified data'); MAX=0
while kill -0 "$TP" 2>/dev/null; do
  r=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo 0); [ "${r:-0}" -gt "$MAX" ] && MAX=$r; sleep 1
done
echo "trainer peak VmRSS: $((MAX / 1024)) MiB"
CK=$(ls .lake/build/vitin_adam128_ckpt_xla_${TAG}.bin 2>/dev/null); echo "checkpoint: ${CK:-MISSING}"
grep -n 'STREAMED\|val = all\|epoch 1:\|bitmap ->\|Epoch 1\|uncaught\|error' $OUT/train.log | cut -c1-200
echo "=== score-checkpoint $(date -Is)"
env "${COMMON[@]}" LEAN_MLIR_CKPT="$CK" LEAN_MLIR_DUMP_CORRECT=$OUT/score \
    .lake/build/bin/score-checkpoint vit-in data > $OUT/score.log 2>&1
grep -o 'checkpoint: acc = [0-9]*/[0-9]* = [0-9.]*%  top5 = [0-9]*/[0-9]*' $OUT/score.log
IN=$(ls $OUT/inrun_*_e1.bin 2>/dev/null | head -1)
if [ -n "$IN" ] && cmp -s "$IN" $OUT/score.bin; then echo "✓ in-run bitmap == score-checkpoint bitmap ($(stat -c %s $OUT/score.bin) bytes)"; else echo "✗ bitmaps differ or missing ($IN)"; cmp -l "$IN" $OUT/score.bin 2>/dev/null | wc -l; fi
