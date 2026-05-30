#!/usr/bin/env bash
# 4-GPU stock-power test with AER watchdog.
# Runs R34-ImageNet on CUDA devices 0,2,3,4 (excludes the two cards that
# threw BadTLP: bus 02=idx1, bus 62=idx5). Polls the kernel log every 2s;
# the instant a new PCIe Hardware Error / BadTLP appears, kills training
# so it can't cascade into a hard reset.
set -u
cd /home/skoonce/lean/klawd_max_power/lean4-jax/jax

LOG=/tmp/r34_4gpu.log
WLOG=/tmp/r34_4gpu_watch.log
: > "$LOG"; : > "$WLOG"
START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[watch] start marker: $START" | tee -a "$WLOG"

CUDA_VISIBLE_DEVICES=0,2,3,4 ../.venv/bin/python -u \
    .lake/build/generated_resnet34_imagenet.py > "$LOG" 2>&1 &
PYPID=$!
echo "[watch] training PID=$PYPID on GPUs 0,2,3,4" | tee -a "$WLOG"

# Stop conditions: AER detected, training exits, or we reach step 300.
while kill -0 "$PYPID" 2>/dev/null; do
  HITS=$(journalctl -k --since "$START" 2>/dev/null | grep -icE "BadTLP|Hardware Error|AER:")
  if [ "$HITS" -gt 0 ]; then
    echo "[watch] !!! AER DETECTED ($HITS lines) — KILLING training NOW" | tee -a "$WLOG"
    kill -9 "$PYPID" 2>/dev/null
    journalctl -k --since "$START" 2>/dev/null | grep -iE "BadTLP|Hardware Error|AER:" | tail -20 >> "$WLOG"
    echo "AER_ABORT" >> "$WLOG"
    exit 0
  fi
  if grep -q "step 300/" "$LOG" 2>/dev/null; then
    echo "[watch] reached step 300 clean — stopping test (success)" | tee -a "$WLOG"
    kill -9 "$PYPID" 2>/dev/null
    echo "CLEAN_300" >> "$WLOG"
    exit 0
  fi
  sleep 2
done
echo "[watch] training process exited on its own" | tee -a "$WLOG"
echo "PY_EXITED" >> "$WLOG"
