#!/usr/bin/env bash
# Full-epoch run on the 4 clean GPUs (0,2,3,4) with AER watchdog.
# Kills training the instant a new PCIe BadTLP/Hardware Error appears so it
# can't cascade into a hard reset. Otherwise lets epoch 1 + validation finish
# (the [Epoch 1] line), then stops before epoch 2 to free the box.
set -u
cd /home/skoonce/lean/klawd_max_power/lean4-jax/jax

TAG="${1:-bf16}"
DEVS="${2:-0,2,3,4}"
LOG=/tmp/r34_${TAG}_epoch.log
WLOG=/tmp/r34_${TAG}_epoch_watch.log
: > "$LOG"; : > "$WLOG"
START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[watch] tag=$TAG devs=$DEVS start=$START" | tee -a "$WLOG"

CUDA_VISIBLE_DEVICES=$DEVS ../.venv/bin/python -u \
    .lake/build/generated_resnet34_imagenet.py > "$LOG" 2>&1 &
PYPID=$!
echo "[watch] training PID=$PYPID on GPUs $DEVS" | tee -a "$WLOG"

while kill -0 "$PYPID" 2>/dev/null; do
  HITS=$(journalctl -k --since "$START" 2>/dev/null | grep -icE "BadTLP|Hardware Error|AER:")
  if [ "$HITS" -gt 0 ]; then
    echo "[watch] !!! AER DETECTED ($HITS) — KILLING NOW" | tee -a "$WLOG"
    kill -9 "$PYPID" 2>/dev/null
    journalctl -k --since "$START" 2>/dev/null | grep -iE "BadTLP|Hardware Error|AER:" | tail -20 >> "$WLOG"
    echo "AER_ABORT" >> "$WLOG"; exit 0
  fi
  # Stop once epoch 1's summary line is printed (validation done).
  if grep -q "\[Epoch 1\]" "$LOG" 2>/dev/null; then
    echo "[watch] epoch 1 complete — stopping before epoch 2" | tee -a "$WLOG"
    sleep 1; kill -9 "$PYPID" 2>/dev/null
    echo "EPOCH1_DONE" >> "$WLOG"; exit 0
  fi
  sleep 3
done
echo "[watch] training exited on its own" | tee -a "$WLOG"
echo "PY_EXITED" >> "$WLOG"
