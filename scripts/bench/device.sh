#!/usr/bin/env bash
# device.sh — the rendered step alone on the two cards: XLA's peak per card, then a bare-device step
# time, no trainer and no loader. Each row is a `*-2card` job's artifact.
#
#     lake run bench device <a2 | a1 | vits | vitb> …
set -u
cd "$(dirname "$0")/../.." || exit 1
[ $# -gt 0 ] || { echo "usage: lake run bench device <a2 | a1 | vits | vitb> …"; exit 2; }
PY="${DEVICE_PY:-.venv/bin/python}"
OUT="${DEVICE_DIR:-runs/$(date -u +%F)-device-$(hostname -s)}"; mkdir -p "$OUT"
MIB=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits)
BUDGET=$(awk -v m="$MIB" 'BEGIN{printf "%.2f", m*0.75/1024}')   # the plugin's default arena
export CUDA_VISIBLE_DEVICES=0,1
for n in "$@"; do
  case "$n" in
    a2)   a=resnet50in_lambaccdp4x256wxclipdropbcebf16 ;;
    a1)   a=resnet50in_lambaccdp4x256wxclipdropbcewd001bf16 ;;
    vits) a=vitsin_emadp512wxdropeps0000001smxerfbf16 ;;
    vitb) a=vitbin_emaaccdp2x256wxdropeps0000001smxerfbf16 ;;
    *) echo "⛔ no device row '$n'"; exit 2 ;;
  esac
  f="verified_mlir/${a}_train_step.mlir"
  { echo "== $n: $f"
    "$PY" scripts/probes/bf16_peak_memory.py --budget-gib "$BUDGET" --replicas 2 "$f"
    "$PY" scripts/probes/bf16_device_step.py --replicas 2 --reps 15 --warmup 5 "$f"
  } 2>&1 | tee -a "$OUT/device.txt"
done
