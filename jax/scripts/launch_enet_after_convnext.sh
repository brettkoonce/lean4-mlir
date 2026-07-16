#!/usr/bin/env bash
# One-shot launcher: wait for the ConvNeXt-T 300ep run to finish cleanly,
# then start the EfficientNet-B0 350ep run on the same 4 clean GPUs.
# Only fires on a clean completion ("TRAINING COMPLETE"); if ConvNeXt instead
# hits MAX_ATTEMPTS it never prints that, so this just keeps waiting (no false
# launch) and a human handles it.
set -u
cd "$(dirname "$0")/.." || exit 1
CVX_MASTER=/tmp/convnext_t_300ep_4gpu_master.log

echo "[launch] $(date '+%F %T') waiting for ConvNeXt 300ep TRAINING COMPLETE ..."
until grep -q "TRAINING COMPLETE" "$CVX_MASTER" 2>/dev/null; do
  sleep 60
done
echo "[launch] $(date '+%F %T') ConvNeXt complete detected."

# Small settle so the ConvNeXt trainer has fully released GPU memory.
sleep 30
setsid nohup bash scripts/supervise_enet_b0_350ep_4gpu_duty.sh \
  > /tmp/enet_b0_350ep_4gpu_sup.out 2>&1 &
disown
echo "[launch] $(date '+%F %T') launched ENet-B0 350ep supervisor."
