#!/bin/bash
# One card's slice of the 80-epoch imagenette-iree run: wait for the card to be free (another
# tier may still be finishing on it), then train each assigned net in sequence through run.sh with
# the environment runDemoGroup sets (LEAN_MLIR_LOWERER=iree, .venv-iree on PATH, IREE_BACKEND=cuda,
# CKPT_TAG=iree so nothing collides with the XLA checkpoints). 80 epochs is the default; no caps.
# Log + wall per net land in runs/2026-11-05-imagenette-iree-cuda/ as run.sh writes them.
#   setsid -f nohup runs/2026-11-05-iree-cuda-tiers/imagenette_card.sh <gpu> <net>... > runs/2026-11-05-iree-cuda-tiers/im8-<gpu>.log 2>&1
cd "$(dirname "$0")/../.."
gpu=$1; shift
export PATH="$PWD/.venv-iree/bin:$PATH"
export LEAN_MLIR_LOWERER=iree IREE_BACKEND=cuda LEAN_MLIR_CKPT_TAG=iree
export RUN_LOG_DIR=runs/2026-11-05-imagenette-iree-cuda
mkdir -p "$RUN_LOG_DIR"
# Wait for the card to be free: no compute process on it. (CIFAR holds a card at <200 MiB, so a
# memory threshold would read it as free — count processes instead.)
while true; do
  napps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$napps" = 0 ] && break
  echo "$(date '+%T') gpu $gpu busy ($napps compute proc), waiting…"; sleep 60
done
for n in "$@"; do
  echo "━━━ $(date '+%F %T') gpu $gpu: $n (80 ep)"
  /usr/bin/time -f "$n wall %e s, maxrss %M KB" -o "$RUN_LOG_DIR/$n.time" nice -n 19 ./run.sh "$n" "$gpu" cuda > "$RUN_LOG_DIR/$n.out" 2>&1 \
    && echo "  ✓ $(date '+%T') $(cat "$RUN_LOG_DIR/$n.time")" || echo "  ✗ $(date '+%T') $n exit $? — $RUN_LOG_DIR/$n.out"
done
echo "━━━ $(date '+%F %T') gpu $gpu: all done"
