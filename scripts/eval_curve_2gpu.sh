#!/usr/bin/env bash
# Waits for the paced ViT run to finish, then rebuilds the full top-1/top-5
# validation curve by sweeping every-5-epoch EMA .bin checkpoints, split across
# BOTH gfx1100s (single-GPU per worker => correct top-5). Merges to vit_curve.csv.
set -u
CKPT=/home/skoonce/vit_tiny_imagenet_bf16.bin
JAXDIR=/home/skoonce/lean/proof_verify_demo/verify-v2/jax
OUTDIR=/home/skoonce/lean/proof_verify_demo/verify-v2/runs
VENV=/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python
TFDS=/home/skoonce/tensorflow_datasets
LOG=$OUTDIR/vit_curve_eval.log

echo "[curve] $(date '+%F %T') waiting for training done (final .bin + trainer exit)" > "$LOG"
while [ ! -f "$CKPT" ]; do sleep 30; done
while pgrep -f "generated_vit_tiny_imagenet.py" >/dev/null; do sleep 30; done
sleep 20

cd "$JAXDIR" || { echo "[curve] no jax dir" >> "$LOG"; exit 1; }
# epochs 5,10,...,300; split even/odd index across the two cards
A=$(seq 5 10 300 | tr '\n' ',' | sed 's/,$//')    # gpu0: 5,15,...,295
B=$(seq 10 10 300 | tr '\n' ',' | sed 's/,$//')   # gpu1: 10,20,...,300
echo "[curve] $(date '+%F %T') GPUs free — 2-GPU sweep  gpu0=[$A]  gpu1=[$B]" >> "$LOG"

HIP_VISIBLE_DEVICES=0 TFDS_DATA_DIR=$TFDS PYTHONUNBUFFERED=1 \
  "$VENV" scripts/eval_curve_worker.py "$OUTDIR/vit_curve_gpu0.csv" "$A" >> "$LOG" 2>&1 &
PA=$!
HIP_VISIBLE_DEVICES=1 TFDS_DATA_DIR=$TFDS PYTHONUNBUFFERED=1 \
  "$VENV" scripts/eval_curve_worker.py "$OUTDIR/vit_curve_gpu1.csv" "$B" >> "$LOG" 2>&1 &
PB=$!
wait $PA; wait $PB

# merge, sorted by epoch
echo "epoch,top1,top5" > "$OUTDIR/vit_curve.csv"
tail -q -n +2 "$OUTDIR/vit_curve_gpu0.csv" "$OUTDIR/vit_curve_gpu1.csv" 2>/dev/null \
  | sort -t, -k1 -n >> "$OUTDIR/vit_curve.csv"
echo "[curve] $(date '+%F %T') DONE -> $OUTDIR/vit_curve.csv" >> "$LOG"
echo "[curve] final (e300): $(tail -1 $OUTDIR/vit_curve.csv)" >> "$LOG"
