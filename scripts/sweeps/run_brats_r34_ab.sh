#!/usr/bin/env bash
# The R34-transfer A/B: same net, same data, same schedule — only the encoder's
# initialization differs. One arm per GPU, concurrently, so wall clock is one
# arm's runtime rather than two.
#
# The arms MUST stay on separate GPUs and separate build tags. They share a
# NetSpec by design, so an untagged parallel run would race on the same
# `_train_step.vmfb` mid-compile and a sequential one would overwrite its own
# checkpoints — the failure the BraTS loss ablation already paid for once.
#
# Usage:
#   ./run_brats_r34_ab.sh [epochs] [data_dir] [skip|noskip] [ctx]
#
# `ctx` (default 0) runs the 2.5D variant on `data/brats224c<2ctx+1>` — pass
# that directory as data_dir — with `ctx=` on both arms and `c<k>` in the log
# names, so a 2D and a 2.5D A/B can sit side by side in runs/.
set -euo pipefail

EPOCHS="${1:-10}"
DATA="${2:-data/brats224}"
# `noskip` runs the skipless v0 decoder. It also renames the logs, because the
# two variants are different experiments and a fixed log name silently
# overwrote one with the other once already.
VARIANT="${3:-skip}"
CTX="${4:-0}"
EXTRA=""
if [ "$VARIANT" = "noskip" ]; then EXTRA="noskip"; fi
if [ "$CTX" != "0" ]; then EXTRA="$EXTRA ctx=$CTX"; VARIANT="${VARIANT}_c$CTX"; fi

if [ ! -f "$DATA/train.bin" ] || [ ! -f "$DATA/val.bin" ]; then
  echo "missing $DATA/{train,val}.bin — build it with:"
  echo "  python3 scripts/datasets/preprocess_brats.py data/brats/Task01_BrainTumour $DATA --size 224 --seed 0 --val-full"
  echo "  (add --context <k> for the 2.5D build)"
  exit 1
fi
if [ ! -f .lake/build/jax_r34_imagenet.bin ]; then
  echo "missing .lake/build/jax_r34_imagenet.bin — the bootstrapped arm IS the demo"
  exit 1
fi

# Build before launching, not during: `lake build A` does not rebuild exe B, and
# two trainers racing a stale exe is a debugging afternoon.
lake build unet-brats-r34

mkdir -p runs
echo "launching both arms for $EPOCHS epochs on $DATA"

A_LOG="runs/brats_${VARIANT}_r34_gpu0.log"
B_LOG="runs/brats_${VARIANT}_scratch_gpu1.log"
for f in "$A_LOG" "$B_LOG"; do
  if [ -f "$f" ]; then
    mv "$f" "$f.$(date +%Y%m%d_%H%M%S).bak"
    echo "  (kept previous $f as a .bak)"
  fi
done

CUDA_VISIBLE_DEVICES=0 nohup ./.lake/build/bin/unet-brats-r34 "$DATA" "$EPOCHS" r34 $EXTRA \
  > "$A_LOG" 2>&1 &
echo "  GPU0  r34 (ImageNet bootstrap)  pid $!  -> $A_LOG"

CUDA_VISIBLE_DEVICES=1 nohup ./.lake/build/bin/unet-brats-r34 "$DATA" "$EPOCHS" scratch $EXTRA \
  > "$B_LOG" 2>&1 &
echo "  GPU1  scratch (He-init control) pid $!  -> $B_LOG"

echo
echo "score with:  python3 scripts/probes/brats_r34_ab.py $A_LOG $B_LOG"
