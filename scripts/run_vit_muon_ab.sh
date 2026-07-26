#!/bin/bash
# Compute-matched A/B: ViT-Tiny on Imagenette — Muon (GPU0) vs AdamW baseline (GPU1).
# Same architecture + recipe; only the optimizer differs. See planning/muon.md.
#   Muon : 2D weight matrices (Q/K/V/O + MLP, ×12 blocks) via Newton-Schulz polar
#          projection; AdamW on the edges (patch conv, LN, biases, head).
#   AdamW: the existing vit-tiny-train baseline.
# Usage: ./run_vit_muon_ab.sh [data/imagenette]
set -u
DATA="${1:-data/imagenette}"
mkdir -p runs
echo "Muon  → GPU0 → runs/vit_muon_gpu0.log"
HIP_VISIBLE_DEVICES=0 IREE_BACKEND=rocm IREE_CHIP=gfx1100 \
  nohup .lake/build/bin/vit-tiny-muon-train "$DATA" > runs/vit_muon_gpu0.log 2>&1 &
echo "  pid $!  (first step is slow: iree-compile of the full train-step ~10-15 min)"
echo "AdamW → GPU1 → runs/vit_adamw_gpu1.log"
HIP_VISIBLE_DEVICES=1 IREE_BACKEND=rocm IREE_CHIP=gfx1100 \
  nohup .lake/build/bin/vit-tiny-train "$DATA" > runs/vit_adamw_gpu1.log 2>&1 &
echo "  pid $!"
echo "watch: tail -f runs/vit_muon_gpu0.log runs/vit_adamw_gpu1.log"
