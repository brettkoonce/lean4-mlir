#!/usr/bin/env bash
# GPU 2: ESM-2 150M embeddings for every chain → the EUs with the same model (CPU; the 150M
# attention maps of a 1,693-residue target would not fit the GPU) → pack esm150 → train 64 ch.
set -euo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
P=.venv-casp/bin/python
export CUDA_VISIBLE_DEVICES=2
echo "[$(date +%H:%M)] embed 150M"; $P scripts/datasets/casp16_embed.py --model esm2_t30_150M_UR50D --out emb150 --device cuda --threads 8 --max-tokens 12288 2>&1 | grep -v Warning | tail -3
echo "[$(date +%H:%M)] targets 150M (cpu)"; $P scripts/datasets/casp16_targets.py --model esm2_t30_150M_UR50D --out targets_esm150 --device cpu 2>&1 | grep -v Warning | tail -2
echo "[$(date +%H:%M)] pack esm150"; $P scripts/datasets/casp16_pack.py --features esm150 2>&1 | tail -2
echo "[$(date +%H:%M)] train esm150"; d=runs/2026-10-02-distogram-r16x64-e30-esm150; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm150 dim=649 epochs=30 batch=32 ch=64 units=16 seed=1 tag=e30-esm150 > $d/train.log 2>&1
echo "[$(date +%H:%M)] done: $(grep trained $d/train.log)"
