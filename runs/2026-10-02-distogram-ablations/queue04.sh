#!/usr/bin/env bash
# Night queue 4 (03:40, Brett: "let's do the 650m model at some point"). GPU 2, once the ω/φ fold of
# the orient run is scored: drop queue03's esm150 seed-2 run if it has started, then the ESM-2 650M
# chain — embeddings streamed straight into the packed pool (no per-chain files: disk), the EUs on
# CPU (the 650M attention maps of a 1,693-residue target do not fit a card), pack targets + valsub,
# train 64 ch × 30 ep, finish (predict / assemble / fold / score).
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
P=.venv-casp/bin/python
O=.lake/build/distogram_r16x64_orient_train_full_e30-orient_targets
while ! [ -f "$O/fold_scores_orient.csv" ] && ! grep -q "esm150 seed-2 run" $A/queue03.log; do sleep 60; done
sleep 90
pkill -f "[s]eed=2 tag=e30-esm150-s2" && echo "[$(date +%H:%M)] esm150 seed-2 run stopped for the 650M chain"
export CUDA_VISIBLE_DEVICES=2
free=$(df --output=avail -BG . | tail -n 1 | tr -dc 0-9)
echo "[$(date +%H:%M)] 650M chain; ${free} GB free"
[ "$free" -ge 19 ] || { echo "not enough disk for the 16.6 GB pool"; exit 1; }
echo "[$(date +%H:%M)] embed 650M -> pool"
$P -u scripts/datasets/casp16_embed.py --model esm2_t33_650M_UR50D --out emb650 --pool-out data/casp16/packed/pool_esm650_feat.bin --device cuda --threads 8 --max-tokens 6144 2>&1 | grep -v Warning | tail -n 4
echo "[$(date +%H:%M)] targets 650M (cpu)"
$P -u scripts/datasets/casp16_targets.py --model esm2_t33_650M_UR50D --out targets_esm650 --device cpu 2>&1 | grep -v Warning | tail -n 3
echo "[$(date +%H:%M)] pack esm650 targets + valsub"
$P -u scripts/datasets/casp16_pack.py --features esm650 --sets targets,valsub 2>&1 | tail -n 3
echo "[$(date +%H:%M)] train esm650 64 ch"
d=runs/2026-10-02-distogram-r16x64-e30-esm650; mkdir -p $d
lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=30 batch=32 ch=64 units=16 seed=1 tag=e30-esm650 > $d/train.log 2>&1
echo "[$(date +%H:%M)] trained: $(grep trained $d/train.log)"
MEMFRAC=0.5 $A/finish_run.sh 2 list=train_full fs=esm650 dim=1289 ch=64 units=16 tag=e30-esm650 > $A/finish_esm650.log 2>&1
tail -n 6 $A/finish_esm650.log
echo "[$(date +%H:%M)] 650M chain done"
