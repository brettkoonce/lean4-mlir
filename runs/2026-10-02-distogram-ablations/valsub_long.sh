#!/usr/bin/env bash
# A second val subset of LONG chains (200–500 residues) for the fold bench: the fold's knobs were
# tuned on 80–200-residue chains, and at 650M 11 of 78 EUs fold to TM < 0.5 from near-perfect
# contact maps, mostly long ones. Pack it, embed its 24 chains with 650M, predict with the 650M
# arm, then bench steps and learning rate (restarts 0, reference state on).
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
P=.venv-casp/bin/python
export CUDA_VISIBLE_DEVICES=2
echo "[$(date +%H:%M)] pack valsub_long (35M features, labels, index)"
$P scripts/datasets/casp16_pack.py --sets valsub_long --val-name valsub_long --val-len 200 500 2>&1 | tail -n 2
printf "id\n%s\n" $(cat data/casp16/packed/valsub_long_order.txt) > data/casp16/packed/valsub_long_ids.csv
echo "[$(date +%H:%M)] embed its chains with 650M"
$P scripts/datasets/casp16_embed.py --model esm2_t33_650M_UR50D --out emb650 --list data/casp16/packed/valsub_long_ids.csv --device cuda --threads 8 --max-tokens 4096 2>&1 | grep -v Warning | tail -n 1
echo "[$(date +%H:%M)] pack valsub_long esm650"
$P scripts/datasets/casp16_pack.py --features esm650 --sets valsub_long --val-name valsub_long --val-len 200 500 2>&1 | tail -n 2
echo "[$(date +%H:%M)] predict"
LEAN_MLIR_MEM_FRACTION=0.5 lake exe distogram-casp predict list=train_full fs=esm650 dim=1289 ch=64 units=16 tag=e30-esm650 pool=valsub_long 2>&1 | grep -E "^predicted|rror" 
vdir=.lake/build/distogram_r16x64_esm650_train_full_e30-esm650_valsub_long
echo "[$(date +%H:%M)] bench steps / lr on $vdir"
$P -u scripts/demos/casp16_valfold.py $vdir --name valsub_long --restarts 0 --threads 4 --grid "ref:1;chain:1;clash:3;sigma:1;steps:1500,4000;lr:0.5,0.2,1.0" 2>&1 | grep -v Warning
echo "[$(date +%H:%M)] and the short set with the 650M arm, for the pairing"
LEAN_MLIR_MEM_FRACTION=0.5 lake exe distogram-casp predict list=train_full fs=esm650 dim=1289 ch=64 units=16 tag=e30-esm650 pool=valsub 2>&1 | grep -E "^predicted|rror"
$P -u scripts/demos/casp16_valfold.py .lake/build/distogram_r16x64_esm650_train_full_e30-esm650_valsub --restarts 0 --threads 4 --grid "ref:1;chain:1;clash:3;sigma:1;steps:1500,4000;lr:0.5,0.2" 2>&1 | grep -v Warning
echo "[$(date +%H:%M)] done"
