#!/usr/bin/env bash
# The fold bench on both val subsets with the 650M arm, on the GPU this time (the bench ran on CPU
# before --device existed). Long chains: steps and learning rate; short chains: the same, for the pairing.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
P=.venv-casp/bin/python
export CUDA_VISIBLE_DEVICES=2
vdir=.lake/build/distogram_r16x64_esm650_train_full_e30-esm650_valsub_long
echo "[$(date +%H:%M)] bench steps / lr on $vdir (GPU)"
$P -u scripts/demos/casp16_valfold.py $vdir --name valsub_long --device cuda --restarts 0 --threads 4 --grid "ref:1;chain:1;clash:3;sigma:1;steps:1500,4000;lr:0.5,0.2,1.0" 2>&1 | grep --line-buffered -v Warning
echo "[$(date +%H:%M)] short set"
[ -d .lake/build/distogram_r16x64_esm650_train_full_e30-esm650_valsub ] || LEAN_MLIR_MEM_FRACTION=0.5 lake exe distogram-casp predict list=train_full fs=esm650 dim=1289 ch=64 units=16 tag=e30-esm650 pool=valsub 2>&1 | grep -E "^predicted|rror"
$P -u scripts/demos/casp16_valfold.py .lake/build/distogram_r16x64_esm650_train_full_e30-esm650_valsub --device cuda --restarts 0 --threads 4 --grid "ref:1;chain:1;clash:3;sigma:1;steps:1500,4000;lr:0.5,0.2" 2>&1 | grep --line-buffered -v Warning
echo "[$(date +%H:%M)] bench done"
