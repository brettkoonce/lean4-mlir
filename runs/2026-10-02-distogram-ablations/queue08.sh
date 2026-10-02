#!/usr/bin/env bash
# Night queue 8 (08:10): the idle cards after the night's rows. GPU 1 — the book-run candidate on the
# right LM: 650M × 64 ch × 100 epochs (+ finish). GPU 2 — after the fold bench: 650M × 128 ch × 30 ep
# (both levers at the top LM, + finish).
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
(
  export CUDA_VISIBLE_DEVICES=1
  echo "[$(date +%H:%M)] 650M × 64 ch × 100 ep on GPU 1"
  d=runs/2026-10-02-distogram-r16x64-e100-esm650; mkdir -p $d
  lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=100 batch=32 ch=64 units=16 seed=1 tag=e100-esm650 > $d/train.log 2>&1
  echo "[$(date +%H:%M)] $(grep trained $d/train.log)"
  MEMFRAC=0.5 $A/finish_run.sh 1 list=train_full fs=esm650 dim=1289 ch=64 units=16 tag=e100-esm650 > $A/finish_esm650_e100.log 2>&1
  tail -n 5 $A/finish_esm650_e100.log
) &
(
  while pgrep -f "[b]ench_only.sh" > /dev/null; do sleep 60; done
  export CUDA_VISIBLE_DEVICES=2
  echo "[$(date +%H:%M)] 650M × 128 ch × 30 ep on GPU 2"
  d=runs/2026-10-02-distogram-r16x128-e30-esm650; mkdir -p $d
  lake exe distogram-casp train list=train_full fs=esm650 dim=1289 epochs=30 batch=32 ch=128 units=16 seed=1 tag=e30-esm650 > $d/train.log 2>&1
  echo "[$(date +%H:%M)] $(grep trained $d/train.log)"
  MEMFRAC=0.5 $A/finish_run.sh 2 list=train_full fs=esm650 dim=1289 ch=128 units=16 tag=e30-esm650 > $A/finish_esm650x128.log 2>&1
  tail -n 5 $A/finish_esm650x128.log
) &
wait
echo "[$(date +%H:%M)] queue08 done"
