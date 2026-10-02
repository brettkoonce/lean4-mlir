#!/usr/bin/env bash
# Night queue 5 (04:50): the finish steps queue02/03 did not include — GPU 0 when the esm150 × 128 ch
# trainer exits, GPU 1 when the crops=3 trainer exits.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
A=runs/2026-10-02-distogram-ablations
(
  while pgrep -f "[f]s=esm150 dim=649 epochs=30 batch=32 ch=128" > /dev/null; do sleep 60; done
  echo "[$(date +%H:%M)] esm150 × 128 ch trainer done; finishing on GPU 0"
  MEMFRAC=0.5 $A/finish_run.sh 0 list=train_full fs=esm150 dim=649 ch=128 units=16 tag=e30-esm150 > $A/finish_esm150x128.log 2>&1
  tail -n 6 $A/finish_esm150x128.log
) &
(
  while pgrep -f "[c]rops=3 epochs=30" > /dev/null; do sleep 60; done
  echo "[$(date +%H:%M)] crops=3 trainer done; finishing on GPU 1"
  MEMFRAC=0.5 $A/finish_run.sh 1 list=train_full ch=64 units=16 tag=e30-crops3 > $A/finish_crops3.log 2>&1
  tail -n 6 $A/finish_crops3.log
) &
wait
echo "[$(date +%H:%M)] queue05 done"
