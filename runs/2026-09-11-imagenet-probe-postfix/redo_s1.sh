#!/bin/bash
# After the sweep: re-run the four seed-1 arms that resumed from an epoch-3 checkpoint left by a
# killed launch (full noaug nocos nowarm). Delete those checkpoints first, keep the spliced logs
# as *_s1_resumed.log, run the arms fresh, move the logs into place, then summarize.
set -u
cd "$(dirname "$0")/../.."
LOG=runs/2026-09-11-imagenet-probe-postfix/then_ablation.log
until grep -q "^ABLATION_DONE" $LOG; do sleep 60; done
A=runs/2026-09-11-r34-ablation-bf16-seeds; T=runs/2026-09-11-r34-ablation-bf16-seeds/.redo_s1
echo "redo_s1: start $(date -u +%FT%TZ)" | tee -a $LOG
for arm in full noaug nocos nowarm; do
  rm -f .lake/build/resnet34_*_ckpt_xla_abl-bf16-$arm-s1.bin .lake/build/resnet34_*_ckpt_xla_abl-bf16-$arm-s1.bin.epoch
  mv $A/${arm}_s1.log $A/${arm}_s1_resumed.log
done
mkdir -p $T
PREC=bf16 SEEDS=1 ARMS="full noaug nocos nowarm" OUT=$T \
  PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
  bash scripts/run_r34_ablation.sh 2>&1 | tee -a $LOG
for arm in full noaug nocos nowarm; do
  grep -q "resuming from checkpoint" $T/$arm.log && echo "⛔ $arm redo RESUMED — check" | tee -a $LOG
  mv $T/$arm.log $A/${arm}_s1.log
done
rm -rf $T
python3 scripts/r34_ablation_ci.py $A --write 2>&1 | tail -3 | tee -a $LOG
echo "REDO_DONE $(date -u +%FT%TZ)" | tee -a $LOG
