#!/bin/bash
# The seed-0 50-epoch rows along their schedule: the 08-28 e10/e20/e30/e40 checkpoints of aug50 and
# long50 (still in .lake/build, copied to the <row>_s0 prefixes) scored under both decodes with
# score_ckpt.sh, so the ladder table has seed 0 beside seeds 1 and 2. aug50 e30 is the archive's
# "best seen" 0.1731 (argmax). ~4.5 min a checkpoint on card 0.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
R=runs/2026-10-27-visdrone-rows-seeds
$R/score_ckpt.sh 0 aug50_s0 10 20 30 40
$R/score_ckpt.sh 0 long50_s0 10 20 30 40
