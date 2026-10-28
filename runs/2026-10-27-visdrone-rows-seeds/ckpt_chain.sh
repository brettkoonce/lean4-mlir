#!/bin/bash
# After score_ckpt.sh's cfoc2_s1 ladder on card 0: the same e6/e8/e10 ladder for seed 0 (the 08-29
# checkpoints, copied to the cfoc2_s0 prefix) and seed 2, so the three seeds of the 12-epoch
# focal row can be read along their schedule (0.1961 / 0.1350 / 0.1516 at e12).
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
R=runs/2026-10-27-visdrone-rows-seeds
until grep -q 'CKPT DONE cfoc2_s1' $R/queue.log; do sleep 30; done
$R/score_ckpt.sh 0 cfoc2_s0 6 8 10
$R/score_ckpt.sh 0 cfoc2_s2 6 8 10
