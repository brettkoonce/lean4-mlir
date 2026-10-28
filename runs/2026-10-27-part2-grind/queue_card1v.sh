#!/bin/bash
# card 1, after queue_card1.sh: two VisDrone rows of tab:visdrone at one seed (§7.4, the user's
# word 10-27 evening: a day or two of compute on Part 2 before the MNv4 pair resumes).
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
while pgrep -f 'bash .*queue_card1\.sh' > /dev/null; do sleep 60; done
runs/2026-10-27-visdrone-rows-seeds/run.sh 1 long50:1 cfoc2:1
