#!/bin/bash
# card 0, after queue_card0b.sh (the dropout arm at 10k): the dropout arm at 20k, then two VisDrone
# rows of tab:visdrone — launched 10-27 evening on the user's word to spend a day or two of
# compute on Part 2 instead of resuming the MNv4 pair.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
while pgrep -f 'bash .*queue_card0b\.sh' > /dev/null; do sleep 60; done
runs/2026-10-27-tinygpt-dropout-xla/run20k.sh 0
runs/2026-10-27-visdrone-rows-seeds/run.sh 0 aug50:1 ctrl12:1
