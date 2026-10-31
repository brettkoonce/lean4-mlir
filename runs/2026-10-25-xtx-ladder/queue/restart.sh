#!/bin/bash
# Relaunch the queue after the 2026-10-25 23:12 hang — only on kernel 6.8.
# ladder-more.sh: the remaining bbn seeds (s2 resumes from its checkpoints) + the nine grid widths;
# miopen.sh and imagenette2.sh wait behind it as before. Blocks <= 55 min, 15 min idle between.
# kernel 7.0 on 2026-10-26 at the user's call (the riser, not the kernel, is the suspect)
cd ~/xtx-queue
echo "=== $(date '+%F %T') restarted after the 23:12 hang (boot $(uptime -s))" | tee -a ladder-more.log miopen.log imagenette2.log
nohup /home/skoonce/xtx-queue/ladder-more.sh xtx-queue/ladder.sh >> ladder-more.log 2>&1 &
sleep 5; nohup /home/skoonce/xtx-queue/miopen.sh >> miopen.log 2>&1 &
sleep 5; nohup /home/skoonce/xtx-queue/imagenette2.sh convnext resnet50 mobilenetv4 >> imagenette2.log 2>&1 &
sleep 2; pgrep -af 'xtx-queue/'
