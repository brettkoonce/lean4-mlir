#!/bin/bash
# planning/part2_ship_audit.md §2.3: seeds 1 and 2 of the three 10-epoch BraTS arms of
# runs/2026-09-25-brats-r34-xla/README.md (r34 bootstrap, scratch, scratch noskip; the 09-25 run is
# the unset seed), one card, one arm at a time, `tag=s<k>` keeping each seed's checkpoints apart
# (the log's `arm:` line reads r34_s1 / scratch_s1 / scratch_noskip_s1). ~50 min an arm.
# Usage: run.sh <gpu> <arm>:<seed> ...   with arm in r34 | scratch | noskip
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
G=$1; shift; R=runs/2026-10-27-brats-r34-seeds
export LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G
log () { echo "$(date -u '+%F %T') $*" >> $R/queue.log; }
for tok in "$@"; do
  arm=${tok%%:*}; s=${tok##*:}
  case $arm in r34) a="r34";; scratch) a="scratch";; noskip) a="scratch noskip";; *) log "bad arm $arm"; continue;; esac
  log "$arm s$s start"
  LEAN_MLIR_SEED=$s .lake/build/bin/unet-brats-r34 data/brats224 10 $a tag=s$s > $R/brats_${arm}_s$s.log 2>&1
  log "$arm s$s exit $?"
done
log "DONE $*"
