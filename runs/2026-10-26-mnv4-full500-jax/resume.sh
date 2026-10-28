#!/bin/bash
# Resume of the 500-epoch pair after the 10-27 pause (unit mnv4-e500-queue stopped at 17:30 UTC
# with the verified half at its epoch-190 checkpoint, the box lent to the Part 2 grind —
# runs/2026-10-27-part2-grind/). Same two halves as queue.sh, without the 100-epoch gate it
# already passed: the verified supervisor picks up at the saved epoch, then JAX runs fresh.
#   systemd-run --user --unit=mnv4-e500-resume runs/2026-10-26-mnv4-full500-jax/resume.sh
R=/home/skoonce/lean/klawd_max_power/lean4-jax-mlir; cd $R || exit 1
J=$R/runs/2026-10-26-mnv4-full500-jax; V=$R/runs/2026-10-26-mnv4-full500-verified
log () { echo "$(date -u '+%F %T') $*" >> $J/queue.log; }
log "resume: verified at epoch $(cat .lake/build/mnv4in256_accdp8x128wxdropdowd01bf16_ckpt_xla.bin.epoch 2>/dev/null || echo '?')"
# 10-28 12:08: the first resume's verified precheck refused (stale ffi/libpjrt_ffi.so, rebuilt
# since) and this script fell through to the JAX half for a minute. A refused verified half now
# ends the resume instead: the supervisor's dry run is the same precheck, nothing launched.
if ! DRY_RUN=1 RUNDIR=$V scripts/supervise.sh mnv4-full-4gpu >> $V/launch.log 2>&1; then
  log "verified precheck refused — nothing started (see $V/launch.log)"; exit 1
fi
log "verified start"; $V/launch.sh >> $V/launch.log 2>&1; log "verified exit $?"
log "JAX start"; $J/launch.sh > $J/launch.log 2>&1; log "JAX exit $?"
