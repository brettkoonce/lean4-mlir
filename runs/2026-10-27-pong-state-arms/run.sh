#!/bin/bash
# planning/part2_ship_audit.md §2.5 (demo_slate.md §6): the state-vs-pixels dissection's arms,
# beside runs/2026-09-25-pong-dqn/'s state (k=1, width 64) and pixel (k=4) rows at the default
# opponent, three seeds each. The six-number state is already scaled to about [-1, 1]
# (`Pong.stateVec`), so the two knobs added to demos/MainPongDqn.lean on 10-27 are
#   k=4      — the last four states stacked, the pixel net's history (the paddles' motion, which
#              the six numbers lack);
#   width=512 — the k=1 MLP at the pixel net's dense width, the capacity arm.
# State runs are ~5 min each; LEAN_MLIR_MEM_FRACTION=0.15 lets them sit beside another job.
# Usage: run.sh <gpu>
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
G=${1:-0}; R=runs/2026-10-27-pong-state-arms
export LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G LEAN_MLIR_MEM_FRACTION=0.15
log () { echo "$(date -u '+%F %T') $*" >> $R/queue.log; }
run () {  # <tag> <args...>
  local tag=$1; shift
  log "$tag start"
  .lake/build/bin/pong-dqn "$@" opp=1.5 tag=$tag 2>&1 | grep --line-buffered -v '^\[pjrt\|^I[0-9]' > $R/$tag.log
  cp .lake/build/*_${tag}_curve.csv $R/ 2>/dev/null
  log "$tag done: $(grep -o 'final.*' $R/$tag.log | tail -1)"
}
for s in 1 2 3; do run state_k4_s$s   mode=state k=4     seed=$s; done
for s in 1 2 3; do run state_w512_s$s mode=state width=512 seed=$s; done
log "ALL DONE"
