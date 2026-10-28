#!/bin/bash
# planning/part2_ship_audit.md §2.1 row T6 — the dropout arm the plan calls the fix for `tiny`
# overfitting from step ~3500 (lm_demos_modernization.md §1.4): GPT-2's residual dropout, added to
# the generic emitter's transformer block on 10-27 (`transformerEncoder`'s `dropout`, in-graph
# Philox masks keyed on the step), at 0.1 and 0.2, 10k steps, three seeds, scored like the ladder
# rows (runs/2026-10-27-tinygpt-ladder-xla/run.sh). The eval forward has no dropout, so `score`
# on the dropout key reads the checkpoint through the same graph as `tiny`'s.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
G=${1:-0}; R=runs/2026-10-27-tinygpt-dropout-xla; EXE=.lake/build/bin/tinygpt-shakespeare
export LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G
log () { echo "$(date -u '+%F %T') $*" >> $R/queue.log; }
job () {  # <name> <key> <seed> <steps> <batch>
  local name=$1 key=$2 s=$3 steps=$4 batch=$5 d=$R/$1
  mkdir -p "$d"; log "$name start (seed $s, $steps steps, batch $batch)"
  LEAN_MLIR_SEED=$s $EXE train "$key" "$steps" "$batch" 30 > "$d/train.log" 2>&1 \
    || { log "$name TRAIN FAILED"; return 1; }
  local p; p=$(grep -o 'saved params to [^ ]*' "$d/train.log" | awk '{print $4}')
  cp "$p" "$d/${key}_params.bin"
  $EXE score "$key" "$d/${key}_params.bin" "$d" > "$d/score.log" 2>&1 || log "$name SCORE FAILED"
  log "$name done: $(grep -o 'final loss: [0-9.]*' "$d/train.log") / $(tail -1 "$d/score.log")"
}
for s in 0 1 2; do
  job tinydo10_s$s tiny-do10 $s 10000 32
  job tinydo20_s$s tiny-do20 $s 10000 32
done
log "ALL DONE"
