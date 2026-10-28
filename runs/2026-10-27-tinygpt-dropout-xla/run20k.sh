#!/bin/bash
# The second half of T6: the same two dropout rates at 20,000 steps (run.sh is 10,000). The
# question the plan asks (lm_demos_modernization.md §1.4) is whether residual dropout turns
# `tiny`'s overfit from step ~3,500 into a schedule that keeps paying — a 10k row shows whether
# the val floor moves, a 20k row whether it keeps moving. Each train.log carries the periodic
# val curve. ~13 min a run; three seeds of each rate.
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
  job tinydo10_20k_s$s tiny-do10 $s 20000 32
  job tinydo20_20k_s$s tiny-do20 $s 20000 32
done
log "ALL DONE (20k)"
