#!/bin/bash
# planning/part2_ship_audit.md §2.1, rows T1–T5, on one 4060 Ti under XLA (the default lowerer),
# three seeds each, one job at a time: every config writes .lake/build/<prefix>_params.bin, so the
# checkpoint is copied into the job's directory before the next train and `score` reads the copy.
# The 10-21 ladder (runs/2026-10-21-tinygpt-ngram-ladder/) ran nano under IREE on the CPU box.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
G=${1:-0}; R=runs/2026-10-27-tinygpt-ladder-xla; EXE=.lake/build/bin/tinygpt-shakespeare
export LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G
log () { echo "$(date -u '+%F %T') $*" >> $R/queue.log; }
job () {  # <name> <key> <seed> <steps> <batch>: train, copy the checkpoint out, score it
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
  job tiny_s$s tiny $s 10000 32                 # T1 — the ladder's empty row
done
for s in 0 1 2; do
  job tiny5k_s$s  tiny $s 5000 32               # T5 — the historical best-val point
  job nano20k_s$s nano $s 20000 32              # T2 — "undertrained" by steps
  if [ "$s" = 0 ]; then                         # T4's contrast: a T=64 nano checkpoint read at T=128 must fail
    $EXE xeval nano 128 > $R/nano20k_s0/xeval_T128.log 2>&1
    log "xeval nano 128 (contrast, exit $?): $(tail -1 $R/nano20k_s0/xeval_T128.log)"
  fi
  job nano40k_s$s nano $s 40000 32              # T2
  job nanob64_s$s nano $s 10000 64              # T3 — more tokens per step, same wall
  job rope_s$s    nano-rope-nopos $s 10000 32   # T4 — RoPE, no learned positions
  for T in 64 128 256; do                       # T4 — the extrapolation gate on the checkpoint just trained
    $EXE xeval nano-rope-nopos $T > $R/rope_s$s/xeval_T$T.log 2>&1
    log "rope_s$s xeval T=$T (exit $?): $(tail -1 $R/rope_s$s/xeval_T$T.log)"
  done
done
log "ALL DONE"
