#!/usr/bin/env bash
# predict → assemble → fold → score for a finished distogram run (the four steps of plan §11).
# usage: finish_run.sh <gpu> <predict args…>     e.g.  finish_run.sh 0 list=train_full fs=onehot dim=30 ch=64 units=16 tag=e30-onehot
# MEMFRAC (default 0.2) caps the predict session's arena so it can share a card with a trainer.
set -uo pipefail
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
gpu=$1; shift
P=.venv-casp/bin/python
export CUDA_VISIBLE_DEVICES=$gpu
echo "[$(date +%H:%M)] predict $*"
# one predict + assemble at a time across the queues: the window accumulators are a ~4 GB transient
exec 9> /tmp/casp16_finish.lock; flock 9
out=$(LEAN_MLIR_MEM_FRACTION=${MEMFRAC:-0.2} lake exe distogram-casp predict "$@" 2>/dev/null | tee /dev/stderr | sed -n 's/.* -> \(.*\)\/$/\1/p' | tail -n 1)
[ -d "$out" ] || { echo "predict did not report an output directory"; exit 1; }
echo "[$(date +%H:%M)] assemble -> $out"; $P scripts/demos/casp16_predict.py $out | tail -n 5
# the summed window logits (f32 [L, L, NC] per EU, ~4 GB per arm) are dead weight once the
# .pred.npz exist; `lake exe distogram-casp predict …` regenerates them in a minute from the kept checkpoint
[ "$(ls "${out:?}"/*.pred.npz | wc -l)" -ge 84 ] && rm -f "${out:?}"/*.acc.bin "${out:?}"/*.cnt.bin
flock -u 9
echo "[$(date +%H:%M)] fold"; $P -u scripts/demos/casp16_fold.py $out --device cuda --max-len 512 --threads 4 > $out/fold_all.log 2>&1
echo "[$(date +%H:%M)] score"; $P -u scripts/demos/casp16_fold_score.py $out > $out/fold_scores.log 2>&1; tail -n 5 $out/fold_scores.log
echo "[$(date +%H:%M)] done $out"
