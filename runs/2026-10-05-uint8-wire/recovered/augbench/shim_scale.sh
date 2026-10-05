#!/bin/bash
# N concurrent shim producers (like SHIM_WORKERS=N), aggregate img/s.
SHIM=$1; N=$2; B=${3:-128}; K=${4:-40}
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
out=$(mktemp -d)
for i in $(seq 0 $((N-1))); do
  ( SHIM_BATCH=$B SHIM_SPLIT=train SHIM_SEED=$i SHIM_SHARD=$i/$N SHIM_NCLASSES=1000 CUDA_VISIBLE_DEVICES= \
    TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets TF_CPP_MIN_LOG_LEVEL=3 \
    timeout 600 .venv/bin/python -u $SHIM 2>/dev/null | .venv/bin/python $(dirname $0)/shim_reader.py 6 $K > $out/$i ) &
done
wait
tot=0; per=""
for i in $(seq 0 $((N-1))); do v=$(cat $out/$i); per="$per $v"; tot=$((tot + ${v:-0})); done
echo "$(basename $SHIM .py) workers=$N batch=$B: total $tot img/s  (per producer:$per)"
rm -rf $out
