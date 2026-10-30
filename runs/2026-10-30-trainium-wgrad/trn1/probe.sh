#!/bin/bash
# probe.sh NAME BIN CKPT VARIANT [ENV=VAL ...] — 160-step timing probe (60 warm), alone on NeuronCore 0.
cd ~/lean4-mlir; n=$1 bin=$2 ck=$3 v=$4; shift 4; O=~/wgrad; mkdir -p $O ~/old-ckpt
mv .lake/build/${ck}_ckpt_xla.bin ~/old-ckpt/ 2>/dev/null
export PLATFORM_BACKEND=neuron; for kv in "$@"; do export "$kv"; done; . scripts/platform/env.sh; export NEURON_RT_VISIBLE_CORES=0 LEAN_MLIR_VARIANT=$v
echo "== $n: $bin variant $v $* NEURON_CC_FLAGS=$NEURON_CC_FLAGS $(date -u +%T)" > $O/$n.log
LEAN_MLIR_MAX_STEPS=160 LEAN_MLIR_PROBE_WARM=60 LEAN_MLIR_SKIP_EVAL=1 LEAN_MLIR_PROBE_DUMP=$O/$n-probe.tsv .lake/build/bin/$bin data 2>&1 | grep -v "^[IW][0-9]" >> $O/$n.log
echo "EXIT ${PIPESTATUS[0]} $(date -u +%T)" >> $O/$n.log
grep -h "compiled verified\|PROBE\|rror\|EXIT\|step 100" $O/$n.log | tail -7 | cut -c1-160
