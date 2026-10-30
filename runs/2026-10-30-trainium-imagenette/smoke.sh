#!/bin/bash
# smoke.sh CORE NAME BIN [ENV=VAL ...] — one 300-step epoch with eval, then a 160-step timing probe.
cd ~/lean4-mlir; core=$1 name=$2 bin=$3; shift 3
O=runs/2026-10-30-trainium-imagenette; mkdir -p $O
export PLATFORM_BACKEND=neuron; for kv in "$@"; do export "$kv"; done; . scripts/platform/env.sh
export NEURON_RT_VISIBLE_CORES=$core
echo "== $name: $* (NEURON_CC_FLAGS=$NEURON_CC_FLAGS)" > $O/$name.log
LEAN_MLIR_G2_STEPS=300 LEAN_MLIR_MAX_EPOCHS=1 .lake/build/bin/$bin data 2>&1 | grep -v '^[IW][0-9]' >> $O/$name.log
echo "EXIT $?" >> $O/$name.log
LEAN_MLIR_MAX_STEPS=160 LEAN_MLIR_PROBE_WARM=60 LEAN_MLIR_SKIP_EVAL=1 LEAN_MLIR_PROBE_DUMP=$O/$name-probe.tsv \
  .lake/build/bin/$bin data 2>&1 | grep -E 'PROBE|rror' >> $O/$name.log
echo DONE >> $O/$name.log
