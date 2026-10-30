#!/bin/bash
# wgrad_probe.sh VARIANT [NAME] — 160-step timing probe (warm 60) of vit_<VARIANT>_train_step on NeuronCore 0, alone.
cd ~/lean4-mlir; v=$1; n=${2:-$1}; O=~/wgrad; mkdir -p $O ~/old-ckpt
mv .lake/build/vit_${v}_ckpt_xla.bin ~/old-ckpt/ 2>/dev/null
export PLATFORM_BACKEND=neuron; . scripts/platform/env.sh; export NEURON_RT_VISIBLE_CORES=0 LEAN_MLIR_VARIANT=$v
echo "== $n: variant $v NEURON_CC_FLAGS=$NEURON_CC_FLAGS $(date -u +%H:%M:%S)" > $O/$n.log
LEAN_MLIR_MAX_STEPS=160 LEAN_MLIR_PROBE_WARM=60 LEAN_MLIR_SKIP_EVAL=1 LEAN_MLIR_PROBE_DUMP=$O/$n-probe.tsv \
  .lake/build/bin/vit-verified-adam data 2>&1 | grep -v "^[IW][0-9]" >> $O/$n.log
echo "EXIT ${PIPESTATUS[0]} $(date -u +%H:%M:%S)" >> $O/$n.log
grep -h "cached neff\|Compiling\|PROBE\|rror\|EXIT\|loss" $O/$n.log | tail -8
