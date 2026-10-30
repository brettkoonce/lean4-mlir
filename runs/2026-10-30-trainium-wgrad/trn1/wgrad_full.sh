#!/bin/bash
# wgrad_full.sh — the full 80-epoch ViT-Tiny Imagenette run under the rewritten render, alone on NeuronCore 0.
cd ~/lean4-mlir; O=~/wgrad
export PLATFORM_BACKEND=neuron; . scripts/platform/env.sh; export NEURON_RT_VISIBLE_CORES=0 LEAN_MLIR_VARIANT=adamwgB
echo "== full: variant adamwgB NEURON_CC_FLAGS=$NEURON_CC_FLAGS start $(date -u +%FT%TZ)" > $O/full.log
.lake/build/bin/vit-verified-adam data 2>&1 | grep -v "^[IW][0-9]" | while IFS= read -r l; do printf "%s %s\n" "$(date -u +%T)" "$l"; done >> $O/full.log
echo "EXIT ${PIPESTATUS[0]} $(date -u +%FT%TZ)" >> $O/full.log
