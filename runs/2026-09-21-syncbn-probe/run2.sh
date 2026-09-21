#!/usr/bin/env bash
# The A3 4x128 row again: synth (compute floor, no producers) then fed (repeat), to tell the
# first fed row's mean tail (546 vs median 282) apart from the graph.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
D=runs/2026-09-21-syncbn-probe
ROWS="$(grep '^r50a3x128|' $D/rows.txt)" PRECS=bf16 ARMS="synth fed" CKPT_TAG=syncbnprobe2 \
  PJRT_PLUGIN=$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so \
  SHIM_PYTHON=$PWD/.venv/bin/python3 \
  scripts/bf16_probe_3060.sh $D/probe2.tsv
