#!/usr/bin/env bash
# Sync-BN probe, 2026-09-21: bf16 ms/step for the §3.5 ResNet legs on ares (4x 4060 Ti), and the
# first run of the A3 4x128 render. Clock from step 200 to 600 (see scripts/bf16_probe_3060.sh).
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir
D=runs/2026-09-21-syncbn-probe
ROWS="$(cat $D/rows.txt)" PRECS=bf16 ARMS=fed CKPT_TAG=syncbnprobe \
  PJRT_PLUGIN=$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so \
  SHIM_PYTHON=$PWD/.venv/bin/python3 \
  scripts/bf16_probe_3060.sh $D/probe.tsv
