#!/bin/bash
# Re-run of the 2026-09-10 fed probe after (a) the owner-thread batch buffers and (b) the shim
# sync (determinism OFF on this box). Same WARM/STEPS window; rows from the confs as of today.
set -u
cd "$(dirname "$0")/../.."
D=runs/2026-09-11-imagenet-probe-postfix
PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
SHIM_PYTHON="$PWD/.venv/bin/python3" DEVS=0,1,2,3 WARM=200 STEPS=600 ARMS=fed \
CKPT_TAG=probe4060b ROWS="$(cat $D/rows.txt)" \
  bash scripts/bf16_probe_3060.sh $D/fed.tsv 2>&1 | tee -a $D/fed.log
# the RSB-A3 bf16 twin's conf runs 8 producers where the f32 twin runs 4 — one extra row for it
PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
SHIM_PYTHON="$PWD/.venv/bin/python3" DEVS=0,1,2,3 WARM=200 STEPS=600 ARMS=fed PRECS=bf16 \
CKPT_TAG=probe4060b ROWS="r50a3w8|resnet50-imagenet-verified|lambaccdp8x64wxclipbce|lambaccdp8x64wxclipbcebf16|64|SHIM_WORKERS=8 LEAN_MLIR_RES=160 LEAN_MLIR_BASE_LR_U=8000 LEAN_MLIR_G2_STEPS=5000" \
  bash scripts/bf16_probe_3060.sh $D/fed.tsv 2>&1 | tee -a $D/fed.log
echo "ALL_DONE $(date -u +%FT%TZ)" | tee -a $D/fed.log
