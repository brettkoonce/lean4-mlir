#!/usr/bin/env bash
# The LIDC denoiser's two training arms (planning/lidc_ct_denoising_demo.md §6 phases 2–3), one GPU
# each: MSE (target = the scanned slice) on device 0, Noise2Noise (target = the other low copy) on
# device 1. 30,000 steps of 8 × 256² crops, Adam 5e-4 cosine, seed 0; each arm then scores the
# 60 val patients' whole slices and runs scripts/probes/lidc_denoise_score.py on its dump.
#
#   runs/2026-10-10-lidc-denoise/launch.sh          # from the lidc worktree root
#
# Refuses unless data/lidc/{train,val}.bin exist, the loader gate passes on val.bin and both GPUs
# are idle. N2N starts only after MSE's data is resident, so the two load peaks (~75 GB each,
# ~38 GB steady) never coincide. Resumable: rerun after a crash and each arm continues from
# data/lidc/out/<arm>/ckpt_* (delete those to start over).
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-10-10-lidc-denoise
STEPS=${STEPS:-30000}
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
for f in data/lidc/train.bin data/lidc/val.bin "$PLUG" ffi/libpjrt_ffi.so; do
  [ -e "$f" ] || { echo "⛔ missing $f"; exit 1; }
done
lake build lidc-denoise-train test-lidc-loader 2>&1 | tail -1
.venv-lidc/bin/python scripts/probes/lidc_loader_check.py data/lidc/val.bin | tail -1 | grep -q PASS \
  || { echo "⛔ loader gate failed on data/lidc/val.bin"; exit 1; }
for g in 0 1; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 500 ] || { echo "⛔ GPU $g busy ($used MiB)"; exit 1; }
done

arm() {  # arm NAME GPU
  local a=$1 g=$2
  mkdir -p $RUN/$a data/lidc/out/$a
  setsid -f nohup env CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG bash -c "
    .lake/build/bin/lidc-denoise-train data/lidc/train.bin data/lidc/val.bin $STEPS --arm=$a --seed=0 --cosine \
      --out=$RUN/$a --dump=data/lidc/out/$a > $RUN/$a/train.log 2>&1 &&
    .venv-lidc/bin/python scripts/probes/lidc_denoise_score.py data/lidc/val.bin data/lidc/out/$a/lidc_val.eps.f32 \
      $RUN/$a/figure.png > $RUN/$a/score.txt 2>&1" > /dev/null 2>&1
  echo "launched $a on GPU $g → $RUN/$a/train.log"
}
arm mse 0
until grep -q "^train: " $RUN/mse/train.log 2>/dev/null; do
  grep -q "uncaught exception\|compile failed\|no lowerer shim" $RUN/mse/train.log 2>/dev/null && { echo "⛔ mse failed before training; see $RUN/mse/train.log"; exit 1; }
  sleep 10
done
arm n2n 1
