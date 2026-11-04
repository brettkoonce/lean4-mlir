#!/usr/bin/env bash
# Plan §12 (§7 b): the two arms on the b50f build, one GPU each, the §10.1 recipe (8 × 256², Adam
# 5e-4 cosine, seed 0) — the same recipe, steps and seed as §9's diff10 / mse10 on data/lidc10, so
# the kernel is the only thing that moves:
#   diff10b50   the generator on data/lidc10b50       60,000 steps   GPU 0
#   mse10b50    the MSE UNet on data/lidc10b50        30,000 steps   GPU 1, then its own val scored
# Artifacts carry the root (`lidc10b50`), so nothing of §9's or §11's is overwritten.
#
#   ARMS="diff10b50 mse10b50" runs/2026-11-03-lidc-kernel/launch_train.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-03-lidc-kernel
D=data/lidc10b50
ARMS=${ARMS:-"diff10b50 mse10b50"}
STEPS_DIFF=${STEPS_DIFF:-60000}
STEPS_MSE=${STEPS_MSE:-30000}
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
for f in $D/train.bin $D/val.bin "$PLUG" ffi/libpjrt_ffi.so; do
  [ -e "$f" ] || { echo "⛔ missing $f"; exit 1; }
done
lake build lidc-denoise-train lidc-denoise-sample test-lidc-loader 2>&1 | tail -1
.venv-lidc/bin/python scripts/probes/lidc_loader_check.py $D/train.bin | tail -1 | grep -q PASS \
  || { echo "⛔ loader gate failed on $D/train.bin"; exit 1; }
gpu_of() { case $1 in diff10b50) echo 0;; mse10b50) echo 1;; esac; }
for a in $ARMS; do
  g=$(gpu_of $a); used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 500 ] || { echo "⛔ GPU $g busy ($used MiB)"; exit 1; }
done
launch() {  # launch NAME GPU EVAL_BIN STEPS ARM DUMP [post]
  local name=$1 g=$2 eval=$3 steps=$4 arm=$5 dump=$6 post=${7:-true}
  mkdir -p $RUN/$name "$dump"
  setsid -f nohup env CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG bash -c "
    .lake/build/bin/lidc-denoise-train $D/train.bin $eval $steps --arm=$arm --seed=0 --cosine \
      --out=$RUN/$name --dump=$dump > $RUN/$name/train.log 2>&1 && $post" > /dev/null 2>&1
  echo "launched $name on GPU $g → $RUN/$name/train.log"
  until grep -q "^train: " $RUN/$name/train.log 2>/dev/null; do
    grep -q "uncaught exception\|compile failed\|no lowerer shim\|⛔" $RUN/$name/train.log 2>/dev/null \
      && { echo "⛔ $name failed before training; see $RUN/$name/train.log"; exit 1; }
    sleep 10
  done
}
for a in $ARMS; do case $a in
  diff10b50) launch diff10b50 0 "" $STEPS_DIFF diff $D/out/diff ;;
  mse10b50)  launch mse10b50 1 $D/val.bin $STEPS_MSE mse $D/out/mse \
    ".venv-lidc/bin/python scripts/probes/lidc_denoise_score.py $D/val.bin $D/out/mse/lidc10b50_val.eps.f32 $RUN/mse10b50/figure.png --by=sigma_add_hu:60,90,120,180 > $RUN/mse10b50/score.txt 2>&1" ;;
  *) echo "⛔ unknown arm $a"; exit 1 ;;
esac; done
