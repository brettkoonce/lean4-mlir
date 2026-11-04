#!/usr/bin/env bash
# Plan §11, R2: the three range arms, one GPU each, the §10.1 recipe (8 × 256², Adam 5e-4 cosine,
# seed 0; 30k steps for the MSE net, 60k for the generators):
#   msemix      MSE UNet on data/lidcmix                            GPU 0
#   diffmix     the generator on data/lidcmix, blind                GPU 1
#   diffmixsig  the generator on data/lidcmix, told σ̂ (--sigma)     GPU 2
# Artifacts carry the root (`lidcmix`) and `sig`, so nothing of §9's is overwritten.
#
#   ARMS="msemix diffmix diffmixsig" runs/2026-11-03-lidc-range/launch_train.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-03-lidc-range
ARMS=${ARMS:-"msemix diffmix diffmixsig"}
STEPS_DIFF=${STEPS_DIFF:-60000}
STEPS_MSE=${STEPS_MSE:-30000}
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
for f in data/lidcmix/train.bin data/lidcmix/val.bin data/lidcmix/train.bin.sigma.f32 "$PLUG" ffi/libpjrt_ffi.so; do
  [ -e "$f" ] || { echo "⛔ missing $f"; exit 1; }
done
lake build lidc-denoise-train lidc-denoise-sample test-lidc-loader 2>&1 | tail -1
.venv-lidc/bin/python scripts/probes/lidc_loader_check.py data/lidcmix/train.bin | tail -1 | grep -q PASS \
  || { echo "⛔ loader gate failed on data/lidcmix/train.bin"; exit 1; }
gpu_of() { case $1 in msemix) echo 0;; diffmix) echo 1;; diffmixsig) echo 2;; esac; }
for a in $ARMS; do
  g=$(gpu_of $a); used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 500 ] || { echo "⛔ GPU $g busy ($used MiB)"; exit 1; }
done
launch() {  # launch NAME GPU EVAL_BIN STEPS ARM EXTRA DUMP [post]
  local name=$1 g=$2 eval=$3 steps=$4 arm=$5 extra=$6 dump=$7 post=${8:-true}
  mkdir -p $RUN/$name "$dump"
  setsid -f nohup env CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG bash -c "
    .lake/build/bin/lidc-denoise-train data/lidcmix/train.bin $eval $steps --arm=$arm $extra --seed=0 --cosine \
      --out=$RUN/$name --dump=$dump > $RUN/$name/train.log 2>&1 && $post" > /dev/null 2>&1
  echo "launched $name on GPU $g → $RUN/$name/train.log"
  until grep -q "^train: " $RUN/$name/train.log 2>/dev/null; do
    grep -q "uncaught exception\|compile failed\|no lowerer shim\|⛔" $RUN/$name/train.log 2>/dev/null \
      && { echo "⛔ $name failed before training; see $RUN/$name/train.log"; exit 1; }
    sleep 10
  done
}
for a in $ARMS; do case $a in
  msemix)     launch msemix 0 data/lidcmix/val.bin $STEPS_MSE mse "" data/lidcmix/out/msemix \
    ".venv-lidc/bin/python scripts/probes/lidc_denoise_score.py data/lidcmix/val.bin data/lidcmix/out/msemix/lidcmix_val.eps.f32 $RUN/msemix/figure.png --by=dose:0.05,0.1,0.15,0.2 > $RUN/msemix/score.txt 2>&1" ;;
  diffmix)    launch diffmix 1 "" $STEPS_DIFF diff "" data/lidcmix/out/diffmix ;;
  diffmixsig) launch diffmixsig 2 "" $STEPS_DIFF diff "--sigma" data/lidcmix/out/diffmixsig ;;
  *) echo "⛔ unknown arm $a"; exit 1 ;;
esac; done
