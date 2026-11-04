#!/usr/bin/env bash
# Phase 5's three trainings (planning/lidc_ct_denoising_demo.md §10.5), one GPU each:
#   diff25  the generator on the 25 % build (data/lidc, N = 2)     60,000 steps   GPU 0
#   diff10  the generator on the 10 % build (data/lidc10, N = 1)   60,000 steps   GPU 1
#   mse10   the MSE UNet on the 10 % build                          30,000 steps   GPU 2
# all 8 × 256² crops, Adam 5e-4 cosine, seed 0 — the 25 % MSE/N2N recipe; the generator takes
# twice the steps (§10.1). mse10 then scores data/lidc10/val.bin and runs the RMSE scorer; the
# diffusion arms write no dump (launch_sample.sh draws the samples). ≈ 2.5 h wall at ≈ 150 ms a
# step for diff; mse10 ≈ 1.3 h.
#
#   STEPS_DIFF=60000 STEPS_MSE=30000 runs/2026-11-02-lidc-diffusion/launch_train.sh
#   ARMS=diff25 runs/2026-11-02-lidc-diffusion/launch_train.sh      # a subset (diff25 needs no 10 % build)
#
# Refuses unless both builds and the loader gates pass and GPUs 0–2 are idle. Launches are
# staggered on each arm's "train:" line so the load peaks (≈ 2× the resident pairs) never
# coincide: resident ≈ 38 GB (diff25) + 19 (diff10) + 19 (mse10). Resumable: rerun after a crash
# and each arm continues from its <dump>/ckpt_* (delete those to start over).
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-02-lidc-diffusion
STEPS_DIFF=${STEPS_DIFF:-60000}
STEPS_MSE=${STEPS_MSE:-30000}
ARMS=${ARMS:-"diff25 diff10 mse10"}
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
need="$PLUG ffi/libpjrt_ffi.so"; gates=""; gpus=""
case " $ARMS " in *" diff25 "*) need="$need data/lidc/train.bin"; gates="$gates data/lidc/train.bin"; gpus="$gpus 0";; esac
case " $ARMS " in *" diff10 "*) need="$need data/lidc10/train.bin"; gates="$gates data/lidc10/train.bin"; gpus="$gpus 1";; esac
case " $ARMS " in *" mse10 "*) need="$need data/lidc10/train.bin data/lidc10/val.bin"; gates="$gates data/lidc10/train.bin"; gpus="$gpus 2";; esac
for f in $need; do
  [ -e "$f" ] || { echo "⛔ missing $f"; exit 1; }
done
lake build lidc-denoise-train lidc-denoise-sample test-lidc-loader 2>&1 | tail -1
for b in $(echo $gates | tr ' ' '\n' | sort -u); do
  .venv-lidc/bin/python scripts/probes/lidc_loader_check.py $b | tail -1 | grep -q PASS \
    || { echo "⛔ loader gate failed on $b"; exit 1; }
done
for g in $gpus; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 500 ] || { echo "⛔ GPU $g busy ($used MiB)"; exit 1; }
done

launch() {  # launch NAME GPU TRAIN_BIN [EVAL_BIN] STEPS ARM DUMP [post-command]
  local name=$1 g=$2 train=$3 eval=$4 steps=$5 arm=$6 dump=$7 post=${8:-true}
  mkdir -p $RUN/$name "$dump"
  setsid -f nohup env CUDA_VISIBLE_DEVICES=$g LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG bash -c "
    .lake/build/bin/lidc-denoise-train $train $eval $steps --arm=$arm --seed=0 --cosine \
      --out=$RUN/$name --dump=$dump > $RUN/$name/train.log 2>&1 && $post" > /dev/null 2>&1
  echo "launched $name on GPU $g → $RUN/$name/train.log"
  until grep -q "^train: " $RUN/$name/train.log 2>/dev/null; do
    grep -q "uncaught exception\|compile failed\|no lowerer shim\|⛔" $RUN/$name/train.log 2>/dev/null \
      && { echo "⛔ $name failed before training; see $RUN/$name/train.log"; exit 1; }
    sleep 10
  done
}
for a in $ARMS; do case $a in
  diff25) launch diff25 0 data/lidc/train.bin ""                 $STEPS_DIFF diff data/lidc/out/diff ;;
  diff10) launch diff10 1 data/lidc10/train.bin ""               $STEPS_DIFF diff data/lidc10/out/diff ;;
  mse10)  launch mse10  2 data/lidc10/train.bin data/lidc10/val.bin $STEPS_MSE mse data/lidc10/out/mse \
    ".venv-lidc/bin/python scripts/probes/lidc_denoise_score.py data/lidc10/val.bin data/lidc10/out/mse/lidc10_val.eps.f32 $RUN/mse10/figure.png > $RUN/mse10/score.txt 2>&1" ;;
  *) echo "⛔ unknown arm $a"; exit 1 ;;
esac; done
