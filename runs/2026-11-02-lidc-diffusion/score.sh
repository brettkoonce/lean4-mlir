#!/usr/bin/env bash
# Phase 5's scoring (planning/lidc_ct_denoising_demo.md §10.2–§10.5), after launch_train.sh and
# launch_sample.sh:
#   1. the regression nets and the filters on the 10 % val and on Mayo (no retraining: the 10 %
#      MSE net's eval forward through the trainer's --eval-only, the filters' knobs re-tuned on
#      the 10 % train split by lidc_classical.py), dumps under data/{lidc10,mayo}/out/<arm>/
#   2. per data root: lidc_denoise_score.py (fidelity), lidc_nodules.py (contrast kept, LIDC only),
#      lidc_texture.py (NPS ratios, blobs, spread) on the §10.5 record set
# GPU: minutes (one eval pass per file); CPU: BM3D ≈ 45 min on 30 cores per file, under nice.
#
#   STAGES="filters mse_mayo tables" runs/2026-11-02-lidc-diffusion/score.sh [gpu]
#   (the filters need only the 10 % build + data/mayo, so they run while the nets train)
#   STAGES=tables ROOTS=lidc TW=5 runs/2026-11-02-lidc-diffusion/score.sh   # one root, in parallel with the others
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/2026-11-02-lidc-diffusion
G=${1:-0}
STAGES=${STAGES:-"filters mse_mayo tables"}
FW=${FW:-24}   # filter workers; 10 beside live trainers (16 physical cores)
ROOTS=${ROOTS:-"lidc lidc10 mayo"}   # the tables stage; one root per process to run them in parallel
TW=${TW:-12}   # texture workers
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
PLUG=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
PY=.venv-lidc/bin/python
mkdir -p $RUN/score data/mayo/out/mse data/lidc10/out/mse

# 1a. the 10 % MSE net on Mayo (its checkpoint lives in data/lidc10/out/mse; the dump goes to Mayo's root)
has mse_mayo && CUDA_VISIBLE_DEVICES=$G LEAN_MLIR_LOWERER=pjrt PJRT_PLUGIN=$PLUG .lake/build/bin/lidc-denoise-train \
  data/lidc10/train.bin data/mayo/val.bin --arm=mse --seed=0 --eval-only \
  --ckpt=data/lidc10/out/mse --dump=data/mayo/out/mse --out=$RUN/mse10 > $RUN/score/mse10_on_mayo.log 2>&1
has mse_mayo && tail -2 $RUN/score/mse10_on_mayo.log
# 1b. the filters, knobs re-tuned on the 10 % train split, on the 10 % val and on Mayo
has filters && for m in gauss tv bm3d; do
  for root in lidc10 mayo; do
    mkdir -p data/$root/out/$m
    if [ ! -e data/$root/out/$m/val.eps.f32 ]; then
      nice -n 10 $PY scripts/probes/lidc_classical.py data/lidc10/train.bin data/$root/val.bin $m \
        data/$root/out/$m/val.eps.f32 --workers $FW > $RUN/score/${m}_$root.tune.txt 2>&1
    fi
  done
done

# 2. the tables
has tables && for root in $ROOTS; do
  for d in data/$root/out/*/; do
    for f in $d*val*.eps.f32; do
      [ -e "$f" ] || continue
      $PY scripts/probes/lidc_denoise_score.py data/$root/val.bin "$f" --records=$RUN/records_val.txt \
        > $RUN/score/fidelity_${root}_$(basename $d).$(basename $f .eps.f32).txt 2>&1 || true
    done
  done
  # the scoring set uses sample seed 0 only (seeds 1–3 cover just the 200 spread records, and a
  # table's record set is the intersection of its arms' sidecars); the spread set takes every seed
  recs=$RUN/records_val.txt; [ $root = mayo ] && recs=$RUN/records_mayo.txt
  arms=$(ls -d data/$root/out/*/ | xargs -n1 basename | sed 's/$/,/' | tr -d '\n' | sed 's/diff,/diff.k0,/; s/,$//')
  if [ $root != mayo ]; then
    $PY scripts/demos/lidc_nodules.py --data=data/$root --records=$recs --arms=$arms \
      --json=$RUN/score/nodules_$root.json > $RUN/score/nodules_$root.txt 2>&1
    $PY scripts/demos/lidc_nodules.py --data=data/$root --records=$RUN/records_spread.txt \
      --json=$RUN/score/nodules_spread_$root.json > $RUN/score/nodules_spread_$root.txt 2>&1
  fi
  # texture + findings on the scoring set, then the spread study: every seed on the 200-record
  # spread set, blobs at 0.15 and at 0.25
  nice -n 10 $PY scripts/demos/lidc_texture.py --data=data/$root --records=$recs --workers=$TW --arms=$arms \
    --json=$RUN/score/texture_$root.json --nps-csv=$RUN/score/nps_$root.csv > $RUN/score/texture_$root.txt 2>&1
  if [ $root != mayo ]; then
    for t in 0.15 0.25; do
      nice -n 10 $PY scripts/demos/lidc_texture.py --data=data/$root --records=$RUN/records_spread.txt --workers=$TW \
        --blob-threshold=$t --json=$RUN/score/spread_${root}_blob$t.json > $RUN/score/spread_${root}_blob$t.txt 2>&1
    done
  fi
done
echo "scores in $RUN/score/"
