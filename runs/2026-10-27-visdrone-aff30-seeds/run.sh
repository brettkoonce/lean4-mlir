#!/bin/bash
# planning/part2_ship_audit.md §2.4: a second and a third seed of the top row of tab:visdrone —
# runs/2026-09-01-visdrone-affine/README.md's aff30 recipe (0.2363 at e28, the unset seed) — one
# card and ~2 h a seed, then the e30 and e28 checkpoints scored under the canonical decode.
# Usage: run.sh <gpu> <seed>
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
G=$1; S=$2; R=runs/2026-10-27-visdrone-aff30-seeds
export LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G FPN_BACKBONE=r34 FPN_TOWER=0
log () { echo "$(date -u '+%F %T') $*" >> $R/queue.log; }
EXE=.lake/build/bin/yolov1-visdrone-fpn
PFX=.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__aff30_s$S
log "train s$S start"
LEAN_MLIR_SEED=$S FPN_TAG=aff30_s$S FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_AFFINE=50 FPN_EPOCHS=30 \
  $EXE data/visdrone_fpn > $R/train_s$S.log 2>&1
log "train s$S exit $?"
score () {  # <tag>: infer, then the README's scoring command; the 267 MB logits dump is dropped
  FPN_TAG=$1 $EXE infer data/visdrone_fpn $R/infer_$1 > $R/infer_$1.log 2>&1
  python3 scripts/demos/yolo_map_visdrone.py $R/infer_$1/logits.bin data/visdrone448/val.bin \
    --fpn data/visdrone --grid 14 --multilabel --topk 3000 --ml-k 3 --ml-floor 0.05 > $R/score_$1.txt 2>&1
  rm -f $R/infer_$1/logits.bin
  log "$1: $(grep -o 'mAP@0.50 = [0-9.]*' $R/score_$1.txt | head -1)"
}
score aff30_s$S                   # the final checkpoint, e30
e=28                              # the row's epoch: the three-file copy to its own tag (README :72–75)
cp ${PFX}_params_e$e.bin ${PFX}e${e}_params.bin
cp ${PFX}_bn_stats_e$e.bin ${PFX}e${e}_bn_stats.bin
cp ${PFX}_fwd_eval.mlir ${PFX}e${e}_fwd_eval.mlir
score aff30_s${S}e$e
log "s$S DONE"
