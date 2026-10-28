#!/bin/bash
# Score a row's intermediate checkpoints (the trainer saves one every two epochs) under both
# decodes: the three-file copy to `<tag>e<N>`, then run.sh's `score`. Written for cfoc2_s1, whose
# e12 landed 0.06 below seed 0's; the ladder says whether that is the level of the run or a late
# fall. Usage: score_ckpt.sh <gpu> <tag> <N> [<N> ...]     e.g. score_ckpt.sh 0 cfoc2_s1 6 8 10
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
G=$1; tag=$2; shift 2; R=runs/2026-10-27-visdrone-rows-seeds
export LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G FPN_BACKBONE=r34 FPN_TOWER=0
EXE=.lake/build/bin/yolov1-visdrone-fpn
PFX=.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__$tag
log () { echo "$(date -u '+%F %T') $*" >> $R/queue.log; }
score () {
  FPN_TAG=$1 $EXE infer data/visdrone_fpn $R/infer_$1 > $R/infer_$1.log 2>&1
  python3 scripts/demos/yolo_map_visdrone.py $R/infer_$1/logits.bin data/visdrone448/val.bin \
    --fpn data/visdrone --grid 14 --multilabel --topk 3000 --ml-k 3 --ml-floor 0.05 > $R/score_$1.txt 2>&1
  python3 scripts/demos/yolo_map_visdrone.py $R/infer_$1/logits.bin data/visdrone448/val.bin \
    --fpn data/visdrone --grid 14 > $R/score_$1_argmax.txt 2>&1
  rm -f $R/infer_$1/logits.bin
  log "$1: multilabel $(grep -o 'mAP@0.50 = [0-9.]*' $R/score_$1.txt | head -1) / argmax $(grep -o 'mAP@0.50 = [0-9.]*' $R/score_$1_argmax.txt | head -1)"
}
for e in "$@"; do
  cp ${PFX}_params_e$e.bin ${PFX}e${e}_params.bin && cp ${PFX}_bn_stats_e$e.bin ${PFX}e${e}_bn_stats.bin \
    && cp ${PFX}_fwd_eval.mlir ${PFX}e${e}_fwd_eval.mlir && score ${tag}e$e || log "${tag}e$e: checkpoint missing"
done
log "CKPT DONE $tag ($*)"
