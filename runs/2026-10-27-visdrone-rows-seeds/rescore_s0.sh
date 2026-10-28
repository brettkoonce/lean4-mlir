#!/bin/bash
# The seed-0 rows of tab:visdrone (ctrl12 / long50 / aug50 of 08-28, cfoc2 of 08-29) rescored under
# both decodes with run.sh's `score`, so the three-seed table prints one decode for every row and
# seed. The book's first three rows were only ever scored under argmax (scorer default top-k 1000),
# cfoc2 under both (0.1961 multilabel / 0.1762 argmax, runs/2026-09-01-cfoc2-rescore). The
# checkpoints are the final ones still in .lake/build; each is copied to a `<row>_s0` tag (the
# three-file copy, as the e<N> checkpoints) so nothing of the originals is touched. ~5 min a row.
# Usage: rescore_s0.sh <gpu> [row ...]      (default: ctrl12 long50 aug50 cfoc2)
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
G=$1; shift; R=runs/2026-10-27-visdrone-rows-seeds
export LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G FPN_BACKBONE=r34 FPN_TOWER=0
EXE=.lake/build/bin/yolov1-visdrone-fpn
B=.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__
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
for row in ${@:-ctrl12 long50 aug50 cfoc2}; do
  tag=${row}_s0
  log "$tag rescore start (seed-0 checkpoint of $(date -r ${B}${row}_params.bin +%F))"
  cp ${B}${row}_params.bin ${B}${tag}_params.bin && cp ${B}${row}_bn_stats.bin ${B}${tag}_bn_stats.bin \
    && cp ${B}${row}_fwd_eval.mlir ${B}${tag}_fwd_eval.mlir && score $tag
done
log "RESCORE DONE (${@:-ctrl12 long50 aug50 cfoc2})"
