#!/bin/bash
# planning/part2_ship_audit.md §2.4 / §7.4 (the user's call, 10-27 evening: "burn a day or two of
# compute on this area"): the other four R34+FPN rows of tab:visdrone at seeds 1 and 2, so every
# row of the table is three seeds like the top one (runs/2026-10-27-visdrone-aff30-seeds/). The
# recipes are the ladder in planning/archive/visdrone_detector.md ("The full ladder"):
#   ctrl12   12 ep, no aug, sqrt class weights            (0.1526 argmax, the unset seed)
#   long50   50 ep, no aug, sqrt                           (0.1243)
#   aug50    50 ep, FPN_AUG=1 (HSV + hflip), sqrt          (0.1674; e30 0.1731 "best seen")
#   cfoc2    12 ep, FPN_AUG=1, FPN_CLSW=none, FPN_CLSFOCAL=2 (0.1961 multilabel / 0.1762 argmax)
# The book's rows for the first three were scored under the historical decode (argmax, the
# scorer's default top-k 1000) and the last two under the canonical multilabel decode
# (runs/2026-09-01-visdrone-decode-sweep/README.md), with no column saying so; every checkpoint
# here is scored under BOTH, so the table can print one decode for all five rows. The 50-epoch
# rows are also scored at e10/e20/e30/e40 (the three-file copy to its own tag, as the aff30 script).
# ~4.1 min an epoch without affine (cfoc2's 49 min), so ~3.4 h a 50-epoch row, ~50 min a 12-epoch one.
# Usage: run.sh <gpu> <row>:<seed> ...
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
G=$1; shift; R=runs/2026-10-27-visdrone-rows-seeds
export LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G FPN_BACKBONE=r34 FPN_TOWER=0
EXE=.lake/build/bin/yolov1-visdrone-fpn
log () { echo "$(date -u '+%F %T') $*" >> $R/queue.log; }
score () {  # <tag>: infer once, score under both decodes, drop the 267 MB logits dump
  FPN_TAG=$1 $EXE infer data/visdrone_fpn $R/infer_$1 > $R/infer_$1.log 2>&1
  python3 scripts/demos/yolo_map_visdrone.py $R/infer_$1/logits.bin data/visdrone448/val.bin \
    --fpn data/visdrone --grid 14 --multilabel --topk 3000 --ml-k 3 --ml-floor 0.05 > $R/score_$1.txt 2>&1
  python3 scripts/demos/yolo_map_visdrone.py $R/infer_$1/logits.bin data/visdrone448/val.bin \
    --fpn data/visdrone --grid 14 > $R/score_$1_argmax.txt 2>&1
  rm -f $R/infer_$1/logits.bin
  log "$1: multilabel $(grep -o 'mAP@0.50 = [0-9.]*' $R/score_$1.txt | head -1) / argmax $(grep -o 'mAP@0.50 = [0-9.]*' $R/score_$1_argmax.txt | head -1)"
}
for job in "$@"; do
  row=${job%%:*}; S=${job##*:}; tag=${row}_s$S
  case $row in
    ctrl12) ROW="FPN_EPOCHS=12" ;;
    long50) ROW="FPN_EPOCHS=50" ;;
    aug50)  ROW="FPN_AUG=1 FPN_EPOCHS=50" ;;
    cfoc2)  ROW="FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_EPOCHS=12" ;;
    *) log "$job: unknown row"; continue ;;
  esac
  PFX=.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__$tag
  log "$tag start ($ROW)"
  env LEAN_MLIR_SEED=$S FPN_TAG=$tag $ROW $EXE data/visdrone_fpn > $R/train_$tag.log 2>&1
  log "$tag exit $?"
  score $tag                                   # the final checkpoint
  case $row in long50|aug50)
    for e in 10 20 30 40; do
      cp ${PFX}_params_e$e.bin ${PFX}e${e}_params.bin && cp ${PFX}_bn_stats_e$e.bin ${PFX}e${e}_bn_stats.bin \
        && cp ${PFX}_fwd_eval.mlir ${PFX}e${e}_fwd_eval.mlir && score ${tag}e$e
    done ;;
  esac
  log "$tag DONE"
done
log "ALL DONE ($*)"
