#!/usr/bin/env bash
# Epoch sweep for a NEU-DET detector arm: infer + score every saved checkpoint
# (e2, e4, …) of one run and print a one-line-per-epoch table.
#
# Unlike the VisDrone watchers (scripts/run_fpn_*_eval_watch.sh), which copied
# `_params_eN.bin` over the run's final `_params.bin`, this passes the epoch to
# the binary (`FPN_EVAL_EPOCH` / `YOLO_EVAL_EPOCH`), so the run's artifacts are
# never touched and a sweep can run beside the training that is producing them.
#
# Usage: scripts/neudet_eval_sweep.sh fpn|grid <tag> <gpu> [split=val] [epochs="2 4 … 30"]
#   scripts/neudet_eval_sweep.sh fpn  run1 3            # val, every saved epoch
#   scripts/neudet_eval_sweep.sh grid run1 3 test "30"  # the table's row
# Output: runs/<date>-neudet-<arm>-<tag>-sweep/{infer,score}_<split>_e<N>.log + table.
set -u
cd "$(dirname "$0")/.."

ARM=$1; TAG=$2; GPU=$3; SPLIT=${4:-val}; EPOCHS=${5:-"2 4 6 8 10 12 14 16 18 20 22 24 26 28 30"}
case "$ARM" in
  fpn)  EXE=./.lake/build/bin/yolov1-neudet-fpn; DATA=data/neu_det_fpn
        PFX=".lake/build/resnet_34___fpn_detector_448__neu_det__$(echo "$TAG" | tr -c 'a-z0-9\n' '_')"
        SCORE_ARGS="--fpn data/neu_det --grid 14 --classes neu"
        ENVTAG=FPN_TAG; ENVEP=FPN_EVAL_EPOCH; ENVSPLIT=FPN_EVAL_SPLIT ;;
  grid) EXE=./.lake/build/bin/yolov1-neudet448; DATA=data/neu_det448
        PFX=".lake/build/resnet_34___yolov1_448__neu_det__$(echo "$TAG" | tr -c 'a-z0-9\n' '_')"
        SCORE_ARGS="--grid 14 --classes neu"
        ENVTAG=YOLO_TAG; ENVEP=YOLO_EVAL_EPOCH; ENVSPLIT=YOLO_EVAL_SPLIT ;;
  *) echo "arm must be fpn|grid"; exit 1 ;;
esac
OUT="runs/$(date +%F)-neudet-${ARM}-${TAG}-sweep"
mkdir -p "$OUT"
GT=data/neu_det448/${SPLIT}.bin      # the scorer reads GT from the single-grid records + sidecar

# Two scorings per checkpoint. `ml` is the VisDrone protocol the 0.2363 was
# measured under (--multilabel --topk 3000 --ml-k 3 --ml-floor 0.05,
# demos/README.md) and is the number the tables quote; `argmax` is the plain
# one-detection-per-cell readout. On VisDrone the two differ by ~0.004 through
# the rare classes; on balanced NEU they should not differ, and printing both is
# the free check that the rare-class machinery is not what the number is made of.
ML_ARGS="--multilabel --topk 3000 --ml-k 3 --ml-floor 0.05"
printf "%-6s %-8s %-8s %-8s %-8s | %s\n" epoch mAP_ml caAP recall mAP_argmax "per-class AP, ml (cr in pa ps rs sc)" | tee "$OUT/table_${SPLIT}.txt"
for EP in $EPOCHS; do
  CK="${PFX}_params_e${EP}.bin"
  if [ ! -f "$CK" ]; then echo "e${EP}: no checkpoint at $CK — skipped" | tee -a "$OUT/table_${SPLIT}.txt"; continue; fi
  env "$ENVTAG=$TAG" "$ENVEP=$EP" "$ENVSPLIT=$SPLIT" CUDA_VISIBLE_DEVICES=$GPU \
    $EXE infer "$DATA" "$OUT/e${EP}_${SPLIT}" > "$OUT/infer_${SPLIT}_e${EP}.log" 2>&1 \
    || { echo "e${EP}: infer FAILED (see $OUT/infer_${SPLIT}_e${EP}.log)"; continue; }
  L="$OUT/e${EP}_${SPLIT}/logits.bin"
  python3 scripts/yolo_map_visdrone.py "$L" "$GT" $SCORE_ARGS $ML_ARGS > "$OUT/score_${SPLIT}_e${EP}.log" 2>&1
  python3 scripts/yolo_map_visdrone.py "$L" "$GT" $SCORE_ARGS          > "$OUT/score_argmax_${SPLIT}_e${EP}.log" 2>&1
  MAP=$(grep -oE 'mAP@0.50 = [0-9.]+' "$OUT/score_${SPLIT}_e${EP}.log" | grep -oE '[0-9.]+$')
  MAPA=$(grep -oE 'mAP@0.50 = [0-9.]+' "$OUT/score_argmax_${SPLIT}_e${EP}.log" | grep -oE '[0-9.]+$')
  CA=$(grep -oE 'localization AP@0.50 = [0-9.]+' "$OUT/score_${SPLIT}_e${EP}.log" | grep -oE '[0-9.]+$')
  REC=$(grep -oE 'recall=[0-9.]+' "$OUT/score_${SPLIT}_e${EP}.log" | head -1 | grep -oE '[0-9.]+$')
  PC=$(grep -E '^\s+(crazing|inclusion|patches|pitted_surf|rolled-in|scratches):' "$OUT/score_${SPLIT}_e${EP}.log" \
       | grep -oE 'AP@0.50 = [0-9.]+' | grep -oE '[0-9.]+$' | tr '\n' ' ')
  printf "%-6s %-8s %-8s %-8s %-8s | %s\n" "e${EP}" "$MAP" "$CA" "$REC" "$MAPA" "$PC" | tee -a "$OUT/table_${SPLIT}.txt"
  # keep the LAST epoch's logits for the figure; drop the rest (267 MB each)
  [ "$EP" = "$(echo $EPOCHS | awk '{print $NF}')" ] || rm -f "$L"
done
echo "SWEEP DONE $(date -Is) → $OUT/table_${SPLIT}.txt"
