#!/usr/bin/env bash
# Watch for T2a (RetinaNet head tower, depth 4) checkpoints and eval each
# on GPU 1 while training continues on GPU 0. planning/yolo_fpn.md train->eval
# recipe. The A/B reference is the T1b arm (figures/yolo_fpn_wcls_e*), which is
# the correct control: a zero-init bias reproduces the biasless head exactly, so
# the ONLY difference here is that objectness starts at logit(0.01).
#
# fpn_obj_separation.py is the measurement that matters: this lever was aimed at
# objectness DYNAMIC RANGE (every e12 logit sat in [-2.7, -1.2], AUC 0.742), so
# watch the logit spread and AUC, not just recall.
set -u
# FPN_TOWER MUST be set on the infer call: it selects the spec, and hence the
# checkpoint prefix. Without it, infer silently evaluates the tower=0 arm's
# checkpoint and reports THAT arm's numbers for every epoch.
cd "$(dirname "$0")"

PFX=.lake/build/resnet_34___fpn_detector_448_wcls_pb_tower4__visdrone_
PY=/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python3
EXE=./.lake/build/bin/yolov1-visdrone-fpn
DEADLINE=$(( $(date +%s) + 8*3600 ))

for EP in 2 4 6 8 10 12; do
  while [ ! -f "${PFX}_params_e${EP}.bin" ]; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT waiting for e${EP}"; exit 1; }
    sleep 60
  done
  sleep 20                      # let the writer finish flushing
  OUT=figures/yolo_fpn_t2a_e${EP}
  echo "=================== e${EP} $(date -Is) ==================="
  cp "${PFX}_params_e${EP}.bin"   "${PFX}_params.bin"
  cp "${PFX}_bn_stats_e${EP}.bin" "${PFX}_bn_stats.bin"
  mkdir -p "$OUT"
  IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 FPN_TOWER=4 $EXE infer data/visdrone_fpn "$OUT" \
    > "runs/fpn_t2a_infer_e${EP}.log" 2>&1 || { echo "infer e${EP} FAILED"; continue; }
  echo "--- mAP e${EP} ---"
  $PY scripts/yolo_map_visdrone.py "$OUT/logits.bin" data/visdrone448/val.bin \
      --grid 14 --fpn data/visdrone 2>&1 | tail -25
  echo "--- objectness separation + class collapse e${EP} ---"
  $PY scripts/fpn_obj_separation.py "$OUT/logits.bin" data/visdrone_fpn/val.bin \
      data/visdrone 2>&1 | tail -16
done
echo "WATCHER DONE $(date -Is)"
