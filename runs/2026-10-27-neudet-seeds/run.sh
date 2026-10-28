#!/bin/bash
# planning/part2_ship_audit.md §2.2: seeds 1 and 2 of the four NEU-DET arms of
# runs/2026-09-17-neudet-fpn-run1/README.md (run1 / noaug / noboot on the FPN, run1 on the grid),
# one card, one run at a time, then the val and test sweeps of every saved epoch. The 09-17 runs
# are the unset seed (0). ~25 min an FPN arm, ~14 min the grid; the sweeps ~10 min a split.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
G=${1:-3}; R=runs/2026-10-27-neudet-seeds
export LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G
log () { echo "$(date -u '+%F %T') $*" >> $R/queue.log; }
FPN=.lake/build/bin/yolov1-neudet-fpn; GRID=.lake/build/bin/yolov1-neudet448
for s in 1 2; do
  log "fpn run1_s$s start";   LEAN_MLIR_SEED=$s FPN_TAG=run1_s$s                    $FPN data/neu_det_fpn > $R/fpn_run1_s$s.log 2>&1;   log "fpn run1_s$s exit $?"
  log "fpn noaug_s$s start";  LEAN_MLIR_SEED=$s FPN_TAG=noaug_s$s  FPN_AUG=0         $FPN data/neu_det_fpn > $R/fpn_noaug_s$s.log 2>&1;  log "fpn noaug_s$s exit $?"
  log "fpn noboot_s$s start"; LEAN_MLIR_SEED=$s FPN_TAG=noboot_s$s FPN_NOBOOTSTRAP=1 $FPN data/neu_det_fpn > $R/fpn_noboot_s$s.log 2>&1; log "fpn noboot_s$s exit $?"
  log "grid run1_s$s start";  LEAN_MLIR_SEED=$s YOLO_TAG=run1_s$s YOLO_EPOCHS=30     $GRID data/neu_det448 > $R/grid_run1_s$s.log 2>&1;  log "grid run1_s$s exit $?"
done
for s in 1 2; do
  for t in run1 noaug noboot; do for sp in val test; do
    log "sweep fpn ${t}_s$s $sp"; scripts/sweeps/neudet_eval_sweep.sh fpn ${t}_s$s $G $sp > $R/sweep_fpn_${t}_s${s}_$sp.log 2>&1
  done; done
  for sp in val test; do
    log "sweep grid run1_s$s $sp"; scripts/sweeps/neudet_eval_sweep.sh grid run1_s$s $G $sp > $R/sweep_grid_run1_s${s}_$sp.log 2>&1
  done
done
log "ALL DONE"
