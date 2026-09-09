# Detector: finish the YOLOv5 recipe on the VisDrone FPN

**Opened 2026-09-09.** Companion to `planning/archive/visdrone_detector.md` (the detector's source
of truth, live-edited through 2026-09-01; §13 is the resume point) and
`planning/archive/yolo_v5_recipe.md` (the forward plan, 2026-09-02). Both stay archived; this doc is
the work list on top of them. Every `yolo_*.md` other than those two carries a ⛔ banner (the
scrambled-data era) or is the Pets demo; read them as history only.

## §0 State, verified against the tree

* **Best arm: mAP@0.5 0.2363** (recall 0.769, ca-AP 0.487), `aff30` = box-aware affine at p=0.50
  for 30 epochs on top of `FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2`, scored `--multilabel --topk
  3000 --ml-k 3 --ml-floor 0.05`. ~2 h on one 4060 Ti. Ledger:
  `runs/2026-09-01-visdrone-affine/README.md`. Checkpoints:
  `.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__aff30e28_params.bin` (+ every even
  epoch, + `_bn_stats.bin`).
* ⛔ The 2026-08-29 finding "12 epochs wins, never run long" was **retracted** for the affine
  pack (`visdrone_detector.md` ~:900–950): it holds for HSV+hflip only. The doc's own header still
  says "STATE 2026-08-29"; §4 still says IREE/vmfb/`IREE_BACKEND=rocm` required; §3/§4 still say
  class weights are on. Only §12b/§13 are current.
* **Engine:** XLA/PJRT by default; IREE only under `LEAN_MLIR_LOWERER=iree`
  (`ffi/lowerer.c:115-175`). The demo's IREE mentions are conditional, not stale. The port measured
  9.47× over IREE (`planning/archive/detector_pjrt_port.md` §9).
* **Recipe status: 0 of 5 items on `main`.** The unmerged branch `yolo-v5-assignment` (one
  commit, `514c5f21`) carries T1 (ratio-match + neighbour cells) and T2 (`2σ−0.5` / `(2σ)²` box
  param) on the ENCODER and SCRIPT side only — `preprocess_visdrone.py`, `ffi/f32_helpers.c`,
  `deploy/orin_detect.py`, `scripts/*` (coverage 88.2% → 93.0%) — and touches **no `LeanMlir/`
  file**. The loss emitter (`MlirCodegen.emitDiouForward`, ~:5171–5227) still decodes the old
  parameterization, so the branch's encoder and main's graph disagree. Its planning doc sits at
  the pre-archive path `planning/yolo_v5_recipe.md` and conflicts on rebase.
* ⚠ The `.lake/build/bin/yolov1-visdrone-fpn` binary on disk (2026-09-02) is from that branch.
  Rebuild on `main` before any `infer` / `emit-deploy`.

## §1 Work packages, ranked by value per cost

1. **Land T1 + T2 completely.** Rebase the branch (move its doc to `planning/archive/`), then
   add the graph half of T2 in `emitDiouForward`: `cx = (j + 2σ(tx) − 0.5)/g`, `w = a·(2σ(tw))²`
   (removes the hand-placed `min(tw, 8)` cap that stops NaNs today). Mirror in
   `scripts/yolo_map_visdrone.py:223-228` and the two `deploy/` decoders. Gates: coverage script
   (seconds, before GPU), `scripts/fpn_loss_probe_check.py` FD arm (needs the throwaway IREE
   venv), then one 30-epoch run against `aff30` at matched schedule. Cost: a day + 2 GPU-h.
2. **Per-level objectness balance `[4.0, 1.0, 0.4]`** (`emitMultiScaleYoloLoss` ~:5649–5695 sums
   the three scales unweighted; P3 outweighs P5 16× by cell count). Three constants; gradient
   shape unchanged. Cheapest loss-side lever and in no doc's "ruled out" list.
3. **Close the dose/length question at 30 epochs:** p=0.25 vs p=0.50 both at 30, then 40–50 at
   p=0.50 (e28 ≈ e30 is a plateau, not a shown ceiling). Config only, ~2 GPU-h each.
4. **A seed knob on `LeanMlir/Train.lean`** (`Train.lean:872,894,915` derive every draw from
   `epoch*10000+bi`; `LEAN_MLIR_SEED` is `VerifiedTrain.lean` only). Every detector number is n=1
   and items 1–2 are expected to move mAP by less than the unmeasured spread. Prerequisite for any
   claim under ~0.02.
5. **Compiled-in defaults = shipped recipe.** `MainYolov1VisdroneFpn.lean:171-193` defaults to 12
   epochs / no aug / sqrt class weights, and `backboneFromEnv` (:233-235) defaults to **r50**
   "because `jax_r34_imagenet.bin` was deleted" — it is on disk (85,138,688 B, 2026-08-28). Three
   docs carry ⚠ banners that exist only because of this.
6. **Bootstrap BN stats.** `Train.lean:766-778` falls back to zeros on every detector run (no
   `jax_r34_imagenet_bn_stats.bin`); the pretrained running statistics are discarded. Export the
   companion file; no codegen.
7. Then, in order: **T4 per-class BCE** (softmax-CE at ~:5504; the +7.6% multilabel decode win
   says the softmax readout is wrong at the source; re-tune `--ml-k` after), **T5a IoU-aware
   objectness** (target is the constant mask channel at ~:5461; needs stop-gradient + an FD arm;
   `yolo_assignment.md` priced it on void data — re-measure), **T3 mosaic** (port
   `preprocess_pets_mosaic.py:49`; depends on T1), **resolution > 448** (24/64 px thresholds and
   the 56/28/14 grids are hardcoded in THREE places: `preprocess_visdrone.py:158-159`,
   `Train.lean:914`, `ffi/f32_helpers.c` — collapse them regardless).
8. **Decode + NMS out of Python** is a project (`scripts/yolo_map_visdrone.py:203-268`, duplicated
   in `deploy/orin_detect.py`). The cheaper on-device win is the u8 preprocess fold —
   see `planning/orin_rerun.md`.

## §2 Copy to fix when the next number lands

`visdrone_detector.md` header date + §4 engine + §3/§4 class weights; `historical/RESULTS.md:188`
(compares YOLOv8 to 0.1526 while :152 says 0.2363; 35.7 fps appears nowhere in it);
`deploy/ORIN_SMOKE_TEST.md:73` (0.1526); `deploy/README.md:27-36` (229 fps, the wrong graph);
`lakefile.lean:1120,1131,1177-1198` cite ⛔-bannered docs and the live exe at :675 has no comment;
`formalization.yaml` has zero detector rows (codegen+FD-gated by decision, `visdrone_detector.md:144`
— say so there).

## §3 Rules

Ask before any run over ~30 min. `FPN_BACKBONE=r34 FPN_TOWER=0 FPN_TAG=<tag>` on every command —
omitting the backbone silently trains r50. Never regenerate anchors by k-means; copy the demo's.
Never mix decodes in a comparison (everything before 2026-08-29 is argmax).
