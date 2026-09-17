# 2026-09-17 — the VisDrone detector, unchanged, on NEU-DET steel defects

The workings behind the book's *Industrial inspection* section and the
`demos/README.md` NEU-DET entry. Plan: `planning/neu_det_fpn_demo.md` (§11 is
the running log). Sibling run dirs, all from this day:

| dir | what |
|---|---|
| `2026-09-17-neudet-fpn-run1/` | R34+FPN, the VisDrone recipe verbatim (this file) |
| `2026-09-17-neudet-fpn-noaug/` | same, HSV+hflip off (affine kept) |
| `2026-09-17-neudet-fpn-noboot/` | same, He init instead of the ImageNet R34 prefix |
| `2026-09-17-neudet-grid-run1/` | the single 14×14 grid arm, 30 epochs |
| `2026-09-17-neudet-*-sweep/` | per-epoch infer + score logs and the `table_{val,test}.txt` files every number below is read from |
| `2026-09-17-visdrone-grid448-remeasure/` | the single-grid arm on VisDrone, re-measured on fixed data |
| `2026-09-17-neudet-smoke/` | the 1-epoch smokes (logits dropped, logs kept) |

Checkpoints (`.lake/build/resnet_34___{fpn_detector,yolov1}_448__neu_det__<tag>_params_e*.bin`)
are not in the repo.

## Data

`download_neu.sh` → `data/neu_det/` from the maintainer's own Google-Drive
`NEU-DET.zip` (26 MB): 1,800 / 1,800 images and VOC XMLs, six class names
exactly as expected, 4,189 boxes, all 200×200 (3-channel JPEGs with grey
content). No official split: `preprocess_neu_det.py` draws **1,080 / 360 / 360**
stratified 180 / 60 / 60 per class from seed 0, and writes the same two record
formats as `preprocess_visdrone.py` — `data/neu_det_fpn/` (FPN, 1.34 MB/record)
and `data/neu_det448/` (single grid + the uncapped `*.full_gt.bin` sidecar the
scorer reads) — for train, val and test. Encoders are imported from the VisDrone
preprocessor, not copied.

**Geometry (Gate 0), train split at 448 px:** 2.35 boxes/image; P3 / P4 / P5
routing **0.0% / 1.9% / 98.1%** (0 / 47 / 2,492 boxes) at the 24 / 64 px
thresholds; 53% of boxes span more than half the frame; slot coverage 99.9% on
both the FPN grids and the single 14×14 grid (VisDrone: 88% / 61%). The
opposite regime in one line.

**Anchors.** `scripts/neu_anchors.py --save data/neu_det` — k-means per level
on the train split by the VisDrone/COCO code path. P4 recall@0.5 0.98; P5, which
carries 98% of the boxes at every aspect from 64 px to the full frame, fits
three anchors at mean wh-IoU 0.52 / recall@0.5 0.52 (k=6 → 0.84, k=9 → 0.94).
The plan's Gate 0 asked for ≥ 0.9, calibrated on VisDrone's homogeneous tiny
cars; that is the prior's *fit*, not a ceiling — the head regresses `exp(t)` and
the k=3 log-residuals top out at |2.0| against a cap of 8. **A=3 kept**, so the
spec is the VisDrone detector layer-for-layer. P3 has no boxes and gets a fixed
fallback table. The values live as constants in `demos/MainYolov1NeuDetFpn.lean`.

## Arms

`demos/MainYolov1NeuDetFpn.lean` is the VisDrone spec with NEU anchors, class
weights off (balanced data), six classes in ids 0–5 of the ten-slot one-hot
(the 5+10 per-anchor width is baked into the `fpnDetect` codegen), and the
0.2363 recipe's env settings as defaults. `demos/MainYolov1NeuDet448.lean` is
`demos/archive/MainYolov1VisDrone448.lean` with a new name and epoch/tag knobs.

```bash
FPN_TAG=run1                    CUDA_VISIBLE_DEVICES=0 lake exe yolov1-neudet-fpn data/neu_det_fpn
FPN_TAG=noaug  FPN_AUG=0        CUDA_VISIBLE_DEVICES=3 lake exe yolov1-neudet-fpn data/neu_det_fpn
FPN_TAG=noboot FPN_NOBOOTSTRAP=1 CUDA_VISIBLE_DEVICES=2 lake exe yolov1-neudet-fpn data/neu_det_fpn
YOLO_TAG=run1  YOLO_EPOCHS=30   CUDA_VISIBLE_DEVICES=1 lake exe yolov1-neudet448  data/neu_det448
scripts/neudet_eval_sweep.sh fpn  run1 0 val      # every saved epoch → table_val.txt
scripts/neudet_eval_sweep.sh fpn  run1 0 test "28 30"
```

FPN: 135 steps/epoch at batch 8, 45–56 s/epoch on one RTX 4060 Ti (≈25 min for
30 epochs); the grid arm 67 steps/epoch at batch 16, 28 s/epoch. Loss start:
79 with the ImageNet prefix, 314 without.

Scoring is the VisDrone protocol (`scripts/yolo_map_visdrone.py … --fpn
data/neu_det --grid 14 --classes neu --multilabel --topk 3000 --ml-k 3
--ml-floor 0.05`) against the uncapped GT sidecar. The sweep also prints the
plain argmax readout: on NEU the two agree to ±0.003 at every epoch, so nothing
in these numbers comes from the rare-class machinery (on VisDrone they differ
by 0.004). The single-grid decode has no multilabel path; its two columns are
identical by construction.

## Table 1 — NEU-DET, 360 test images, at each arm's val-peak epoch

| arm | epoch | mAP@0.5 | recall | class-agnostic AP | val mAP@0.5 |
|---|---|---|---|---|---|
| **R34+FPN, VisDrone recipe** (bootstrap, HSV+hflip, affine p=0.5, class focal) | 28 | **0.623** | 0.992 | 0.625 | 0.636 |
| R34+FPN, VisDrone recipe, at e30 | 30 | 0.615 | 0.989 | 0.617 | 0.631 |
| R34+FPN, HSV+hflip off (affine kept) | 30 | 0.630 | 0.987 | 0.640 | 0.652 |
| R34+FPN, no ImageNet bootstrap | 26 | 0.555 | 0.986 | 0.555 | 0.581 |
| R34+FPN, no ImageNet bootstrap, at e30 | 30 | 0.559 | 0.988 | 0.552 | 0.573 |
| **R34 single grid 14×14**, bootstrap, 30 ep | 28 | **0.607** | 0.917 | 0.609 | 0.642 |
| R34 single grid 14×14, at e30 | 30 | 0.614 | 0.922 | 0.615 | 0.636 |

Published, same dataset, other splits and recipes (mAP@0.5): stock detectors at
640 px / 300 epochs on 1,440/180/180 — Faster R-CNN 0.766, YOLOv5s 0.762,
YOLOv7 0.730, YOLOv8s 0.781, their improved YOLOv5 0.832 (Zhou, Wang & Wang,
*Sci. Rep.* 15:44492, 2025); at ~63 epochs on 70/20/10 — YOLO11s 0.716,
YOLOv8s 0.687, RetinaNet 0.462, Faster R-CNN 0.301 (Maity & Ghosh,
arXiv:2510.21811, 2025); a self-supervised-pretrained Faster R-CNN at 200 px on
1,080/360/360 — 0.768 (Hu, Ma & Xu, *J. Mater. Inform.* 5(4), 2025).

Per-class test AP at the val-peak epoch, FPN run1 / single grid:
crazing 0.29 / 0.36, inclusion 0.66 / 0.64, patches 0.84 / 0.82, pitted surface
0.74 / 0.68, rolled-in scale 0.41 / 0.43, scratches 0.79 / 0.72. Crazing and
rolled-in scale — diffuse textures whose "box" is most of the crop, every NEU
paper's worst two — are the worst two here as well, for both heads.

Val sweeps, mAP@0.5 every other epoch (`*-sweep/table_val.txt`):

| arm | e10 | e14 | e18 | e22 | e26 | e28 | e30 |
|---|---|---|---|---|---|---|---|
| FPN, recipe | 0.350 | 0.499 | 0.526 | 0.618 | 0.625 | **0.636** | 0.631 |
| FPN, HSV+hflip off | 0.381 | 0.185 | 0.476 | 0.626 | 0.642 | 0.638 | **0.652** |
| FPN, no bootstrap | 0.143 | 0.313 | 0.414 | 0.524 | **0.581** | 0.577 | 0.573 |
| single grid | 0.198 | 0.392 | 0.491 | 0.589 | 0.621 | **0.642** | 0.636 |

Mid-schedule checkpoints dip by a factor of two on single epochs (recipe e12,
noaug e14) and recover two epochs later — the learning rate is still near its
peak there; the late rows are the ones to read.

## Table 2 — VisDrone-DET val, 548 images, uncapped GT (38,759 boxes)

| arm | epochs | mAP@0.5 | recall | class-agnostic AP |
|---|---|---|---|---|
| R34+FPN, 30 ep, +scale aug (`runs/2026-09-01-visdrone-affine/`) | 30 | **0.2363** | 0.769 | 0.487 |
| R34 single grid 14×14, archived recipe, **re-measured 2026-09-17** | 12 | **0.0391** | 0.184 | 0.107 |

⛔ The "single grid, mAP 0.0000" the plan quoted is `planning/archive/yolo_fpn.md`'s
opening line from 2026-07-17, five days before the shuffle-pairing fix, and that
doc's own banner says every number in it describes the bug. The archived
448/14 arm had never been scored on fixed data. Re-measured here on the same
lowerer and scorer as the NEU rows: 0.039 — car 0.20, every other class ≤ 0.07,
people / bicycle / awning-tricycle at zero.

## What the two tables say

- **On steel the single grid matches the multi-scale head; on drones it is six
  times behind.** NEU test: FPN 0.623 vs grid 0.607 (recall 0.99 vs 0.92).
  VisDrone: 0.236 vs 0.039 (recall 0.77 vs 0.18). Same backbone, same
  bootstrap, same scorer, same lowerer. The FPN pays for itself only where
  objects are small — 98% of NEU's boxes land on P5, so on NEU the neck is
  carrying three heads for one level's worth of work.
- **Recall is not the problem on NEU** — the FPN finds 99% of defects; the
  number is precision and ranking, and it is two classes: crazing 0.29 and
  rolled-in scale 0.41 against 0.66–0.84 for the other four.
- **The ImageNet prefix is worth +0.07** (0.623 vs 0.555 test) at 1,080 images
  — where BraTS said "one epoch" at 14k slices, here the gap holds to epoch 30.
- **Photometric augmentation is within noise**: off is +0.007 on test, +0.016
  on val, n=1 each, under the ~0.02 floor the VisDrone table established. The
  box-aware affine was kept in both arms and is untested here.
- **Both arms are still rising at e26–e30** under a 30-epoch cosine: 30 epochs
  here is 4,050 steps where VisDrone's 30 were 24,000. The gap to the published
  0.73–0.78 stock rows is schedule, resolution (448 vs 640) and split before it
  is architecture — per the plan's §9, said, not chased.

## Figure

`demos/figures/neudet_fpn.png` (`scripts/fpn_render.py … --classes neu
--compare-grid … --indices 33,87,267,327 --topk-per-gt --layout cols`): truth /
R34+FPN / single grid on one crazing, inclusion, rolled-in-scale and scratches
crop from val, each with two GT boxes. The book's copy is
`blueprint/src/figures/demos/neudet_fpn.png`.

## Two things fixed on the way

- `ffi/pjrt_ffi.c`: `iree_ffi_train_step_adam_yolov1` was a `not_ported` stub
  on the XLA shim (the VisDrone single-grid arms were IREE-era), so the grid arm
  could not train. Ported — the DDPM protocol plus the mask input — and the shim
  rebuilt by rename so running jobs kept their mapped copy. Not step-tied against
  IREE; the evidence is the training curve and the 0.61.
- `scripts/yolo_map_visdrone.py --classes neu`, `scripts/fpn_render.py
  --classes / --compare-grid`, `FPN_EVAL_EPOCH` / `YOLO_EVAL_EPOCH` on both infer
  paths so a sweep never copies checkpoints over a run's own files.
