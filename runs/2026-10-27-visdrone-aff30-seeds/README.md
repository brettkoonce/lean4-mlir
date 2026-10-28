# 2026-10-27 — VisDrone, the top row of tab:visdrone at two more seeds (planning/part2_ship_audit.md §2.4)

`runs/2026-09-01-visdrone-affine/README.md`'s recipe — R34 backbone bootstrapped from this
stack's ImageNet run, towerless FPN head, HSV + hflip + box-aware affine at p = 0.5, class
weights off, class focal γ = 2, 30 epochs — gave the book's bold 0.2363 (e28) at the unset seed,
with the README's own warning that no error bar existed because `Train.lean` had no seed knob.
It has one now (`LEAN_MLIR_SEED`, `LeanMlir/Train.lean:651–654`, printed as "seed offset" at
launch); this is the same recipe at seeds 1 and 2, one card each (cards 1 and 2 of the grind,
2 h 44 min for 30 epochs beside the other chains), the e30 and e28 checkpoints scored under
the canonical decode exactly as the 09-01 rows (`run.sh`; the 267 MB logits dumps dropped
after scoring).

```
LEAN_MLIR_SEED=$S LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G \
  FPN_BACKBONE=r34 FPN_TOWER=0 FPN_TAG=aff30_s$S \
  FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_AFFINE=50 FPN_EPOCHS=30 \
  .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
FPN_TAG=aff30_s$S … infer data/visdrone_fpn <dir>          # e28: the three-file copy to aff30_s${S}e28
python3 scripts/demos/yolo_map_visdrone.py <dir>/logits.bin data/visdrone448/val.bin \
  --fpn data/visdrone --grid 14 --multilabel --topk 3000 --ml-k 3 --ml-floor 0.05
```

## The row, three seeds (VisDrone-DET val, 548 images, 38,759 uncapped GT boxes)

| seed | epoch | mAP@0.5 | recall | class-agnostic AP |
|---|---|---|---|---|
| 0 (09-01, the book's row) | 28 | **0.2363** | 0.7688 | 0.4872 |
| 0 | 30 | 0.2362 | 0.7685 | 0.4890 |
| 1 | 28 | 0.2311 | 0.7696 | 0.4856 |
| 1 | 30 | 0.2313 | 0.7694 | 0.4874 |
| 2 | 28 | 0.2204 | 0.7684 | 0.4484 |
| 2 | 30 | 0.2289 | 0.7691 | 0.4685 |
| **mean ± sd, e30** | | **0.2321 ± 0.0037** | 0.769 ± 0.000 | 0.482 ± 0.011 |
| mean ± sd, e28 | | 0.2293 ± 0.0081 | 0.769 ± 0.001 | 0.474 ± 0.022 |

Per class at e28 (the book's epoch), AP@0.5:

| class (val GT) | seed 0 | seed 1 | seed 2 | spread |
|---|---|---|---|---|
| car (14,064) | 0.6851 | 0.6842 | 0.6189 | 0.066 |
| van (1,975) | 0.2900 | 0.2698 | 0.2539 | 0.036 |
| pedestrian (8,844) | 0.2255 | 0.2227 | 0.1999 | 0.026 |
| truck (750) | 0.1707 | 0.1489 | 0.1471 | 0.024 |
| bus (251) | 0.2645 | 0.2599 | 0.2569 | 0.008 |
| motor (4,886) | 0.2658 | 0.2628 | 0.2644 | 0.003 |
| people (5,125) | 0.2202 | 0.2215 | 0.2211 | 0.001 |
| tricycle (1,045) | 0.1438 | 0.1425 | 0.1425 | 0.001 |
| awning-tri (532) | 0.0623 | 0.0618 | 0.0634 | 0.002 |
| bicycle (1,287) | 0.0356 | 0.0365 | 0.0361 | 0.001 |

## What it says

- **The row is 0.232 ± 0.004 at e30, and 0.2363 was the best of three.** The book's bold number
  is seed 0's; the seed sd is 0.004 at e30 and recall is the same 0.769 on every seed and epoch.
  The 09-01 reading "e28 ≈ e30 is what makes this a settled measurement" holds for seeds 0 and 1
  (Δ ≤ 0.0002) and not for seed 2, whose e28 is 0.0085 below its e30 with the drop entirely in
  car and pedestrian (class-agnostic AP 0.448 against 0.469): single late-epoch checkpoints
  move by ~0.01 on their own, which is the floor for any one-seed comparison in this table.
- **The seed spread lives in the big classes, not the rare ones.** Car moves 0.066 across seeds
  and van / pedestrian / truck 0.02–0.04, while the four rarest classes agree to 0.001–0.002.
  The 09-01 per-class claim — `cfoc2` → `aff30` lifts every class, the rare ones most
  (awning-tri +59%, bicycle +41%, tricycle +40%) — is far outside the seed spread for those
  classes; its car gain (+6.6%, 0.643 → 0.685) is inside it.
- For the table: the top row prints as 0.232 ± 0.004 (three seeds, e30) or keeps 0.2363 with
  "best of three seeds; mean 0.232" — either way the 0.1961 second row is a different arm by
  more than seed noise, and the rows below it get their own seeds in
  `runs/2026-10-27-visdrone-rows-seeds/` the same night.
