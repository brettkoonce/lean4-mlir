# 2026-10-27 — NEU-DET, the four arms of Table 1 at seeds 1 and 2 (planning/part2_ship_audit.md §2.2)

`runs/2026-09-17-neudet-fpn-run1/README.md`'s Table 1 — the VisDrone R34+FPN recipe verbatim on
NEU-DET steel defects, with HSV+hflip off, without the ImageNet bootstrap, and the single
14×14 grid head — was one seed per arm, and its three readings (the bootstrap is worth +0.07; the
single grid matches the FPN on steel; photometric augmentation is within noise) were each n = 1
under a floor the 09-17 README put at ~0.02. These are the same four arms, same data split
(`data/neu_det_fpn`, `data/neu_det448`, the 1,080 / 360 / 360 draw of 09-17), same binaries, at
`LEAN_MLIR_SEED=1` and `2` (the He-init of what is not bootstrapped, the per-epoch shuffle and
the augmentation draws), one card (card 3 of the grind), one run at a time, then the val and
test sweeps of every saved epoch (`run.sh`, `queue.log`: 24–28 min an FPN arm, 14 the grid;
the sweeps 33–36 min a split beside three busy cards, 2 min for the grid).

```
LEAN_MLIR_SEED=$s FPN_TAG=run1_s$s                    CUDA_VISIBLE_DEVICES=3 .lake/build/bin/yolov1-neudet-fpn data/neu_det_fpn
LEAN_MLIR_SEED=$s FPN_TAG=noaug_s$s  FPN_AUG=0         …
LEAN_MLIR_SEED=$s FPN_TAG=noboot_s$s FPN_NOBOOTSTRAP=1 …
LEAN_MLIR_SEED=$s YOLO_TAG=run1_s$s  YOLO_EPOCHS=30    CUDA_VISIBLE_DEVICES=3 .lake/build/bin/yolov1-neudet448 data/neu_det448
scripts/sweeps/neudet_eval_sweep.sh fpn  run1_s$s 3 val    # → runs/<date>-neudet-fpn-run1_s$s-sweep/table_val.txt
scripts/sweeps/neudet_eval_sweep.sh fpn  run1_s$s 3 test
python3 runs/2026-10-27-neudet-seeds/summary.py            # → summary.md, the tables below
```

The sweep dirs are `runs/2026-10-27-neudet-fpn-{run1,noaug,noboot}_s1-sweep/` and
`runs/2026-10-28-neudet-{fpn-{run1,noaug,noboot}_s2,grid-run1_s1,grid-run1_s2}-sweep/` (named
by the day each sweep ran); seed 0 is the 09-17 sweeps. Scoring is unchanged: the VisDrone
multilabel protocol against the uncapped GT sidecar; the argmax column agrees to ±0.004 on
every row and ±0.001 on the late ones, as on 09-17. Checkpoints stay on ares; the training logs and sweep logs are here.

## Table 1, three seeds — NEU-DET, 360 test images, at each arm's val-peak epoch

| arm | seed | val-peak epoch | test mAP@0.5 | recall | class-agnostic AP | val mAP@0.5 | test at e30 |
|---|---|---|---|---|---|---|---|
| **R34+FPN, VisDrone recipe** | 0 (09-17, the book's row) | 28 | 0.623 | 0.992 | 0.625 | 0.636 | 0.615 |
| | 1 | 28 | 0.636 | 0.990 | 0.642 | 0.653 | 0.636 |
| | 2 | 30 | 0.637 | 0.993 | 0.651 | 0.651 | 0.637 |
| | **mean ± sd** | | **0.632 ± 0.008** | 0.992 ± 0.001 | 0.640 ± 0.013 | 0.647 ± 0.009 | 0.630 ± 0.012 |
| R34+FPN, HSV+hflip off (affine kept) | 0 | 30 | 0.630 | 0.987 | 0.640 | 0.652 | 0.630 |
| | 1 | 30 | 0.659 | 0.994 | 0.671 | 0.667 | 0.659 |
| | 2 | 28 | 0.649 | 0.993 | 0.666 | 0.663 | 0.648 |
| | **mean ± sd** | | **0.646 ± 0.015** | 0.991 ± 0.004 | 0.659 ± 0.017 | 0.660 ± 0.008 | 0.646 ± 0.015 |
| R34+FPN, no ImageNet bootstrap | 0 | 26 | 0.555 | 0.986 | 0.555 | 0.581 | 0.559 |
| | 1 | 28 | 0.576 | 0.987 | 0.583 | 0.603 | 0.577 |
| | 2 | 28 | 0.554 | 0.987 | 0.564 | 0.577 | 0.548 |
| | **mean ± sd** | | **0.562 ± 0.012** | 0.986 ± 0.001 | 0.567 ± 0.014 | 0.587 ± 0.014 | 0.561 ± 0.015 |
| **R34 single grid 14×14**, bootstrap, 30 ep | 0 | 28 | 0.607 | 0.917 | 0.609 | 0.642 | 0.614 |
| | 1 | 28 | 0.628 | 0.932 | 0.632 | 0.648 | 0.617 |
| | 2 | 30 | 0.593 | 0.920 | 0.578 | 0.599 | 0.593 |
| | **mean ± sd** | | **0.609 ± 0.018** | 0.923 ± 0.008 | 0.606 ± 0.027 | 0.630 ± 0.027 | 0.608 ± 0.013 |

Paired on test at the val peak, by seed (mean):

| contrast | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| HSV+hflip off − recipe | +0.007 | +0.023 | +0.012 | **+0.014** |
| no bootstrap − recipe | −0.068 | −0.061 | −0.083 | **−0.071** |
| single grid − FPN recipe | −0.016 | −0.008 | −0.044 | **−0.023** |

Per-class test AP at the val peak, mean ± sd over the three seeds:

| arm | crazing | inclusion | patches | pitted surface | rolled-in scale | scratches |
|---|---|---|---|---|---|---|
| R34+FPN, recipe | 0.34 ± 0.05 | 0.67 ± 0.02 | 0.84 ± 0.01 | 0.73 ± 0.03 | 0.43 ± 0.02 | 0.79 ± 0.01 |
| R34+FPN, HSV+hflip off | 0.35 ± 0.04 | 0.71 ± 0.04 | 0.84 ± 0.01 | 0.74 ± 0.02 | 0.43 ± 0.04 | 0.80 ± 0.04 |
| R34+FPN, no bootstrap | 0.33 ± 0.01 | 0.60 ± 0.01 | 0.79 ± 0.01 | 0.57 ± 0.04 | 0.40 ± 0.03 | 0.69 ± 0.04 |
| R34 single grid | 0.33 ± 0.03 | 0.63 ± 0.03 | 0.81 ± 0.01 | 0.70 ± 0.03 | 0.48 ± 0.05 | 0.70 ± 0.03 |

Val sweeps, mAP@0.5 (`table_val.txt`; bold = the arm's peak at that seed):

| arm | seed | e10 | e14 | e18 | e22 | e26 | e28 | e30 |
|---|---|---|---|---|---|---|---|---|
| FPN, recipe | 0 | 0.350 | 0.499 | 0.526 | 0.618 | 0.625 | **0.636** | 0.631 |
| | 1 | 0.411 | 0.545 | 0.464 | 0.588 | 0.645 | **0.653** | 0.649 |
| | 2 | 0.423 | 0.571 | 0.591 | 0.594 | 0.643 | 0.650 | **0.651** |
| FPN, HSV+hflip off | 0 | 0.381 | 0.184 | 0.475 | 0.626 | 0.642 | 0.638 | **0.652** |
| | 1 | 0.436 | 0.531 | 0.618 | 0.645 | 0.662 | 0.665 | **0.667** |
| | 2 | 0.432 | 0.521 | 0.534 | 0.635 | 0.655 | **0.663** | 0.660 |
| FPN, no bootstrap | 0 | 0.143 | 0.313 | 0.414 | 0.524 | **0.581** | 0.577 | 0.573 |
| | 1 | 0.126 | 0.365 | 0.493 | 0.536 | 0.601 | **0.603** | 0.601 |
| | 2 | 0.235 | 0.369 | 0.417 | 0.484 | 0.554 | **0.577** | 0.576 |
| single grid | 0 | 0.198 | 0.392 | 0.491 | 0.589 | 0.621 | **0.642** | 0.636 |
| | 1 | 0.301 | 0.441 | 0.539 | 0.589 | 0.624 | **0.648** | 0.638 |
| | 2 | 0.120 | 0.330 | 0.472 | 0.550 | 0.597 | 0.585 | **0.599** |

## What the three seeds say

- **The book's row is the low seed.** The recipe arm is 0.632 ± 0.008 on test; seed 0's 0.623
  is the lowest of the three, and recall is 0.99 on every seed and arm of the FPN. The seed sd
  of the FPN arms (0.008–0.015) is below the 09-17 README's ~0.02 floor; the grid's (0.018) is
  at it.
- **The bootstrap is worth 0.071 ± 0.011.** −0.068 / −0.061 / −0.083 without it, on test at
  the val peak, with the no-bootstrap arm's val curve 0.04–0.09 lower at every epoch from e26.
  The gap is five times either arm's sd; the 09-17 "+0.07 at 1,080 images, holding to epoch 30"
  is a measurement now. Per class the prefix buys most on pitted surface (−0.16 without it),
  scratches (−0.10), inclusion (−0.07) and patches (−0.05); crazing and rolled-in scale, the
  two it cannot help, lose 0.01–0.03.
- **"Photometric augmentation is within noise" becomes "off is ahead on every seed."** HSV+hflip
  off is +0.007 / +0.023 / +0.012 on test and +0.016 / +0.014 / +0.011 on val — the same sign six
  times, mean +0.014, about one recipe-arm sd. NEU's crops are grey 200 × 200 JPEGs; hue and
  saturation jitter on grey content is input noise, and the box-aware affine (kept in both arms)
  is the augmentation doing the work. Small, consistent, and not the table's point: the row the
  book bolds is the VisDrone recipe unchanged, and a NEU-tuned recipe would start by switching
  the HSV jitter off.
- **The single grid is 0.023 behind the FPN, not level with it.** −0.016 / −0.008 / −0.044, with
  seed 2 the outlier on an arm whose val curve at that seed was still rising at e30 (its best is
  the last epoch, 0.599 against 0.642–0.648 at the other seeds). The recall gap is the same on
  every seed, 0.92 against 0.99. The reading survives in its contrast form — on steel the single
  grid is within 0.01–0.04 of the multi-scale head, on drones it is six times behind
  (0.039 against 0.236, Table 2 of the 09-17 README) — and the word "matches" does not: "within
  0.02 ± 0.02" is the sentence.
- **Crazing and rolled-in scale stay the worst two on every arm and seed** (0.33–0.35 and
  0.40–0.48, against 0.60–0.84 for the other four), and crazing carries the largest seed spread
  (sd 0.05 on the recipe arm) — the diffuse-texture classes are where the ranking is least
  stable.
- **The mid-schedule dips recur at different epochs at different seeds**: recipe e12 at seed 0
  and e18 at seed 1 (0.464 between 0.545 and 0.588), no-bootstrap e16 at seed 2, grid e20 at
  seed 2, each recovered within two epochs; the late rows are the ones to read, as the 09-17
  README said. The val peak is e28 or e30 on eleven of the twelve runs (no-bootstrap seed 0's
  e26 is the exception), and test at e30 is within 0.001 of test at the peak on seven of twelve
  and within 0.011 on all of them.
- For the book's Table 1: the four rows become 0.632 ± 0.008 / 0.646 ± 0.015 / 0.562 ± 0.012
  / 0.609 ± 0.018 (three seeds, test at the val-peak epoch), or keep the seed-0 numbers with the
  spread in the caption; the three readings become "+0.07 ± 0.01", "within 0.02", and "off is
  +0.014 on every seed".
