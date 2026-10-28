# 2026-10-27 — VisDrone, the other four R34+FPN rows of tab:visdrone at three seeds (planning/part2_ship_audit.md §7.4)

`runs/2026-10-27-visdrone-aff30-seeds/` gave the table's top row (R34+FPN, 30 epochs, scale
augmentation) its seeds: 0.232 ± 0.004. The four rows under it — the archive's ladder in
`planning/archive/visdrone_detector.md` ("The full ladder") — were still one run each from
08-28/08-29, and the book printed three of them under the historical decode (argmax, the
scorer's default top-k 1000) and one under the canonical multilabel decode without a column
saying so. The user's call on the evening of 10-27 ("burn a day or two of compute on this area")
put seeds 1 and 2 of all four rows on the box behind the §2 chains (`run.sh`, one row at a time
per card: ~4.1 min an epoch without affine beside the other chains, 3.4 h a 50-epoch row), every
checkpoint scored under **both** decodes, the 50-epoch rows also at e10/e20/e30/e40. Then the
seed-0 checkpoints, which are all still in `.lake/build`, were rescored the same way
(`rescore_s0.sh`, `ladder_s0.sh`; every historical argmax number reproduced to the fourth
decimal), and the 12-epoch focal row's e6/e8/e10 checkpoints were scored for all three seeds
(`score_ckpt.sh`, `ckpt_chain.sh`) once its e12 numbers came in 0.06 apart.

| row | recipe (`FPN_BACKBONE=r34 FPN_TOWER=0` + …) | the book's number, its decode |
|---|---|---|
| `cfoc2` | `FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_EPOCHS=12` | 0.1961, multilabel |
| `aug50` | `FPN_AUG=1 FPN_EPOCHS=50` (sqrt class weights, no focal) | 0.1674, argmax |
| `ctrl12` | `FPN_EPOCHS=12` | 0.1526, argmax |
| `long50` | `FPN_EPOCHS=50` | 0.1243, argmax |

```
LEAN_MLIR_SEED=$S FPN_TAG=${row}_s$S <recipe> LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G \
  .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
FPN_TAG=<tag> … infer data/visdrone_fpn infer_<tag>                      # the e<N> tags by the three-file copy
python3 scripts/demos/yolo_map_visdrone.py infer_<tag>/logits.bin data/visdrone448/val.bin \
  --fpn data/visdrone --grid 14 --multilabel --topk 3000 --ml-k 3 --ml-floor 0.05   # score_<tag>.txt
python3 scripts/demos/yolo_map_visdrone.py … --fpn data/visdrone --grid 14            # score_<tag>_argmax.txt
python3 summary.py                                                                    # summary.md, the tables below
```

`queue.log` has every launch, exit and score line with its time; the logits dumps (406 MB each)
were deleted after scoring; checkpoints stay on ares. Seed 2 of `aug50` and `ctrl12` ran last,
on card 3 after the NEU-DET sweeps (landed 07:58 UTC 10-28, the grind's last run; the 50-epoch
row took 3.2 h with the other three cards idle), and closed the table.

## The rows, final checkpoint, three seeds (VisDrone-DET val, 548 images, 38,759 uncapped GT boxes)

| row | seed | multilabel mAP@0.5 | recall | class-agnostic AP | argmax mAP@0.5 | recall | class-agnostic AP |
|---|---|---|---|---|---|---|---|
| `cfoc2` | 0 (the book's) | **0.1961** | 0.749 | 0.441 | **0.1762** | 0.708 | 0.441 |
| | 1 | 0.1350 | 0.748 | 0.323 | 0.1182 | 0.704 | 0.324 |
| | 2 | 0.1516 | 0.749 | 0.355 | 0.1326 | 0.706 | 0.355 |
| | **mean ± sd** | **0.161 ± 0.032** | 0.749 ± 0.001 | 0.373 ± 0.061 | **0.142 ± 0.030** | 0.706 ± 0.002 | 0.373 ± 0.061 |
| `aug50` | 0 (the book's) | 0.1718 | 0.739 | 0.431 | **0.1674** | 0.704 | 0.429 |
| | 1 | 0.1668 | 0.739 | 0.436 | 0.1624 | 0.705 | 0.434 |
| | 2 | 0.1708 | 0.738 | 0.438 | 0.1666 | 0.704 | 0.436 |
| | **mean ± sd** | **0.170 ± 0.003** | 0.739 ± 0.001 | 0.435 ± 0.004 | **0.166 ± 0.003** | 0.704 ± 0.001 | 0.433 ± 0.004 |
| `ctrl12` | 0 (the book's) | 0.1588 | 0.725 | 0.395 | **0.1526** | 0.682 | 0.393 |
| | 1 | 0.1527 | 0.726 | 0.393 | 0.1460 | 0.683 | 0.390 |
| | 2 | 0.1606 | 0.726 | 0.399 | 0.1538 | 0.682 | 0.396 |
| | **mean ± sd** | **0.157 ± 0.004** | 0.726 ± 0.001 | 0.396 ± 0.003 | **0.151 ± 0.004** | 0.682 ± 0.001 | 0.393 ± 0.003 |
| `long50` | 0 (the book's) | 0.1287 | 0.710 | 0.370 | **0.1243** | 0.669 | 0.369 |
| | 1 | 0.1297 | 0.710 | 0.374 | 0.1254 | 0.670 | 0.372 |
| | 2 | 0.1276 | 0.708 | 0.371 | 0.1232 | 0.666 | 0.369 |
| | **mean ± sd** | **0.1287 ± 0.0011** | 0.709 ± 0.002 | 0.372 ± 0.002 | **0.1243 ± 0.0011** | 0.668 ± 0.002 | 0.370 ± 0.002 |

The top row, for the same table: `aff30` 0.2362 / 0.2313 / 0.2289 = **0.232 ± 0.004** at e30,
multilabel (`…-aff30-seeds/README.md`); 0.2363 at e28 is the book's.

## Along the schedule, multilabel mAP@0.5 (argmax in brackets)

| row | seed | e10 | e20 | e30 | e40 | e50 |
|---|---|---|---|---|---|---|
| `aug50` | 0 | 0.1684 (0.1592) | 0.1766 (0.1703) | **0.1777** (0.1731, the archive's "best seen") | 0.1738 (0.1694) | 0.1718 (0.1674) |
| | 1 | **0.1788** (0.1690) | 0.1628 (0.1563) | 0.1707 (0.1655) | 0.1681 (0.1637) | 0.1668 (0.1624) |
| | 2 | 0.1703 (0.1614) | **0.1806** (0.1738) | 0.1797 (0.1748) | 0.1743 (0.1700) | 0.1708 (0.1666) |
| `long50` | 0 | 0.1277 (0.1223) | **0.1355** (0.1313) | 0.1327 (0.1283) | 0.1312 (0.1268) | 0.1287 (0.1243) |
| | 1 | **0.1450** (0.1383) | 0.1362 (0.1319) | 0.1358 (0.1315) | 0.1307 (0.1265) | 0.1297 (0.1254) |
| | 2 | **0.1385** (0.1319) | 0.1354 (0.1311) | 0.1346 (0.1304) | 0.1289 (0.1247) | 0.1276 (0.1232) |

The 12-epoch focal row, multilabel mAP@0.5 (class-agnostic AP / recall); the cosine schedule has
lr 3.0e-4 at e6, 1.65e-4 at e8, 4.7e-5 at e10, 0 at e12:

| seed | e6 | e8 | e10 | e12 | train loss at e12 |
|---|---|---|---|---|---|
| 0 | 0.1366 (0.381 / 0.719) | 0.1386 (0.347 / 0.733) | 0.1944 (0.439 / 0.746) | **0.1961** (0.441 / 0.749) | (08-29, no log kept) |
| 1 | 0.1344 (0.379 / 0.714) | 0.1399 (0.309 / 0.736) | 0.1298 (0.311 / 0.744) | **0.1350** (0.323 / 0.748) | 131.0 |
| 2 | 0.1264 (0.334 / 0.709) | 0.1438 (0.350 / 0.741) | 0.1687 (0.404 / 0.748) | **0.1516** (0.355 / 0.749) | 130.4 |

## Per class, final checkpoint, multilabel

`cfoc2` by seed, beside the top row's e28 (three seeds agree to 0.002 on these classes):

| class (val GT) | `cfoc2` s0 (the book's base) | s1 | s2 | `aff30` e28, 3-seed mean |
|---|---|---|---|---|
| car (14,064) | 0.643 | 0.531 | 0.593 | 0.663 |
| pedestrian (8,844) | 0.175 | 0.116 | 0.119 | 0.216 |
| people (5,125) | 0.193 | 0.156 | 0.174 | 0.221 |
| motor (4,886) | 0.211 | 0.164 | 0.190 | 0.264 |
| van (1,975) | 0.215 | 0.173 | 0.160 | 0.271 |
| bicycle (1,287) | 0.025 | 0.023 | 0.026 | 0.036 |
| tricycle (1,045) | 0.103 | 0.049 | 0.108 | 0.143 |
| truck (750) | 0.133 | 0.061 | 0.073 | 0.156 |
| awning-tri (532) | 0.039 | 0.019 | 0.043 | 0.063 |
| bus (251) | 0.225 | 0.058 | 0.031 | 0.260 |

The other rows, mean ± sd over three seeds:

| row | pedestrian | people | bicycle | car | van | truck | tricycle | awning-tri | bus | motor |
|---|---|---|---|---|---|---|---|---|---|---|
| `aug50` | 0.157 ± 0.005 | 0.140 ± 0.004 | 0.015 ± 0.001 | 0.610 ± 0.002 | 0.179 ± 0.006 | 0.101 ± 0.010 | 0.070 ± 0.003 | 0.029 ± 0.003 | 0.223 ± 0.027 | 0.174 ± 0.001 |
| `ctrl12` | 0.137 ± 0.005 | 0.138 ± 0.005 | 0.017 ± 0.001 | 0.580 ± 0.002 | 0.181 ± 0.007 | 0.097 ± 0.006 | 0.081 ± 0.004 | 0.028 ± 0.002 | 0.153 ± 0.019 | 0.161 ± 0.007 |
| `long50` | 0.108 ± 0.001 | 0.097 ± 0.005 | 0.009 ± 0.001 | 0.540 ± 0.003 | 0.136 ± 0.009 | 0.069 ± 0.008 | 0.044 ± 0.002 | 0.017 ± 0.001 | 0.138 ± 0.014 | 0.127 ± 0.003 |

## What it says

- **Three of the four rows are settled to ±0.005; the fourth has a seed sd of 0.032.** `long50`
  is 0.1287 ± 0.0011, `ctrl12` 0.157 ± 0.004 and `aug50` 0.170 ± 0.003 on three seeds, every
  seed within 0.006 of seed 0; recall agrees to 0.002 on every row and seed. `cfoc2` — the 12-epoch row with photometric
  augmentation, class weights off and the class term under focal γ = 2 — is 0.1961 / 0.1350 /
  0.1516, and the book's 0.1961 is the best of three by 0.045. The recipe is the same function on
  all three: the seed-0 training graph (emitted 08-29 01:46, two minutes before the commit that
  added the class-focal term, 8afd515e, the only commit ever to touch it) differs from today's
  only in comment lines, the identical change the `aff30` graph shows, and `aff30` reproduced
  on seeds 1 and 2 to 0.007.
- **Where the 0.06 comes from: the cosine tail.** At e8, with the learning rate at 1.65e-4, the
  three seeds are 0.139 / 0.140 / 0.144 — level with `ctrl12`, which has no augmentation and no
  focal. The last four epochs, lr 4.7e-5 → 0, move seed 0 by +0.058, seed 2 by +0.025 then
  −0.017, seed 1 by −0.005. Recall is 0.744–0.749 on every seed from e10 and the training loss
  ends at 131.0 / 130.4: what settles in the tail is the ranking — the class-agnostic AP goes
  0.347 → 0.441 at seed 0 and stays at 0.31–0.32 at seed 1 — and nothing the trainer prints
  sees it (`fpn eval skipped — … train loss above is the signal` is wrong for this row). The
  same loss at 30 epochs with scale augmentation (`aff30`) has sd 0.004, so the instability is
  the short schedule's tail, not the focal term as such. Which classes land is also the seed's
  choice: bus is 0.225 / 0.058 / 0.031, truck 0.133 / 0.061 / 0.073, tricycle 0.103 / 0.049 / 0.108.
- **The decode column.** Multilabel adds +0.004 to +0.007 on the three rows without focal, on
  every seed (means: `ctrl12` 0.151 → 0.157, `long50` 0.1243 → 0.1287, `aug50` 0.166 → 0.170) and +0.017 to
  +0.020 on `cfoc2` (0.1762 → 0.1961, 0.1182 → 0.1350, 0.1326 → 0.1516) — the top-3-per-cell
  decode is worth most to the arm whose class probabilities the focal term flattened. The table
  can print the multilabel column for all five rows now; the ordering is the same under either.
- **What the book's "augmentation and schedule" paragraph keeps and loses.** Keeps: the long
  schedule without augmentation costs — `long50` ends at 0.129 on three seeds, 18% under
  `ctrl12`'s 0.157, and *no checkpoint of the 50-epoch run reaches the 12-epoch run's number*
  (its best is 0.136 / 0.145 / 0.139 at e20 / e10 / e10, under `ctrl12`'s lowest seed 0.153, and
  every seed falls 0.007–0.015 from there to e50); photometric augmentation recovers it (`aug50`
  0.170 at 50 epochs); scale augmentation with the schedule it needs is a different tier
  (0.232). Loses: "at that strength 12 epochs still beats 50" rested on 0.1961 against 0.1674 —
  at three seeds the 12-epoch focal row is 0.161 ± 0.032 against `aug50`'s 0.170 ± 0.003, i.e.
  not above it. The ladders give the paragraph's claim its honest form: without augmentation
  the long schedule is worse than the short one at every checkpoint and declines after
  e10–e20; with photometric augmentation it is flat from e10 to e50 (0.163–0.181 across
  checkpoints and seeds, the peak at e30 / e10 / e20 on seeds 0 / 1 / 2 — the checkpoint noise is
  the size of any trend, and the archive's "best seen" 0.1731 at e30 was that noise); with
  scale augmentation it climbs to e28–30. **The pack sets
  whether the long schedule pays**, which is what "one decision, not two" meant, and it no
  longer needs the focal row to say it.
- **The per-class lift of scale augmentation holds in direction on every seed and its
  percentages depend on the base.** The book's +59% / +41% / +40% (awning-tri, bicycle,
  tricycle) against car's +7% are `aff30` e28 over `cfoc2`, both at seed 0. With three seeds on
  both sides they are +87% / +45% / +65% against car's +13%; over `cfoc2` seed 2 alone +47% /
  +38% / +32% against +12%. The rare classes gain more than car on every base, bicycle is the
  stable one (+38% to +56%), and the sentence should carry ratios rounded to the half ("the rare classes
  gain 1.4–1.9×, car 1.1×") or use the table's row means rather than one seed of a row whose sd
  is a fifth of its value.
- For the table: five rows, one decode (multilabel), three seeds each — `aff30` 0.232 ± 0.004,
  `aug50` 0.170 ± 0.003, `cfoc2` 0.161 ± 0.032, `ctrl12` 0.157 ± 0.004, `long50` 0.1287 ± 0.0011
  — and the second row's bold moves to its sd. By the mean the 50-epoch photometric row now
  sits above the 12-epoch focal row, and the book's row order follows the means.
