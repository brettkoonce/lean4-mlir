# neu_det_fpn_demo.md — the VisDrone detector on NEU-DET steel defects

Goal: an industrial-inspection entry for the bestiary. Run the R34+FPN detector
that scores 0.2363 mAP on VisDrone, unchanged, on the NEU-DET hot-rolled-steel
defect set, and run the single-grid YOLOv1 arm that collapsed to 0.0000 on
VisDrone beside it. NEU is VisDrone's opposite regime — one or two large defects
per 200-px crop instead of seventy 20-px cars per frame — so the pair of tables
shows *when* multi-scale detection matters, which neither dataset can show
alone. Written 2026-09-17; the synthetic-data tool that shares this dataset is
deferred (§9).

Prerequisite reading: `planning/archive/visdrone_detector.md` (the ladder from
7×7 collapse to FPN, the traps), `demos/README.md` (the FPN detector's current
recipe and its three silent flags), `planning/gw_detection_demo.md` (the
bracketed-table shape these demos use).

## 0. The one-paragraph version

NEU-DET is 1,800 grayscale 200×200 crops of hot-rolled steel strip, 300 per
class across six defect types — crazing, inclusion, patches, pitted surface,
rolled-in scale, scratches — with Pascal-VOC bounding boxes (Song & Yan 2013,
Appl. Surf. Sci. 285; boxes from He et al. 2020, IEEE TIM 69). The published
detectors sit at mAP@0.5 0.75–0.79 for stock Faster R-CNN / YOLOv5s / YOLO11s
and ~0.85 for the improved variants, on a 1,080/360/360 split. The demo is the
VisDrone R34+FPN trainer with a NEU preprocessor and NEU anchor priors, the
ImageNet-trained R34 bootstrapped exactly as on VisDrone and BraTS, and the
archived single-grid 14×14 arm run on the same records. Table 1 is the two arms
on NEU beside the published rows; Table 2 is the same two arms on VisDrone,
already measured. The claim the section makes is the second table's shape: the
grid head is a wash on steel and a zero on drones, and the FPN pays for itself
only where objects are small.

## 1. Why this and not another detection demo

- The detector is reused verbatim — backbone, neck, heads, loss, scorer,
  bootstrap. The one Lean file that changes is a sibling with different
  constants (§3). Nothing in the codegen moves for phase 1.
- It is the dataset the industrial-inspection literature actually reports on
  (with MVTec AD, which is CC BY-NC-SA and has no boxes). NEU's license is
  "free for research" on the NEU page, not a stated CC; say so in the data
  chapter the way the BraTS paragraph does.
- The lesson is a comparison the book can only make with two datasets in the
  same binary: the single-grid arm is not wrong, it is wrong *for drones*.
- The data is a minute's download; the runs are under an hour each on one
  card; the figure is grayscale steel with coloured boxes — clean.

## 2. The data

`scripts/datasets/download_neu.sh` → `data/neu_det/` with `IMAGES/*.jpg` (200×200, 8-bit
grayscale) and `ANNOTATIONS/*.xml` (VOC: one `<object>` per box, class name in
`<name>`). Mirrors: the NEU page (faculty.neu.edu.cn/me/songkc), IEEE DataPort,
Kaggle. ⚠ Verify the count is 1,800 and the six class names match exactly
(`crazing inclusion patches pitted_surface rolled-in_scale scratches`);
mirrors differ in file extension and in whether the names carry underscores.

No official split. Use 1,080/360/360 by image with a fixed seed, which is the
split the Faster R-CNN and YOLO rows above used, so the comparison is loose but
not meaningless; report which split every row is on.

`scripts/datasets/preprocess_neu_det.py data/neu_det data/neu_det_fpn --size 448 --fpn data/neu_det`
writes the FPN record format `scripts/datasets/preprocess_visdrone.py` writes (image + flat
`[P3|P4|P5]` target of `Σ_s A_s·15·g_s²` floats), and `--grid 14` the
single-grid format, so `F32.loadDetBinFpn` and the archived single-grid loader
read them unchanged. The 200-px grayscale crop is replicated to three channels
and upsampled to 448 (2.24×; the NEU-DET YOLO papers train at 640), so the R34
stem is still a byte prefix of `.lake/build/jax_r34_imagenet.bin` and the
bootstrap self-check passes as-is. Also write `val.full_gt.bin` as the VisDrone
preprocessor does, since the scorer reads it for uncapped GT.

Box statistics to print first, because they decide §3 and §4: the (w, h)
distribution in source pixels and in 448-px input, and the fraction that
`fpn_scale_of` (`scripts/datasets/preprocess_visdrone.py:159`, thresholds 24 / 64 px on
max(w, h) at input) assigns to P3 / P4 / P5. Expectation: almost everything on
P5, inclusions and thin scratches on P4, nothing on P3 — which is the regime
claim in one line.

## 3. What is and is not "just a dataloader"

Three constants in the detector are VisDrone's, and one of them is in the
codegen:

1. **Anchor priors** — `demos/MainYolov1VisdroneFpn.lean:31-41`, k-means over
   VisDrone boxes, the largest P5 prior 0.18 × 0.15 of the image side. A NEU
   defect often spans half the crop. With VisDrone's priors every NEU box is
   assigned to P5 against anchors several times too small and the box loss
   starts from a residual `exp` cannot reach. Run the k-means on NEU boxes:
   `scripts/probes/visdrone_anchors.py` imports `parse_visdrone_txt` for its box
   filter, so give it a `--parser` hook or a sibling `scripts/probes/neu_anchors.py`
   that reads the VOC XML; it already prints recall@0.5 of the priors, which
   is the coverage ceiling to record. Three per scale as on VisDrone.
2. **Class count** — `NUM_CLASSES_A = 10` in `scripts/datasets/preprocess_visdrone.py:101`
   sets the per-anchor width 5 + 10 = 15, and ⛔ that 15 is baked into the
   `fpnDetect` codegen (`Types.lean:263`, head `oc → A·15`,
   `Ntot = A·15·Σg²`; `emitFpnDetectForward/Backward`), not a spec field.
   Phase 1 therefore keeps ten classes and maps NEU's six to ids 0–5, leaving
   four logits that never see a positive. Cost: nothing — the scorer averages
   AP over classes present in the GT (`yolo_map_visdrone.py:504`, "mean over
   {len(valid)} classes"), and four dead softmax slots are four biases that
   learn to be −∞. Parameterising the 15 as `5 + nClasses` is a later cleanup
   that touches an FD-verified emitter and must re-run
   `scripts/probes/fpn_loss_probe_check.py` and the `fpn-detect` FD gate.
3. **Class weights and names** — `fpnClsWeights` (sqrt-inverse VisDrone
   frequencies, `MainYolov1VisdroneFpn.lean:52`) and `CLASS_NAMES` in
   `scripts/demos/yolo_map_visdrone.py:59`. NEU is balanced (300 per class, boxes
   roughly so): `FPN_CLSW=none`. The scorer takes a `--classes` file or a
   sibling name table.

The Lean side is a sibling file, `demos/MainYolov1NeuDetFpn.lean`, with NEU's
anchor table, `name := "... (NEU-DET)"` so its build prefix and checkpoints
never collide with the VisDrone arms, and everything else imported from the
VisDrone file or copied line-for-line — the way every VisDrone rung was added.
Reading the anchors from `data/<dir>/anchors_fpn_*.txt` at startup so one
binary serves both datasets is the cleanup after both tables exist, not
before: the VisDrone binary's checkpoints are live references.

The single-grid arm, `demos/archive/MainYolov1VisDrone448.lean`
(`lake exe yolov1-visdrone448`, 448 / 14×14, the same R34 and the deep conv
head), comes out of the archive as `MainYolov1NeuDet448.lean` with the same
two edits (name, no anchors to change). ⚠ Its record format is the
`--grid 14` single-grid one, a different file from the FPN records.

## 4. The arms

Table 1, NEU-DET, 1,080/360/360, mAP@0.5 / recall / class-agnostic AP as the
VisDrone table prints them:

| arm | what it tests |
|---|---|
| R34+FPN, NEU anchors, bootstrap, `FPN_AUG=1 FPN_AFFINE=50`, 30 ep | the VisDrone recipe, unchanged |
| R34+FPN, no bootstrap | what ImageNet features buy at 1,080 images (BraTS said "one epoch" at 14k slices; here it should be more) |
| R34 single grid 14×14, bootstrap, same epochs | the arm that scored 0.0000 on VisDrone |
| Faster R-CNN / YOLOv5s / YOLO11s, published | the bracket: 0.768 / ~0.767 / 0.747 |
| improved YOLOv10 / YOLOv5, published | the ceiling rows: ~0.85 |

Table 2 is the same two arms on VisDrone, all four numbers already in
`runs/` and the book (0.2363 FPN, 0.0000 single grid), re-printed beside NEU.
That table is the section.

Recipe knobs stay VisDrone's except `FPN_CLSW=none`; `FPN_AFFINE` is worth
keeping (scale/translate is exactly the augmentation steel crops want) and
`FPN_AUG=1` photometric is a question — NEU's classes are partly defined by
intensity texture (rolled-in scale vs pitted surface), so run it both ways once.
⚠ The three silent flags from `demos/README.md` apply verbatim: `FPN_TAG` on
`infer` too, `FPN_BACKBONE` defaults to `r50`, `--topk` defaults to 1000.

## 5. The instrument

`scripts/demos/yolo_map_visdrone.py logits.bin data/neu_det448/val.bin --fpn
data/neu_det --grid 14 --classes neu` — per-class AP@0.5, mAP over present
classes, recall, class-agnostic AP, the same code and protocol as VisDrone so
the two tables are one scorer. Add the argmax-vs-multilabel note: on VisDrone
`--multilabel` moved mAP by 0.004 through the rare classes; on balanced NEU it
should move nothing, and printing both is a free check that the scorer's
rare-class machinery is not what the NEU numbers are made of.

Per-class AP is the second thing to read. Crazing and rolled-in scale are
diffuse textures whose "box" is most of the crop — every NEU paper's worst
two classes — and the single-grid head's argument is strongest there.

## 6. Figure and section

Figure: four NEU val crops, one per hard class, with GT and the FPN's boxes,
`scripts/demos/fpn_render.py --layout cols --n 4` as for VisDrone. Beside it, or as a
second row, the same crops under the single-grid arm.

Section: *Industrial inspection — demo: steel-surface defects on NEU-DET*,
under the bestiary beside the VisDrone detection demo, two tables and the
figure, ~1 page. The data-chapter row: NEU-DET, Song & Yan 2013 / He et al.
2020, "free for research" (no stated CC), `scripts/datasets/download_neu.sh` → `data/neu_det/`.

## 7. Phases

```
Phase 0 (½ session, CPU):   download, preprocess, box statistics, anchors
                            Gate 0: 1,800 files, six names, P3/P4/P5 fractions
                                    printed; NEU anchors' recall@0.5 ≥ 0.9
                                    (VisDrone's was 0.76 with A=6)
Phase 1 (1 session, GPU):   sibling FPN trainer, bootstrap arm, 30 ep
                            Gate 1: mAP@0.5 ≥ 0.60 (a wrong anchor table or a
                                    class-id off-by-one reads as < 0.3)
Phase 2 (½ session, GPU):   single-grid arm out of the archive, same epochs
                            no-bootstrap FPN arm on the other card
Phase 3 (½ session):        tables, figure, section, data-chapter row
Cleanup (optional):         anchors read from the data dir; `5 + nClasses` in
                            the emitter with the FD gate re-run
```

## 8. Gates that fail loudly

- Gate 0 is the geometry: if the P3 fraction is not ~0, the upsampling or the
  thresholds are wrong, and the whole "opposite regime" claim is void.
- Gate 1 is the plumbing: class ids in the XML parser, the anchor table's
  order (P3 → P4 → P5, matching the codegen concat), and the bootstrap
  self-check. Each fails as a low number, not a crash; the per-class AP
  printout tells them apart (one class at 0 = id map; all classes low =
  anchors; training loss flat from step 1 = bootstrap or record stride).
- The single-grid arm scoring *above* the FPN is not a failure; it is a
  possible result and belongs in the table as-is.

## 9. Out of scope

- The synthetic-rare-defect tool (class-conditional DDPM on NEU-CLS at 10–300
  real images per class, CFG, the classifier retrained on real + generated,
  with ImageNet-R34 fine-tuning as the honest baseline column). Same dataset,
  its own plan; deferred 2026-09-17 while the shape is decided.
- Leave-one-class-out novelty detection on NEU (train on five defect types,
  hold one out, image AUROC): the anomaly-detection demo's NEU form. Its own
  plan if wanted; it shares this preprocessor.
- MVTec AD: CC BY-NC-SA, no boxes; cited in the section's prose as the
  benchmark, not shipped.
- Any detector head change. If the FPN arm underperforms the published rows
  by more than ~0.1, the gap is the recipe (640-px input, mosaic, longer
  schedules), not the architecture, and the section says so rather than
  chasing it.

## 10. Notes before starting

- ⛔ `lake run <job>` launches; this demo is `lake exe`, but the FPN trainer's
  default is a real 30-epoch run — pass `FPN_EPOCHS=1` for the smoke.
- The R34 bootstrap file is `.lake/build/jax_r34_imagenet.bin`; if it is
  missing the trainer must refuse, not He-init silently — check the existing
  self-check fires on a wrong path.
- Two arms on two cards; the box crashes under long multi-card loads, so keep
  each run under an hour and checkpoint every few epochs (`FPN_CKPT_EVERY`).

## 11. Log

### Phase 0 — 2026-09-17, DONE

- Data: the maintainer's own Drive copy (`NEU-DET.zip`, 26 MB) via
  `scripts/datasets/download_neu.sh`; 1,800 / 1,800, six names exactly as §2 expected, all
  200×200, 4,189 boxes. The JPEGs are 3-channel files with grey content.
- Split: 1,080 / 360 / 360 **stratified** (180 / 60 / 60 per class), seed 0,
  `preprocess_neu_det.split_stems`; `scripts/probes/neu_anchors.py` imports it so the
  priors see exactly the train images. Records: `data/neu_det_fpn/` (FPN,
  1.34 MB/record) and `data/neu_det448/` (single grid + `*.full_gt.bin`), each
  with train/val/test.
- **Gate 0 geometry: P3 0.0% / P4 1.9% / P5 98.1%** of 2,539 train boxes at
  448 px (0 / 47 / 2,492); 53% of boxes span more than half the frame; 2.35
  boxes per image; slot coverage 99.9% on both the FPN grids and the single
  14×14 grid (VisDrone: 88% / 61%). The regime claim holds in one line, and
  the P4 residue is inclusions (6%) and patches (1%).
- **Gate 0 anchors: NOT met as written, and the gate was wrong.** P4 recall@0.5
  0.98, but P5 — 98% of the boxes, every aspect from 64 px to the full frame —
  fits three k-means anchors at mean wh-IoU 0.52 / recall@0.5 0.52 (k=6 → 0.84,
  k=9 → 0.94). The ≥ 0.9 threshold was calibrated on VisDrone's homogeneous
  tiny cars. It measures the prior's fit, not a ceiling: the head regresses
  `exp(t)` off the anchor, the k=3 log-residuals top out at |2.0| against the
  training cap of 8, and the ceiling that does bind — a unique (level, cell,
  anchor) slot per GT — is 99.9%. **Decision: keep A=3** (the spec's `fpnDetect
  … 3` is the detector verbatim; A=9 on P5 is a possible ablation if Gate 1
  fails on box quality specifically, i.e. low class-agnostic AP with the
  per-class argmax fine). P3 has no boxes at all, so its table is a fixed
  fallback spanning its band.
- Anchors live as constants in `demos/MainYolov1NeuDetFpn.lean` (the source of
  truth, as VisDrone's do); `data/neu_det/anchors_fpn_*.txt` are regenerated by
  the download script and read by the scorer's `--fpn`.
- The `A·15` class width: as §3 planned, six classes in ids 0–5, four dead
  slots; `scripts/demos/yolo_map_visdrone.py --classes neu` scores the six.

### Phase 1 — 2026-09-17, in flight

- `demos/MainYolov1NeuDetFpn.lean` (`lake exe yolov1-neudet-fpn`): the VisDrone
  spec layer-for-layer, NEU anchors, class weights off, and the 0.2363 recipe's
  env settings folded in as DEFAULTS (r34, aug, affine 50, class focal 2, 30 ep)
  so the bare command is the measured arm. `FPN_EVAL_SPLIT=test` for the table.
- Smoke (1 epoch, tag `smoke`): loss 79 → 26, 40 s/epoch on one 4060 Ti,
  bootstrap line present (the same BN-stats WARN as the VisDrone log), infer +
  scorer end-to-end (mAP 0.0085 after one epoch, class-agnostic recall 0.60).
  ⚠ A missing bootstrap file refuses via `IO.FS.readBinFile`
  (`SpecHelpers.patchInitWithPretrainedPrefix`), not a silent He init — §10 checked.
- The single-grid arm (`demos/MainYolov1NeuDet448.lean`, `lake exe
  yolov1-neudet448`) hit `train_step_adam_yolov1 is not implemented on the XLA
  backend` — the YOLOv1 masked-loss step was an unported stub in
  `ffi/pjrt_ffi.c` (the VisDrone 0.0000 was measured under IREE). Ported: the
  DDPM protocol plus the mask input, ~60 lines; shim rebuilt by rename so the
  running FPN jobs keep their mapped copy.
- ⛔ **Table 2's single-grid row has no valid measurement.** The "single 7×7
  grid, mAP 0.0000" the plan quotes is `planning/archive/yolo_fpn.md`'s opening
  line, written 2026-07-17 — before the 2026-07-22 shuffle-pairing fix, and
  that doc's own banner says every number in it "describes the bug, not the
  detector". The 448/14 rung (`demos/archive/MainYolov1VisDrone448.lean`) was
  never scored on fixed data either. With the yolov1 step now on XLA the
  archived arm builds and runs as-is (`lake exe yolov1-visdrone448
  data/visdrone448`, records regenerated 2026-08-28), so the VisDrone
  single-grid row is RE-MEASURED today on the same lowerer and scorer as the
  NEU rows rather than quoted from the archive. Whatever it scores — "a zero on
  drones" is the plan's expectation, not yet a number.

### Phase 2 — 2026-09-17, val sweeps

- **Single grid 14×14 on NEU (val, 360 imgs, VisDrone scoring protocol):**
  climbs monotonically to **mAP@0.5 0.642 at e28** (0.636 at e30), recall 0.92,
  class-agnostic AP 0.63; per class at e28: crazing 0.35, inclusion 0.69,
  patches 0.81, pitted 0.78, rolled-in 0.47, scratches 0.76 — crazing and
  rolled-in scale worst, as §5 predicted. Still rising at e26–28 under a
  30-epoch cosine, so the arm is schedule-limited, not capacity-limited.
  `mAP_ml == mAP_argmax` here by construction (the single-grid decode has no
  multilabel path). Table: `runs/2026-09-17-neudet-grid-run1-sweep/table_val.txt`.
- **VisDrone single grid 14×14, RE-MEASURED on fixed data** (`yolov1-visdrone448`,
  archived recipe, 12 ep, XLA, `runs/2026-09-17-visdrone-grid448-remeasure/`):
  **mAP@0.5 0.0391, recall 0.184, class-agnostic AP 0.107** on the uncapped
  38,759-box val GT. Not the 0.0000 the plan quoted — that was scrambled-data
  July — but 6× below the FPN's 0.2363 with recall 0.18 against 0.77: car 0.20,
  everything else ≤ 0.07, people/bicycle/awning-tricycle at zero. The single
  grid on drones is a collapse in every class but the one big enough to see,
  which is the claim in its honest form. Table 2's grid row is this number.

### Phases 2–3 — 2026-09-17, DONE

NEU-DET test at the val-peak epoch: **FPN recipe 0.623** (e28; recall 0.992,
ca-AP 0.625), photometric off 0.630 (e30), no bootstrap 0.555 (e26), **single
grid 0.607** (e28; recall 0.917). VisDrone: FPN 0.2363 / grid **0.0391**
(re-measured). Gate 1 (≥ 0.60) passed by both heads. The section's claim in
its measured form: **grid ≈ FPN on steel (0.61 vs 0.62), grid = FPN/6 on
drones (0.04 vs 0.24)**. Tables, figure and workings:
`runs/2026-09-17-neudet-fpn-run1/README.md`; book section inserted before the
BraTS demo, data-chapter row + paragraph added; `demos/README.md` section
added. Published bracket sources: Zhou, Wang & Wang, Sci. Rep. 15:44492 (2025);
Maity & Ghosh, arXiv:2510.21811 (2025); Hu, Ma & Xu, J. Mater. Inform. 5(4)
(2025).

Open, not chased (§9): the schedule (both arms still rising at e30; 4,050
steps), an affine-off arm, A=9 on P5, the `5 + nClasses` emitter cleanup, an
IREE step-tie for the ported yolov1 train step.
