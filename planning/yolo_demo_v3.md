# yolo_demo_v3.md — v3 follow-up: real VOC, real training, real mAP

Successor stub to `yolo_demo_v2.md`. v2 ships Phase 1 only:
the `useYolov1` codegen extension + smoke test on synthetic
targets. v3 picks up at Phase 2 (real VOC data) and lands the
remaining work for an end-to-end detection demo.

This file is a **stub**: real scoping happens when v2 lands and we
have ground truth on what Phase 1 cost vs. estimate.

## Inheritance from v2

v2 Phase 1 establishes:

- `useYolov1 : Bool` flag in `MlirCodegen.generateTrainStep`.
- `trainStepAdamF32Yolov1` opaque FFI entrypoint with separate
  `target` + `mask` `ByteArray` args.
- 5-term masked-MSE forward + hand-derived backward, with an
  ε-floor on the √(ŵ, ĥ) terms to avoid the `1/√ŵ` singularity.
- In-MLIR reshape `[B, 1470] → [B, 30, 7, 7]` inside the loss
  block.
- "Always predictor 0" responsibility assignment (Option A from v2).
- ResNet-34 + YOLO head spec compiled and exercised via the smoke
  test with batch=1.

v3 inherits all of those decisions. The codegen does not change
shape; v3 is data, training, and evaluation work on top of it.

## What v3 owns (Phases 2-6 from v2 + accumulated refactors)

### Phase 2 — VOC 2007 data pipeline

Carry over verbatim from `yolo_demo_v2.md` Phase 2 section:

- VOC 2007 download + XML parse + 224×224 resize.
- Target-tensor encoding `target[i, j, ...] = (cx_cell, cy_cell,
  √w_rel, √h_rel, 1, one_hot_class)` written to `.bin` files
  matching the existing dataset format.
- Per-image mask `[7, 7]` written alongside.
- `DatasetKind.pascalVoc` added to `LeanMlir/Types.lean`.
- New `pascalVocIO : DatasetIO` in `LeanMlir/Train.lean`. The
  current `DatasetIO` signature assumes a single label buffer; v3
  must extend it to a **(label, mask)** pair OR split the loader
  to return a tuple including the mask. The smaller diff is
  probably "label buffer carries `target ++ mask` concatenated,
  Train.lean splits at call site." Pin this when starting Phase 2.

Estimate: 2-3 days (data pipeline) + 1 day (`DatasetIO` extension).

### Phase 3 — Bbox-aware augmentation

Carry over from v2 Phase 3:

- Horizontal flip (image + box x-coords).
- Random crop (image + clip boxes to crop, drop boxes losing > 50%
  area).
- Targets recomputed after augmentation per step.

This requires either:
- New C kernels `F32.randomCropBoxes`, `F32.randomHFlipBoxes` that
  return `(image, target, mask)` — mirroring `F32.randomCrop` /
  `F32.randomHFlip` for Imagenette but bbox-aware. ~2 days C work.
- OR pre-compute K augmented versions of each image during VOC
  preprocessing and sample at training time. Avoids C kernels but
  loses true per-step augmentation. ~1 day.

Pick at Phase 3 start.

### Phase 4 — First training run

Carry over from v2 Phase 4:

- Initialize backbone from existing `resnet34` weights. Open
  decision in v2: torchvision ImageNet vs. local Imagenette. The
  Imagenette checkpoint is on disk now; torchvision requires a
  weight-conversion script (~half-day to a day given conv/BN
  parameter-naming differences). Recommendation: try Imagenette
  first since it's zero-effort, fall back to torchvision if mAP
  plateaus too low.
- Freeze schedule: v2 recommended freeze first 2 stages for 50
  epochs, then unfreeze. Implementable in v3 via a simple
  `paramFreezeMask` ByteArray applied in the SGD/Adam update
  step. Estimate: 1 day.
- Training: 100 epochs at batch 32 on one 7900 XTX. Expect ~6-10
  hours wall-clock per full run.
- Target: mAP@0.5 in the 40-50% range (v2's honest tradeoff
  estimate).

### Phase 5 — NMS + mAP eval + visualization

Carry over from v2 Phase 5. All Python, all post-inference.

- Greedy NMS per class at IoU=0.5.
- Borrow mAP@0.5 from torchvision or pycocotools. Verify on a
  tiny subset with known answers before trusting the full run.
- 4-image inference visualization grid (PIL).

Open decision from v2: mAP eval lives in `scripts/voc_map.py` and
runs against a saved checkpoint, not inline in the trainer's eval
loop. (User recommendation, decided.)

### Phase 6 — Bestiary entry

Carry over from v2 Phase 6. ~1 day.

## Architectural refactors deferred from v2 review

The Phase-1-only review identified three refactors that v2 punts
to v3 to keep Phase 1 small. v3 should land them before or
alongside Phase 2:

### Refactor R1 — `LossKind` enum

Replace the boolean matrix
(`useFocal × useSoftLabels × useSeg × useDdpm × useYolov1`) in
`compileVmfbs` and `generateTrainStep` with a single
`LossKind` parameter. Sketch:

```lean
inductive LossKind where
  | classCE              -- int32 [B] label, softmax CE
  | softLabelCE          -- float [B, NC] soft labels (mixup/cutmix)
  | focal                -- modifier on classCE
  | perPixelCE           -- int32 [B, H, W] label (segmentation)
  | floatTargetMse       -- float [B, C, H, W] (DDPM, etc.)
  | yolov1Masked         -- float [B, 30, 7, 7] target + [B, 7, 7] mask
deriving Repr, BEq
```

Replaces the ~10-line mutex check at `Train.lean:100-109` with a
single match. Touches every existing trainer's `compileVmfbs`
call site (mechanical rename). ~1 day.

### Refactor R2 — `emitMaskedMseTerm` helper

Once YOLOv1's 5 terms + DDPM's per-pixel term + (eventually)
DETR's box-regression term all exist, factor the common code into
one `private def emitMaskedMseTerm (predSlice targetSlice mask :
String) (weight : Float) (applySqrt : Bool) : String × String`.

Rule of three: only refactor when the third use site exists. DETR
may or may not land in v3; if it doesn't, leave R2 alone.
Estimate when triggered: ~2 hours.

### Refactor R3 — `validateLossSelection` helper

Replace the inline pairwise checks at `Train.lean:100-109`+
with a single helper that collects active loss flags and rejects
combinations. Falls out for free when R1 lands (just a single
match on `LossKind`). Don't do R3 standalone — bundle with R1.

## Open performance question (carried from v2 Section 4 review)

- Does IREE fuse the 5 masked-MSE term emits into one elementwise
  pipeline, or emit 5 separate kernels? Run `iree-opt` on the
  emitted MLIR after Phase 1 lands and check. If unfused, the
  Phase-1 codegen is doing ~5× the backward work it needs to. Not
  blocking; affects v3's training-run wall-clock.

## Open decisions deferred from v2

- **Pretrained weights source**: Imagenette local vs.
  torchvision ImageNet. v2's recommendation was torchvision; v3
  starts with Imagenette for zero-cost, escalates if mAP is too
  low. Re-decide at Phase 4 start.
- **Backbone freezing schedule**: v2 recommended freeze 50 epochs
  + unfreeze 50. Confirm at Phase 4 start.
- **mAP eval location**: Python script in `scripts/voc_map.py`,
  decided in v2.

## Sequencing (rough)

| Phase | Estimate | Gate |
|---|---|---|
| R1 (LossKind enum) | 1 day | mechanical refactor, no behavior change |
| Phase 2 (VOC pipeline) | 3-4 days | `.bin` files smoke-loadable |
| Phase 3 (bbox aug) | 2 days | unit tests on synthetic boxes |
| Phase 4 (training run) | 3-5 days incl debug | mAP rising on `runs/` logs |
| Phase 5 (NMS + mAP + viz) | 2-3 days | `figures/yolo_voc.png` + mAP number |
| Phase 6 (bestiary entry) | 1 day | blueprint update |

Total: ~3-4 weeks. Same as v2's Phase 2-6 estimate, plus 1 day
for R1.

## Out of scope (still)

Inherits v2's deferral list:

- VOC 2012 data.
- 448×448 input resolution.
- Color augmentation, scale jitter, random affine.
- Multi-scale evaluation.
- mAP at multiple IoU thresholds.
- Multi-backbone ablation (EnetB0, ConvNeXt-T).
- YOLOv2/v3 (anchor boxes, multi-scale heads).
- COCO.

If any of these end up wanted, they're v4 work.
