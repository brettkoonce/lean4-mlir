# yolo_demo_v2.md — YOLOv1 on VOC 2007, MLIR-first, smallest viable

Successor to `yolo_demo.md` (v1, untouched). Same architecture and
dataset choice; this revision tightens the scope to **smallest
viable**, makes the codegen-extension work explicit, and pins
decisions that v1 left open.

## What's different from v1

| | v1 | v2 |
|---|---|---|
| Scope | MVP + polish, 5-7 weeks total | Smallest viable only, ~3-4 weeks |
| Backbone | R34 or EnetB0 (choice) | **R34** (pinned) |
| Pipeline | Implied phase-3 IREE | **Phase-3 IREE, explicit** |
| Codegen extension | "Already in scope" | **Explicit Phase 1: 1 week of new codegen** |
| Responsibility assignment | "Forward-only computation" | **Option A: always pred 0 — explicit, ~-2-4% mAP cost** |
| Dataset | VOC 2007+2012 (~16K imgs) | **VOC 2007 only (~5K imgs)** |
| Box-aware aug | Random crop / flip / scale / hue (~2 weeks) | **Flip + crop only, ~2 days** |
| mAP eval | "Borrow logic" (~1 week) | **Borrow + verify on a tiny subset, ~2 days** |
| Phase commit pattern | Not specified | **One commit per phase, see below** |

The v1 estimate (5-7 weeks) was honest but front-loaded the loss
work and back-loaded the codegen extension. v2 explicitly orders
codegen-first since that's the load-bearing blocker.

## Scope (smallest viable)

Goal: a runnable end-to-end detection demo on VOC 2007 with the
phase-3 MLIR/IREE pipeline. Target: ~45-50% mAP@0.5 (vs paper's
63% at 448×448 with full Darknet backbone + 2012 data).

- **Backbone**: ResNet-34, ImageNet-pretrained, freeze the first
  two stages for first run, unfreeze later if mAP plateaus.
- **Head**: strip classifier, add `dense → reshape(7, 7, 30)`.
  No FPN, no multi-scale.
- **Input**: 224×224 (not paper's 448). Smaller is faster and our
  existing ResNet-34 already accepts this size; the accuracy hit is
  real but acceptable for a demo.
- **Loss**: full 5-term MSE with paper weights (λ_coord=5,
  λ_noobj=0.5), masked.
- **Responsibility**: **always predictor 0** (Option A, see below).
- **Data**: Pascal VOC 2007 trainval only (~5K images). Skip 2012.
- **Augmentation**: random horizontal flip + random crop, both
  bbox-aware. No HSV / color / scale jitter.
- **Inference**: greedy NMS per class at IoU=0.5.
- **Eval**: mAP@0.5 on VOC 2007 test (~5K images).

What we're explicitly NOT doing in v2 (deferred to v1's polish
phase or out of scope):
- VOC 2012 data (would lift mAP +3-5%).
- 448×448 input (would lift mAP +5-10%).
- Full color augmentation, scale jitter, random affine.
- mAP at multiple IoU thresholds.
- Multi-scale evaluation.
- Multi-backbone ablation (EnetB0, ConvNeXt-T variants).
- YOLOv2/v3 (anchors, multi-scale heads).

## Architecture

```lean
-- NetSpec: ResNet-34 minus the classification head, plus a YOLOv1 head.
def r34Yolov1 : NetSpec where
  name := "ResNet-34 + YOLOv1 head"
  imageH := 224
  imageW := 224
  layers := [
    -- ResNet-34 body (identical to resnet34Imagenet through .globalAvgPool drop):
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    -- YOLOv1 head: 7×7 spatial feature map -> 1470 outputs -> reshape:
    -- (paper had 2 FC layers; we use one for simplicity)
    .dense 25088 1470 .identity     -- 7*7*512 -> 7*7*30 flat
    -- Reshape to (B, 7, 7, 30) happens host-side / in loss codegen
  ]
```

ResNet-34 with 224×224 input outputs `(B, 512, 7, 7)` after stage 4
(no GAP for detection — keep the spatial grid). Flatten to
`(B, 7*7*512 = 25088)`, single dense to `(B, 1470)`, host-side
reshape to `(B, 7, 7, 30)`.

Total params: ~21M (backbone) + ~37M (head). Big head; could
factor into two FCs to shrink, paper-faithful is 4096-then-1470.
For v2 just one FC layer is enough — minimal.

## Loss + responsibility

Per cell `(i, j)` and per predictor `b ∈ {0, 1}`, with masks
`mask_obj[i, j] ∈ {0, 1}` (1 if any GT box's center is in this
cell) and `mask_resp[i, j, b] ∈ {0, 1}` (1 if predictor `b` is
"responsible"):

```
loss
  = λ_coord * Σ mask_obj * mask_resp * [(x - x̂)² + (y - ŷ)² + (√w - √ŵ)² + (√h - √ĥ)²]
  +          Σ mask_obj * mask_resp * (C - 1)²                                     -- objectness positive
  + λ_noobj * Σ (1 - mask_obj * mask_resp) * Ĉ²                                    -- objectness negative
  +          Σ mask_obj * Σ_classes (p_c - p̂_c)²                                   -- per-cell classification
```

λ_coord = 5, λ_noobj = 0.5. `(x̂, ŷ, ŵ, ĥ, Ĉ)` is the model
output; `(x, y, w, h, C=1, one_hot_class)` is the target.

### Responsibility assignment: Option A — always predictor 0

In strict YOLOv1, `mask_resp[i, j, b] = 1` iff `b` is the predictor
whose IoU with the GT box is highest. This computation depends on
the model's **current** output and so would have to be recomputed
each training step.

**v2 uses Option A**: `mask_resp[i, j, 0] = 1, mask_resp[i, j, 1] = 0`
always. Predictor 0 is always responsible.

| | Option A (chosen) | Option B (host roundtrip) | Option C (in MLIR) |
|---|---|---|---|
| Codegen complexity | minimal | medium | hard |
| Per-step host I/O | none | yes (read predictions back) | none |
| mAP cost vs paper | -2-4% | 0 | 0 |
| Implementation time | 0 | ~1 week | ~2 weeks |

Option A keeps `mask_resp` host-precomputable (it's a static `[1, 0]`
per cell), which means the whole loss is just masked MSE over the
target tensor — no per-step host roundtrip, no in-MLIR conditional
selection. The mAP cost (-2-4%) is the price of simplicity for v2.

Upgrade path to Option B or C is one design iteration on top of v2,
not a rewrite.

## Phase 1: codegen extension (load-bearing)

`MlirCodegen.lean` today supports int32 `(B,)` class labels with
softmax-CE loss. YOLOv1 needs:

1. **Float-tensor target plumbing**: dispatcher/loader accepts
   `(B, 7, 7, 30)` float target instead of `(B,)` int32.
2. **Float-tensor mask plumbing**: `(B, 7, 7)` float mask alongside
   target.
3. **`.yolov1Loss` (or `.maskedMse`) variant** in the loss enum:
   emit masked-MSE MLIR with the 5-term breakdown above. λ_coord
   and λ_noobj as compile-time scalars.
4. **Backward derivation**: masked MSE's backward is `2 * mask *
   (pred - target)` — hand-derivable, no need to invoke autodiff
   machinery. Emit directly.

Estimate: **~1 week** of focused codegen work + a smoke test.

Smoke test: synthesize a random `(B, 7, 7, 30)` target + matching
mask, compile train-step MLIR, run one optimizer step, verify loss
decreases. No VOC data needed yet.

## Phase 2: VOC 2007 data pipeline

```
data/voc2007/
  train.bin      -- (B*224*224*3) float32 images
  train_targets.bin  -- (B, 7, 7, 30) float32 targets
  train_masks.bin    -- (B, 7, 7) float32 masks
  val.bin
  val_targets.bin
  val_masks.bin
```

Preprocessing (Python, one-time):
1. Download VOC 2007 (~870MB) — train/val 5011 images, test 4952.
2. Parse each `.xml` annotation to `[(class_id, xmin, ymin, xmax, ymax), ...]`.
3. For each image: load + resize to 224×224, compute target tensor:
   - For each GT box: find cell `(i, j)` containing the center.
   - Fill `target[i, j, 0..4] = (cx_cell, cy_cell, √w_rel, √h_rel, 1)`.
   - Fill `target[i, j, 10..30] = one-hot(class_id)`.
   - Set `mask[i, j] = 1`.
4. Save as `.bin` matching existing dataset format.

Estimate: ~2-3 days. Most of the work is the XML parser + target
encoder; the rest is byte-pack file writing.

## Phase 3: bbox-aware augmentation

Two augmentations, both bbox-tracking:
1. **Horizontal flip** (p=0.5): flip image, flip each box's
   x-coordinates around the image center.
2. **Random crop** (paper's `±20%`): pick a random crop of size
   ≥80% of original, clip boxes to the crop region, drop boxes
   that lose >50% of area.

Both happen on the host before targets are recomputed. ~2 days.

## Phase 4: first training run + iteration

- Initialize backbone from existing `resnet34` weights (transfer
  learn), random init for the head.
- Train 100 epochs at batch 32 on one 7900 XTX.
- Watch: loss components separately (coord, obj_pos, obj_neg,
  class), mAP@0.5 every 10 epochs.
- Expect: loss decreases, mAP rises to 30-45% over 50-100 epochs.
- Estimated wall-clock: ~6-10 hours per run depending on per-step
  cost.

If mAP plateaus at <30%: debug — most likely culprit is target
encoding (off-by-one on cell assignment), loss weighting, or the
"always pred 0" assumption being too constraining for this
backbone.

## Phase 5: NMS + mAP eval + viz

| Item | Where | Effort |
|---|---|---|
| Greedy NMS | Python (host-side, post-inference) | 1 day |
| mAP@0.5 | Python, borrow from torchvision or pycocotools VOC | 2 days |
| Inference viz: 4-image grid with boxes drawn | Python + PIL | 0.5 days |

mAP eval verification: run the borrowed mAP impl on a tiny subset
(10 GT images + 10 predicted images) with known answers, confirm
agreement. Catches off-by-one bugs in IoU / class indexing.

## Phase 6: bestiary write-up

One bestiary entry under "Object detection" (new subsection or
folded into "Vision classifiers" with a sub-heading). Format:
NetSpec listing + final mAP + sample inference grid. ~1 day.

## Sequencing summary (commit boundaries)

| Phase | Commit | Verifiable result | Estimate |
|---|---|---|---|
| 0 | this doc | planning v2 lands | — |
| 1 | `.yolov1Loss` codegen | smoke test: random target, loss decreases over 10 steps | 1 week |
| 2 | VOC data pipeline | bin files produced, smoke-loadable | 2-3 days |
| 3 | Bbox-aware aug | unit tests on synthetic boxes | 2 days |
| 4 | First training run | loss-vs-epoch + mAP-vs-epoch logs in `runs/` | 3-5 days (incl debug) |
| 5 | NMS + mAP eval + viz | `figures/yolo_voc.png` + final mAP number | 2-3 days |
| 6 | Bestiary entry | blueprint update | 1 day |

**Total: ~3-4 weeks** of plumbing work, plus the training run
(~1 week of iteration including debug). Six commits, each at a
clean stopping point.

## Open decisions for the user

- **Backbone freezing**: freeze first 2 stages of ResNet-34 for
  first run? (Conservative: yes. Aggressive: full-finetune.)
  My recommendation: **freeze first 2 stages for first 50 epochs,
  then unfreeze for last 50**. Costs ~30% per-step throughput when
  unfrozen.
- **Where mAP eval lives**: Python script in `scripts/voc_map.py`
  with `traces/voc_map_results.json` output? Or inline in the
  trainer's eval pass via FFI? My recommendation: **Python
  script, called post-checkpoint**.
- **Pretrained-from-where**: our existing `resnet34` Imagenette
  checkpoint, or pull torchvision's ImageNet checkpoint?
  My recommendation: **torchvision's ImageNet** — Imagenette
  pretraining is on 10 classes, torchvision's 1000-class model
  has richer features. Need a one-time weight-conversion script.

If the recommendations are fine, decisions are made. Override
where needed.
