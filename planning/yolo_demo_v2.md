# yolo_demo_v2.md — YOLOv1 on VOC 2007, MLIR-first, smallest viable

Successor to `yolo_demo.md` (v1, untouched). Same architecture and
dataset choice; this revision tightens the scope to **smallest
viable**, makes the codegen-extension work explicit, and pins
decisions that v1 left open.

## Status (2026-05-28, post `/plan-eng-review`)

**Scope reduced to Phase 1 only.** Phase 2-6 (real VOC pipeline,
training, mAP) moved to `planning/yolo_demo_v3.md`. This
document still describes the full v2 design as context, but the
committable scope is the **Phase 1 codegen extension + smoke
test** (Section "Phase 1: codegen extension" below).

11 architectural decisions captured during review — see "Phase 1
decisions (pinned)" section after Phase 1.

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

## Phase 1: codegen extension (load-bearing) — THIS IS THE COMMITTABLE SCOPE

`MlirCodegen.lean` today supports int32 `(B,)` class labels with
softmax-CE loss. YOLOv1 needs:

1. **Float-tensor target plumbing**: dispatcher/loader accepts
   `(B, 7, 7, 30)` float target instead of `(B,)` int32. The
   closest existing analogue is **`useDdpm`** (not `useSeg`):
   `trainStepAdamF32Ddpm` (IreeRuntime.lean:107) already plumbs a
   float `[B, C, H, W]` target through with per-pixel MSE. Phase
   1 reuses that pattern.
2. **Float-tensor mask plumbing**: separate `(B, 7, 7)` float
   mask ByteArray, alongside target (see D3 decision below).
3. **`useYolov1 : Bool` flag** in `MlirCodegen.generateTrainStep`:
   emit masked-MSE MLIR with the 5-term breakdown above. λ_coord
   and λ_noobj as compile-time scalars. (Loss enum refactor is
   v3 work — see D2 + R1 in `yolo_demo_v3.md`.)
4. **Backward derivation**: hand-derived per-term. For (x, y),
   confidence, and class terms, the derivative is the standard
   `2 * mask * (pred - target)`. For **(√w, √h) terms**, the
   derivative includes a `1/√ŵ` factor that has a singularity at
   ŵ = 0. The codegen applies the √ via `stablehlo.sqrt(max(ŵ,
   ε))` with `ε = 1e-6` to keep gradients finite (see D4).
5. **In-MLIR reshape**: model output is `[B, 1470]` flat; the
   loss block does `stablehlo.reshape` to `[B, 30, 7, 7]` (NCHW)
   before the elementwise masked MSE. Free at runtime; matches
   the existing `lmHead` reshape pattern at MlirCodegen.lean:4704.

Estimate: **~3-5 days** of focused codegen work + the smoke
tests (down from ~1 week in the pre-review estimate, given the
DDPM-template insight).

### Tests (Phase 1 deliverables)

All six tests are in scope (D10 decision). Spec: full ResNet-34
backbone + YOLO head, batch=1 (D11 decision — realistic spec
prioritized over fast CI).

| ID | File | What | Pattern |
|---|---|---|---|
| T1 | `tests/TestYolov1Emit.lean` | Compile-only: emit train-step MLIR with `useYolov1 := true`, run iree-compile, assert exit 0 | matches `tests/TestDdpmTrainEmit.lean` |
| T2 | `tests/TestYolov1TrainStep.lean` | Loss-decreases: synthesize random `[B, 7, 7, 30]` target + `[B, 7, 7]` mask, run 10 optimizer steps, assert `loss[9] < 0.5 * loss[0]` | new — first runtime test of its kind |
| T3 | (same file as T2) | Mask correctness: target with `mask = 0` everywhere ⇒ loss matches closed-form noobj-conf-only expectation | new |
| T4 | (same file as T2) | Per-term decomposition: 5 sub-tests, each setting target with only one of (coord, sqrt-coord, obj-conf, noobj-conf, class) active, verify only that term contributes non-zero loss | new — load-bearing for correctness |
| T5 | (same file as T2) | √ ε-floor stability: target with predicted ŵ → ε, verify no NaN gradient | new |
| T7 | `tests/TestYolov1Mutex.lean` | Mutex rejection: `useYolov1 + useSeg`, `useYolov1 + useFocal`, etc. each throw at `compileVmfbs` | mirrors existing useFocal mutex tests |

### Smoke-test responsibility-assignment

Always predictor 0 (Option A from "Loss + responsibility" section
above). For the smoke test this means `mask_resp[i, j, 0] = 1,
mask_resp[i, j, 1] = 0` precomputed once per batch and passed in
the mask ByteArray (combined: `mask_obj * mask_resp` since
they're always multiplied together in the loss).

## Phase 1 decisions (pinned by `/plan-eng-review` on 2026-05-28)

| ID | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Scope | **Phase 1 only** | Smallest viable cut; Phases 2-6 → `yolo_demo_v3.md` |
| D2 | Loss-path routing | **`useYolov1` bool, enum refactor as v3 R1** | Smallest diff; matches existing pattern (useFocal/Seg/Ddpm) |
| D3 | Mask plumbing | **Separate `ByteArray` arg** | Generalizes to other masked losses; cleaner FFI |
| D4 | √ stability | **ε floor on √(ŵ, ĥ)**, `ε = 1e-6` | Paper-faithful math without `1/√ŵ` NaN |
| D5 | Reshape site | **In-MLIR, inside loss block** | Free at runtime; matches `lmHead` pattern |
| D6 | ABI name | **`trainStepAdamF32Yolov1`** | Matches existing naming (Ddpm/Seg/Soft); defers abstraction debate |
| D7 | DRY for 5 MSE terms | **Inline 5 terms in Phase 1; factor in v3 (R2) when 3rd use site exists** | Rule of three; avoid premature abstraction |
| D8 | Mutex checks | **Inline ~5 throws in `compileVmfbs`; refactor in v3 (R3) bundled with R1** | Smallest diff |
| D9 | TODO sink | **`planning/yolo_demo_v3.md` stub** | Mirrors v1/v2 evolution; doubles as v3 design doc |
| D10 | Test scope | **T1 + T2 + T3 + T4 + T5 + T7 all in scope** | "Loss decreases" alone is too weak — per-term decomp catches real bugs |
| D11 | Smoke-test spec | **Full R34 + YOLO head, batch=1** | Realistic spec over fast CI; catches scale-related issues |
| D12 | Outside voice | **Skipped** | Phase-1-only scope is small enough; subagent dispatch deferred |

## What already exists (reuse map for Phase 1)

| Reuse target | Source | Phase 1 usage |
|---|---|---|
| Float `[B, ...]` target plumbing | `trainStepAdamF32Ddpm` (IreeRuntime.lean:107) | Template for new `trainStepAdamF32Yolov1` |
| Per-pixel MSE emit | MlirCodegen.lean:4847-4869 (DDPM block) | Template for masked-MSE emit |
| Flat → spatial reshape pattern | `lmHead` `[B, T*V] → [B, V, T, 1]` (MlirCodegen.lean:4704) | Template for `[B, 1470] → [B, 30, 7, 7]` |
| Compile-only smoke test pattern | `tests/TestDdpmTrainEmit.lean` | Template for T1 |
| Mutex check pattern | `Train.lean:100-109` | Template for T7 + the new throws |
| ResNet-34 spec | `MainResnetTrain.lean`, `MainResnet34Imagenet.lean` | Backbone of the smoke-test spec |
| BN-free head ordering | Existing dense layers | YOLO head is dense-only, no BN |

## NOT in scope (Phase 1 — all deferred to v3)

| Deferred | Reason | Lands in |
|---|---|---|
| Pascal VOC data pipeline (Phase 2) | Codegen-first cut | v3 Phase 2 |
| Bbox-aware augmentation (Phase 3) | No data yet | v3 Phase 3 |
| First real training run (Phase 4) | No data, no aug | v3 Phase 4 |
| NMS + mAP eval + viz (Phase 5) | No inference target | v3 Phase 5 |
| Bestiary entry (Phase 6) | Nothing to show yet | v3 Phase 6 |
| `DatasetKind.pascalVoc` addition | No loader yet | v3 Phase 2 |
| `Loss` enum refactor (R1) | Defer for v3 | v3 pre-Phase 2 |
| `emitMaskedMseTerm` helper (R2) | Rule of three not yet triggered | v3 if/when DETR lands |
| `validateLossSelection` helper (R3) | Bundles with R1 | v3 pre-Phase 2 |
| IREE fusion check on 5-term backward | Smoke test doesn't expose this | v3 perf observation |

## Failure modes (Phase 1)

| Codepath | Realistic failure | Caught by? |
|---|---|---|
| 5-term masked-MSE forward | wrong slice indices, off-by-one on (x, y, w, h, conf, class) channel layout | T4 (per-term decomposition) |
| √ ε floor | ε too small ⇒ NaN; ε too large ⇒ gradient bias | T5 (ε-floor stability) |
| Mask broadcast `[B,7,7]` × `[B,30,7,7]` | wrong axis broadcast, silent wrong-loss | T3 (mask = 0 closed-form) |
| Backward 5-term sum | one term's gradient missing, model still trains on the others | T4 catches partial failures |
| Mutex check at compileVmfbs | new flag combo not rejected ⇒ codegen runs both paths | T7 (mutex rejection) |
| `useYolov1` flag forgotten in `generateTrainStep` recursion | silent fallback to classCE | T1 catches (different MLIR output length) |
| `trainStepAdamF32Yolov1` C-side glue | wrong number of bound args ⇒ IREE runtime error | T2 (loss-decreases requires real execution) |

**Critical gaps?** None. Every failure mode has a test, error
handling exists (mutex throws + IREE runtime errors are visible),
and the user-facing surface in Phase 1 is "iree-compile fails" or
"loss-doesn't-decrease test fails" — both loud.

## Worktree parallelization

Phase 1 work is largely sequential — codegen → opaque FFI → C
glue → tests all depend on the prior step. Two lanes possible:

- Lane A: codegen + FFI + C glue + T1 + T2 (sequential, all
  touch `LeanMlir/` + `ffi/`)
- Lane B: T7 mutex-rejection test (independent file, just needs
  the `useYolov1` flag to exist in the codegen call signature)

Lane B can launch as soon as Lane A's codegen-signature change
is in. Not a strong parallelization win for Phase 1; mostly
sequential.

---

## Phase 2-6 reference (MOVED to yolo_demo_v3.md)

The sections below describe the full v2 design and remain useful
as context — but the committable scope is Phase 1 only (above).
v3 inherits these phases verbatim.

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

## Open decisions for the user (RESOLVED for Phase 1, deferred for v3)

Phase 1 has no remaining open decisions — see "Phase 1 decisions
(pinned)" section above for the 11 decisions captured during
review.

Phase 2-6 open decisions (deferred to v3, see `yolo_demo_v3.md`):

- **Backbone freezing schedule** (v3 Phase 4): recommendation
  remains **freeze first 2 stages for first 50 epochs, then
  unfreeze for last 50**.
- **mAP eval location** (v3 Phase 5): recommendation remains
  **Python script `scripts/voc_map.py`, called post-checkpoint**.
- **Pretrained source** (v3 Phase 4): v3 starts with existing
  local Imagenette weights (zero-effort), escalates to
  torchvision ImageNet if mAP plateaus too low.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 10 issues, 0 critical gaps, scope reduced Phase 1-only |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **UNRESOLVED:** 0
- **VERDICT:** ENG CLEARED — ready to implement Phase 1

