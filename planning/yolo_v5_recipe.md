# yolo_v5_recipe.md — v5's training recipe on the R34+FPN body

**Written 2026-09-02**, at the end of the session that took the detector to
**mAP@0.5 0.2363** (`planning/visdrone_detector.md` §13a-bis). That is the number
to beat; everything here is measured against it.

**This is self-contained.** A session that has read nothing else can execute it.
Companions: `visdrone_detector.md` (the detector's single source of truth) and
`runs/2026-09-01-visdrone-affine/README.md` (the run that motivated this).

---

## 0. The thesis, and why it is not "port YOLOv5"

⭐⭐ **What separates YOLO releases splits in two, and only one half is worth
buying here.**

| half | examples | cost in THIS stack |
|---|---|---|
| backbone / neck architecture | CSPDarknet, C3k2, C2PSA | ⛔ high — `cspBlock`, `darknetBlock` and `fpnModule` exist in `Spec.lean` and `Bestiary/YOLO.lean` but have **zero occurrences in `MlirCodegen.lean`**. Nothing emits them. Each needs a forward AND a hand-derived backward. |
| training recipe | assignment, augmentation, loss/readout | ⭐ low — assignment and augmentation are **host-side**, so they cost *no new VJPs at all* |

▶ So: **take v5's recipe, keep the R34+FPN body.** The backbone is also the part
`visdrone_detector.md` §13 ranked last ("do not swap on a recipe still being
tuned"), and swapping it would forfeit the ImageNet-pretrained R34 bootstrap.

⚠ **Why v5 and not v8/v11.** v8's anchor-free head is reachable, but what makes
v8 good is **TAL — dynamic assignment from the current predictions**, and that
cannot be precomputed. `encode_targets_fpn` bakes targets into
`data/visdrone_fpn/*.bin` at preprocess time, and `generateTrainStep` emits
forward+loss+backward+Adam as ONE fused module with the target as an *input*
(`trainStepAdamF32`, one call). Dynamic assignment needs either in-graph
assignment with a stop-gradient region, or splitting that fused step — a runtime
project, not a head swap. Shipping v8's shape with v3-era assignment buys the
new-op cost and none of the benefit. ⭐ **v5's assignment is STATIC** — a pure
function of GT geometry — so it drops into the existing design with no runtime work.

---

## 1. The target: the 88.2% ceiling

`encode_targets_fpn` (`preprocess_visdrone.py:167`) assigns **one slot per GT**:
`fpn_scale_of` picks the level by `max(w,h)·448` against 24/64 px, `best_anchor`
picks ONE anchor by max wh-IoU, the centre cell is `floor(c·g)`. On a collision,
**later GT overwrites earlier** (its own docstring says so).

| quantity | today |
|---|---|
| encodability | **88.2%** |
| collisions | last-write-wins |
| val GT | 38,759 boxes over 548 images, **70.7/img** |

⛔ **11.8% of the ground truth is discarded before training starts**, and at 70
boxes an image the collisions compound it. No recipe tuning reaches that ceiling.
v5's assignment attacks exactly this, and it is the reason to do this work.

⭐⭐ **T1's success is measurable in SECONDS, with no training.** Re-run
`scripts/visdrone_fpn_coverage.py` after the encoder change: encodability should
go 88.2% → high 90s and collisions → near zero. **Do not train until that number
moves** — if it does not, the rest of the plan is pointless and costs nothing yet.

---

## 2. The changes, in dependency order

### T1 — v5 assignment (**no new VJPs**) ⭐ do this first

Two rules replace `best_anchor`'s argmax:

1. **Ratio match, not best-IoU.** For each GT against each anchor at its level,
   `r = gt_wh / anchor_wh`; accept if `max(r, 1/r).max() < 4.0` (v5's `anchor_t`).
   A GT may match **several anchors, or none** — where today it always matches
   exactly one, however badly.
2. **Neighbour cells.** Assign the centre cell plus the **two nearest** by the
   fractional part of the centre (frac < 0.5 → also the left/upper neighbour,
   > 0.5 → right/lower). Up to 3 cells × n matched anchors per GT.

▶ The on-disk format does **not** change: `PER_ANCHOR = 15`, targets stay
`[A·15, g, g]`. More slots are filled and the mask carries more ones. That is the
whole reason this is cheap.

### T2 — v5 box parameterization (**small VJPs**) ⚠ REQUIRED BY T1

⚠⚠ **T1 does not work without T2, and this is the easiest thing in the plan to
get wrong.** Today `cx = (j + σ(tx))/g` (`decode_anchor`, `yolo_map_visdrone.py:122`),
and σ ∈ (0,1) means **a cell can only predict a centre inside itself**. A
neighbour cell physically cannot reach the object it has just been assigned. So:

| | today | v5 |
|---|---|---|
| xy | `σ(t)` ∈ (0,1) | `2σ(t) − 0.5` ∈ (−0.5, 1.5) |
| wh | `a · exp(t)` — unbounded | `a · (2σ(t))²` ∈ (0, 4a) |

The wh change is a bonus, not a cost: `exp` is unbounded and is the classic source
of early-training box blowups. Both derivatives are scalar reworks of the existing
sigmoid path (`d/dt [2σ−0.5] = 2σ'`; `d/dt [a(2σ)²] = 8a·σσ'`), so this is a small,
well-contained backward change — but it is a backward change, and it needs its FD
gate (§4).

### T3 — mosaic (**no new VJPs**)

`preprocess_pets_mosaic.py:49` already has `def mosaic(pets_dir, names4, rng)`.
Port it to the FPN encoder: 4 images into one canvas, boxes transformed and
re-encoded by the T1 rules.

▶ `visdrone_detector.md` §13 deferred mosaic "on the grounds that 4-into-1 halves
already-tiny objects", pending evidence that scale aug helps. **`aff30` is that
evidence** (+20.5%, every class up). The objection is still real — VisDrone objects
are 2–5 px — so keep `fpnAffineWhThrPx`-style filtering and A/B it, do not assume it.

### T4 — per-class BCE instead of softmax CE (**small VJPs**) ⭐ evidence-backed

v3 got this right in 2018 and v5 kept it: **independent per-class logistic outputs**,
not a softmax. ⭐ This repo already has the evidence — multilabel decode is worth
**+7.6%** (§13b), i.e. reading the top-3 classes per cell instead of the argmax. That
is what you would expect if a softmax head is the wrong readout and the decode was
compensating in post. T4 tests it at the source. ▶ `bce` already exists as a loss
type in `Types.lean`, so the pattern is there to copy.

### T5 — IoU-aware objectness target (**small VJPs, in-graph**)

v5's objectness target is `IoU(pred_box, gt_box)` rather than a constant 1.0.
⚠ This is *dynamic in value but not in assignment* — the slot is already chosen by
T1, and both boxes are available inside the loss, so it computes **in-graph** and
needs no runtime change. ⛔ **The IoU target must be stop-gradient**, or gradient
flows into the box head through the objectness term and the two losses fight.

Also v5's per-level objectness balance `[4.0, 1.0, 0.4]` for P3/P4/P5 — scalar
multipliers, free.

---

## 3. Every file this touches

| file | change | new VJP? |
|---|---|---|
| `preprocess_visdrone.py` | `encode_targets_fpn`, `best_anchor` → ratio+neighbour (T1); mosaic (T3) | no |
| **`ffi/f32_helpers.c`** | ⚠⚠ `lean_f32_fpn_affine` **re-encodes targets from decoded boxes using the anchor rules** (`fpn_encode_boxes`). It must port in lockstep with T1 **or the +20.5% affine win silently breaks.** | no |
| `LeanMlir/MlirCodegen.lean` | `emitAnchorYoloLoss`: xy/wh reparam (T2), BCE class (T4), IoU obj target + level balance (T5) — **and their backwards** | **yes** |
| `scripts/yolo_map_visdrone.py` | `decode_anchor` box param (T2), class readout (T4) | no |
| `deploy/orin_detect.py`, `deploy/export_onnx.py` | the deployed decode and the PyTorch replica, both must match T2/T4 | no |
| `data/visdrone_fpn/` | **regenerate** (8.7 GB train) | no |

⚠ `.fpnDetect (oc c3 c4 c5 g5 A tower)` already takes `A` as a parameter, so the
head shape itself needs no change — anchors stay, this is not an anchor-free port.

---

## 4. Gates — run these, in this order

1. ⭐ **`scripts/visdrone_fpn_coverage.py`** — encodability and collisions after T1.
   Seconds, no training. **Gate the whole plan on this.**
2. **`scripts/fpn_loss_probe_check.py`** — finite-difference the new loss. Add an arm
   per change (T2, T4, T5) rather than one combined arm: ⚠ its own header warns that
   errors "would cancel in the product if only the combined arm were checked."
   ⚠ Needs IREE, which is **not** in the repo `.venv` and must never be — use a
   throwaway venv with a matched `iree-base-compiler`/`iree-base-runtime` pair; the
   recipe is in that script's header.
3. **`scripts/check_fpn_affine.py`** — after the `f32_helpers.c` change. ⭐ It compiles
   the real C and references the real encoder; it does not test a copy. Keep it that
   way: `export_onnx.py` shipped a different model for a day because its "validation"
   compared against a re-derivation that had drifted.
4. **`deploy/export_onnx.py --verify-frame`** + `bespoke/diff_lean.py --lean-logits` —
   the replica, after the decode changes. ⛔ Without `pad="lean"` the twin is a
   different function.

---

## 5. Traps, all paid for already

- ⚠⚠ **Scoring needs `--topk 3000 --ml-k 3 --ml-floor 0.05`.** `--topk` defaults to
  1000 and truncates the multilabel candidates; the same checkpoint reads 0.1919
  instead of 0.1961. §13b.
- ⚠⚠ **A schedule finding is a property of the augmentation PACK.** 12 epochs won
  for HSV+hflip; 30 beats 12 by 44% with scale jitter. Caught this project twice.
  **Every arm here needs its own epoch curve** — do not assume 30 transfers.
- ⛔ **Do not compare at 12 epochs.** No affine arm is near converged there; the
  p ladder came out non-monotonic and measured nothing (§13a-ter).
- ⚠ **`FPN_BACKBONE` defaults to `r50`, not `r34`.** Omitting it silently trains a
  different arm.
- ⚠ **`FPN_TAG` is not optional on `infer`** — an untagged eval silently scores a
  different arm, and the only tell is identical rows across an epoch sweep.
- ⚠ Scoring an epoch checkpoint needs **three** files copied to a new tag prefix:
  `_params_eN.bin`, `_bn_stats_eN.bin` **and `_fwd_eval.mlir`**.
- ⛔ **No seed knob exists.** `Train.lean` has none (`LEAN_MLIR_SEED` is
  `VerifiedTrain.lean`, a different path); init and data order are fixed and
  augmentation is seeded `epoch*10000 + bi`. Every number is n=1 and two arms that
  differ in any augmentation parameter also differ in realization. ▶ **Add the seed
  env var before T3/T4/T5**, whose expected effects are small enough to hide in it.
  T1 should be large enough not to need it.

---

## 6. What to run first, in order

1. **T1 + the `f32_helpers.c` port**, then `visdrone_fpn_coverage.py`. If
   encodability does not move off 88.2%, stop — nothing has cost a GPU-hour yet.
2. **T2**, with its FD arm. T1 is inert without it.
3. Regenerate `data/visdrone_fpn/`, retrain at **30 epochs** with the `aff30`
   recipe otherwise unchanged, score the epoch curve. **Target: beat 0.2363.**
4. **A seed knob + two replicates**, before the small levers.
5. **T4**, then **T5**, then **T3** — each A/B'd on its own, each with an epoch curve.

---

## 7. ⚠ What this is NOT — say it accurately

This is **not YOLOv5**. The backbone is ResNet-34, not CSPDarknet; `cspBlock` is not
even emittable. Claiming "we did YOLOv5" would be the one overclaim on the marquee
demo, in a repo whose habit is marking its own retractions in place.

▶ The accurate claim is also the stronger one: **a YOLO-family anchor-based detector
trained end-to-end by this stack, from a verified ImageNet backbone, with v5's
assignment and augmentation.** The differentiator was never the architecture — it is
that the whole thing is ours down to the MLIR.
