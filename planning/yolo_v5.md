# yolo_v5.md — the simplest detector that actually works

Successor to `yolo_v4.md`. v4 documented the 20-class VOC YOLOv1 attempt and
its class-collapse-to-"person". This doc resets the **goal** around what we
learned the hard way in the 2026-05-31/06-01 session, and lays out the path.

## Goal (deliberately narrow)

**A single-class (person), single-scale object detector that clearly works** —
boxes that land on people, varying per image. Not historical fidelity, not
multi-class, not SOTA mAP. The simplest thing in the detector design space that
detects, as a clean pedagogical demo. Scale/extend *after* it works.

Explicitly **out of scope** (the complexity we're avoiding):
- **R-CNN family** — two-stage (region proposals + RoI pooling + per-region
  heads). Too much machinery.
- **SSD / YOLOv3** — multi-scale feature pyramids + dense anchor boxes. The
  anchor + FPN machinery is exactly the complexity to avoid for a *working*
  demo. (v3's *training* fixes — conv head, sigmoid/BCE objectness — we DO
  adopt; its *structural* extras — anchors, multi-scale — we don't.)

Single grid (7×7), one responsible box per cell, no anchors, no proposals, no
multi-scale. You cannot get structurally simpler and still detect.

## What this session established (the diagnostic chain)

We hit, in order, the three classic detection failure modes — each one a thing
the literature spent a paper on. This chain IS the pedagogical content:

1. **20-way class collapse → "person" everywhere.** With 20 classes on ~5k
   VOC07 images, "predict the marginal class (person)" is a loss minimum. The
   class head's *input-dependent signal was only ~10% of its output* (measured:
   per-cell class map identical across all 16 probe images). No LR/clip/warmup
   setting escaped it; hot head LR made it *worse* (WD zeroed the input-
   dependent weights faster). **Fix taken: drop to person-only** (`--only-classes
   person`), which makes the marginal == the truth and removes the problem.

2. **FC head can't localize → center-blanket.** The `flatten(25088)→dense(1470)`
   head has to learn spatial correspondence through a giant dense layer; it
   collapses box/objectness to the *marginal location* (boxes always near
   center). Measured: objectness across-image std ≈ 0.002, identical firing
   cells on every image, even person-only. **Fix taken: 1×1 conv detection
   head** (`conv2d 512→30 → [B,30,7,7] → flatten`). Each cell predicts from its
   own feature column with shared weights → localization is structural.
   **Result: conv head localizes per-image from e2 (15/16 distinct firing
   patterns) vs the FC head's permanent 2/16. This is the session's key win.**

3. **Foreground/background imbalance → objectness collapse.** ~1–2 person cells
   vs ~47 background cells per image. Even with λ_noobj=0.5, background
   objectness outweighs foreground ~10:1, so "predict low objectness
   everywhere" is the loss minimum. The conv head localized early (e2–e6) then
   **decayed back** to a uniform center-prior by e20–e30 (objectness std →
   0.0009, all images peak at the same cell; conf briefly spiked to 1.37 at e10
   fitting foreground, then background pressure crushed it). Cosine LR decay did
   NOT settle it — it converged *harder* to the marginal. **Fix (PENDING):
   focal loss on the objectness.** This is the one remaining piece.

Net recipe for a working detector: **conv head + person-only + focal objectness.**
We have the first two; #3 is the immediate next step.

## What's done / validated (reusable infra)

- **grad-clip in the IREE train path** (global L2 norm) — committed (`c01509f`).
  Essential for from-scratch heads at hotter LR.
- **env-resume** — `LEAN_MLIR_INIT_LOAD` + `LEAN_MLIR_START_STEP` in
  `Train.lean` (load full checkpoint + resume epoch/LR schedule). Validated
  live on GPU 1. Plus `scripts/yolo_autoresume.sh` (mars-segfault auto-resume).
- **per-step warmup** — `Train.lean` LR is now per-step (was per-epoch), which
  stopped Adam's early steps from blowing up a hot head (the 45k-loss spike).
- **conv detection head** — `demos/MainYolov1VocTrainBootstrap.lean` +
  `MainYolov1VocInfer.lean`, net name "ResNet-34 + YOLOv1 conv-head (person VOC)".
- **person-only data** — `data/voc2007_person/{train,val}.bin` via
  `preprocess_voc.py --only-classes person` (keeps the 30-ch layout → no
  FFI/loader changes).
- **per-cell diagnostic tooling** (ad-hoc python): class-map variance,
  objectness across-image std, firing-pattern count — the metrics that exposed
  each collapse. Worth formalizing into a small eval script.

All but grad-clip are currently **uncommitted** (one working tree). Commit
before the focal change so the conv-head win is locked + focal is isolated.

## Next steps (ranked by leverage)

1. **Focal objectness (THE unblock).** Convert the conf terms (T3 obj, T4 noobj
   box0, T5 noobj box1) from raw SSE to **sigmoid + focal-BCE**, gated by the
   existing `cfg.useFocal` / `cfg.focalGamma` (γ=2). Use the detached-focal-
   weight gradient — clean and low-risk:
   `d/dlogit = (1−p_t)^γ · (p − t)`, where `p = sigmoid(pred_c0)`, `t` = mask.
   Keep λ_noobj as the α-balance. **Also update the render** to apply sigmoid to
   the objectness channel before thresholding (it currently reads raw conf).
   Retrain person-only conv-head; expect objectness to stay on foreground →
   boxes that land on people. ~one codegen session.

2. **mAP eval (carry-over from v4, still valid).** Replace qualitative renders
   with a number: infer over all 4,952 VOC07 test images, person AP @ IoU 0.5,
   NMS. Gives a metric to optimize and a "how good" answer.

3. **Confidence/threshold calibration for the demo.** Once focal is in, pick a
   render threshold that gives clean boxes (and verify NMS dedups the grid).

4. **(Optional) higher input resolution (224 → 448).** v4's "biggest single mAP
   lever" — needs imageH/W + grid (7×7 → 14×14) rescale + recompile. Bigger
   change; only if the 224 person detector isn't crisp enough.

5. **(Optional, deferred) multi-class.** The hard wall (the v4 class collapse).
   If wanted: class-balanced sampling (`W ∝ 1/n_class`) + independent sigmoid
   class outputs (multi-label, v3-style) instead of softmax. Do this only after
   the single-class detector clearly works, as a separate effort.

6. **(Explicitly NOT next) anchors / multi-scale / v3.** The SSD-flavored
   complexity. Revisit only if a concrete accuracy need forces it.

## Reproduce / run

```bash
# data (person-only, one-time)
python3 preprocess_voc.py data/voc2007 data/voc2007_person --only-classes person

# train (conv head, person, on mars)
IREE_BACKEND=rocm IREE_CHIP=gfx1100 \
  .lake/build/bin/yolov1-voc-train-bootstrap data/voc2007_person
# or under auto-resume: bash scripts/yolo_autoresume.sh <rundir> data/voc2007_person

# render (GPU 1, alongside a training run): cp params_eN→params, then
HIP_VISIBLE_DEVICES=1 IREE_BACKEND=rocm IREE_CHIP=gfx1100 \
  .lake/build/bin/yolov1-voc-infer 16 data/voc2007 /tmp/render
python3 scripts/yolo_render.py /tmp/render --score-thresh 0.2
```

## Artifacts

- Conv-head person run (no focal): `runs/2026-06-01-yolov1-voc-person-convhead/`
  — 80/80 epochs, no segfault; objectness collapsed (the focal motivation).
- Earlier FC-head runs: `runs/2026-05-31-yolov1-voc-{gradclip,person,difflr}/`.
- Net prefix: `.lake/build/resnet_34___yolov1_conv_head__person_voc_*`.
- Backbone bootstrap: `.lake/build/jax_r34_imagenet.bin`, prefix 21,284,672.

## One-line summary

The simplest working detector = single-scale YOLO + conv head + person-only +
focal objectness. We have conv head + person-only (localization works, proven);
**focal is the last piece.** Don't climb to v3/SSD/RCNN — that's the complexity
this goal exists to avoid.
