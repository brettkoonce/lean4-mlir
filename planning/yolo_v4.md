# yolo_v4.md — YOLOv1 on VOC: where it's at, what's next

Status + next-steps handoff after the 2026-05-30 session. Picks up from
`planning/yolo_with_r34_imagenet.md` (the R34-ImageNet backbone bootstrap)
and `planning/yolo_demo_v3.md` (Phases 1-5).

## TL;DR

- **We have a working *localizer*, not yet a working *classifier*.** The
  full pipeline trains end-to-end (Lean codegen → IREE → boxes on objects),
  and the detector puts 2-3 boxes on the right objects per image. But the
  **class head collapsed to "person"** (the VOC majority class) — it labels
  ~everything person, so multi-class mAP would be low.
- **The session's real win: SSE → cross-entropy class loss.** That took the
  detector from **0 confident detections → 16/16 images detecting.** The
  original YOLOv1 SSE class term couldn't sharpen the 20-way logits.
- **No mAP measured** — eval is train-loss-only; the 16-image grid is
  qualitative.

## What works (don't redo)

- R34-ImageNet backbone (69.26% top-1) bootstraps cleanly into YOLOv1.
- Full train→infer→render loop on VOC07, gfx1100/IREE (needs the
  reduction-distribute flag, already baked into `Types.lean` rocm path).
- **CE class loss** in `LeanMlir/MlirCodegen.lean` (commit `8bc767d`):
  forward = numerically-stable log-softmax over the 20 class channels,
  masked NLL; backward = `mask·(softmax−tgt)/B`.
- Inference batch fix (commit `55b85da`): infer runs one batch of 16 to
  match the eval vmfb's compiled batch size.

## The key finding: class collapse to "person"

Predicted class over all 16×49 = 784 cells of the final (e40) checkpoint:
```
person 704 (90%) · chair 64 · car 16    — only 3 of 20 classes ever predicted
```
Cause: **class imbalance + under-training.** Person dominates VOC; the
weakly-trained class head defaulted to the majority-class prior instead of
learning to discriminate. Class quality **plateaued by epoch 10** (e10 ≈
e30 ≈ e40 on class metrics) — it found the "predict person" mode fast and
stopped. CE fixed "predict *something* confident" (vs SSE's uniform mush),
but ~10 effective epochs on sparse, imbalanced cells wasn't enough for
20-way discrimination.

**Implication:** more of the *same* training won't help (plateaued). The
fix is the imbalance + capacity levers below.

## Detection behavior (final e40 checkpoint)

Score = sigmoid(conf) × max softmax(class). Threshold sweep:

| thresh | detections/image | read |
|---|---|---|
| 0.10 | 16 × 1 | the single most-confident box |
| **0.09** | **11×2, 5×3** | clean multi-box — best for the figure |
| 0.08 | 4-5 | over-detecting |
| 0.05 | 6-7 | mostly false positives |
| 0.01 | ~30 | noise (every cell) |

Best detection score ~0.28 (a strong detector peaks much higher). Figure
rendered at **0.09** → `blueprint/src/figures/yolo_voc/grid.png`.

## Honest gap to "real" YOLOv1

Paper: ~63% mAP@0.5 (VOC07, 448 input, VOC07+12). Ours is **unmeasured**
but estimated low — and the class collapse caps it hard (person AP maybe
okay, other 19 classes ≈ 0). Handicaps, ranked:

1. **224 input vs paper's 448** — the big one; detection needs resolution,
   small/rare-class objects are unclassifiable at 224.
2. **Class collapse to person** — see above; the binding constraint on mAP.
3. **VOC07 only (5,011 imgs)** vs VOC07+12 (~16.5k) — ~3× less data, and
   worsens imbalance for rare classes.
4. **CE class head only ~10 effective epochs**, plateaued.
5. Single-scale, 2 boxes/cell, no test-time tricks.

## Next steps (ranked by leverage)

1. **Build an mAP eval** to replace the qualitative read with a number.
   Run infer over all 4,952 VOC test images + per-class AP (NMS IoU 0.5).
   ~1 hr. Gives the real "how close" answer and a metric to optimize.
2. **Class-balanced sampling** — oversample rare classes (`W ∝ 1/n_class`)
   to break the person-collapse. ~30 lines in the dataloader (see
   `planning/SIDE_QUESTS.md`). Highest-leverage fix for the *classifier*.
3. **448 input** — the paper's detection resolution. Biggest single mAP
   lever; needs the spec's imageH/W bumped + the YOLO head/grid rescaled
   (7×7 → 14×14) and a recompile. Bigger change.
4. **+ VOC2012 data** — ~3× more training data, eases imbalance.
5. Longer / more-diverse class training only *after* 2-4 (it's plateaued
   on the current setup).

## How to reproduce / run

The working config used a **continue-train** from the 80-epoch checkpoint
with CE loss:
- `demos/MainYolov1VocTrainBootstrap.lean` config (currently a ONE-OFF edit,
  uncommitted): full-checkpoint bootstrap (prefix = 58,165,502 = all params,
  path `_params.bin`), LR 3e-4, 40 epochs, CE loss.
- Train: `IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 lake exe yolov1-voc-train-bootstrap data/voc2007`
- Infer: `IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 .lake/build/bin/yolov1-voc-infer 16 data/voc2007 <outdir>`
  (runs one batch of 16; loads `{buildPrefix}_params.bin` + `_bn_stats.bin` + `_fwd_eval.vmfb`)
- Render: `python3 scripts/yolo_render.py <outdir> --score-thresh 0.09`
- Test an intermediate snapshot: swap `_params_eN.bin`/`_bn_stats_eN.bin`
  over `_params.bin`/`_bn_stats.bin`, infer on GPU 1, restore.

## Checkpoints / artifacts

- `.lake/build/resnet_34___yolov1_head__voc_bootstrap__params.bin` — final
  e40 CE-trained detector (+ `_e10/_e20/_e30/_e40` snapshots, `_bn_stats*`).
- Original 80-epoch (SSE) checkpoint backed up: `/tmp/yolo_voc_80ep_params.bak.bin`.
- Figure: `blueprint/src/figures/yolo_voc/grid.png` (+ `det_*.png`).

## Open housekeeping

- **Commit the figures** (`blueprint/src/figures/yolo_voc/`) when wanted.
- **Reconcile `MainYolov1VocTrainBootstrap.lean`** — it currently holds the
  one-off continue-train edit; either keep it as a documented "continue"
  config or revert to the canonical ImageNet-backbone-from-scratch version.
- CE change is committed (`8bc767d`); infer fix (`55b85da`) — both pushed.

## Loss trajectory (CE continue-train, for reference)

epoch 1: 10.87 → epoch 40: 9.84. Note: total loss is dominated by
coord/conf; it barely moves as the class head sharpens, so it's NOT a good
proxy for detector quality — always confirm with inference. (This is the
trap that hid the dead-class-head for the whole original 80-epoch SSE run.)
