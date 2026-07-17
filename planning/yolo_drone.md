# yolo_drone.md — modern multi-scale detection on VisDrone

Goal: replace the YOLOv1-on-Pets detection demo (cat-vs-dog head boxes, a
single 7×7 grid) with a real detector on a real task — **modern multi-scale
YOLO on VisDrone** — and, once it lands alongside the BraTS segmentation demo,
retire the Oxford-IIIT Pets dataset entirely.

This is the detection twin of the BraTS thesis, and the parallel is exact:

- **BraTS** — the class you care about (enhancing tumour) is ~0.5% of *pixels*,
  and the naïve loss collapses onto background. Weighting + init + schedule is
  what recovers it.
- **VisDrone** — the objects you care about (pedestrians, cars from altitude)
  are *tiny and densely packed*, and the naïve detector (YOLOv1's single coarse
  grid) structurally cannot emit them. **Multi-scale detection** is what
  recovers them.

Both are the same lesson: real data exposes what the toy hid. Pets head-boxes
are one big centered object per image — the regime where a 7×7 grid is fine and
nothing is learned about *why* modern detectors look the way they do.

Prerequisite reading: `planning/yolo_final.md` (the existing YOLOv1 path — what
detection infrastructure already exists). Sibling demo: `planning/brats_demo.md`
(the same real-data-over-toy move, already executed for segmentation).

**Audience, same as BraTS: a working demo a newcomer can pick up and run.**
Reproducibility and legibility outrank verification depth; the correctness story
is FD validation of the new loss/assignment blocks plus a runnable eval (mAP).

## The parallel that makes this a coherent chapter, not a rewrite

The detection section currently makes its point on Pets: "YOLO works, here's a
7×7 grid finding pet heads." That teaches the *mechanics* of detection (grid,
box regression, objectness) but nothing about the *architecture* — because on
one centered object, single-scale is enough and anchors are unnecessary.

VisDrone forces the architecture into view. It is the detection analogue of the
segmentation chapter's structure:

| | segmentation (BraTS) | detection (VisDrone) |
|---|---|---|
| the hard thing | thin class (0.5% of pixels) | tiny dense objects |
| the naïve failure | loss collapses to background | grid can't resolve overlapping small boxes |
| the fix | class-balanced loss + init + LR | multi-scale heads + anchors/assignment |
| the metric | WT/TC/ET Dice | mAP@0.5 / mAP@[.5:.95], per-scale |
| honest expectation | modest, region-not-type | modest, VisDrone is a hard benchmark |

## Two decisions, up front

### Data: VisDrone-DET (object detection track)

VisDrone-DET2019 — the standard aerial-detection benchmark. ~6,471 train / 548
val / 1,610 test-dev images, **10 classes** (pedestrian, people, bicycle, car,
van, truck, tricycle, awning-tricycle, bus, motor). Images are large
(~2000×1500) and the objects are small and dense — the whole point.

Annotations are per-image `.txt`, one line per box:
`bbox_left, bbox_top, width, height, score, category, truncation, occlusion`.
Note: `category 0` (ignored regions) and `category 11` (others) are **excluded**
from evaluation, and `score = 0` marks regions to ignore — the preprocessor must
honor both or the numbers are wrong (the VisDrone analogue of BraTS's
brain-voxel-only normalization: a data subtlety that silently corrupts the
result if skipped).

**Download reality — the one thing to verify first.** The official source
(`github.com/VisDrone/VisDrone-Dataset`) links to Google Drive + Baidu, and the
Drive links are large-file-flaky under `gdown`. **Workstream 0 is to pin a
reproducible fetch** — a HuggingFace mirror or AWS copy, `curl`-able without an
account, the way `download_brats.sh` fetches MSD over unauthenticated S3. If no
clean source exists, that is a finding that reshapes the demo (same as BraTS
choosing MSD over Synapse-gated 2021). Do not build on a fetch that only works
on one machine.

Citation: Zhu et al., *Detection and Tracking Meet Drones Challenge*, IEEE TPAMI
(2021); the VisDrone2019 dataset paper.

### Architecture: multi-scale, and anchor-based first

"Modern YOLO" spans a decade. The feature that *matters for this task* — and the
one YOLOv1 lacks — is **multi-scale detection**: a feature-pyramid neck feeding
detection heads at several strides, so small objects are found on the
high-resolution head and large ones on the coarse head. That single change is
the v1→v3 leap and it is what makes VisDrone tractable at all.

**Recommendation: build the multi-scale anchor-based detector (YOLOv3/v5-class)
first, with a documented path to anchor-free (v8/X-class).** Reasoning:

- **The neck is nearly free.** An FPN/PANet neck is upsample + concat + conv —
  the *exact* primitives segmentation already ships (`unetUp`,
  `bilinearUpsample`, concat). The multi-scale structure reuses the verified kit.
- **Anchor-based label assignment is simpler codegen** than anchor-free's
  dynamic assignment (SimOTA / task-aligned). Anchors are a k-means over the
  VisDrone box sizes (host-side, once) + IoU matching — no new in-graph
  machinery. Anchor-free adds Distribution Focal Loss and a dynamic assigner,
  which are genuinely novel blocks.
- **It is still a real modern detector.** Multi-scale + anchors + IoU loss +
  proper NMS is v3/v5, which is the honest "not the og way" the demo wants. v8
  anchor-free is the *stretch*, scoped below, not the entry point.

Backbone: **reuse an existing verified backbone** (ResNet-34, the current
YOLOv1 bootstrap, or a lighter one) rather than porting CSP-Darknet — the
framework's whole approach is backbone-reuse (YOLOv1 already bootstraps R34).
The neck + heads are what's new.

## What's reusable vs what's new

**Reusable (measured 2026-07-16):**
- Verified backbones (ResNet-34 etc.) — the feature extractor.
- **FPN neck primitives** — `unetUp` / `bilinearUpsample` / channel concat
  already exist and compile; the multi-scale fusion is these, not new codegen.
- The detection training ABI scaffold (`LossKind.yolov1Masked`, the
  `trainStepAdamF32Yolov1` dispatch, the det-bin loader) — a starting point,
  though the loss itself is replaced.
- The demo/figure infrastructure pattern (per-arm artifacts, `evalEveryNEpochs`,
  best-by-val checkpointing) landed for BraTS transfers directly.

**New (the real build):**
1. **Multi-scale detection head** — N output tensors at strides 8/16/32, each
   `[B, A*(5+C), Hi, Wi]` (A anchors, 5 box+obj, C classes). New codegen, but
   it is convs on the neck outputs — no new *primitive*, a new *emit pattern*
   (like the seg loss blocks were).
2. **Anchors + label assignment** — k-means box priors (host, once), IoU-match
   each GT box to (scale, anchor, cell). Host-side target encoding, the
   detection analogue of the seg mask encoding.
3. **Detection loss** — objectness BCE (with the fg/bg imbalance the existing
   focal path already addresses), class BCE, and an **IoU-family box loss**
   (CIoU/DIoU) rather than v1's raw MSE. FD-verify the box-loss gradient the way
   the seg losses were — CIoU's gradient is the one genuinely new piece worth a
   numeric check.
4. **NMS** — multi-scale non-max suppression at inference. Appears thin/absent
   in the current infer path; needed for real mAP. Host-side is fine.
5. **mAP eval harness** — per-class AP + mAP@0.5 and mAP@[.5:.95], the field's
   metric. The detection analogue of the region-Dice harness.
6. **VisDrone data pipeline** — `download_visdrone.sh` + `preprocess_visdrone.py`
   (parse the `.txt` boxes, honor ignored/other categories, resize/letterbox to
   the model input, encode multi-scale targets).

## The hard part, stated honestly up front

VisDrone is a **hard** benchmark. Strong published models land around
**mAP@0.5 ≈ 0.35–0.55**; a from-scratch demo detector on a modest budget will be
well under that, and that is fine — the same honesty the BraTS demo carries
(WT Dice ~0.33, stated as modest, not sold as SOTA). The deliverable is *a
working multi-scale detector on real dense-small-object data*, with the figure
showing why single-scale fails and multi-scale helps — not a leaderboard entry.

The likely money-slide, mirroring BraTS's collapse triptych: **YOLOv1's single
grid on VisDrone → almost nothing detected** (the grid can't resolve the
density), next to **the multi-scale detector → tiny objects actually boxed**.
That contrast *is* the demo, and it rhymes with the segmentation chapter's
empty-brain-vs-segmented-brain figure.

## Workstreams (gates)

```
WS-0  DATA     pin a reproducible VisDrone fetch          (Gate 0 — do FIRST)
WS-A  baseline YOLOv1 single-grid on VisDrone → it fails  (Gate A: the collapse)
WS-B  neck     FPN/PANet from unetUp+concat, 3 scales     (Gate B: 3 heads emit)
WS-C  anchors  k-means priors + IoU label assignment      (Gate C: targets encode)
WS-D  loss     obj/cls BCE + CIoU box loss, FD-verified   (Gate D: grad vs FD)
WS-E  NMS+mAP  multi-scale NMS + AP harness               (Gate E: a real number)
WS-F  figure   v1-vs-multiscale on identical images       (Gate F: the money slide)
WS-G  train    matched-budget run, honest mAP             (Gate G: the result)
      (stretch) anchor-free head + DFL + dynamic assign   (→ its own doc)
```

WS-A first is deliberate and cheap: running the *existing* YOLOv1 on VisDrone
reproduces the collapse and motivates everything after it, exactly as CE-on-BraTS
did for the loss ablation. It costs one training run and buys the whole
narrative.

## Sequencing with the Pets retirement — the hard rule

**Pets is deleted LAST, only after both replacements are landed and working.**
Delete-first leaves the segmentation section pointing at a mid-migration demo
and the detection section pointing at nothing.

Order:
1. BraTS → blueprint §10.2.3 (segmentation), Pets demoted to a one-line mention.
2. This demo built through Gate G; VisDrone → blueprint detection section.
3. **Then** the single clean Pets sweep: `download_pets.sh`,
   `preprocess_pets*.py`, `MainUnetPetsTrain`, `MainAutoencoderPetsTrain`,
   `MainPetsPredict`, `MainYolov1Pets*`, the Pets loaders, and the Appendix A
   row — all removed together.

**Two things Pets was quietly doing — handle deliberately, do not lose by
accident:**
- **The skip-connection ablation** (`autoencoderPets`, the skipless UNet twin).
  BraTS has no such twin. It is currently *unpublished* (the 3-epoch A/B has the
  autoencoder ahead, 0.360 vs 0.344 — a budget artifact), so deleting it loses a
  *latent* result, not a delivered one. If the skip point is worth keeping: a
  skipless BraTS autoencoder (~a day), or make the argument architecturally in
  the Bestiary UNet definition without a training run. Decide, don't drift.
- **The cheap smoke test.** Pets is 738 MB / minutes; the fast seg/det iteration
  path. Replacement already exists in embryo — the 64-slice **mini-BraTS** used
  this session to validate changes in seconds. Formalize it (`data/brats_mini/`
  or a flag) and the fast path survives Pets.

## Out of scope (for this demo's first result)

Anchor-free (YOLOv8/X) head — its own follow-on doc, gated on the anchor-based
version working. Tracking (VisDrone-MOT is a separate task). Test-set submission
to the VisDrone server (we report val mAP, reproducible; not a challenge entry).
Full CSP-Darknet backbone (reuse an existing verified backbone instead).
Rotated/oriented boxes (VisDrone is axis-aligned). TTA / model ensembling.
