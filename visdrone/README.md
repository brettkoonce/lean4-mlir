# visdrone/ — standalone PyTorch reference detector

A conventional PyTorch detector on VisDrone-DET, built to give the Lean FPN arm
something to be measured **against**. It is not a demo and it is not the
deliverable; it is the missing rung on the validation ladder.

## Why this exists

`planning/yolo_assignment.md` records four trained arms and eight numpy probes,
all reporting mAP@0.5 = 0.0001, and a chain of four mutually-overturning
explanations for why. Every one of those probes was conditioned on "the pipeline
is correct and the detector is merely mediocre" — a condition that was never
tested, because there was no known-good number on this dataset to test it
against. Every other brick in this repo had a reference partner to diff against
(FD oracles for the codegen, Lean ref trainers for the optimizers, numpy replicas
for the loss). The detector never did.

## Layout

| file | what it does |
|---|---|
| `prepare_yolo.py` | VisDrone-DET → YOLO labels. Uses the **identical** class filter as the Lean pipeline's `parse_visdrone_txt`, so the two arms score the same GT. |
| `train_baseline.py` | stock ultralytics recipe; `--no-aug` mirrors the Lean arm's `augment := false` |
| `inspect_lean_bin.py` | renders a record straight out of `data/visdrone_fpn/train.bin` with encoded targets + raw annotations overlaid |
| `check_bin_alignment.py` | numerical version of the above: does the encoder round-trip? |

Setup (already done; recorded for reproducibility):

```
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
.venv/bin/pip install ultralytics pillow numpy
.venv/bin/python3 prepare_yolo.py          # writes labels/ beside the source images
HIP_VISIBLE_DEVICES=0 .venv/bin/python3 -u train_baseline.py --epochs 100 --name v8s_640_100ep
```

## Cross-checks that the two arms are measuring the same thing

`prepare_yolo.py` independently reproduces the Lean pipeline's GT exactly:

| quantity | Lean pipeline | this converter |
|---|---|---|
| train boxes | 343,204 | **343,204** |
| val boxes/img | 70.7 | **70.7** |

Same 10 classes, same `score==0` / category-0 / category-11 filter, same
`cat-1` remap.

## Findings so far

**The encoder is not the bug.** `check_bin_alignment.py` decodes 38,421 encoded
targets out of `train.bin` across 600 records and matches each against its
annotation: centre residual max **2.11e-09** normalized (= 0.000 fed pixels),
w/h max 2.95e-08, **0 boxes missing their annotation**. The image↔target
correspondence the Lean trainer is taught is exact. This was the leading
"something is silently broken" hypothesis and it is now ruled out.

**The dataset is not the problem.** Numbers below are `metrics/mAP50(B)` on the
548-image val split.

| arm | recipe | init | epochs | recall | mAP@0.5 |
|---|---|---|---|---|---|
| Lean FPN (T2-bias, best of four) | no aug, 448 squash | ImageNet R34 | 12 | 0.118\* | **0.0001** |
| YOLOv8s `scratch_noaug_12ep_448` | no aug, 448 letterbox | **random** | 12 | 0.164 | **0.114** |
| YOLOv8s `scratch_squash_12ep_448` | no aug, 448 squash | **random** | 12 | 0.197 | **0.140** |
| YOLOv8s `leanmatch_noaug_12ep_448` | no aug, 448 letterbox | COCO-det | 12 | 0.240 | **0.192** |
| YOLOv8s `v8s_640_100ep` | full aug, 640 | COCO-det | 100 | 0.400 | **0.391** (best, e60) |

The reference number is **mAP@0.5 = 0.391 / mAP@0.5:0.95 = 0.225**, which sits in
the normal published band for a YOLOv8s on VisDrone — so the ladder is
trustworthy, and the Lean arm is **3900× below an ordinary result.**

\* class-agnostic, and measured against GT that `MAX_BBOXES = 56` truncation had
already cut by 34.9% — i.e. flattered relative to the rows below it.

**The recipe is not the gap, and neither is the initialization.** The
from-scratch row is matched to the Lean arm on augmentation, resolution and
epoch budget, and is *strictly worse* on initialization (random weights versus
an ImageNet-pretrained R34 backbone). It still lands **1140× higher**.

**Where the gap actually lives: score ordering, not localization.** The Lean arm
recalls 11.8% of objects and the from-scratch YOLOv8s recalls 16.4% — the same
ballpark. Their mAP differs by three orders of magnitude. So the Lean detector
is finding roughly the objects a working detector finds and is unable to rank
them above background. That independently corroborates the last probe in
`planning/yolo_assignment.md` (objectness AUC 0.741, ranking is the binding
constraint) and supplies the calibration that probe was missing: a detector at
this recall *should* score 0.114, so 0.741 is not "mediocre", it is broken.

Note also the no-aug arm peaks at epoch 6 (0.198) and drifts to 0.192 by 12 —
textbook no-augmentation overfitting, and the shape the Lean arm's flat loss
curve should have had if it were training normally.

## Squash vs letterbox — my aspect-ratio hypothesis, REFUTED

`make_squash_dataset.py` pre-squashes VisDrone to 448×448 with the Lean
pipeline's exact `resize((448,448))` call, so the subsequent letterbox is a
no-op and the model sees precisely what the Lean trainer sees. Identical recipe
otherwise (random init, no aug, 448, 12 epochs) — the only difference is the
resize.

| epoch | 6 | 8 | 10 | **12** |
|---|---|---|---|---|
| letterbox (control) | 0.0946 | 0.1014 | 0.1103 | **0.1140** |
| **squash** (Lean-style) | 0.1139 | 0.1362 | 0.1401 | **0.1399** |

**Squashing is 23% BETTER, at every epoch.** The aspect distortion is real —
0.750 for 4:3 sources, 0.562 for 16:9, and the measured width-vs-height error
asymmetry in `planning/yolo_assignment.md` is consistent with it — but it is
swamped by the pixel budget. Letterboxing a 16:9 frame into a 448 square spends
~44% of the canvas on grey padding, shrinking objects that are already only a
few pixels across. On this dataset that costs far more than the distortion.

So "letterbox the input" was a wrong recommendation and is withdrawn. Worth
recording as the one place a confident, well-quantified geometric argument lost
to a direct A/B.

## The scoring comparison — `compare_scoring.py`

Both detectors, one architecture-neutral metric: a candidate slot is POSITIVE if
its cell centre falls inside a GT box. 60 val images, full annotations (no
`MAX_BBOXES` truncation on either side).

| detector | score | AUC | pos mean | bg mean | separation | in top-1000/img |
|---|---|---|---|---|---|---|
| Lean FPN | `sigmoid(obj)` | 0.5505 | 0.1471 | 0.1407 | 0.17 σ | 10.2% |
| Lean FPN | `obj × clsprob` (production) | 0.5311 | 0.0497 | 0.0480 | 0.11 σ | 9.1% |
| **YOLOv8s** (scratch, no aug, 448) | `max sigmoid(cls)` | **0.7385** | 0.0673 | **0.0025** | **3.18 σ** | **57.3%** |

**The Lean detector's score is very nearly a constant function of the input.**
Objectness puts positives at 0.1471 and background at 0.1407 — a gap of 0.0065,
or 0.17 background sigma — and never exceeds **0.2417 on any of 740,880 slots**.
The production score never exceeds 0.1121. A working detector routinely emits
0.9; this one has never learned to say "yes, definitely".

**And 0.14 is the base rate.** Positives are 11.06% of slots. A head emitting
≈0.14 everywhere is predicting the marginal P(object) and almost nothing
conditional on the pixels. That reframes the "converged objectness equilibrium
at p≈0.14" in `planning/yolo_assignment.md`: it is not a fixed point of the loss
to be respected, it is a head that did not learn.

Three further readings:

- **The class multiplier makes ranking worse, not better** (0.5505 → 0.5311).
  The Lean class head is a *softmax masked to positive cells*, so it is never
  trained on background and cannot suppress it; multiplying by it injects noise
  into the one channel that was doing the work. YOLOv8's per-class *sigmoids*
  are trained on every anchor with target 0 on background — which is why its
  background mean is 0.0025, **56× lower**.
- **AUC alone understates the gap.** 0.55 vs 0.74 is a factor of ~2 in error
  terms, but top-1000 survival differs 6× (9.1% vs 57.3%), and mAP is driven by
  the extreme tail, not the average pair. YOLOv8's distribution is bimodal —
  median background 0.0000, positives reaching 0.9178 — so its top-1000 is
  mostly real objects. The Lean distribution is unimodal mush, so its top-1000
  is close to a random sample of slots.
- **This is one score doing two jobs versus two scores doing neither.** YOLOv8
  trains a single class score against a soft, IoU-aware target over all anchors.
  The Lean detector splits the job: objectness (trained to a constant 1.0 on
  positives only) and a positives-only class softmax. `yolo_assignment.md`
  already measured the consequence — objectness is good at object-vs-background
  and *anti-correlated* with box quality (AUC 0.3497), the class head is the
  reverse — and its IoU-target probe tested the fix as a **rescoring** of frozen
  channels. The comparison above says the missing ingredient is not the target
  shape but **training the score on background at all**.
