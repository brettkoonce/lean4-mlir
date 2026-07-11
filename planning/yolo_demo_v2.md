# yolo_demo_v2.md — YOLO detection demo, second pass

Goal: take the Pets cat/dog detector from "works on the mosaics it
was trained on" to a detector that (1) handles ordinary single
full-frame images, (2) is scored by a real metric instead of manual
counting, and (3) optionally grows the one architectural rung v1
skipped — multi-scale features — now that the prerequisite resize
codegen exists. The v1 deliverable was a *lesson* (marginal
dominance); the v2 deliverable is a *detector*.

Prerequisite reading: `planning/yolo_final.md` (the consolidated v1
history — marginal-dominance diagnosis, mosaic + class-balance
fixes, e20 results) and `planning/bifpn_demo.md` (the multi-scale
plan this doc partially re-prices). This doc assumes both.

## Where v1 landed (recap, one paragraph)

One trainer and one infer exe ship today
(`demos/MainYolov1Pets{TrainBootstrap,Infer}.lean`): R34-ImageNet
backbone (21,284,672-float bootstrap prefix) + deep conv head →
7×7×30 YOLOv1 grid, `yolov1Masked` 5-term loss with sigmoid
focal-BCE objectness (γ=2, detached focal weight), grad-clip
codegen, bbox-aware augment, trained on class-balanced 2×2 Pets
head-box mosaics (`preprocess_pets_mosaic.py`). At e20: 64/64
localization on mosaic val, balanced classes at ~64% cat-vs-dog.
Decode + per-class greedy NMS live in `scripts/yolo_render.py`.
The two data lessons (positional marginal → mosaic; class marginal
→ balance) are documented and load-bearing.

## What v1 left open (its own words, plus two audit findings)

From `yolo_final.md` §caveats:

1. **No single-frame transfer.** Trained on mosaics only → 2/16 on
   full-frame pets (the centered marginal reasserts itself).
2. **cat-vs-dog tops out ~64%** at quadrant scale.
3. Suggested-but-unneeded-for-v1: per-quadrant random crop, mixed
   single+mosaic training, 448 input / 14×14 grid.

From re-reading the code:

4. **No quantitative eval anywhere.** "64/64" and "30/34" were
   hand-counted off renders; there is no mAP in the repo (checked).
   This is the detection analogue of the DDPM lesson ("loss ≠
   quality") — v1 even states it: the loss dropped the whole time
   objectness was collapsing. Every v2 A/B is blind until this
   exists.
5. **Eval vmfb is locked to the training batch (16)** — the infer
   main hard-codes `batch := 16` and can't score an arbitrary val
   set or a single image without padding games.
6. **Score decoding is patched around, not fixed**: focal conf
   saturates ~0.55, `obj×class` is dog-biased in ranking, and the
   render script compensates with `--sigmoid-conf` + rank-by-
   objectness. Fine for a demo render, wrong for mAP.

And one repricing that changes the strategic picture:

7. **`bilinearUpsample` now has real forward+backward codegen**
   (landed with the UNet work). `bifpn_demo.md` priced multi-scale
   detection at 7–10 weeks *mostly because resize ops didn't exist*
   (~2–3 weeks) and multi-scale backbone hooks didn't exist
   (~1 week). The first cost is already paid, and the UNet skip
   stack (`MlirCodegen.lean:1490`) is a working precedent for
   threading intermediate features by position. FPN-lite is now a
   1–2 session codegen job, not a quarter-long project.

So v2's core bet: **metric first, then data, then resolution, then
architecture** — each gated on the previous one actually moving the
number.

## Workstream A — mAP harness + flexible eval (do first, no codegen)

**Status 2026-07-10: DONE — Gate A MET.** All four items shipped and
the e20 checkpoint scored on both val sets (RESULTS.md §Pets detection):

| Val set | mAP@0.5 | mAP@0.3 |
|---|---|---|
| mosaic (trained) | 0.041 | 0.227 |
| single-frame (transfer) | 0.0002 | 0.005 |

- **Item 1** `scripts/yolo_map.py`: per-class AP@0.5 (all-point VOC
  integration), reads `logits.bin` + GT boxes straight from the
  157,728-byte `val.bin` records (Python-side, no Lean loader change);
  AP integrator unit-tested (perfect→1, random→0, TP-then-FP→0.5).
- **Item 2** flexible-N in `MainYolov1PetsInfer.lean`: `n=0` scores the
  whole set by looping ⌈N/16⌉ batches and padding the final partial
  batch (both val sets are ×16, but padding is there); the old
  `batch:=16, n:=batch` lock is gone. Images dump skipped for n>64.
- **Item 3** both standing val sets scored (mosaic + single-frame det).
- **Item 4** decode ranks by sigmoid(conf), class = argmax(class slots);
  obj×class product dropped.

**Gate A verdict — met, with a sharper finding than the doc expected.**
Qualitatively as predicted: mosaic ≫ single-frame, single-frame ≈ 0
(quantifies the "2/16 doesn't transfer" caveat — Workstream B's target).
But the *absolute* mosaic mAP@0.5 is only 0.04, not "high": v1's "64/64"
was **peak-in-right-cell** localization, not IoU@0.5. The localization
ceiling (best IoU over all 49 cells per GT) averages 0.497 (50% ≥0.5,
91% ≥0.3) — the detector finds the region but its boxes run ~20% large
and sit on the IoU=0.5 edge, so mAP@0.3 (0.23) ≫ mAP@0.5 (0.04). The
focal head is confidence-saturated (~4 cells/img >0.5), weakening
top-rank ordering — known, not a bug. **This baseline is the honest
before-number for Workstreams B/C/D.**

Operational note for the next session: the e20 checkpoint was recovered
from the pre-rename `lean4-mlir-blueprint` checkout
(`resnet_34___yolov1_deep_head__person_voc__{params,bn_stats}_e20.bin`,
architecture-identical, copied to the `..._pets_` slug); the eval vmfb
was regenerated under the new slug by launching the trainer to its
compile step and killing before bootstrap. `data/pets_mosaic_bal` and
`data/pets_det` are symlinked from that checkout. The R34 bootstrap file
`.lake/build/jax_r34_imagenet.bin` is absent here — needed only to
*retrain* (Workstream B), not to score.

Nothing else in this doc is measurable without it.

1. **mAP@0.5 scorer** (~1 session, Python). Extend
   `scripts/yolo_render.py`'s decode/NMS into
   `scripts/yolo_map.py`: read `logits.bin` + GT boxes, produce
   per-class AP@0.5 and mAP over the *whole* val set. The GT boxes
   are already in the 157,728-byte record format (up to
   `MAX_BBOXES=56` per record) — the loader side exists; the
   Python just needs to read them back out.
2. **Arbitrary-N inference.** Either compile a second eval vmfb at
   B=1 (specs already parameterize batch) or loop full batches +
   pad the tail. Removes the `batch := 16` lock in the infer main.
   ~1 hr.
3. **Two standing val sets, scored every run**: mosaic-val (the v1
   regime) and **single-frame-val** (plain `preprocess_pets_det.py`
   output). Single-frame mAP is *the* v2 headline number; v1's
   2/16 anecdote becomes a baseline row.
4. **Fix the score decode for real**: calibrate or drop the
   `obj×class` product — for mAP, rank by sigmoid(conf logit) and
   report class from argmax(class slots), matching how the loss
   actually shaped the head. Document the equilibrium-≈0.55 focal
   conf as expected behavior, not a bug.

Gate A: harness reproduces v1's e20 checkpoint qualitatively
(mosaic mAP high, single-frame mAP near zero) — then we have honest
before/after numbers for everything below.

## Workstream B — single-frame transfer (data only, the main event)

**Status 2026-07-10: 50/50 blend run DONE — Gate B PARTIAL.** Shipped
items 1+2 (`--single-frac` + box-aware single crop in
`preprocess_pets_mosaic.py`), R34 bootstrap recovered, full 80-ep run
on `data/pets_mixed` (50/50, class-balanced) via `run_yolo_mixed.sh`
(clean, zero crashes). Scored vs both standing val sets (RESULTS.md
§Workstream B):

| ckpt | mos@0.5 | mos@0.3 | sng@0.5 | sng@0.3 |
|---|---|---|---|---|
| v1 (mosaic-only e20) | 0.041 | 0.227 | 0.0002 | 0.005 |
| mixed e80 | 0.034 | 0.217 | 0.011 | 0.049 |

- **Single-frame rose ~50× @0.5 / 10× @0.3** (the doc's literal Gate B
  criterion — "rises substantially" — is MET), and **mosaic recovered to
  within noise of v1** by e80 (mild tradeoff). So the *mechanism* works.
- **BUT single-frame plateaus at ~0.01 @0.5 from e20→e80** (more epochs
  don't help), and it's NOT a usable single-frame detector yet.
  **Root cause (diagnosed from e80 preds):** box regression is per-cell,
  so 50/50-by-record is ~4:1-by-box toward small quadrant boxes (mosaic
  = 4 small-box cells/record, single = 1 large-box cell). The width head
  collapses to a near-constant **w=0.23±0.01** while single heads need
  w≈0.40 → **0% of top single boxes reach IoU≥0.5**. Class head is fine
  (68% top-box class match); the failure is box *scale*, not class.
- **Next lever = box-distribution balance, not epochs.** Higher single
  fraction (≈0.8 balances box counts) or larger/uncropped singles. This
  is exactly the doc's "report the frontier — 25/50/75% blend sweep"
  fallback below; **a --single-frac 0.75 run is launched to test it**
  (checkpoints of the 50/50 run backed up as `..._params_mix50_e80.bin`).

Recipe/data notes for the next session: mixed data at `data/pets_mixed`
(regen via `preprocess_pets_mosaic.py data/pets <out> --single-frac F`);
score any checkpoint by copying `..._params_eN.bin`/`..._bn_stats_eN.bin`
→ base names then `yolov1-pets-infer 0 <valdir> <out>` +
`scripts/yolo_map.py <out>/logits.bin <valdir>/val.bin`.

v1 proved the failure is distributional; the fix is too.

1. **Mixed single+mosaic training** (~1 session in
   `preprocess_pets_mosaic.py`). Emit a configurable blend — start
   50% mosaic / 50% single full-frame (class-balanced on the
   singles too). Mosaics keep killing the positional marginal;
   singles teach the "one big centered object" mode as a *real*
   mode instead of a prior to exploit. Same record format, zero
   Lean changes.
2. **Global random crop/zoom on singles** (~half session). A
   random-scale crop (e.g. 60–100% of frame, box-aware, reject
   crops that cut the head) decenters singles *without* leaving the
   natural-background distribution — the same reasoning that made
   mosaic beat gray-canvas. Also directly implements v1's
   "per-quadrant random crop" suggestion when applied inside
   mosaic quadrants.
3. **Retrain the v1 spec unchanged** (80 ep config exists;
   checkpoint-every-2 already handles the mars segfault pattern).

Gate B: single-frame mAP rises substantially with mosaic mAP within
noise of the v1 baseline. This is the "v2 is a detector" claim; if
mixed training trades the two off hard, report the frontier (25/50/
75% mosaic blend sweep — three overnight runs).

## Workstream C — resolution: 448 input, 14×14 grid

The codegen already parameterizes `yoloGridH/W`; R34 at 448 gives
14×14 at stride 32 naturally, so this is a spec + preprocessing
change (re-emit records at 448, `MAX_BBOXES` layout scales), not
new machinery. Costs ~4× per step (batch 16 → likely 8 for VRAM,
~4 s/step territory on mars — overnight for 40 ep). Expected wins:
finer localization and, per v1's diagnosis, a higher cat-vs-dog
ceiling (the 64% was blamed on quadrant scale — at 448 a mosaic
quadrant is 224², exactly the scale the backbone was pretrained
at).

Gate C: 448/14×14 vs 224/7×7 at matched wall-clock on both val
sets. Class accuracy is the number to watch, not just mAP.

## Workstream D — FPN-lite multi-scale head (the new codegen)

Only after B (and ideally C) land — this is the architectural rung,
now cheap enough to justify:

- **Expose backbone stage features.** The UNet skip stack already
  demonstrates emit-time side-channel plumbing; add a `.saveFeature`
  marker Layer kind (no params) that pushes the current SSA +
  shape, placed after the stride-16 and stride-32 residual stages.
  This is `bifpn_demo.md`'s "option 2" (3-day estimate), with the
  abstraction leak contained by the existing stack precedent.
- **FPN-lite fusion**: lateral 1×1 conv on P4 + `bilinearUpsample`
  P5 + add + 3×3 smooth — all existing primitives with real
  codegen. Head predicts on the fused stride-16 map: a 14×14 grid
  *at 224 input* (an alternative route to Workstream C's
  resolution, at ~1× instead of ~4× compute).
- Explicitly **not** anchors/RetinaNet-style matching — stay in the
  YOLOv1 per-cell formulation so `yolov1Masked` + the record format
  carry over with only grid-size changes. Anchor machinery is where
  the old BiFPN estimate's weeks lived; skip it entirely.

Estimate: 1–2 sessions codegen (forward + backward arms for
`.saveFeature` threading; the pidx/BN-stat audit that bit both v1
efforts is the tax to budget for) + overnight A/B.

Gate D: FPN-lite@224 vs plain@224 and vs plain@448 at matched
wall-clock. If plain@448 wins outright, document it and don't ship
the complexity — resolution beating architecture is a fine result.

## Workstream E — verified-gradient tie-in (stretch, on-brand)

The train step already runs through the verified train-step
codegen, but the `yolov1Masked` + focal-BCE loss branch (masked
5-term MSE, sigmoid focal with *detached* weight) has no VJP
theorem. Adding it to the proved-VJP set is more interesting than
the DDPM MSE case precisely because of the detachment — the proof
documents that the implemented gradient is the gradient of a
*modified* objective (focal weight treated as constant), which is
exactly the kind of honesty-about-what's-actually-computed the
proof arc exists for. ~1–2 sessions; not on the demo's critical
path.

## Sequencing

```
Phase 0 (1 session, no codegen):   A  mAP harness + flexible eval + decode fix
                                   → score the existing e20 checkpoint (Gate A)
Phase 1 (1–2 sessions + runs):     B  mixed single+mosaic + crop aug (Gate B)
Phase 2 (1 session + overnight):   C  448 / 14×14 (Gate C)
Phase 3 (1–2 sessions + runs):     D  .saveFeature + FPN-lite (Gate D)
Phase 4 (optional):                E  focal/masked-loss VJP proof
```

Phases 0–1 are the committed core (≈2–3 sessions + two or three
overnight runs) and alone convert v1's honest caveat into the v2
headline ("now works on ordinary photos, mAP attached"). 2–4 are
independently droppable; D depends on nothing in C but should be
judged against it.

## Deliverables

- `scripts/yolo_map.py` + mAP rows in RESULTS.md (mosaic-val AND
  single-frame-val, before/after)
- `preprocess_pets_mosaic.py` gains `--single-frac` / crop-aug
  flags (or a thin `preprocess_pets_mixed.py` wrapper)
- Retrained checkpoint + updated `demos/figures/yolo_pets.png`
  featuring *single full-frame* detections — the money shot v1
  couldn't produce
- If D lands: `.saveFeature` Layer kind + FPN-lite spec, and a
  re-priced note in `bifpn_demo.md` (its resize-op premise is
  stale — point it here)
- Bestiary: `YOLO.lean` prose gains the v2 result; the multi-scale
  variants (`yoloV3`+) stay shape-only unless D ships

## Out of scope (unchanged from v1, plus)

Full VOC/COCO multi-class (VOC scripts stay retired in git history
until the mAP harness proves out on Pets), anchor-based heads and
IoU-aware losses (GIoU/CIoU — worthwhile only after anchors, which
we're avoiding), full EfficientDet/BiFPN (re-priced but still its
own doc), DETR-style set prediction (`detrHeads` stays shape-only),
video/tracking.
