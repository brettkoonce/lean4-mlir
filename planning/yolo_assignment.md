# yolo_assignment.md — target assignment (detection-infra brick #4)

Handoff for the VisDrone detector's **last untested hypothesis**. Prereqs: `yolo_fpn.md`
(the FPN build + the Tier 1/2 refutations), `yolo_drone.md` (overall plan), memories
`yolo-fpn-thread` + `visdrone-fetch-and-wsa`. Written 2026-07-19, after Tier 2 was
measured out.

## Why this doc exists

Four levers have been built, trained, and refuted. mAP@0.5 was **0.0001 in every one**:

| lever | what it changed | result |
|---|---|---|
| T1a objectness `/num_pos` | loss balance | **refuted before building** — background is 6% of total loss |
| T1b class-weighted CE | class term | works (class spread 5/10 → 7/10 predicted), mAP unmoved |
| T2-bias + RetinaNet prior | head init | **washed out** — converged objectness identical to baseline |
| T2a 4-conv head tower (+7M) | head capacity | **refuted** — didn't even lower the TRAIN loss |

Recall sits at 11.8–12.6% across all of them: one noise band. Every arm converges to the
same objectness equilibrium (gap ≈ +0.25, std ≈ 0.24/0.33, means ≈ −1.57/−1.83).

## The measured constraint — read this before proposing anything

`scripts/fpn_neighbor_separation.py` splits background into cells **8-adjacent to a
positive** ("ring") vs everything else. On the T2a e12 logits (and it replicates on the
T2-bias arm to 3 decimals):

| population | n | mean objectness logit | std |
|---|---|---|---|
| positive | 34,341 | **−1.5818** | 0.249 |
| ring (adjacent) | 192,747 | **−1.6035** | 0.261 |
| far background | 6,539,616 | −1.8509 | 0.329 |

    ring − far  = +0.247     the head KNOWS where the objects are
    pos  − ring = +0.022     it CANNOT split the assigned cell from its neighbour
    ring sits 92% of the way from far background to positive

**The head has solved localization at neighbourhood resolution.** On top of that it is
being asked to pick which one of ~6 cells in the neighbourhood is "the" assigned cell. For
a 2–5px VisDrone object on the 56² P3 grid those cells see essentially the same receptive
field — they are **near-identical inputs, and no function of them can rank them apart**.

That is why capacity, initialization and loss weighting all washed out: none of them
changes the fact that the task as posed is unlearnable. It also explains the pinned mAP
mechanically — **5.6 ring cells per positive, all within 0.022 of the positives**, so the
top of the score-ranked detection list is a near-miss-duplicate avalanche that no
confidence threshold can clear.

**Corollary: stop tuning the network. The next change must be to WHAT IS ASKED OF IT.**

## The hypothesis

Make the ring cells positive too — **multi-anchor-per-GT / FCOS-style center sampling**.

1. It stops demanding an impossible discrimination.
2. It multiplies positives ~6.6×, which the T1a measurement says the loss can absorb
   (positives already carry 75% of objectness loss; this is not a balance regression, but
   see Risk 1).
3. Near-duplicate detections become **NMS-mergeable clusters** instead of false positives.

## ❌ BITE 0 RESULTS — measured 2026-07-19. THE HYPOTHESIS IS REFUTED. DO NOT BUILD IT.

All three sub-measurements were run on existing artifacts (no training, no GPU, no Lean).
Every one of them says no, and 0c says the doc is asking the wrong question entirely.
**Measure-before-build pays off a fifth time — this saved the 8.3 GB re-encode and a
~4.5 h training run.**

### 0a — the oracle does not move (`visdrone_fpn_coverage.py --center-sampling`)

Two ceilings, and the second is the one that matters. `encodable` = the GT keeps ≥1 slot
after collisions (today's 88.2% metric). `reachable` = it keeps ≥1 slot that can actually
**decode** an IoU≥0.5 box — because `cx=(j+σ(tx))/g` confines a cell's predicted centre
strictly inside itself, so a ring cell whose neighbour owns the GT centre cannot emit a box
on that GT at all. Making a cell positive that cannot represent its target does not add a
detection; it trains a guaranteed false positive.

Full train split, 343,204 GT boxes:

| assignment | slots/GT | encodable | reachable (naive) | encodable | reachable (priority) |
|---|---|---|---|---|---|
| baseline (1 cell/GT) | 0.88 | 88.2% | **88.2%** | 88.2% | **88.2%** |
| FCOS: centre inside GT box | 1.90 | 88.3% | 88.3% | 89.5% | **89.5%** |
| radius 1 → 3×3 | 5.80 | 86.3% | **68.9%** | 88.2% | **88.2%** |
| radius 2 → 5×5 | 12.64 | 84.5% | **51.4%** | 88.2% | **88.2%** |

*naive* = today's last-write-wins encoder; *priority* = own-centre beats any other GT's ring
(the best case, tested so the hypothesis is not judged on an incidental encoder detail).

- **Radius center sampling buys exactly 0.0 points** in the best case — the centre cell was
  already the only reachable one, so protecting it reproduces the baseline exactly.
- With the encoder **as actually written** it is *destructive*: −19 to −37 points of
  reachable recall, because ring cells steal neighbouring GTs' centre cells in
  53-objects/image scenes.
- FCOS "centre inside box" is near-degenerate here (2–5px objects vs 8px P3 cells) and buys
  at most **+1.3 points**.

Per this doc's own stopping rule — *"if the oracle does not move, stop"* — **stop.**

### 0b — ring boxes do not merge, and mostly cannot even be right

Measured on the T2a e12 logits, 270,838 ring cells:

| quantity | mean | median | p90 | frac ≥ NMS 0.5 |
|---|---|---|---|---|
| IoU(ring box, centre box) | 0.125 | 0.063 | 0.338 | **0.41%** |
| IoU(ring box, GT) | 0.104 | 0.054 | 0.296 | **0.71%** |

Risk 2 was the right thing to worry about and it is fatal: **99.6% of ring boxes would
survive NMS as separate detections.** Claim 3 of the hypothesis is false — the duplicates
do not become mergeable clusters, they become more false positives.

Geometry bound (best box a ring cell could *ever* emit, centre-in-cell, w/h free):

    span 0.0  (today,  cx=(j+σ)/g)          only 30.6% of ring cells can reach IoU≥0.5
    span 0.5  (YOLOv5, cx=(j+2σ−0.5)/g)          80.8%

So center sampling is **not a standalone change**: without also widening the centre range to
YOLOv5's `2σ−0.5`, ~69% of the new positives are trained toward boxes they cannot represent.

### 0c — duplicates are not what costs the mAP, and the real constraint is elsewhere

`fpn_duplicate_oracle.py`, T2a e12, IoU@0.5. ORACLE = keep only the single best correct
detection per GT and drop everything else — perfect duplicate suppression *and* perfect FP
rejection, unreachable by any real detector:

| variant | dets/img | recall | ca-AP | mAP |
|---|---|---|---|---|
| baseline IoU-NMS@0.5 | 942.7 | 0.1180 | 0.0009 | 0.0001 |
| centre-NMS r=1 | 582.2 | 0.0812 | 0.0006 | 0.0001 |
| centre-NMS r=2 | 235.3 | 0.0325 | 0.0003 | 0.0000 |
| **ORACLE 1-det-per-GT** | 4.2 | 0.0914 | 0.0914 | **0.0232** |

Duplicate suppression *lowers* recall at IoU 0.5 — the higher-scoring detection in a
neighbourhood is usually not the one matching the GT, so clustering deletes true positives.
And perfect duplicate handling plus perfect FP rejection still caps mAP at **0.0232**.

**The binding constraint: the detector emits a correct-class IoU≥0.5 box for only 9.1% of GT
(11.8% class-agnostic) — against an 88.2% encodability ceiling.** Ranking, assignment and
duplicates are all downstream of the fact that the right box is simply not in the output.

### The reframing — it is box PRECISION, not assignment

Rescoring the same logits at looser IoU:

| TP IoU threshold | 0.50 | 0.25 | 0.10 |
|---|---|---|---|
| class-agnostic recall | 0.1180 | 0.3998 | **0.6804** |
| ORACLE mAP ceiling | 0.0232 | 0.0620 | 0.0768 |

**The detector already finds 68% of the objects.** It puts a box in the right place and
cannot make it precise enough to clear IoU 0.5 on a 2–5px target, where a one-pixel error
is fatal. That is consistent with — and a better explanation of — the `ring−far = +0.247`
separation: the head localizes to *neighbourhood* resolution because that is all the
448px-downsampled input retains.

Note this also inverts Risk 4: **P2/resolution is not a deferred Tier 3 nicety, it is the
constraint.** VisDrone at 448px destroys the pixels the box regressor needs; a pedestrian
~20px in the source image is ~6px after the resize.

### Suggested next levers (measure first, as always)

1. **Input resolution / P2 scale.** The only lever the 0c curve supports. Cheapest probe:
   re-encode + eval at 768px or add a stride-4 P2 level, and watch recall@0.5, not mAP.
2. **Report mAP@0.25 alongside mAP@0.5** so the demo has a metric with signal in it while
   resolution work lands. At 0.5 every arm reads 0.0001 and the thread is flying blind —
   four levers were judged against a metric that was saturated at zero.
3. Center sampling stays dead **unless** resolution first lifts recall@0.5, at which point
   re-run 0b: bigger objects make ring boxes genuinely mergeable and the calculus changes.

### Incidental: a real bug in `fpn_neighbor_separation.py` (fixed, verdict unchanged)

It read `val.bin` from byte 0, never skipping the 4-byte `<I` record-count header that
`process_split_fpn` writes — shifting every target by one float32 = one cell along `j`, so
each positive's right-hand neighbour was labelled the positive. **This did not change its
verdict** (`pos−ring` +0.0217 → +0.0250 corrected; see `fpn_neighbor_align_check.py`), and
the C loader `lean_f32_load_voc_fpn` reads the header correctly, so **training was never
affected**. Fixed anyway.

## ⚠️ BITE 0 — PRE-MEASURE. Do NOT write codegen first.  ✅ DONE — see results above.

Measure-before-build has now paid off four times in this thread (T1a refuted on paper,
T1b's ceiling predicted, T2-bias and T2a both explained by one measurement). Do the same
here. **All of bite 0 is numpy on existing artifacts — no training, no GPU, no Lean.**

**0a — oracle recall.** Extend `scripts/visdrone_fpn_coverage.py` with a center-sampling
assignment (every cell whose centre falls inside the GT box, or within radius r of the box
centre, at the scale the size-threshold picks) and report the achievable recall@0.5 ceiling
the way it already reports the 88.2% encodability ceiling. **If the oracle does not move,
stop — the hypothesis is dead and this doc is wrong.**

**0b — the NMS question, which is the real risk.** With ~6.6× positives, adjacent cells
each emit a box. Those merge only if their mutual IoU exceeds the NMS threshold. **At 2–5px
box sizes IoU is brutal** — a one-pixel centre shift can drop IoU below 0.5. So measure, on
the *existing* e12 logits: take the boxes predicted by ring cells around each GT and
compute their pairwise IoU distribution against the positive cell's box. If those IoUs sit
below the NMS threshold, center sampling converts an FP avalanche into a *differently
shaped* FP avalanche and buys nothing without also fixing NMS (see Risk 2).

**0c — duplicate accounting.** From the same logits, quantify how much of the current mAP
loss is actually duplicate-driven: re-score with an oracle that keeps only the
highest-objectness detection within each GT's neighbourhood. That gives an upper bound on
what perfect duplicate handling alone would buy, independent of assignment.

Write results into this doc before touching `preprocess_visdrone.py`.

## The build, if bite 0 says go  — ⛔ BITE 0 SAID STOP. Retained for reference only.

*(0a moved the oracle by 0.0 points, 0b showed 99.6% of ring boxes never merge, and 0c
showed duplicates are not the cost. Do not execute the bites below. The encoder analysis
in bite 1 remains accurate and would be reusable if resolution work ever revives this.)*

**The good news: this is very likely a ZERO-CODEGEN change.** The multi-scale loss masks
positives by the **target's own objectness channel** (`%{pa}_m4`, sliced from the target at
`base+4`) — there is no FFI mask and no hard-coded positive count anywhere in
`emitAnchorYoloLoss`. Making more cells positive in the encoded target is therefore picked
up by the existing, FD-verified loss with **no emitter change at all**.

- **Bite 1 — encoder.** `encode_targets_fpn` in `scripts/preprocess_visdrone.py`. Each
  newly-positive cell needs its OWN box target: the DIoU block predicts
  `w = anchor·exp(tw)` and centres relative to that cell, so a ring cell must encode the
  same GT box as offsets from *its* cell origin and *its* anchor — not a copy of the
  centre cell's numbers. Getting this wrong is silent (training runs, boxes are just
  wrong), so `--smoke` it the way the original encoder was smoke-verified: decode the
  encoded targets back to boxes and assert they reproduce the GT.
- **Bite 2 — data.** Re-run `process_split_fpn --fpn` to a NEW dir (`data/visdrone_fpn_cs`)
  — 8.3 GB, do not clobber the existing set, which is the control's input.
- **Bite 3 — train + A/B.** New spec name (checkpoint prefix!) so the four existing arms
  survive. **Control = the T2-bias arm** (`…_448_wcls_pb__visdrone_`, recall 12.62%,
  mAP 0.0001) — same weights, same bias, same everything but the assignment.
- **Bite 4 — the diagnostic that matters.** Re-run `fpn_neighbor_separation.py`. Under
  center sampling the "ring" is now mostly positive, so the meaningful new question is
  whether the boundary between the enlarged positive region and the background outside it
  is separable. If `pos − ring` is still ≈0 at the NEW boundary, the problem has just moved
  outward one cell and the hypothesis is refuted.

## Risks, in the order they are likely to bite

1. **Loss re-balance.** 6.6× positives shifts the focal loss further toward positives
   (already 75% of the objectness term). This may need `λ_noobj` or focal α revisited — but
   **measure before adjusting**, exactly as T1a demanded. Do not pre-emptively "fix" it.
2. **NMS at tiny box sizes** (bite 0b). If ring boxes don't merge, consider a lower NMS IoU,
   or soft-NMS, or scoring by cluster rather than by cell. This is the most likely reason
   for the hypothesis to fail in a way that still leaves it *fixable*.
3. **Recall vs precision trade.** More positives should raise recall; mAP only moves if
   ranking improves too. Recall has been stuck at ~12% across four arms, so treat a recall
   jump as necessary-but-not-sufficient and always read mAP alongside it.
4. **The 88.2% encodability ceiling still stands.** Center sampling does not fix cell
   collisions on the 56² grid — that is a P2-scale question (Tier 3, deferred).

## Inherited gotchas — all of these have already cost time

- **`FPN_TOWER` selects the SPEC, so set it on `infer` too, not just training.** Forgetting
  it silently evaluates a *different arm's* checkpoint: every prefix/size/vmfb is
  self-consistent for the wrong spec, so no size check catches it. The tell is an epoch
  sweep with **zero variation between checkpoints**. `inferDump` now prints
  spec/prefix/expected-floats on line 2 — read it. (`runs/fpn_t2a_eval_watch_BROKEN_wrong_arm.log`
  is the failure kept for reference.)
- **Build/run the train step with `IREE_BACKEND=rocm`.** Without it `ireeCompileArgs`
  defaults to CUDA/sm_86 and drops
  `--iree-codegen-llvmgpu-use-reduction-vector-distribution=false`, and the loss reduce dies
  with `'func.func' op failed to distribute`. The flag exists for exactly that failure.
- **Any unbounded op (`exp`) + global-norm grad clip = latent `inf·0 = NaN`.** Cap at
  source; DIoU's `tw,th ≤ 8` cap is why the FPN arm trains at all.
- **An interim measurement confirming the MECHANISM is not evidence the LEVER works.**
  T2-bias moved its target metric 3–5× at e4 and gave every bit of it back by e12. Judge at
  e12, on mAP, against a named control.
- `lake build <exe>` mid-run does NOT kill a running trainer (new inode) — verified.
- Checkpoint prefix = sanitized `NetSpec.name`; give every arm a distinct name.

## The train → eval loop (current, for reference)

```
lake build yolov1-visdrone-fpn
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 ./.lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
# eval a checkpoint on the OTHER gpu while training runs (see run_fpn_pb_eval_watch.sh):
B=.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone_
cp ${B}_params_e12.bin ${B}_params.bin; cp ${B}_bn_stats_e12.bin ${B}_bn_stats.bin
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 ./.lake/build/bin/yolov1-visdrone-fpn infer \
  data/visdrone_fpn figures/yolo_fpn_pb_e12
<jax-venv>/bin/python3 scripts/yolo_map_visdrone.py figures/yolo_fpn_pb_e12/logits.bin \
  data/visdrone448/val.bin --grid 14 --fpn data/visdrone
<jax-venv>/bin/python3 scripts/fpn_neighbor_separation.py \
  figures/yolo_fpn_pb_e12/logits.bin data/visdrone_fpn/val.bin
```
`<jax-venv>` = `/home/skoonce/lean/claude_max/lean4-jax/.venv`. ~22 min/epoch at tower=0
(34 at tower=4) on gfx1100, + ~7 min one-time compile; checkpoints every 2 epochs. GT for
mAP comes from the single-box `data/visdrone448/val.bin`.

## Diagnostics available

| script | answers |
|---|---|
| `fpn_neighbor_separation.py` | is the cap the assignment? (positive vs ring vs far) |
| `fpn_loss_breakdown.py` | where does the loss actually go? (box/obj/cls, pos vs neg) |
| `fpn_obj_separation.py` | objectness AUC + logit spread + class-head collapse |
| `fpn_class_freq.py` | encoded-target class frequencies + weight literals |
| `visdrone_fpn_coverage.py` | encodability ceiling per assignment scheme (`--center-sampling` = 0a) |
| `fpn_prior_bias_check.py` | did a prior-bias init land on the right channels? |
| `fpn_ring_boxes.py` | 0b: do ring boxes merge under NMS? + the centre-reachability bound |
| `fpn_duplicate_oracle.py` | 0c: mAP under centre-NMS and a 1-det-per-GT oracle; `--iou` sweeps |
| `fpn_neighbor_align_check.py` | header-alignment control for `fpn_neighbor_separation.py` |
