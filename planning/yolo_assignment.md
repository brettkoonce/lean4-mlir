# yolo_assignment.md — target assignment (detection-infra brick #4)

> # ⛔ RESOLVED 2026-07-22 — EVERY REFUTATION IN THIS DOC WAS MEASURED ON SCRAMBLED DATA
>
> `lean_f32_shuffle` permuted images by a full record but swapped only a hardcoded 4 bytes of
> each label, so the detector trained on mismatched image/target pairs on every epoch and
> could only ever learn the marginal target distribution. **That is why mAP was 0.0001 for
> every lever: no lever could have worked.** Fixed ⇒ mAP 0.0001 → **0.1167**, recall 0.118 →
> **0.7353**. See `planning/yolo_scoring.md`'s banner and `planning/jax_gradient_oracle.md` §0.
>
> The four refutations are **void as statements about the levers** — they measured a broken
> data path, not T1a/T1b/T2-bias/T2a. The "target assignment is the constraint" thesis and
> the ring/neighbour analysis are likewise void. Re-measure before re-adopting or re-refuting
> any of it. The numpy probes themselves are still sound instruments.

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

### ⚠️ SUPERSEDED — suggested next levers (measure first, as always)

*(The resolution probe below refuted lever 1 and overturned the reframing this list rests
on. Lever 2 — reporting mAP@0.25 — still stands. Kept for the record.)*

1. **Input resolution / P2 scale.** The only lever the 0c curve supports. Cheapest probe:
   re-encode + eval at 768px or add a stride-4 P2 level, and watch recall@0.5, not mAP.
2. **Report mAP@0.25 alongside mAP@0.5** so the demo has a metric with signal in it while
   resolution work lands. At 0.5 every arm reads 0.0001 and the thread is flying blind —
   four levers were judged against a metric that was saturated at zero.
3. Center sampling stays dead **unless** resolution first lifts recall@0.5, at which point
   re-run 0b: bigger objects make ring boxes genuinely mergeable and the calculus changes.

## ❌ RESOLUTION PROBE RESULTS — measured 2026-07-19. RESOLUTION IS REFUTED TOO.

`scripts/fpn_resolution_probe.py`, numpy on the existing T2a e12 logits — no training,
no GPU, no Lean. **Measure-before-build pays off a sixth time: this cancels the 768px
re-encode + ~4.5 h train.** It also **overturns bite 0c's reframing above**: the
constraint is not box precision, and the "unlearnable neighbour" story is not what is
costing the recall.

### The headline — resolution's entire headroom is 5 points; ranking's is 24

Recall of encoded positives with BOTH gates applied (box must clear IoU 0.5, conf must
clear the scorer's top-1000 cut):

| scenario | box ok | in top-k | **BOTH** |
|---|---|---|---|
| today (measured) | 33.06% | 14.10% | **9.19%** |
| resolution 768, optimistic | 51.18% | 14.10% | **12.97%** |
| PERFECT boxes (∞ resolution), ranking as measured | 100% | 14.10% | **14.10%** |
| PERFECT ranking, boxes as measured | 33.06% | 100% | **33.06%** |

**Resolution taken to perfection buys +4.9 points. Fixing the ranking buys +23.9.**
Resolution multiplies a term that is gated downstream, so it cannot pay.

### The probe reproduces the measured pipeline — this is why the numbers can be trusted

The FPN encoder sees 62.7 positives/img; the scorer's GT sees 46.0/img, so encoded
positives = 1.362 × scorer GT. Converting: 9.19% of encoded positives = **12.51% of
scorer GT, against a measured end-to-end class-agnostic recall of 11.80%.** The probe
predicts the real pipeline to within 0.7 points, which is what licenses the
counterfactual rows above.

### Why 0c's reframing was wrong

**The right box IS in the output.** At its own assigned cell the detector emits an
IoU≥0.5 box for **33.06%** of encoded positives (45.0% of scorer GT). Bite 0c concluded
"the right box is simply not in the output" — but it read that off an oracle applied to
an already **top-1000-truncated** detection list, so its 0.0232 "unreachable" ceiling is
a truncation artifact, not a property of the detector. The correct box exists and is
discarded for its *rank*: median rank **2256 of 12348** candidate slots. All 12348 slots
clear the scorer's `conf>=0.001` filter, so top-k is the sole binding cut.

| top-k | IoU≥0.5 kept | + right class |
|---|---|---|
| 500 | 5.05% | 3.91% |
| **1000 (today)** | **9.19%** | **7.07%** |
| 4000 | 20.85% | 14.96% |
| 12348 (all) | 33.06% | 19.19% |

*This is not a licence to raise top-k* — VisDrone protocol caps detections per image, so
the cap is legitimate. It is a measurement that the **confidence ordering is the failure**.

The truncation-artifact claim rests on the table above, which is exact: 0c's oracle chose
from a list that had already discarded three quarters of the correct boxes, so it could
not have exceeded what the cut left behind. A direct re-scoring of 0c's oracle at
`--topk 12348` was attempted and **abandoned, not completed** — `_nms_per_class` is
O(n²) in pure Python and does not finish at 12k dets/img within 90 min. If a confirming
mAP number is wanted, vectorize the NMS first.

### Why the "unlearnable neighbour discrimination" story was wrong

For the 11,354 positives that do emit an IoU≥0.5 box, the slots outranking them split:

| category | mean count outranking | share | census share |
|---|---|---|---|
| **far background** | **3230** | **84.5%** | 91.3% |
| ring (adjacent to a positive) | 555 | 14.5% | 8.2% |
| other positives | 36 | 0.9% | 0.5% |

Ring cells *are* over-represented relative to census (14.5% vs 8.2%) — the `ring−far =
+0.247` separation is real. But they are only 555 of the 3821 slots in the way. **Perfect
ring suppression would still leave a correct box at rank ≈3266, nowhere near the top
1000.** The binding failure is ordinary object-vs-background discrimination: overall
objectness **AUC = 0.741** across 12,348 candidates with ~63 positives.

That also re-opens the four refuted levers. They were aimed at objectness and judged
against an mAP saturated at zero; the doc then explained their failure with a
pos-vs-ring argument that this measurement says was never the operative constraint.

### Incidental findings worth acting on

- **Width regression is twice as bad as height.** `|log w_pred/w_gt|` mean **0.478** vs
  `|log h|` **0.241** — width is off by a factor of e^0.478 ≈ 1.61 on average. Substituting
  the GT centre (leaving predicted w,h) reaches only 59.5% at IoU≥0.5, so size is the
  *harder* of the two box terms, not the centre. Suspect the P3 width anchors.
- **The eval GT is silently truncated.** `pack_raw_boxes` caps at `MAX_BBOXES = 56`, but
  VisDrone val averages **70.7** boxes/img (median 65, max 317). **59.1% of val images
  exceed the cap and 34.9% of all val GT boxes are dropped** — every recall/mAP number in
  this thread is computed against 65% of the GT, biased toward the first 56 boxes in
  annotation order and likely *inflating* recall (dense images are the hardest). Relative
  comparisons between arms are unaffected (same truncated GT), so the refutations stand,
  but the absolute numbers are not VisDrone protocol.

### Suggested next levers, revised

1. **Objectness discrimination is the only lever with real headroom** (+23.9 pts vs
   resolution's +4.9). AUC 0.741 → the target is separating positives from *far
   background*, not from neighbours.

   **Measured across all four arms (`fpn_obj_separation.py`, e12) — nothing moved it:**

   | arm | baseline | T1b wcls | T2-bias | T2a tower |
   |---|---|---|---|---|
   | objectness AUC | 0.7417 | 0.7400 | 0.7393 | 0.7410 |

   A spread of 0.0024 across a class-weighting change, a prior-bias init and +7M
   parameters of head capacity. This is the same "every arm converges to one
   equilibrium" finding as the objectness-logit table above, now measured on the
   quantity that actually gates recall — and it means the lever is genuinely
   **unexplored**, not already-tried. Nothing built so far targets it.
2. **Fix the `MAX_BBOXES = 56` eval cap** before trusting any further absolute number.
3. Resolution / P2 stays dead as a *primary* lever. It remains the right *second* move
   once ranking is fixed — at perfect ranking, box precision becomes the binding gate
   (33.06% ceiling), and that is exactly where section C's 768px → 51.18% would pay.

## ✅ OBJECTNESS-DISCRIMINATION PROBE — measured 2026-07-19. RESCORING REFUTED; THE LEVER IS NAMED.

`scripts/fpn_objectness_readout_probe.py` + an end-to-end A/B through the real scorer.
The resolution probe said ranking was the lever with the headroom; this probe asks
whether that ranking can be fixed by **rescoring** (cheap, no retrain) or only by
**retraining** (expensive) — and it answers the question by identifying the specific
defect.

### Objectness is ANTI-correlated with box quality

There are two distinct ranking jobs, and no single existing channel does both:

| score | object vs background | box quality (IoU≥0.5 among positives) |
|---|---|---|
| `sigmoid(obj)` alone | **0.7504** | **0.3497** ← worse than chance |
| `max softmax(cls)` alone | 0.4172 | **0.7447** |
| production `obj × clsprob` | 0.6373 | 0.6713 |

**The head is systematically more confident about positives whose boxes are wrong.**
And the class head — which has nothing to do with localization — is accidentally the
best box-quality predictor the score has. The production `obj × clsprob` is a
compromise that wins neither job outright but is the best available on the one that
matters.

### Rescoring is exhausted — the readout ceiling is +0.4 points

Fitting rankers on the head's own 16 output channels, trained on one set of images and
scored on a disjoint set. `USABLE` = positive AND IoU≥0.5 AND inside the top-1000; it is
the column that predicts recall.

| ranker | positives in top-1000 | **USABLE** |
|---|---|---|
| production `obj × clsprob` | 15.29% | **9.71%** |
| `sigmoid(obj)` alone | 28.74% | 5.14% |
| logistic regression, USABLE objective | 16.53% | 9.44% |
| MLP (16-32-1), USABLE objective | 15.62% | **10.08%** |

**Best readout 10.08% vs production 9.71%.** No function of the head's output channels
meaningfully beats what is already deployed. This is a **feature failure, not a readout
failure** — the fix has to be a training change.

### The trap this probe fell into, recorded because it nearly shipped

Ranking by `sigmoid(obj)` alone nearly **doubles** the positives reaching the top-1000
(15.29% → 28.74%) and looks like a free win. It is not: those extra positives carry
unusable boxes, so USABLE *halves* (9.71% → 5.14%). Measured end-to-end through the real
scorer, `--score obj` takes **recall@0.5 from 0.1180 down to 0.0699** and ca-AP from
0.0009 to 0.0002.

This is the doc's own gotcha — *"an interim measurement confirming the MECHANISM is not
evidence the LEVER works"* — and the intermediate metric ("positives in top-k") was the
thing that lied. The end-to-end A/B is what caught it. `--score obj` is retained in
`yolo_map_visdrone.py` as a **diagnostic, defaulting off**; the control run reproduces
mAP 0.0001 / recall 0.1180 exactly.

### The named lever: IoU-aware objectness

AUC 0.3497 is not a small miss, it is a **sign error in what objectness is being asked to
predict**. Positives are all trained toward a constant target of 1.0 regardless of whether
the box that slot emits is good, so nothing ever teaches objectness to rank a good box
above a bad one — and the measurement says it has settled on the opposite.

The standard fix is to make the objectness target the **achieved IoU** of the predicted
box rather than a constant 1.0 (YOLOv5-style IoU-aware objectness, or quality focal loss /
GFL). That attacks the 0.3497 directly, and it is the first lever in this thread aimed at
a defect that was measured rather than guessed.

Two reasons to think it is tractable here:

- It is a **target** change, not an architecture change. The multi-scale loss already
  masks positives by the target's own objectness channel (`%{pa}_m4`), so the value in
  that channel is what would change.
- The four refuted arms all left objectness AUC at 0.7393–0.7417 because none of them
  changed *what objectness is trained to mean*. This one does.

**Measure first, as always:** before building, check on existing logits what the objectness
AUC-vs-quality would be under an IoU target — i.e. how much of the 0.3497→0.75 gap is
recoverable from the boxes the detector already emits. That is another pure-numpy probe.

## ⚠️ IoU-TARGET PROBE — measured 2026-07-19. THE RUN IS NOT WORTH SPENDING AS FRAMED.

`scripts/fpn_iou_target_probe.py`. Gates the IoU-aware-objectness run above.

### The achieved IoU IS predictable — that part of the theory holds

Fit on positives in train images, scored on positives in disjoint test images, with the
objectness channel excluded (it is the thing being retrained, so feeding it in is circular):

| predictor of IoU≥0.5 among positives | AUC |
|---|---|
| `sigmoid(obj)` — today's signal | 0.3497 |
| `max softmax(cls)` — best single channel | 0.7447 |
| **MLP on all non-objectness channels** | **0.8602** |

### But production already captures the benefit accidentally

The controlled test: same features, same network, same capacity, **only the target differs.**

| ranker | positives in top-1000 | **USABLE** |
|---|---|---|
| production `obj × max softmax(cls)` | 15.29% | **9.71%** |
| MLP, BINARY target (today's objectness semantics) | 26.57% | 3.12% |
| MLP, **IoU target** (the proposed change) | 14.46% | **9.39%** |
| ORACLE object/background | 100% | **33.65%** |

**Like for like, the target change is worth +6.27 points** (3.12% → 9.39%) — the theory is
right. But production's class multiplier is *already* a crude quality signal (AUC 0.7447),
so measured against what is deployed the change is worth **−0.32 points**. The run would be
buying something the scorer already gets for free.

### Why it cannot break past ~10%, and what that means

Look at the `positives` column against `USABLE`. Every score faces the same trade: admit
many positives and they are quality-blind (binary target: 26.57% in, 3.12% usable), or
admit few and good (IoU target: 14.46% in, 9.39% usable). Both land near 9–10% USABLE.
**Breaking past that needs admitting MORE positives while keeping them good — which is
object/background discrimination (AUC 0.7504), not quality ranking.** IoU targeting does
not touch that axis, and in fact *hurts* it: it downweights hard positives, so raw
positives-in-top-k falls from 26.57% to 14.46%.

The 23.94-point headroom to the oracle is a **feature** problem. Four arms already failed
to move it.

### ⚠️ Correction to the resolution verdict above — one axis was never tested

The resolution probe's lever table held `in top-k` FIXED at 14.10% for the 768px row, i.e.
it assumed resolution changes box precision only and **not** objectness AUC. That was
conservative on the ranking axis, and this probe shows the ranking axis is the one that
matters. **Whether resolution improves object/background AUC is untested and cannot be
tested from 448px logits** — bigger objects are plausibly easier to *detect*, not just to
box. So "resolution refuted" stands only for its box-precision mechanism; resolution
remains the one untested lever that plausibly touches the binding feature constraint.

### The trap this probe fell into first (recorded — it is subtle)

Ranking by `obj × TRUE IoU` looks like a clean "oracle quality" ceiling and reported a
tidy 33.65%. It is **label leakage**: `iouv` is 0 on every background slot, so that score
is an oracle object/background detector wearing a quality costume — it reproduced the
oracle row to the decimal. True IoU is not a usable stand-in for a quality signal. The
controlled same-features/different-target design above is the fix.

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
| `fpn_resolution_probe.py` | the resolution probe: box-error decomposition, resolution transfer, rank survival, and the lever comparison |
| `fpn_objectness_readout_probe.py` | can rescoring fix the ranking? readout ceiling, obj-vs-quality AUC split, USABLE metric |
| `fpn_iou_target_probe.py` | gates the IoU-aware-objectness run: IoU predictability + same-features/different-target comparison |
| `yolo_map_visdrone.py --score obj` | diagnostic only — objectness-only ranking, measured WORSE end-to-end (recall 0.1180→0.0699) |
