# yolo_scoring.md — the detector's scoring design (detection-infra brick #5)

> # ⛔ RESOLVED 2026-07-22 — READ THIS BEFORE ANYTHING BELOW
>
> **The detector was training on mismatched image/target pairs.**
> `ffi/f32_helpers.c: lean_f32_shuffle` permuted images by a full record but swapped a
> **hardcoded 4 bytes** of each label — one float, a classification scalar. The FFI had no
> label-stride parameter. An FPN label record is 740,880 bytes. So on every epoch the images
> were shuffled and the targets were not, and the only learnable thing was the marginal
> target distribution.
>
> Fixed (`label_stride` threaded through `F32.shuffle` from `dio.labelBytesPerRecord`):
>
> | | before | after |
> |---|---|---|
> | mAP@0.5 | 0.0001 | **0.1167** |
> | recall | 0.118 | **0.7353** |
> | 8-img overfit objectness | 27.550 | **0.526** |
> | 12-ep train loss | 300.9, flat from e4 | **112.62**, still descending |
>
> **Everything below this banner was measured on scrambled data.** The scoring analysis, the
> objectness-AUC story, the "converged equilibrium at p≈0.14", the readout/resolution/
> IoU-target probes and their lever recommendations were all describing the bug rather than
> the detector. The *methods* remain sound and the scripts are still useful; the *conclusions*
> are void. Nothing here should be actioned without re-measuring on the fixed shuffle.
>
> Full write-up: `planning/jax_gradient_oracle.md` §0. The probe that found it, and which
> should be run first on any future "cannot descend" report: set LR to 0 so the parameters
> are provably frozen, run several steps, and check the loss is constant. It swung 12%.

Where the VisDrone detector goes from here. Written 2026-07-21, after a standalone
PyTorch reference was built (`visdrone/`) and re-measured the whole thread.

**This doc supersedes the "suggested next levers" lists in `yolo_assignment.md`.**
Those lists were written without a working detector to compare against; every one of
them was chasing a sub-point improvement in a regime where the detector was ~3900×
below an ordinary result. Read `yolo_assignment.md` for the six refutations (they
stand, as relative comparisons) and this doc for what to do.

Prereqs: `visdrone/README.md` (the reference numbers), `yolo_assignment.md`,
`yolo_fpn.md`, memories `yolo-fpn-thread` + `visdrone-fetch-and-wsa`.

## Why this doc exists

The thread spent four training runs and eight numpy probes arguing about assignment,
resolution, duplicates, ranking and IoU-aware targets. All of that was conditioned on
"the pipeline is correct and the detector is merely mediocre." A standalone reference
was built in one afternoon and the condition turned out to be false.

| arm | recipe | init | ep | recall | mAP@0.5 |
|---|---|---|---|---|---|
| **Lean FPN** (best of four) | no aug, 448 squash | ImageNet R34 | 12 | 0.118\* | **0.0001** |
| YOLOv8s | no aug, 448 letterbox | random | 12 | 0.164 | 0.114 |
| YOLOv8s | no aug, 448 squash | random | 12 | 0.197 | **0.140** |
| YOLOv8s | no aug, 448 letterbox | COCO-det | 12 | 0.240 | 0.192 |
| YOLOv8s | full aug, 640 | COCO-det | 100 | 0.400 | **0.391** |

\* class-agnostic, and against GT already cut 34.9% by the `MAX_BBOXES = 56` truncation
— i.e. flattered relative to every row below it.

Row 3 is the one that matters: matched to the Lean arm on augmentation, resolution,
resize convention and epoch budget, and **strictly worse on initialization** (random
weights vs a pretrained ImageNet backbone). It still lands **1400× higher**.

**So the gap is not the dataset, not the recipe, and not the initialization.**

## The measured constraint — the score is nearly a constant function

`visdrone/compare_scoring.py`, both detectors on one architecture-neutral metric (a slot
is POSITIVE iff its cell centre falls inside a GT box), 60 val images, full annotations:

| detector | score | AUC | pos mean | bg mean | separation | in top-1000/img |
|---|---|---|---|---|---|---|
| Lean FPN | `sigmoid(obj)` | 0.5505 | 0.1471 | 0.1407 | 0.17 σ | 10.2% |
| Lean FPN | `obj × clsprob` (production) | 0.5311 | 0.0497 | 0.0480 | 0.11 σ | 9.1% |
| YOLOv8s (scratch, no aug, 448) | `max sigmoid(cls)` | **0.7385** | 0.0673 | **0.0025** | **3.18 σ** | **57.3%** |

- Lean objectness never exceeds **0.2417 on any of 740,880 slots**; production never
  exceeds 0.1121. A working detector routinely emits 0.9.
- **0.14 is the base rate.** Positives are 11.06% of slots. A head emitting ≈0.14
  everywhere is predicting the marginal P(object) and almost nothing conditional on the
  pixels. That reframes `yolo_assignment.md`'s "converged objectness equilibrium at
  p≈0.14": it is not a fixed point of the loss to be respected, it is a head that did
  not learn.
- **The class multiplier makes ranking worse** (0.5505 → 0.5311).

## The hypothesis — one score doing two jobs vs two scores doing neither

The structural diff between the two detectors' scoring:

| axis | Lean FPN | YOLOv8 |
|---|---|---|
| score channels | objectness (1) **+** class softmax (10) | class sigmoid (10), no objectness |
| class activation | softmax over 10 classes | per-class sigmoid |
| class loss support | **masked to positive cells** | **all anchors**, target 0 on background |
| objectness target | constant 1.0 | n/a |
| assignment | static best-anchor, 1 cell/GT | dynamic TAL top-k by alignment |
| ranking score | `sigmoid(obj) × max softmax(cls)` | `max sigmoid(cls)` |

`yolo_assignment.md` already measured the consequence from the inside: objectness is
decent at object-vs-background (0.7504 under its narrower positive definition) and
**anti-correlated with box quality (0.3497)**; the class head is the reverse (0.4172 /
0.7447). Neither channel does both jobs, and their product does neither well.

**The named suspect: nothing in the Lean detector's class score is ever trained on
background, and its objectness is trained to a constant regardless of box quality.**
Its IoU-target probe tested the target-shape half of this as a *rescoring of frozen
channels* and found +6.27pt like-for-like but −0.32pt vs production. It never tested
training the score on background at all, which is the half the reference says matters.

## ⚠️ BITE 0 — PROVE IT IN PYTORCH. Do NOT touch MlirCodegen first.

Demo-first protocol, and the direction matters:

> **Cripple the detector that works. Do not fix the one that doesn't.**

Every step is a small delta from a known-good baseline, so the epoch the number craters
names the cause. Fixing the broken arm instead means 4.5 h per attempt with no signal
until the end — which is exactly how the thread burned four runs.

Control = `visdrone/runs/scratch_squash_12ep_448` (**mAP 0.140**): YOLOv8s, random init,
no aug, 448 squash, 12 ep. Every rung below is that run plus one change, ~25 min each on
one 7900 XTX. All of it is a patch to ultralytics' `v8DetectionLoss` + `Detect` head, not
a from-scratch detector.

### ❌ BITE 0 RESULTS — measured 2026-07-21. THE HYPOTHESIS IN THIS DOC IS REFUTED.

`visdrone/train_cripple.py`, all arms 12 ep on the squash dataset, random init, no aug,
448. Predictions were written down first; they were wrong.

| rung | change | predicted | **measured** | cost |
|---|---|---|---|---|
| L0 | control, unmodified | 0.140 | **0.1399** | 1.00× |
| L3 | target scores binarized to 1.0 (no quality awareness) | < 0.02 | **0.1164** | 1.20× |
| L4 | static best-cell assignment, 1 slot/GT, last-write-wins | < 0.01 | **0.1144** | 1.22× |
| L2 | class loss masked to foreground (no background training) | **< 0.03** | **0.0978** | **1.43×** |
| L2+L4 | both together | — | **0.0445** | **3.14×** |
| | *(Lean FPN, for scale)* | | *0.0001* | *1399×* |

**No rung craters.** The two strongest ingredients together cost 3.14×, against a gap of
1399×. They do compound superadditively (1.43 × 1.22 = 1.75 expected, 3.14 observed), so
the scoring design is a real contributor — but it is ~0.2% of the gap, and the remaining
unmodelled pieces (3 anchors/cell, a separate objectness head, focal loss, DIoU-vs-DFL)
would have to contribute ~450× between them to close it. Nothing measured here suggests
they would.

**Per this doc's own stopping rule, stop: the Lean detector's problem is not its scoring
design, and the class-BCE port sketched in bite 2 would have bought ~1.4×.** That is
exactly what the ladder was for — it cost 2.5 GPU-hours and cancelled a codegen cycle.

### What is now ruled out

| candidate | verdict | evidence |
|---|---|---|
| dataset difficulty | ✗ | stock recipe reaches 0.391 |
| recipe (aug / resolution / epochs) | ✗ ~1.7× | matched-recipe arm still 1140× higher |
| initialization | ✗ | random init beats ImageNet-bootstrapped Lean by 1140× |
| encoder / data alignment | ✗ | 38,421 targets round-trip at 2.1e-09, 0 mismatches |
| aspect squash | ✗ (helps) | 0.140 squash vs 0.114 letterbox |
| **scoring design** | **✗ ~3×** | **the table above** |

### The redirect — it is the TRAINER, and FD verification does not cover this

Three independent signatures already in the record all say the Lean model **is not
fitting its own training data**, which is an optimization failure, not a design one:

1. Train loss moves 315 → 300.9 over 12 epochs, then flattens.
2. **+7.08M parameters (T2a tower) did not lower the TRAIN loss** (320.37 vs 319.24).
3. Objectness outputs ≈0.14 = the base rate, on positives and background alike — a head
   emitting a constant is a head receiving almost no usable gradient.

**FD verification proves `∂L/∂θ` matches `L`. It does not prove that `L` is the intended
loss, that the optimizer applies its updates correctly, that the LR/schedule is sane,
that BN running stats update properly, or that grad clipping is not eating the signal.**
Every FD probe in this thread is consistent with a trainer that computes a correct
gradient for a correct loss and then fails to descend.

**Next measurement — cheap, decisive, and never run in this thread: can the Lean trainer
OVERFIT A TINY SUBSET?** Train on 8–32 images for a few hundred steps and watch the train
loss.

- If it drives loss toward 0 and predicts those images near-perfectly, the trainer is
  sound and the failure is genuinely in generalization — go back to design questions.
- **If it cannot overfit 8 images, there is a bug in the trainer**, and the search space
  collapses from "detector design" to optimizer / LR / loss-scale / update path. That is
  where every remaining measurement should go.

This is the standard first check for "model won't fit" and it takes minutes, not the
4.5 h a full arm costs. It should have been rung 0 of the whole thread.

### ✅ OVERFIT PROBE RESULTS — measured 2026-07-21. THE TRAINER CANNOT FIT 32 IMAGES.

`visdrone/make_overfit_subset.py` slices 32 records VERBATIM out of
`data/visdrone_fpn/train.bin` (same encoder, same images, same targets — the only
variable is dataset size) into `data/visdrone_fpn_overfit`, train.bin == val.bin.
Run with the arm's own hyperparameters, only `epochs` changed:

    IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 FPN_TAG=overfit FPN_EPOCHS=200 \
      FPN_CKPT_EVERY=100 ./.lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn_overfit

| epoch | 1 | 25 | 50 | 100 | 150 | **200** |
|---|---|---|---|---|---|---|
| train loss (lr ×1) | 915 | 363 | 312 | 262 | 227 | **215** |

**200 epochs = 800 Adam steps over 32 images the model has now seen 200 times each,
and the loss converges at 215** (the cosine schedule reaches lr=0 by then, so this is
converged, not truncated). A 21.5M-parameter model that cannot memorize 32 images is
not a detector with a design problem.

For scale: the FULL 6471-image run converges at ~300. **On 200× less data with 16× more
passes it gets only 28% lower.**

The per-batch losses are frozen. Across four consecutive epochs the same four batches
report 220.5/163.7/399.4, 220.8/154.2/404.3, 221.2/161.8/394.4, 220.0/165.7/397.3 —
under 1% movement per batch over 16 gradient steps.

**But the update path is LIVE.** A second probe at 10× LR (`FPN_LR_MULT=10`) *diverges*
— 872 → 468 (e5) → 510 (e10) → 714 (e15) — so gradients do reach the parameters and do
move them. This is not a dead-gradient or disconnected-update bug. The optimization is
real, responsive to LR, and still cannot descend.

    lr ×1   glacial, converges at 215
    lr ×10  unstable, diverges by epoch 10

### ⚠️ THE NEXT MEASUREMENT — what is the loss FLOOR?

Everything above assumes the achievable loss is ≈0. **That has never been checked, and
if it is false the whole reading inverts.** If a PERFECT prediction scores ~215 on this
subset, then the model has already converged and the defect is a mis-specified loss —
a term that cannot be driven to zero — rather than a broken optimizer.

Cheapest possible probe, pure numpy, no GPU: feed the ENCODED TARGET IN AS THE
PREDICTION and evaluate the emitted loss (`scripts/fpn_loss_breakdown.py` already
replicates it from a dumped checkpoint). Report the floor per term (box / obj / cls).

- **Floor ≈ 0** ⇒ the trainer genuinely cannot descend. Next: Adam moment handling,
  the grad-clip interaction, LR schedule, BN in train mode.
- **Floor ≈ 215** ⇒ the loss is mis-specified and every "converged equilibrium" reading
  in `yolo_assignment.md` was measuring the floor, not the model. That would also
  explain why +7.08M parameters did not lower the train loss.

Do this before touching the optimizer. It is minutes of work and it decides which of
two very different investigations to run.

### ✅ FLOOR MEASURED 2026-07-21 — THE FLOOR IS ZERO. THE TRAINER CANNOT DESCEND.

`visdrone/make_perfect_logits.py` inverts the encoder to synthesize a perfect
prediction — `logit()` on the in-cell centre fraction, `log(w_rel/anchor)` on the size
(clipped at the emitter's own `tw ≤ 8` cap), ±12 logits for objectness and the one-hot
class — and feeds it to the already-validated `scripts/fpn_loss_breakdown.py`
**unchanged**, so the floor cannot disagree with the loss for a reason of its own
making.

On the same 32 images the probe converged on:

| term | perfect prediction | trainer at convergence |
|---|---|---|
| box | **0.000** | — |
| obj | **0.000** | — |
| cls | **0.000** | — |
| **TOTAL loss/img** | **0.000** | **215** |

Sanity indicators from the same run: mean IoU **1.0000**, mean p(obj+) **1.0000**, mean
p(obj−) **0.0000** — every term sits exactly at its optimum. That the synthesized logits
decode to IoU 1.0 against the targets is also an independent re-verification that the
emitted decode path matches the encoder.

**So the loss is well-specified and the achievable loss is 0.** The alternative reading
is dead: the model has NOT converged to a floor, it has stalled at 215 out of an
achievable 0.

### The problem statement, now tight

Everything below is simultaneously true and measured:

1. The loss is correct and its optimum is 0 *(this section)*.
2. The gradient matches the loss — FD-verified to ~1e-6 across neck, multi-scale loss,
   whole-detector backward, head bias and tower *(`yolo_fpn.md`)*.
3. The data the trainer is fed is correct — 38,421 targets round-trip at 2.1e-09
   *(`visdrone/check_bin_alignment.py`)*.
4. The update path is live and influential — 10× LR diverges *(overfit probe)*.
5. **And 800 Adam steps over 32 images leave the loss at 215.**

A correct gradient on a correct loss over correct data, with updates that demonstrably
move the parameters, that still cannot descend. The remaining surface is small:

- **Adam moment handling** — `m`/`v` init, bias correction, the eps placement, and
  whether the slots are being read/written to the right offsets. The Shampoo work
  reused these slots (`[[shampoo-demo]]`), so the plumbing is worth re-reading.
- **The grad-clip interaction.** Adam is scale-invariant to a *constant* gradient
  rescale, so `gradClipNorm=4.0` should be nearly inert — but a clip that varies
  per-step, or one that drives gradients toward the eps floor, is not. Measure the
  actual global norm against the 4.0 threshold; it has never been logged.
- **Weight decay**, applied to 21.5M params — check it is not overwhelming a small
  update, and whether it is coupled or decoupled.
- **BN in train mode.** The bootstrap emits `WARN: bootstrap BN stats size 87190688 ≠
  expected 68096; using zeros`. Benign for training in principle (batch stats are used),
  but it has never been verified that the BN forward actually uses batch statistics on
  this path rather than the zeroed running stats.

### ✅ CLIP PROBE — measured 2026-07-22. THE CLIP IS EXONERATED, AND "CANNOT DESCEND" IS TOO STRONG.

Rather than log `%gcnorm` (which would only say whether the clip is ACTIVE), added
`FPN_CLIP` and turned it OFF, which says whether it is CAUSAL. Tightest possible probe:
**8 images = ONE batch**, so every step is the same batch, 500 steps.

| step | 1 | 100 | 200 | 300 | 400 | **500** |
|---|---|---|---|---|---|---|
| clip = 4.0 (arm default) | 800 | 162 | 106 | 86 | 66 | **55.0** |
| clip = OFF | 800 | 187 | 126 | 103 | 79 | **66.0** |

**The clip is not the problem** — turning it off makes things marginally *worse*, which
is what Adam's scale-invariance to a constant gradient rescale predicts. Exonerated.

**And the trainer is NOT stuck.** On 8 images it descends 800 → 55, a 15× reduction, and
was still falling when the cosine schedule zeroed the LR. 3× LR descends faster still
(137.8 vs 165.6 at step 100). So the previous section's "cannot descend" is too strong
and is hereby weakened to: **it descends, slowly, and is step/LR-limited rather than
broken.** The correct statement is about RATE, not capability.

That does not close the gap — a working detector reaches useful mAP in 12 epochs on the
full split while this arm reads 0.0001 — but it does redirect again: the question is no
longer "what is broken in the update path" (clip, gradient, loss, data are all now
cleared) but **"why is this trainer's sample efficiency so much lower than a
conventional one's."**

In flight at time of writing: 8 images × 2000 steps at 1× and 3× LR, to establish
whether it reaches the ~0 floor at all or plateaus short of it. A plateau would still
indicate a residual defect; convergence to ~0 would mean the optimization is sound and
the whole gap is sample efficiency / schedule.

### ✅ THE RESIDUAL IS OBJECTNESS — measured 2026-07-22. AND CAPACITY DOES NOT FIX IT.

2000 steps on the SAME 8 images (pure memorization, no generalization required):

| | lr ×1 | lr ×3 |
|---|---|---|
| step 1 | 800 | 800 |
| step 1000 | 54.0 | 46.5 |
| **step 2000** | **36.0** | **32.4** |
| achievable floor | **0** | **0** |

4× the steps bought 1.5×. It plateaus around 32–36 against a floor of 0. So the
optimization is *not* merely slow — there is a real barrier well above the optimum.

Decomposing that residual (`infer` the of8long checkpoint on those 8 images →
`scripts/fpn_loss_breakdown.py`):

| term | loss/img | share |
|---|---|---|
| box | 6.16 | 17.4% |
| **objectness** | **27.55** | **77.8%** |
| cls | 1.72 | 4.8% |

**Box and class fit; objectness does not.** Box fell from ~193 (full run) to 6.2 and
class from ~99 to 1.7, but objectness is stuck at 27.6 — split roughly evenly between
positives (14.8) and negatives (12.8).

The asymmetry explains itself: box and class are **masked to positives**, so they only
have to fit ~45 cells/image. Objectness has to classify all **12,348**. The model can
memorize the boxes of 8 images and cannot learn which cells contain objects — even with
2000 passes over them.

**Head capacity is not the answer.** Re-running the same 8-image overfit with
`FPN_TOWER=4` (+7.08M params):

| step | 100 | 200 | 300 | 400 | **500** |
|---|---|---|---|---|---|
| tower=0 | 162.2 | 105.8 | 86.2 | 66.1 | **55.0** |
| tower=4 | 164.6 | 103.3 | 77.9 | 65.9 | **55.2** |

Identical to within noise, on a task that is pure memorization. This independently
confirms T2a's original refutation and removes the confound that made it arguable —
T2a was judged on mAP after 12 epochs on the full split, where "didn't help" could have
meant "didn't generalize". Here it cannot even help the model MEMORIZE. The 1×1 head's
capacity is not the constraint.

**So the objectness task itself is what the model cannot fit**, and that is the same
constraint `yolo_assignment.md` originally measured from the outside (`pos − ring =
+0.022`: the head cannot separate an assigned cell from its 8-neighbours). What this
probe adds is that it is not a data, generalization, optimizer or capacity problem — it
survives 2000 passes over 8 images with 7M extra parameters.

Note this does NOT revive center sampling: bite 0 measured that fix directly and the
oracle moved 0.0 points. The constraint is real; the previously-proposed remedy is
still refuted.

**Superseded next probe: log the global gradient norm per step** (the value is already
computed in the emitted IR as `%gcnorm`, it is simply not returned). If it sits far above
4.0, the clip is active on every step and the effective step size is not what the
schedule says. That is one extra return value from the train step, not a new experiment.

### ✅ CHECKPOINT FORENSICS — measured 2026-07-22. TWO MORE REFUTATIONS, AND A STRUCTURAL FINDING.

Both probes read checkpoints already on disk. No GPU, no training, no codegen.

**❌ "the backbone is not learning" — REFUTED.** The signature invited it: box and class
are masked to positives (~45 constraints/img) while objectness is dense (~12,348), and a
head on *frozen* features fits sparse targets trivially and fails dense ones. Comparing
the lr×1 and lr×3 overfit checkpoints at step 2000 (same spec, same data, same steps —
only the LR differs, so any parameter identical across them is not being updated):

| block | n | bitwise-identical | mean abs delta, relative |
|---|---|---|---|
| backbone | 21,284,672 | 4 (0.00%) | 0.570 |
| neck + head | 264,071 | 0 (0.00%) | 0.147 |

Uniform across all 16 forward-order chunks of the backbone (0.43–0.75). Gradients reach
every stage, including the C3 tap. The `fpnTapGrad` integration seam — the one join never
covered by an FD probe — is live.

**❌ "the objectness weight vector has collapsed to zero" — REFUTED.** Mean L2 norm per
1×1 head row, by slot kind, at overfit step 2000: box 1.32 / **obj 1.46** / cls 1.57 on
P3, and the same story on P4/P5 and across every production checkpoint e2..e12. The
objectness weights are as healthy as the box weights.

**⚠️ BUT THAT FORCES AN ARITHMETIC CONSEQUENCE, AND IT IS THE FINDING.** The objectness
bias sits at **−4.5890**, against its RetinaNet prior init of −4.5951: it has moved 0.006
in 2000 steps (production e12: −4.4986). Measured logits average −1.80. So

```
  z = b + w·f  =>  w·f  ~  +2.79 mean,  0.30 std     (measured std 0.24/0.31)
```

**The readout is a large constant plus a wiggle ~9× smaller.** That is what "the score is
nearly a constant function" is, at the level of the weights.

**This also corrects the record: `yolo_assignment.md`'s "the prior init is washed out by
training" is WRONG.** The prior is intact at −4.59, exactly where it was initialized.
What collapsed is the dynamic range of `w·f`, which is a different defect with a
different fix.

**🔎 STRUCTURAL FINDING — THE NECK AND HEAD ARE A LINEAR MAP WITH NO NORMALIZATION.**
Reading `emitFpnNeckForward` (MlirCodegen ~1863) and `emitFpnDetectForward` (~2059):
neck = three `emitConv1x1Fwd` + bilinear upsample + add; head at `tower=0` = one
`emitConv1x1Fwd` + a broadcast bias. **No BN, no ReLU, no normalization of any kind
between the backbone output and the objectness logit** — the whole detector head is an
affine functional of C3/C4/C5.

**And that is a drift from this project's own design.** `yolo_fpn.md` bite 4 (line ~508)
specifies each head as `conv2d 256→256 3×3 relu → 256→A·15 1×1`, built from "plain
verified `convBn` emitters — to be composed into an `emitFpnHead` (fwd = 2×
`emitConvBnTrain`; bwd = 2× `emitConvBnBackward`)". Bite 7 shipped a "minimal head, NO
tower/bias" as an FD-probe simplification and **it was never upgraded to the designed
head.** T2a's tower added the 3×3+ReLU back but explicitly **without norm** ("Plain
conv+bias+ReLU, NO norm — keeps the tower out of BN running-stats plumbing entirely").
So normalization has never been present in the neck or head, in any of the four arms.

**Hypothesis H8 — the objectness readout is DC-dominated because nothing normalizes it.**
It fits every observation: near-constant output with healthy weights; a bias pinned at its
prior; box/class fitting anyway (45 constraints can exploit a small wiggle, 12,348 cannot);
capacity not helping (T2a added nonlinearity but no norm, so the conditioning is unchanged);
and the reference detector not having the problem (YOLOv8 has BN in every head conv,
RetinaNet uses GroupNorm). It also rhymes with [[ddpm-v2-gateA-verdict]], where all three
v2 failures were normalization.

**H8 IS NOT YET MEASURED. Phase A below exists to kill it cheaply before spending codegen** —
this thread's track record on plausible-but-unmeasured stories is 0 for 8.

> **❌❌ H8 IS REFUTED, AND SO IS THE PHASE A/B PLAN BELOW IT. SEE "THE BESPOKE TWIN" SECTION —
> the twin runs `tower=0, norm=None` (the same bare, un-normalized, linear head) and scores
> mAP 0.1299. The head drift from `yolo_fpn.md` bite 4 is REAL but is NOT the defect.
> Track record now 0 for 9. Everything from "📋 THE PLAN" to the end of this block is DEAD —
> kept only so the reasoning that produced it stays auditable.**

### 📋 THE PLAN — Phase A measures, Phase B builds, and A can cancel B

**Phase A — confirm the mechanism. No codegen, no training.**

- **A1. Dump P3/P4/P5 (and C3) for the 8 overfit images.** This is the enabling step and
  the only new plumbing: a feature dump in `inferDump`, alongside the existing logits dump.
- **A2. Fit an unconstrained linear probe** (logistic regression, pos = cell centre inside
  a GT box) on those frozen features, and measure per-channel mean vs std to quantify the
  DC domination directly. **This is the decisive fork:**
  - probe AUC high (≥0.95) ⇒ a linear readout of these features *is* sufficient, and
    training simply cannot find it ⇒ conditioning/normalization. **H8 confirmed, Phase B
    is the right build.**
  - probe AUC ≈ 0.74, i.e. no better than the deployed head ⇒ the features genuinely do
    not carry it ⇒ the defect is upstream in backbone/neck, Phase B is mis-aimed, and
    resolution re-enters as a live question.
- **A3. Same measurement on YOLOv8s' pre-head features** as the calibrated control — the
  reference already proves the task is learnable at 448/stride-8, so this says by how much.
- Note `sklearn` is **not** installed in `visdrone/.venv`; either add it or hand-roll the
  probe in numpy.

**Phase B — the fix, gated by the 8-image overfit (only if A2 says so).**

- **B1. Put normalization back in the head, as originally designed.** Default to the
  already-FD-verified `convBn` path rather than building GroupNorm from scratch — bite 4
  already scoped `emitFpnHead` as 2× `emitConvBnTrain`/`emitConvBnBackward`, so this is
  composition of verified pieces, not new math. GroupNorm is the fallback if BN's
  batch-size dependence or running-stats plumbing bites.
- **B2. FD-verify**, per thread convention.
- **B3. GATE: the 8-image overfit, not a 12-epoch run.** The objectness term must fall
  well below 27.55/img. This is 30–90 min, and it is now a calibrated instrument with a
  known floor of 0.
- **B4. Only if B3 passes:** the full 12-epoch arm, judged on mAP against 0.0001 and
  against the reference's 0.114 (scratch) / 0.140 (scratch, squash).

**Phase C — two standing decisions that are Brett's, not the measurement's.**

- **Fix the `MAX_BBOXES = 56` eval-GT truncation** (34.9% of val GT silently dropped,
  59.1% of val images over the cap) before *any* absolute number from this thread is
  quoted as VisDrone protocol. Relative arm comparisons are unaffected.
- **Decide whether VisDrone is the demo target at all.** 70 objects/image at 2–5 px is a
  punishing setting for the "the verified stack trains a real detector" claim. If that
  claim is the demo's job, it lands far sooner elsewhere with VisDrone as the stretch.

### 🎯 THE BESPOKE TWIN — 2026-07-22. THE THREAD IS RESOLVED: IT IS A LEAN BACKWARD BUG.

`visdrone/` had built a **YOLOv8** reference — a *different architecture* — so every arm and
probe in this thread conflated "our architecture is bad" with "our implementation is bad."
Nobody had built a PyTorch twin of **our** detector. That was the missing rung.

New `visdrone/bespoke/`: `model.py` (R34+FPN transcribed from `emitFpnNeckForward` /
`emitFpnDetectForward`), `loss.py` (from `emitMultiScaleYoloLoss` / `emitAnchorYoloLoss` /
`emitDiouForward`), `data.py` (reads the SAME `data/visdrone_fpn/*.bin` bytes),
`train.py`, `lean_ckpt.py` (loads Lean flat checkpoints), `infer.py` (dumps logits in Lean's
`inferDump` format so `yolo_map_visdrone.py` scores both arms with ONE scorer), `diff_lean.py`.

| arm (no aug, 448, 12 ep) | mAP@0.5 | recall |
|---|---|---|
| **Lean FPN** (best of four) | **0.0001** | 0.118 |
| **the twin**, ImageNet R34 | **0.1299** | **0.738** |
| YOLOv8s scratch, letterbox, random init | 0.114 | — |
| YOLOv8s scratch, squash, random init | 0.140 | — |

**1299×, same scorer, same GT, same encoded bytes.** The twin lands *between the two
YOLOv8s-from-scratch rows*, so the "RetinaNet is the weakest VisDrone baseline" worry does not
bite at this scale — the architecture is fine. The 8-image overfit is even starker: objectness
**27.55 (Lean) vs 0.35 (twin)** at step 2000, total 36.0 vs 1.08.

**THE TWIN IS VERIFIED AGAINST LEAN AT THREE LEVELS**, which is what makes it an oracle
rather than another opinion:
1. **Forward faithful** — on Lean's own `of8long_e2000` weights, twin objectness logits are
   mean −4.2467 / std 4.7337 / min −153.43 vs Lean's dumped −4.3237 / 4.8138 / −155.83
   (~2%; residual is BN train-vs-eval mode).
2. **Loss exact** — run on Lean's OWN logits dump it reproduces `fpn_loss_breakdown.py` to
   every digit: total 35.422, box 6.155, obj 27.550 (pos 14.761 / neg 12.789), cls 1.717.
   (The mirror applies no class weights; `--no-cls-weights` is the byte-matching arm.)
3. **Logit gradient exact** — vs the emitted `α·w·(p−t)/B` with detached focal weight
   (`anchor_loss_probe_check.py:88-93`): max abs diff **6.9e-18**.

**⇒ THE DEFECT IS IN THE BACKWARD PASS FROM THE LOGIT GRADIENT TO THE PARAMETERS.** Walking
that path by what is actually verified: loss grad ✓ (1e-18); `emitFpnDetectBackward` ✓ (bite 7
FD at γ=0); `residualBlock` backward ✓ (long-standing); grad clip ✓ (clip-off probe); Adam ✓
(shared with all five verified classifier nets, which train correctly). **That leaves exactly
one unverified join: the `fpnTapGrad` injection of dC3/dC4 into the backbone's backward walk at
the residualBlock markers.** The bite-7 probe stops at dC3/dC4/dC5; everything past it was
validated only by `iree-compile` passing, which is a TYPE check. It also fits the symptom — drop
or misplace the C3/C4 cotangents and the backbone trains mostly through C5, so P3 never adapts,
objectness (dense, P3-dependent) cannot fit while box/class (sparse, 45 cells) fit fine.
**"The backbone moves" does NOT clear it: moving 57% is not moving CORRECTLY.**

**🔪 THE NEXT PROBE, AND IT NEEDS NO NEW LEAN CODE.** Adam's first step from m=v=0 is
`Δw = −lr·sign(g)` (bias correction gives m̂/√v̂ = g/|g|), and checkpoints are exactly
`totalParams` floats ⇒ m/v are NOT saved ⇒ a fresh run starts at zero state. Run Lean ONE step
from a known checkpoint, run the twin one step from the same weights, diff: that reads out
**the sign of every emitted gradient, per layer**, with no gradient-dump plumbing. Head agrees
+ early backbone disagrees ⇒ tap injection confirmed. Gate the fix on the 8-image overfit
(objectness must fall well below 27.55), then the 12-epoch arm.

**COST, MEASURED — why the detector work belongs in PyTorch:**

| | Lean | twin | ratio |
|---|---|---|---|
| 12-epoch VisDrone run | 4.4 h (22 min/ep) | **11 min** (55 s/ep) | **24×** |
| 2000-step 8-image overfit | ~90 min | ~4 min | **22×** |
| codegen change → runnable | +6.5 min vmfb compile | 0 | ∞ |

**KNOWN TWIN DIVERGENCES (none can produce 1299×, all documented):** ⚠️ **Lean's spec is
`.maxPool 2 2` but torchvision's ResNet stem is `MaxPool2d(3, stride=2, padding=1)`** — same
112² output, different function; `pool="lean"` is REQUIRED for any checkpoint diff (it moved the
forward diff 219.9→196.5) and the 0.1299 run used torchvision's, so a byte-exact re-run is the
control. Also: the stem's `.same` padding is asymmetric (2,3) vs symmetric 3; decoupled AdamW
vs Lean Adam+wd; `align_corners=False` unchecked against Lean's upsample; torch kaiming vs Lean
He init (start loss 489 vs ~800); torchvision ImageNet weights vs `jax_r34_imagenet.bin`.

**STILL TRUE AND STILL BLOCKING ANY EXTERNAL NUMBER:** `pack_raw_boxes` caps at
`MAX_BBOXES = 56` while VisDrone val averages 70.7 boxes/img ⇒ **34.9% of val GT is silently
dropped**. Both arms are scored against the same truncated GT so 0.0001-vs-0.1299 is valid, but
**no absolute number here is VisDrone protocol.**

### Probe harness added to the trainer (2026-07-21)

`demos/MainYolov1VisdroneFpn.lean` gained four env overrides on the `FPN_TOWER` pattern,
all defaulting to the arm's configured values so every existing runbook is byte-identical:

| env | effect | why |
|---|---|---|
| `FPN_TAG` | suffix folded into `NetSpec.name` | **the name IS the checkpoint prefix** — a probe without a tag silently overwrites the live arm's e2..e12 checkpoints, which every measurement in `yolo_assignment.md` is computed from |
| `FPN_EPOCHS` | epoch count | the probe needs 200, the arm wants 12 |
| `FPN_CKPT_EVERY` | checkpoint interval | 200 epochs at every-2 writes ~8.6 GB of useless 86 MB snapshots |
| `FPN_LR_MULT` | integer LR multiplier | separates "throttled" from "broken"; integer because this toolchain has no `String.toFloat?` |

## ⛔ Bite 1 / Bite 2 — SUSPENDED by the bite 0 result

*(Both were scoped to port a scoring fix. Bite 0 priced that fix at ~1.4× against a
1399× gap, so neither should be executed until the trainer question above is settled.
The MlirCodegen facts recorded in bite 2 stay accurate and stay useful if scoring ever
becomes the binding constraint again — which it is not today.)*

## Bite 1 (SUSPENDED) — the fix ladder, also in PyTorch

Only once bite 0 has named the rung. Climb back up from the crater, cheapest ingredient
first, and record what each one is worth:

1. **Class score trained on background** (per-class sigmoid BCE over all anchors,
   target 0 on background) — predicted to recover most of it.
2. **Quality-aware target** (objectness/class target = achieved IoU rather than 1.0).
3. **Dynamic assignment** (TAL) — the most expensive to port, so measure it last and
   only if 1+2 leave a gap worth paying for.

The output of bite 1 is a **single number per ingredient**, which is what makes the port
scopeable instead of speculative.

## Bite 2 (SUSPENDED) — the port back to Lean

Only the ingredients bite 1 proved. What already exists on the Lean side (all
FD-verified and committed): the FPN neck, `emitMultiScaleYoloLoss`, the whole-detector
DAG backward, head biases, the tower. None of that needs to change.

Two facts checked in `LeanMlir/MlirCodegen.lean` that make this cheaper than it sounds:

- **The class term's positives-mask is one broadcast.** `%{pa}_cls_maskb` =
  `broadcast_in_dim %{pa}_m4` (the target's own objectness channel), applied to the loss
  at `_cls_nllm` and to the gradient at `_cls_dm`. Un-masking the class term is a
  localized edit at those two sites — plus swapping softmax-NLL for per-class sigmoid
  BCE, which is new math and needs its own FD probe.
- **The achieved IoU is already in the graph.** `emitDiouForward` computes `%{pfx}_iou`
  and only then subtracts the DIoU penalty. A quality-aware target can read that SSA
  value directly — no new geometry, and the value is already FD-covered by
  `scripts/diou_probe_check.py`.

Sequence: FD-probe the new loss term at tiny scale (the established pattern), A/B against
a named control arm at e12, then judge on mAP — never on an interim metric.

## Do regardless of which lever wins

1. **Fix `MAX_BBOXES = 56`.** VisDrone val averages 70.7 boxes/img; 59.1% of val images
   exceed the cap and **34.9% of all val GT is dropped**, biased toward the dense (hard)
   images. Relative comparisons between arms survive (same truncated GT), so no
   refutation is at risk, but no absolute number in this thread is VisDrone protocol.
   `visdrone/prepare_yolo.py` independently reproduces the full GT (343,204 train boxes,
   70.7 val/img) and is the cross-check.
2. **Report mAP@0.25 alongside mAP@0.5** until the detector clears zero. Four levers were
   judged against a metric saturated at 0.0001.
3. **Keep the standalone as the permanent control.** Any future Lean arm gets diffed
   against `visdrone/runs/scratch_squash_12ep_448` at matched recipe, not against its own
   previous epoch.

## Refuted — do not rebuild

| lever | verdict |
|---|---|
| T1a objectness `/num_pos` | refuted on paper — background is 6% of total loss |
| T1b class-weighted CE | mechanism works, mAP unmoved (keep it, it is free) |
| T2-bias + RetinaNet prior | washed out by e12 |
| T2a 4-conv head tower (+7M) | did not even lower the TRAIN loss |
| Tier 3 FCOS/center sampling | oracle moves 0.0pt; 99.6% of ring boxes never merge |
| input resolution / P2 | +4.9pt vs ranking's +23.9; gated downstream |
| IoU-aware objectness as **rescoring** | −0.32pt vs deployed (as a *training* change it is untested — see bite 1.2) |
| **letterboxing the input** | **refuted 2026-07-21: squash BEATS letterbox 0.140 vs 0.114 at matched recipe.** Padding 16:9 into a square burns ~44% of the canvas and shrinks already-tiny objects; the pixel budget dominates the aspect distortion. |

## Risks, in the order they are likely to bite

1. **Bite 0's crater lands on a rung we cannot port cheaply.** If TAL (L4) carries it and
   background-training (L2) does not, the port is a dynamic assigner in MLIR — a much
   bigger job than a loss-term edit. Measure L2 first precisely so this is known early.
2. **Per-class sigmoid BCE over all anchors is a new loss term**, not a re-mask. It needs
   its own FD probe and it changes the loss's numerical scale, so λ weights and the grad
   clip may need re-checking. **Measure before adjusting** — do not pre-emptively retune.
3. **The 448-vs-640 question is still open on the ranking axis.** `yolo_assignment.md`
   only refuted resolution's *box-precision* mechanism; whether it lifts object/background
   AUC was never testable from 448px logits. The reference now makes it a cheap A/B, but
   it is a second-order move — fix scoring first.
4. **Reference drift.** `visdrone/prepare_yolo.py` must keep the identical class filter as
   `parse_visdrone_txt`. If it drifts, the two arms stop being comparable and this whole
   doc loses its footing.

## Inherited gotchas — all of these have cost time

From today:
- **`close_mosaic` hangs with augmentation off.** Ultralytics rebuilds the dataloader 10
  epochs from the end even when mosaic is already 0; the rebuild hung indefinitely (epoch
  3, no batch in 10 min, GPU pinned at 100%). `close_mosaic=0` whenever `--no-aug`.
- **Ultralytics resolves a symlinked image dir** and then looks for labels inside the
  *source* tree. Write `labels/` beside the real images.
- **A relative `project=` resolves against the global `runs_dir`**, which defaults to this
  repo's `runs/`. Pass an absolute path or it lands in the Lean training logs.
- **`pkill -f <pattern>` kills the calling shell** when the pattern matches your own
  command line. Collect PIDs with `pgrep` and kill those.
- **Eyeballing a 53-object drone scene cannot distinguish "shifted box" from "different
  truck".** Match numerically — and match on centre AND size jointly, because VisDrone has
  concentric annotations (a truck and its cab share a centre) that make a centre-only
  matcher report phantom size errors.

Still live from the Lean side:
- **`FPN_TOWER` selects the SPEC, so set it on `infer` too.** The tell is an epoch sweep
  with zero variation between checkpoints.
- **Build/run the train step with `IREE_BACKEND=rocm`**, or the loss reduce dies with
  `'func.func' op failed to distribute`.
- **Any unbounded op (`exp`) + global-norm grad clip = latent `inf·0 = NaN`.** Cap at
  source; DIoU's `tw,th ≤ 8` cap is why the FPN arm trains at all.
- **An interim measurement confirming the MECHANISM is not evidence the LEVER works.**
  Judge at e12, on mAP, against a named control.

## Runbooks

Standalone (`visdrone/`, torch 2.9.1+rocm6.4 + ultralytics 8.4.103):

```
cd visdrone
.venv/bin/python3 prepare_yolo.py                    # labels beside the source images
.venv/bin/python3 make_squash_dataset.py             # 448 squashed copy, Lean-style resize
HIP_VISIBLE_DEVICES=1 .venv/bin/python3 -u train_baseline.py \
  --model yolov8s.yaml --data data/visdrone_squash.yaml \
  --epochs 12 --imgsz 448 --no-aug --name <arm>
.venv/bin/python3 compare_scoring.py --yolo runs/<arm>/weights/best.pt
```

Lean arm (unchanged, for reference):

```
lake build yolov1-visdrone-fpn
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 ./.lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
```

~22 min/epoch at tower=0 on gfx1100, +7 min one-time compile, checkpoints every 2 epochs.

## Diagnostics available

| script | answers |
|---|---|
| `visdrone/compare_scoring.py` | **both detectors on one metric**: obj-vs-bg AUC, score dynamic range, top-k survival |
| `visdrone/check_bin_alignment.py` | does the Lean encoder round-trip? (it does: 2.1e-09, 0 mismatches over 38,421 boxes) |
| `visdrone/inspect_lean_bin.py` | renders a `train.bin` record with encoded targets + raw annotations overlaid |
| `visdrone/prepare_yolo.py` | independent GT cross-check (343,204 train boxes, 70.7 val/img) |
| `scripts/fpn_obj_separation.py` | objectness AUC + logit spread + class-head collapse |
| `scripts/fpn_loss_breakdown.py` | where does the loss go? (box/obj/cls, pos vs neg) |
| `scripts/visdrone_fpn_coverage.py` | encodability ceiling per assignment scheme |
| `scripts/fpn_resolution_probe.py` | box-error decomposition, rank survival, lever comparison |
| `scripts/fpn_objectness_readout_probe.py` | readout ceiling; obj-vs-quality AUC split |
| `scripts/fpn_iou_target_probe.py` | IoU predictability + same-features/different-target test |

## The demo, once it works

The stated goal is a demo piece plus an inference/runtime playground. Two notes:

- **VisDrone is a punishing choice for the "verified stack trains a real detector" claim**
  — 70 objects/image at 2–5 px. If the demo's job is that claim, it lands far sooner on an
  easier task with VisDrone as the stretch. If the demo's job is specifically drone
  imagery, the dataset tax is just the cost of the thesis. Worth deciding explicitly
  rather than by default.
- The reference detector is itself the runtime playground's first target: it exports to
  ONNX/IREE and gives a known-good latency and accuracy baseline to port against.
