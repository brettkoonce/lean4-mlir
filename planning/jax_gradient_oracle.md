# jax_gradient_oracle.md — a differential gradient oracle for the FPN detector

**Status:** the bug this was written to find is **FOUND AND FIXED** (2026-07-22, same day).
It was not in the backward pass. See §0. The oracle itself remains unbuilt, and the
structural case for building it (§11) is now much stronger, not weaker.
**Prereq reading:** `planning/yolo_scoring.md` §"THE BESPOKE TWIN".

---

## 0. RESOLUTION — the defect was the DATA SHUFFLE, not the gradient

`ffi/f32_helpers.c: lean_f32_shuffle` permuted the images with a Fisher–Yates swap of
`pixels_per*4` bytes per record, and permuted the labels with a swap of a **hardcoded 4
bytes** — one float, i.e. a classification scalar. The FFI had no label-stride parameter
at all. For the FPN detector a label record is 185,220 floats = **740,880 bytes**.

**Every epoch, the images were permuted and the targets were not.** The detector was
trained on mismatched image/target pairs for its entire life. All it could learn was the
marginal distribution of the targets, which is precisely what mAP 0.0001 and an objectness
floor of 27.55 look like.

### How it was found

Not by comparing gradients. By running the trainer with **`FPN_LR_MULT=0`**, which zeroes
both the Adam update and the decoupled weight-decay term, so the parameters are provably
frozen. Three consecutive steps on the same 8 images then returned:

| | loss |
|---|---|
| frozen params, shuffle ON (before fix) | 35.831673, 36.196793, **32.388531** |
| frozen params, `LEAN_MLIR_NO_SHUFFLE=1` | 35.235260, 35.235260, 35.235260 |
| frozen params, shuffle ON (after fix) | 35.235374, 35.235291, 35.235306 |
| faithful PyTorch twin, same weights | 35.2359 |

A fixed batch at frozen weights **must** be a pure function. A 12% spread across identical
steps is not a gradient bug, an optimizer bug, or a numerical-precision story — it is the
input changing. The `NO_SHUFFLE` control isolated it in one run, and the residual ~1e-4
after the fix is fp32 reduction-order noise from the differing batch order.

**This probe cost about two minutes and needed no codegen change.** It should be the first
thing tried on any "the trainer cannot descend" report, ahead of every measurement in §8.

### Both gates passed — the detector works

Same schedule, same data, same recipe; only the shuffle differs.

| | broken | fixed |
|---|---|---|
| 8-img overfit, total @2000ep | 35.97 | **3.145** |
| 8-img overfit, **objectness** | **27.550** | **0.526** (pos 0.221 / neg 0.305) |
| 12-ep train loss @e12 | 300.9 (flat since e4) | **112.62** (still descending) |
| **12-ep mAP@0.5** | **0.0001** | **0.1167** |
| 12-ep recall | 0.118 | **0.7353** |
| 12-ep class-agnostic AP | 0.0009 | 0.3162 |

Against the faithful twin (mAP 0.1391, recall 0.738), **recall now matches** and mAP is at
84%, with per-class ordering and magnitudes tracking it closely. The residual is plausibly
Lean's Adam+wd vs the twin's decoupled AdamW and the jax-ImageNet bootstrap vs torchvision
init — not a bug hunt.

⇒ Every "converged equilibrium", "unlearnable task", and "objectness is the term the model
cannot fit" conclusion in this thread and in `yolo_assignment.md` was measuring this bug.

⚠️ The mAP is still scored against the `MAX_BBOXES = 56` truncated eval GT (34.9% of val GT
dropped). Fine for arm-vs-arm; **not VisDrone protocol** — fix before quoting 0.1167 outside.

### Why nine previous investigations missed it

- It is **host-side C**, not emitted IR. `iree-compile`, every MLIR-level review, and every
  proof obligation are looking at the wrong artifact entirely.
- **FD probes cannot see it.** An FD probe checks d(emitted loss)/dθ against the *same*
  emitted forward on the *same* batch. A mispaired batch is self-consistent under FD, so
  every probe passes.
- The **loss and its VJP are genuinely correct** — the verified 6.9e-18 logit-gradient
  match was real, and measuring it harder would never have helped. The targets fed *into*
  that correct loss were the wrong ones.
- **Classification is unaffected** (`labelBytesPerRecord = 4` is exactly the hardcoded
  value), which is why the five verified classifier nets trained correctly and kept
  pointing suspicion away from shared infrastructure.
- "The backbone moves a lot but not correctly" (measured lr×1-vs-lr×3 delta 0.570) was
  true and correctly interpreted — it was descending on scrambled targets.

### The fix

`lean_f32_shuffle` now takes an explicit `label_stride` and swaps whole label records under
the same permutation, and errors out rather than shuffling a partial label if the stride is
inconsistent with the buffer. `F32.shuffle` gained a `labelBytes` argument; `Train.lean`
passes `dio.labelBytesPerRecord`, and the two `VerifiedTrain.lean` classifier call sites
pass a commented literal 4.

### ⚠️ BLAST RADIUS — every detection and segmentation trainer

Any `DatasetIO` with `labelBytesPerRecord ≠ 4` ate this corruption on every epoch:

| dataset | label bytes/record |
|---|---|
| `petsIO` (Oxford Pets segmentation) | 224·224 |
| `bratsIO` (BraTS brain tumour) | 240·240 |
| YOLOv1 pets detection | 7,200 |
| anchor YOLO | A·15·gH·gW·4 |
| YOLO mosaic | 30·gH·gW·4 + gH·gW·4 + 4 + 1,120 |
| FPN detector | 740,880 |

**Every measurement taken on any of these arms was taken on a mispaired dataset**, and any
conclusion drawn from one needs re-checking. The BraTS thread in particular concluded that
`ce` == `dicece` == a trivial predictor and root-caused it to Dice's gradient vanishing
∝ p_i. Mispaired image/mask data *also* produces exactly a trivial predictor, and it is the
simpler explanation. That diagnosis is now confounded and should be re-run before the
class-weighted-CE fix is credited or dismissed.

### Second bug found on the way: the twin was never faithful

§6 validation was run first, as this doc insists, and check 2 **failed**. `MlirCodegen.samePad`
is TensorFlow-style asymmetric SAME (`pad_low = t/2`, `pad_high = t − t/2`); torchvision uses
a symmetric `padding=`. The stem 7×7/s2 gets Lean **(2,3)** vs torchvision **(3,3)**, and every
stride-2 3×3 conv gets Lean **(0,1)** vs torchvision **(1,1)**. Identical output sizes, so no
shape check, parameter count, or compile step can catch it — but the sampling grid is offset
by one pixel and the offset compounds through four downsampling stages.

The original forward check compared **mean, std and min** — all three permutation-invariant.
It reported "~2%, residual is BN train-vs-eval" while the per-element disagreement was
rel 0.327 with max|diff| 74.4. Fixed via `apply_lean_padding` (`pad="lean"`, a forward
pre-hook so `named_parameters()` ordering stays intact for the flat-checkpoint loader):

| | mean\|diff\| | max\|diff\| | loss |
|---|---|---|---|
| `pad="torchvision"` | 0.7898 | 74.4306 | 196.4932 |
| `pad="lean"` | **0.0000** | **0.0020** | **35.2359** |

**Never accept summary statistics as a forward tie.** Compare per element, and compare
sorted values too — matching sorted values with mismatched positions is the tell that
distinguishes a layout/alignment bug from a numerical one.

### New tooling (all under `visdrone/bespoke/`)

| path | what |
|---|---|
| `validate_oracle.py` | the §6 gate: loss on Lean's dump, per-element forward tie, logit-grad tie |
| `bn_stats.py` | loads Lean `*_bn_stats.bin` into the twin's BN buffers (eval mode is meaningless without it) |
| `layout_hunt.py` | tests candidate index permutations when a forward tie fails |
| `grad_dump.py` | twin parameter gradients in Lean's flat checkpoint order (for §8a) |

---

## 1. The problem this exists to solve

> **SUPERSEDED BY §0.** Kept as written for the record. Two of its load-bearing claims are
> now known false: the "forward measured identical" row compared only permutation-invariant
> summary statistics (the twin was off by a one-pixel padding shift), and the conclusion
> "⇒ the defect is in the backward pass" does not follow — the defect was in the host-side
> data shuffle. The `fpnTapGrad` injection it fingers was never implicated.

The Lean FPN detector reaches **mAP@0.5 = 0.0001**. A PyTorch twin of *the same
architecture*, trained on *the same encoded bytes* with *the same loss* and scored by *the
same scorer*, reaches **0.1391** — a 1391× gap. On an 8-image pure-memorization probe the
twin drives objectness to 0.35 while Lean plateaus at 27.55 against a floor of zero.

As of 2026-07-22 the following are **measured identical** between the two:

| | evidence |
|---|---|
| data | byte-identical; both read `data/visdrone_fpn/train.bin` |
| architecture | 21,548,743 params on both, exactly |
| forward | twin on Lean's weights vs Lean's own logits dump: obj mean −4.2467 vs −4.3237, std 4.7337 vs 4.8138, min −153.43 vs −155.83 (~2%, residual is BN train-vs-eval mode) |
| loss | twin on Lean's OWN logits dump reproduces `scripts/fpn_loss_breakdown.py` to every digit: 35.422 / box 6.155 / obj 27.550 / cls 1.717 |
| gradient **at the logits** | vs emitted `α·w·(p−t)/B`: max abs diff **6.9e-18** |

**⇒ The defect is in the backward pass from the logit gradient to the parameters.** Within
that path everything is FD-verified *except one join*: the **`fpnTapGrad` injection** of
dC3/dC4 into the backbone's backward walk at the residualBlock markers. The bite-7 probe
stops at dC3/dC4/dC5; past that point the only check was `iree-compile` succeeding, which is
a **type** check. Adam is shared with five verified classifier nets that train correctly, and
grad clipping was exonerated by a clip-off probe.

Note: "the backbone moves" does **not** clear the tap. Measured lr×1-vs-lr×3 relative delta
is 0.570 across all 16 backbone chunks — it moves a lot. Moving is not moving *correctly*.

---

## 2. Why JAX specifically

`jax/` is **Phase 2: Lean → JAX** — Lean 4 as a metaprogramming layer emitting idiomatic JAX
Python, walking the **same `NetSpec`** (`jax/Jax.lean` re-exports `LeanMlir.Types/Spec`).
Two properties make it the right oracle for *this* bug:

1. **The bug class cannot exist there.** The MLIR backend hand-writes every backward
   (`emit*Backward`), and the defect is a seam between two of them. JAX writes no backward
   at all — `value_and_grad` derives it. A dropped cotangent tap is structurally impossible.
2. **No transcription gap.** The PyTorch twin required hand-transcribing the architecture,
   which introduced six divergences; one (the stem maxpool) was caught only by checking
   before believing a number. A JAX arm driven by the same `NetSpec` has none of that — any
   numerical disagreement is *definitionally* the codegen.

This is strictly better than the torch twin for localization. The torch twin remains useful
as a fast experiment platform and as a cross-check on the JAX oracle itself.

---

## 3. Scope — this is a DIAGNOSTIC, not a training backend

**Build:** a JAX forward for `r34FpnDet` + the multi-scale loss, fed one batch of numpy
straight from the bin, differentiated with `jax.grad`, producing reference parameter
gradients.

**Do NOT build:** a data pipeline, an eval path, a training loop, or `LossKind` plumbing.

That boundary matters, because the full JAX *training* backend is a much bigger lift than it
first appears. Surveyed 2026-07-22:

- `cfg.lossKind` exists and is pluggable (good), but the emitter does **not** read
  `fpnScales`, `focalGamma`, or `yoloClsWeights` — none appear in its `cfg.*` usage.
- The eval path (`Jax/Codegen.lean` ~2390) is hardcoded classification: `one_hot(y, nClasses)`,
  `correct1/correct5`.
- Data loading is classifier-shaped.
- `Jax/Codegen.lean:278` states outright: *"YOLOv1/detection is phase-3-only."* There is no
  detection support in that backend today.

Scoped as a diagnostic, none of that is on the critical path.

**⚠️ Also: the `jax/` track is NOT dead weight.** `.lake/build/jax_r34_imagenet.bin` (87 MB,
21,797,672 floats = R34 + 1000-class head) is the ImageNet backbone the detector bootstraps
from — `r34FpnDetConfig.bootstrapBackbone := some (".lake/build/jax_r34_imagenet.bin", 21284672)`,
taking the first 21,284,672 floats. The codegen path is dormant; its output is load-bearing.
Separate those two before removing anything.

---

## 4. The architecture to emit

From `demos/MainYolov1VisdroneFpn.lean` and `LeanMlir/MlirCodegen.lean`.

```
layers = [ convBn 3 64 7 2 .same          -- NOTE .same on 7x7/s2 is ASYMMETRIC pad (2,3)
         , maxPool 2 2                    -- k=2 s=2. NOT torchvision's k=3 s=2 p=1.
         , residualBlock  64  64 3 1      -- stride 4
         , residualBlock  64 128 4 2      -- C3: 128ch @ 56x56
         , residualBlock 128 256 6 2      -- C4: 256ch @ 28x28
         , residualBlock 256 512 3 2      -- C5: 512ch @ 14x14
         , fpnDetect 256 128 256 512 14 3 0 ]   -- oc c3 c4 c5 g5 A tower
```

**Neck** (`emitFpnNeckForward`, MlirCodegen ~1863) — 1×1 convs, **no bias, no norm, no
activation**; the whole neck+head is a linear map at `tower=0`:

```
P5 = conv1x1(C5, Wn5)
P4 = conv1x1(C4, Wn4) + upsample2(P5)
P3 = conv1x1(C3, Wn3) + upsample2(P4)
```

**Head** (`emitFpnDetectForward`, ~2059): `conv1x1(Pn, Whn) + bhn`, per level.

**Output:** per level `[B, A*15, g, g]` → C-order flatten → concat `[P3|P4|P5]` →
`[B, 185220]`. Flat index within a scale is `((a*15 + c)*g + i)*g + j`.

**Upsample:** bilinear, and `bilinearWeights1D` (MlirCodegen ~846) computes
`src = (2i+1−scale)/(2·scale) = (i+0.5)/scale − 0.5` with border clamping — this is exactly
`align_corners=False` / half-pixel. **Verified 2026-07-22.** `jax.image.resize` with
`method="bilinear"` uses the same convention; confirm on a small array rather than assuming.

---

## 5. The loss to emit

Transcribed from `emitMultiScaleYoloLoss` / `emitAnchorYoloLoss` / `emitDiouForward`.
**A working, digit-exact PyTorch reference already exists at `visdrone/bespoke/loss.py`** —
it is pure array ops and ports to JAX nearly line-for-line. Start from it, not from the Lean.

Slot layout per anchor (`base = a*15`): `+0..3` box (tx,ty,tw,th), `+4` objectness,
`+5..14` class. **The target's objectness channel IS the assignment mask** — there is no
separate mask tensor.

```
loss = Σ_scales (1/B) Σ_a [ 5.0·Σ mask·(1−DIoU) + Σ α·w_foc·BCE + Σ mask·w_cls·CE ]
```

Constants: `λ_box = 5.0`, `γ = 2.0`, `α = 1.0` positive / `0.5` negative, exp cap `8.0`,
eps `1e-9` (union, c²) and `1e-12` (focal). Everything is **summed** over cells, anchors and
scales; the only division is by batch size — never by grid area or positive count.

Box decode: `cx = (j + σ(tx))/g`, `w = anchor_w · exp(min(tw, 8))`. Targets use raw
image-normalized `gw, gh` and `(j + gx)/g`.

### ⚠️⚠️ THE TWO TRAPS THAT WILL SILENTLY BREAK THIS ORACLE

**1. The objectness gradient is deliberately NOT the true VJP.** Lean emits
`grad = α·w_foc·(p−t)/B` with the focal modulating factor **held constant**
(MlirCodegen:5456-5459; `scripts/anchor_loss_probe_check.py:12-15` says it is *"intentionally
NOT the FD of the forward"*). A naive `jax.grad` will differentiate *through* `w_foc` and
produce a different gradient — **and that difference is intentional, not the bug.**
**You must wrap the focal weight in `jax.lax.stop_gradient`.** Get this wrong and the oracle
will report a spurious mismatch on objectness, which is precisely the term under
investigation. This thread's record on plausible-but-unverified stories is 0 for 9; do not
add a tenth by mis-building the instrument.

**2. The `min(tw, 8)` cap has no backward indicator.** The emitted backward computes
`dw/dtw = w` unconditionally, so above the cap Lean passes gradient where a plain clamp would
zero it. Reproduce with a straight-through clamp:
`t + stop_gradient(clamp(t, max=8) − t)`. Only matters when `tw > 8`, but it is free.

---

## 6. Validate the oracle BEFORE trusting it

An unverified oracle is just another opinion. Reproduce the same three-level check that
validated the torch twin — every reference number below is already on disk:

1. **Loss.** Run the JAX loss on Lean's own logits dump
   `figures/yolo_of8long/logits.bin` (8 × 185220 f32) with targets from
   `data/visdrone_fpn_of8/train.bin`. **Must produce total 35.422, box 6.155,
   obj 27.550 (pos 14.761 / neg 12.789), cls 1.717** — i.e. match
   `scripts/fpn_loss_breakdown.py` exactly. *(That mirror applies no class weights; run the
   unweighted arm for the byte match.)*
2. **Logit gradient.** `jax.grad` w.r.t. the logits must match `α·w_foc·(p−t)/B` to ~1e-16.
   This is the check that catches a missing `stop_gradient`.
3. **Forward.** Load a Lean checkpoint (§7) and compare emitted logits against
   `figures/yolo_of8long/logits.bin`. Expect agreement to ~2% in train mode (BN batch stats
   vs the dump's running stats), or tight if you load `bn_stats` and run eval mode.

Cross-check against `visdrone/bespoke/` at every step — it is verified and fast.

---

## 7. Loading Lean checkpoints

`visdrone/bespoke/lean_ckpt.py` already does this for PyTorch (117 tensors, 21,548,743
floats, 0 shape mismatches) and documents the layout. Reuse its `r34_fpn_param_shapes()`.

Order is `NetSpec.paramShapes` (`LeanMlir/SpecHelpers.lean:23`): per conv
`[oc,ic,k,k]`, `gamma[oc]`, `beta[oc]`; per residual block conv1,bn1,conv2,bn2 and — for
block 0 of a **projecting** stage — the 1×1 projection **last**. Then `.fpnDetect`:
`Wn3, Wn4, Wn5, Wh3, Wh4, Wh5, bh3, bh4, bh5`, with head biases **last** (`applyDetPriorBias`
splices the tail).

- 1×1 weights are stored 2-D `[oc, ic]` row-major (`emitConv1x1Fwd` uses `tensor<oc×ic>`,
  contracting `[1]×[1]`).
- Backbone prefix = first **21,284,672** floats; head = the remaining **264,071**
  (oc=256 ⇒ 256·(128+256+512) + 3·45·256 + 3·45 = 264,071 exactly).
- BN running mean/var are in a **separate** `*_bn_stats.bin` (68,096 bytes = 2 × 8,512
  channels) and are **not needed** to reproduce a *training* loss, which uses batch stats.
- Objectness prior: `bias[a*15+4] = −log((1−π)/π)`, π=0.01 ⇒ **−4.595120**; all other head
  bias entries 0.

Useful checkpoints on disk: `resnet_34___fpn_detector_448_wcls_pb__visdrone__of8long_params_e2000.bin`
(the 8-image overfit at step 2000) and the production arm's `..._params_e{2..12}.bin`.

---

## 8. The comparison protocol

### 8a. First: the sign probe — free, no codegen change

Adam's first step from `m=v=0` is `Δw = −lr·sign(g)` (bias correction gives `m̂/√v̂ = g/|g|`).
Checkpoints are **exactly `totalParams` floats**, so m/v are *not* saved and a fresh run
starts at zero state. Therefore:

1. Resume Lean from a known checkpoint `W0`, run **one** step on a fixed batch
   (`FPN_EPOCHS=1 FPN_CKPT_EVERY=1`, and **always set `FPN_TAG`** — see gotchas), giving `W1`.
2. Compute the JAX oracle's gradient on the same `W0` and the same batch.
3. Compare `sign(W0 − W1)` against `sign(g_jax)`, **per layer**.

**Reads out the sign of every emitted gradient with zero new plumbing.** Head agrees +
early backbone disagrees ⇒ the tap injection is confirmed. Report agreement as a percentage
per parameter block, not in aggregate — a tap bug will be localized, and an aggregate number
will hide it.

### 8b. Then, if needed: magnitudes

Return `%d_W*` from the emitted train step (or add a debug dump) and compare per-parameter
gradients against the oracle's, as relative error per block. More informative than signs, but
it costs a codegen change and a ~6.5 min vmfb recompile, so do 8a first.

### 8c. The fix gate

**The 8-image overfit, not a 12-epoch run.** Objectness must fall well below **27.55**; the
floor is a measured **0**. That is 30–90 min in Lean and a known-calibrated instrument. Only
after it passes is the full 12-epoch arm worth 4.4 hours. Target: beat 0.0001 and land near
the twin's **0.1391**.

---

## 9. Decision tree

| outcome | reading | next |
|---|---|---|
| oracle fails §6 validation | the instrument is wrong | fix it; do **not** compare yet |
| signs agree everywhere | the backward is fine; bug is in the optimizer/update path after all | dump magnitudes (8b); re-examine Adam despite the classifier evidence |
| head agrees, backbone signs disagree | **tap injection confirmed** | fix the `fpnTapGrad` wiring, gate on 8c |
| disagreement starts at a specific stage | localized | read that stage's backward |
| everything disagrees | forward differs after all, or the oracle is mis-wired | re-run §6.3 |

---

## 10. Gotchas — every one of these has already cost time

- **`FPN_TAG` is not optional.** The spec name IS the on-disk checkpoint prefix; a probe
  without a distinct tag silently overwrites the live arm's e2..e12 checkpoints, which every
  measurement in `yolo_assignment.md` is computed from. Nothing errors.
- **`FPN_TOWER` selects the SPEC, so set it on `infer` too.** This already cost a full
  12-epoch eval sweep. The tell: an epoch sweep with *zero* variation between checkpoints.
- **Run the train step with `IREE_BACKEND=rocm`** or the loss reduce dies with
  `'func.func' op failed to distribute`.
- **`.maxPool 2 2` ≠ torchvision's `MaxPool2d(3, stride=2, padding=1)`.** Same 112² output,
  different function. This silently corrupted the first forward diff.
- **Any unbounded op (`exp`) + global-norm grad clip = latent `inf·0 = NaN`.** Cap at source;
  the DIoU `tw,th ≤ 8` cap is why the arm trains at all.
- **An interim measurement confirming the MECHANISM is not evidence the LEVER works.**
- **`pack_raw_boxes` caps at `MAX_BBOXES = 56`** while VisDrone val averages 70.7 boxes/img
  ⇒ 34.9% of val GT is silently dropped. Both arms are scored against the same truncated GT
  so 0.0001-vs-0.1391 is valid, but **no absolute number in this thread is VisDrone protocol.**
- **jax on this box is pinned:** jax 0.10.0 + rocm7-plugin 0.9.1.post4; 0.10.2 is
  incompatible. `jax.lax.top_k` is broken on gfx1100 — not needed here.
- **`pkill -f <pattern>` kills the calling shell** when the pattern matches your own command
  line. Collect PIDs with `pgrep` and kill those.

---

## 11. Cost, and the honest alternative

Measured 2026-07-22: a 12-epoch VisDrone run is **4.4 h in Lean vs 11 min in torch (24×)**;
the 2000-step 8-image overfit is **~90 min vs ~4 min (22×)**; and any codegen change costs
**+6.5 min of vmfb compile** where torch costs zero.

**The honest alternative to this whole doc:** the sign probe (§8a) can be run against the
*existing, already-verified* PyTorch twin instead of a new JAX oracle, today, in about an
hour. If it localizes the bug, the JAX oracle was not needed to find *this* one.

Build the JAX oracle anyway — but for the right reason. Its value is **structural, not
diagnostic**: a permanent differential reference derived from the same `NetSpec`, immune to
the entire hand-written-VJP bug class, for every future detector change. That is the thing
that would have caught this months ago and will catch the next one. Do not justify it on
this single bug, which a cheaper probe may well close first.

---

## 12. Assets that already exist

| path | what |
|---|---|
| `visdrone/bespoke/model.py` | verified torch twin (`pool="lean"` for byte-exact) |
| `visdrone/bespoke/loss.py` | **digit-exact loss — port this to JAX** |
| `visdrone/bespoke/lean_ckpt.py` | Lean flat checkpoint → torch; has the layout |
| `visdrone/bespoke/data.py` | reads `data/visdrone_fpn/*.bin` |
| `visdrone/bespoke/infer.py` | dumps logits in Lean's `inferDump` format |
| `visdrone/bespoke/diff_lean.py` | forward-diff harness |
| `scripts/fpn_loss_breakdown.py` | golden numpy loss mirror |
| `scripts/anchor_loss_probe_check.py` | reference gradient incl. the detached focal weight |
| `figures/yolo_of8long/logits.bin` | Lean logits, 8 overfit images |
| `data/visdrone_fpn_of8/train.bin` | the 8 records (= first 8 of the main train.bin, verified) |
| `scripts/yolo_map_visdrone.py --fpn` | the one scorer both arms use |

Record format: `<I count>` header, then per record uint8 `[3,448,448]` (602,112 B) +
float32 `[185220]` (740,880 B) = **1,342,992 B**. Normalization is `/255` then ImageNet
mean `(0.485,0.456,0.406)` / std `(0.229,0.224,0.225)` (`ffi/f32_helpers.c:529`).
