# yolo_fpn.md — the multi-scale FPN neck (detection-infra brick #3)

Handoff doc for the VisDrone detection build. Prereqs: `yolo_drone.md` (the overall
plan), `yolo-fpn-thread` + `visdrone-fetch-and-wsa` memories (session logs). Written
2026-07-17 after the anchor detector landed; **updated 2026-07-18: the FPN detector is
now fully wired, trained, and evaluated — it beats the anchor baseline 2.44× on recall.
The forward-looking content is now the improvements roadmap (§ below); bites 1–8 are DONE.**

## Where we are (what's built + committed)

The detection stack was rebuilt from "single 7×7 grid, mAP 0.0000" into a working
anchor detector, one FD-verified brick at a time. Anchor-detector commits on `main`:

- `31ad181` — VisDrone data pipeline (Ultralytics mirror) + the **DIoU box loss**
  (math/forward/backward/integration, all FD-verified) + anchor priors + target encoding.
- `ca68a3e` — the **anchor-YOLO loss codegen** (`emitAnchorYoloLoss`): per-anchor DIoU
  (with prior) + focal objectness + softmax class + gradient assembly. FD-verified.
- `e4bc585` — the anchor detector **wired end-to-end** (loader → codegen routing → host
  → spec/demo → decode). Builds, trains, exe `yolov1-visdrone-anchor`.
- `26ed613` — **FPN neck topology + DAG backward FD-verified** in numpy
  (`scripts/fpn_neck_check.py`). Bite 1 of this doc.

**FPN session (2026-07-17/18) — all COMMITTED on `main`:**
- `40ce38e` bite 2 — `emitFpnNeck` StableHLO fwd+bwd + `fpn-neck-probe` (FD-verified, 5 configs).
- `6cb7be5` bite 5 — coverage thesis **60.9% → 88.2%** (`scripts/visdrone_fpn_coverage.py`) +
  `encode_targets_fpn` (smoke-verified) + `data/visdrone/anchors_fpn_{p3,p4,p5}.txt`.
- `e23d695` bites 4+6 — `emitMultiScaleYoloLoss` + concat plumbing + tag-parameterized
  `emitAnchorYoloLoss`; `fpn-loss-probe` FD-verified.
- `b299520` **bite-7 CORE** — whole-detector DAG backward (`emitFpnDetectForward`/`Backward`:
  neck + 1×1-conv heads + concat + full backward); `fpn-detect-probe` at focal γ=0, all 9
  grads (dC3/dC4/dC5 + 6 param grads) FD-verified ~1e-4.
- `ca310e8` bite-7 plumbing (types) — `Layer.fpnDetect (oc c3 c4 c5 g5 A)` + `TrainConfig.fpnScales`
  + Spec/paramShapes/heInit cases. Build-green, additive, unused-until-wired.

**BITE 7/8 DONE — the detector is wired, trained, and evaluated. Commits on `main`
(pushed, HEAD `d31fa49`):**
- `c9aca30` bite-7 wiring — `.fpnDetect` in the train-step codegen (FwdRec fields, forward
  walk tap, loss branch, backward tap-injection, optimizer, sig, `fpnScales` threading).
- `1eb45f1` bite-8 host/data/spec/eval — `lean_f32_load_voc_fpn` + `F32.loadDetBinFpn`,
  DatasetIO/dispatch (`useFpnRun`, via the single-target DDPM FFI with `%y_fpn:[B,Ntot,1,1]`),
  `process_split_fpn`+`--fpn` → `data/visdrone_fpn`, `yolov1-visdrone-fpn` exe, forward/eval
  generator `.fpnDetect`, `yolo_map_visdrone.py --fpn` (3-scale decode + merge + 1 NMS).
- `3c3b809` NaN fix — cap `tw,th ≤ 8` before `exp` in `emitDiouForward` (see roadmap gotcha).
- `d31fa49` eval speed — vectorized `decode_fpn` + top-1000/img cap (was O(hours)).

**RESULT (12-epoch run, R34-ImageNet bootstrap, lr 4e-4, GPU-0 train / GPU-1 eval):
FPN e12 recall@0.5 = 12.38% (3123 TP) vs anchor A=6 5.08% (1281 TP) = 2.44×** (climbing
e2 9.82% → e6 9.34% → e12 12.38%). Multi-scale thesis CONFIRMED — P3 56² localizes the
<24px objects (77% of GT) the 14×14 grid couldn't. See the A/B table below.

## The improvements roadmap — from "beats anchor on recall" to "a good detector"

The gap now is detector QUALITY, not wiring/encoding: recall 12.38% against an 88.2%
encodability ceiling, and per-class mAP@0.5 pinned at 0.0001.

### ⚠️ MEASURED 2026-07-18 — T1a as originally written is WRONG. Do not do it.

The roadmap below originally led with "objectness is swamped by ~200:1 negatives,
normalize by `num_pos`." That was inferred from the flat loss curve, never measured.
Measuring it on the e12 val logits (`scripts/fpn_loss_breakdown.py`,
`scripts/fpn_obj_separation.py`) **refutes it**:

| term | loss/img | share |    | objectness | share of obj loss |
|---|---|---|---|---|---|
| box  | 193.2 | 49.3% |  | **positives** | **75.2%** |
| obj  |  99.5 | 25.4% |  | negatives     | 24.8% |
| cls  |  98.9 | 25.3% |  | | |

Background is only 24.8% of the objectness loss — i.e. **6% of total loss**. Focal γ=2
plus α=0.5 on negatives has already done its job; the count ratio is 1:197 but the *loss*
ratio is 3:1 the other way. Per-cell gradient magnitudes confirm an equilibrium rather
than a collapse (pos ≈ −0.64/cell × 63 cells vs neg ≈ +0.0014/cell × 12,285 cells), which
is exactly why objectness settled at p≈0.14 instead of running to zero. Dividing the
objectness term by `num_pos` would scale it by **1/63**, deleting the one term that is
already working; hard-negative mining targets that same 6%. **Both are contraindicated.**

The real objectness problem is **dynamic range, not balance**: every objectness logit in
the val set lies in ≈[−2.7, −1.2] (p5..p95), with pos/neg means −1.549 / −1.803 and
std 0.24 / 0.31. Overall AUC is **0.742** — real signal, but the head is barely modulating,
and the top-100 highest-objectness cells per image are only 0.2–2.6% true positives. A
single 1×1 conv **with no bias** has to manufacture the constant background offset out of
its weights, which is precisely what it is spending them on. That is a capacity/init
problem → **Tier 2 (head tower) + a prior-initialized bias**, not a loss-normalization one.
Note `TrainConfig.headPriorBias` already exists for the RetinaNet log-π trick.

### Tier 1 — loss rebalancing (loss-only, cheapest)

Edits to `emitAnchorYoloLoss` (`MlirCodegen.lean:~5146`), which `emitMultiScaleYoloLoss`
calls 3×. Validate each on the eval's recall/mAP — one lever at a time.
- **[T1a] ~~Normalize objectness by positive count / hard-negative mining~~ — REFUTED
  by measurement, see above. Skipped deliberately, not forgotten.**
- **[T1b] Class-balanced classification — DONE (uncommitted).** The collapse is real and
  worse than "everything is car": on positive cells the argmax emits only **5 of 10**
  classes, and class 3 (car) takes **75.9%** of predictions while being 38.4% of GT. The
  class term was already correctly masked to positive cells (`%{pa}_cls_maskb`), so this
  is genuine frequency imbalance, not background leakage. Encoded-target frequencies
  (`scripts/fpn_class_freq.py`): car 44.1%, pedestrian 21.2%, awning-tricycle 1.0%.
  Implemented as `TrainConfig.yoloClsWeights` → a per-cell weight `w_{c(cell)}` on both
  the loss and the gradient. Weights are **target-only**, hence exactly constant w.r.t.
  the logits, so the weighted gradient stays finite-difference checkable — `fpn-loss-probe
  --clsw` PASSES all 3 configs (fwd ~1e-8, grad vs numpy ~1e-6, grad vs f64 FD ~1e-6).
  The unweighted path is guarded by `wOn` and its SSA names are unchanged, so it emits
  the same MLIR as before: the unweighted emission contains zero `clsw` SSAs and its FD
  numbers still match the bite-6 values. (Not diffed against a HEAD-built binary —
  rebuilding mid-run would have swapped the exe out from under the eval watcher.)
  Using sqrt-inverse frequency (6.7× spread) normalized to `Σ_c f_c·w_c = 1` so the class
  term's magnitude — and its balance against box/obj — is unchanged. Full inverse is 45×.

#### T1b RESULT (12 ep, A/B vs the unweighted baseline, everything else identical)

`runs/fpn_wcls_t1b_gpu0.log`, checkpoints `…_448_wcls__visdrone__*`, evals
`figures/yolo_fpn_wcls_e{2..12}`. **T1b fixed the mechanism it targeted and did NOT move
the end metric.**

| | baseline e12 | T1b e12 |
|---|---|---|
| recall@0.5 | 12.38% (3123 TP) | 12.51% (3155 TP) |
| class-agnostic AP@0.5 | 0.0009 | 0.0009 |
| car AP@0.5 | 0.0014 | 0.0014 |
| **mAP@0.5** | **0.0001** | **0.0001** |
| classes predicted, positive cells | 5/10 | **7/10** |
| top-class share, positive cells | 75.9% | **60.3%** (GT 38.4%) |

Recall +0.13pp is noise. Class concentration on positive cells genuinely halved its excess
over GT (+37.5pp → +21.9pp), so the weights work — but mAP did not move at all. Per-epoch
recall: 4.67 → 8.39 → 9.83 → 9.99 → 10.77 → 12.51; T1b starts *worse* (baseline e2 was
9.82%) and only catches up by e6, i.e. the weighting costs early localization and repays
it later.

**Why mAP could not move — the measurement that matters for what to do next.** The class
term is masked to POSITIVE cells, so the class head is never trained on background. But
the detection list is background: 6,732,361 background cells vs 34,343 positives
(**196:1**), ~539k detections survive the 0.001 conf floor, and **537,886 of them (99.8%)
are labeled car**. Class predictions on those untrained background cells are what per-class
AP actually scores. T1b did improve background spread as a shared-weight side effect
(7/10 → 10/10 classes, top share 62.1% → 48.4%), and it still didn't matter.

So **per-class mAP is bounded by objectness ranking, not by class balance.** With
objectness AUC 0.742 and only 0.2–2.6% of each image's top-100 objectness cells being real,
the detection list is flooded with background no matter how well the class head is
calibrated on positives. This is the same conclusion the T1a refutation reached from the
other direction, and it makes Tier 2 the only lever left that can move mAP.

**Recommendation: stop working Tier 1. Both of its items are now measured out** — T1a
attacks a 6%-of-loss non-problem, T1b works but is downstream of a binding constraint.
Go to Tier 2, and specifically:
1. **Give the head a bias** (it currently has none — 6 params, all conv weights) and
   initialize it to the RetinaNet prior −log((1−π)/π), π≈0.01. `TrainConfig.headPriorBias`
   already exists for the classifier; the detector head needs the analogous hook. Without
   a bias the 1×1 conv must manufacture the constant background offset out of its weights,
   which is exactly why every objectness logit is squeezed into [−2.7, −1.2].
2. **Then the 3–4 conv 3×3 tower (T2a).** Do (1) first — it is a handful of params and
   directly targets the measured dynamic-range failure, so it isolates cleanly from the
   capacity question.
Keep T1b on (it is free, FD-verified, and strictly better on class spread) but do not
expect it to show up in mAP until objectness ranks.

### Tier 2 — head capacity (codegen, medium effort, the big jump)

- **[T2a] A real head tower.** Head is minimal: one 1×1 conv (256→A·15), no hidden layers
  (`emitFpnDetectForward`, `:1920`). The flat-by-e6 loss smells capacity-limited, not
  undertrained. Add a 3–4 conv 3×3 tower per branch (RetinaNet); new `emitFpnHead` emitter,
  FD-checkable in isolation like the other bricks. **Lets recall climb toward 88.2%.**
- **[T2b] Decouple cls/box subnets** — separate towers so each specializes (do with T2a).

### Tier 3 — refinements (only after T1/T2 move the numbers)

- 3×3 smoothing convs on P3/P4/P5 (deferred as "skip for v1"; reduce upsample-add aliasing).
- A P2 scale at 112² for the tiniest objects (coverage now cell-collision-bound on 56²).
- Multi-anchor-per-GT (IoU) or anchor-free center sampling (FCOS).
- conf-thresh / NMS-IoU / per-class score calibration — tune only AFTER classification works.

### The train → eval loop (exact recipe)

```
lake build yolov1-visdrone-fpn
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 ./.lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
# eval a checkpoint on the OTHER gpu while training runs:
B=.lake/build/resnet_34___fpn_detector_448__visdrone_
cp ${B}_params_e12.bin ${B}_params.bin; cp ${B}_bn_stats_e12.bin ${B}_bn_stats.bin
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 ./.lake/build/bin/yolov1-visdrone-fpn infer data/visdrone_fpn figures/yolo_fpn_e12
<jax-venv>/bin/python3 scripts/yolo_map_visdrone.py figures/yolo_fpn_e12/logits.bin \
  data/visdrone448/val.bin --grid 14 --fpn data/visdrone
```
`<jax-venv>` = `/home/skoonce/lean/claude_max/lean4-jax/.venv` (has iree.runtime). ~22
min/epoch on gfx1100 + ~15 min one-time compile; ckpts every 2 ep. GT for mAP is the
single-box `data/visdrone448/val.bin` (the FPN val.bin has no boxes on disk).

### Gotchas from the build + first run
- **Any unbounded op (exp) + global-norm grad-clip = latent `inf·0 = NaN`.** DIoU
  `w=anchor·exp(tw)` overflowed under the 3-scale sum's ~10× grads → `dw/dtw=inf` → clip
  `inf·(clip/inf)=NaN` → NaN at step 300. Capped `tw,th≤8` before `exp`. **When adding
  T1/T2 watch for new unbounded ops — cap at source.**
- Scorer was O(hours) at diffuse objectness (sigmoid(~0)≈0.5 > 0.001 conf floor ⇒ all
  12,348 cells pass ⇒ O(n²) NMS) → vectorized + top-1000 cap. (T1a sharpening obj fixes this.)
- FPN reuses the single-target **DDPM** train-step FFI (`%y_fpn` rank-4); do NOT add a
  mask arg. Checkpoint prefix = sanitized `name` (DISTINCT so no anchor-arm clobber).
- `{ x with … }` update rejects the `pos` field-abbrev + multi-line field lists →
  single-line `r := { r with … }`.

Everything with a gradient was finite-difference scaffolded during the build. FD probes:
`scripts/{diou_probe_check,anchor_loss_probe_check,fpn_neck_probe_check,fpn_loss_probe_check,fpn_detect_probe_check}.py`
and exes `diou-loss-probe`, `anchor-loss-probe`, `fpn-neck-probe`, `fpn-loss-probe`, `fpn-detect-probe`.

## The result that motivates the neck (A/B, VisDrone val, class-agnostic localization AP@0.5)

| arm (448 input) | epoch | true positives | recall@0.5 |
|---|---|---|---|
| single-grid √-MSE (14×14) | e12 (converged) | 0 | 0% |
| DIoU box loss (14×14) | e12 (converged) | 805 | 3.2% |
| anchor A=6 detector (14×14) | e12 (converged) | 1281 | **5.08%** (mAP@0.5 0.0001) |
| **FPN 3-scale (56/28/14)** | **e12** | **3123** | **12.38%** (class-agn AP 0.0009, mAP@0.5 0.0001) |

The ladder is clean and monotonic — every FD-verified brick moves the number, and the
**FPN breaks the single-scale wall (5.08% → 12.38%, 2.44×)**. The anchor detector
**plateaued at ~5% recall**: only ~61% of GT boxes are encodable at a
single 14×14 scale (cell collisions), and IoU 0.5 on 2–5 px objects is unforgiving.
√-MSE box widths stay 100% negative (ε-floor gradient death) even converged; the DIoU
exp param fixes that but plain-DIoU's `exp` explodes (~1e13) on some cells — which the
anchor prior (`w = anchor·exp`) fixes for free (bounded boxes). **The single-scale wall
is what the FPN neck is for.**

## The key discovery — the neck needs NO new primitives

- `bilinearUpsample` **forward AND backward VJP already exist** in the codegen
  (`emitBilinearUpsample` ~MlirCodegen.lean:885 = separable matmul with
  `bilinearWeights1D`; backward ~6935 = transpose matmul-pair). Used by `unetUp`,
  validated by BraTS UNet training. (An old memory note said "shape-only" — wrong.)
- The `unetDown`/`unetUp` **skip machinery** (encoder saves `%unet_skip_g{e}`, decoder
  restores it; gradient accumulated by name in the reverse-record backward loop) is
  structurally FPN's tap + lateral-merge, already differentiable.

So the neck is: compose 1×1 lateral convs + `bilinearUpsample` + adds, plus multi-tap
and a single concatenated multi-scale output.

## The tractability trick — concat output, no multi-output codegen

The current train step is single-output (`forward → [B,N] → loss → grad [B,N] →
backward`). Do NOT generalize to multi-output. Instead: the 3 detection heads each
flatten and **concatenate into one `[B, Ntot]` tensor**, where
`Ntot = A·15·(56² + 28² + 14²)`. The multi-scale structure lives in the loss and decode:
split the concat into 3 scales, run the (already-verified) `emitAnchorYoloLoss` on each,
sum the 3 losses; the summed-loss gradient is the 3 head-grads concatenated, which the
single-threaded backward un-concats and flows through heads → neck → backbone.

## FPN neck math (bite 1 DONE, FD-verified ~1e-8 — `scripts/fpn_neck_check.py`)

Top-down (RetinaNet-style), all levels reduced to 256 channels:

    P5 = conv1x1(C5, W5)                    # [B,256,14,14]  (stride 32)
    P4 = conv1x1(C4, W4) + upsample2(P5)    # [B,256,28,28]  (stride 16)
    P3 = conv1x1(C3, W3) + upsample2(P4)    # [B,256,56,56]  (stride 8)

Backward (each Pn cotangent routes to its lateral AND up the pyramid):

    dP4_tot = dP4 + upsample2^T(dP3);  dC3 = W3^T·dP3
    dP5_tot = dP5 + upsample2^T(dP4_tot);  dC4 = W4^T·dP4_tot;  dC5 = W5^T·dP5_tot

(Optionally a 3×3 smoothing conv on each Pn before the head — RetinaNet has it; can skip
for v1.) The numpy `upsample2`/`upsample2_T` match `bilinearWeights1D` exactly.

## Build record (bites 1–8, ALL DONE — kept for reference)

2. **Emit `emitFpnNeck` (StableHLO) + `fpn-neck-probe`.** — **DONE, FD-verified.**
   `emitFpnNeckForward`/`emitFpnNeckBackward` + helpers (`emitConv1x1Fwd`/`BwdDC`/`BwdDW`,
   `emitBilinearUpsampleBwd`) at `MlirCodegen.lean` ~5009 (defined BEFORE `emitTrainStepBody`
   so bite 6/7 can reuse them). 1×1 conv = `dot_general` (contract channel dim) + transpose to
   NCHW; reuse `emitBilinearUpsample`; add. Probe `fpnNeckProbeModule` (~9218):
   `@main(C3,C4,C5,W3,W4,W5,dP3,dP4,dP5) -> (P3,P4,P5, dC3,dC4,dC5, dW3,dW4,dW5)` — **cotangents
   are explicit inputs, NOT a scalar-loss reduce** (dodges the `[B,NC,H,W]→scalar` reduce gotcha
   AND mirrors the real train-step wiring where head backwards feed dPn straight in). Exe
   `fpn-neck-probe`, checker `scripts/fpn_neck_probe_check.py`. All 5 shape configs PASS: fwd vs
   numpy ~1e-7, dC vs oracle ~1e-6, dW vs oracle ~1e-5 (oracle self-gated by f64 FD ~1e-8).

3. **Multi-tap R34 backbone.** The R34 residualBlocks at strides 8/16/32 (after the
   `64→128`, `128→256`, `256→512` stages) must expose their outputs C3/C4/C5. Use the
   `unetDown` named-SSA save pattern: record the stage output SSA; on backward, ADD the
   FPN-routed gradient at that point (like `addSkipGrad`). Backbone = R34-ImageNet bootstrap
   (prefix 21,284,672) so keep the residualBlock structure.

4. **3 heads → flatten → concat.** Each head = `conv2d 256→256 3×3 relu → 256→A·15 1×1`
   on Pn → flatten `[B, A·15·gn²]` → concatenate the three → single `[B, Ntot]` output.
   — **Concat/split plumbing FD-verified** (with bite 6, see below). The head convs are
   plain verified `convBn` emitters — to be composed into an `emitFpnHead` (fwd = 2×
   `emitConvBnTrain`; bwd = 2× `emitConvBnBackward`) during the bite-7 wiring; no new
   math, so nothing left to FD-check for the heads (conv is ROCm-only anyway).

5. **Multi-scale target encoding** — **THESIS VALIDATED + encoder+anchors DONE; 4 GB disk
   write deferred.** Coverage measured by `scripts/visdrone_fpn_coverage.py`: single-scale
   14×14 A=6 reproduces the wall at **60.9%**; FPN 3-scale (56/28/14, 3 anchors/scale, size
   thresholds max(w,h)px 24/64) lifts it to **88.2%** — and the joint-best-anchor upper bound
   is also 88.2%, so the size-threshold assignment is near-optimal and 3 anchors/scale is
   enough (coverage is now cell-collision-bound on the 56² grid, not anchor-bound). 77% of GT
   are tiny (<24px) → the new P3 scale, exactly the objects the 14×14 grid collapsed. Per-scale
   k-means anchors saved: `data/visdrone/anchors_fpn_{p3,p4,p5}.txt`. Encoder
   `encode_targets_fpn` (in `preprocess_visdrone.py`) written + smoke-verified (`--smoke`:
   slot count == coverage recompute, shapes, **mask == obj-channel** so the loader derives
   masks, no FFI mask). **DONE (`1eb45f1`):** `process_split_fpn` + `--fpn` CLI wrote
   `data/visdrone_fpn` (6471 train 8.3 GB / 548 val) at 88.2% encoded (byte-exact 1342992/rec,
   image u8 + flat [P3|P4|P5] target only — no mask/boxes on disk).

6. **Multi-scale loss** — **DONE, FD-verified.** `emitMultiScaleYoloLoss` (MlirCodegen ~5009)
   splits the `[B,Ntot]` concat per scale, runs `emitAnchorYoloLoss` 3× (now takes a `tag`
   param — `s{i}` per scale — so its loss-level SSAs don't collide; `tag=""` keeps the two
   single-scale callers byte-identical), sums the losses, re-concats the grads. Probe
   `fpnLossProbeModule` + exe `fpn-loss-probe` + `scripts/fpn_loss_probe_check.py` (conv-free
   ⇒ CPU-compiles for FD). All configs PASS: fwd vs numpy Σ-loss ~1e-9, grad vs concat-grad
   ~1e-6, grad vs f64 FD through the concat ~1e-6. Still to wire: route on the new
   `TrainConfig.fpnScales : List (Nat × List (Float×Float))` field inside `emitTrainStepBody`.

7. **DAG backward wiring.** concat grad → un-concat to 3 head grads → each head's conv
   backward → 3 Pn grads → `emitFpnNeck` backward → C3/C4/C5 grads → seed the backbone
   backward at the tap points (accumulate via the unet skip-grad named-SSA mechanism).
   — **DAG math DONE + FD-verified.** `emitFpnDetectForward`/`emitFpnDetectBackward`
   (MlirCodegen ~5196): neck + per-scale 1×1-conv heads (minimal head, no tower/bias — 6
   params: neck laterals Wn3/4/5 + head convs Wh3/4/5) + flatten/concat; backward un-concats
   → head VJP → neck VJP, returning (dC3,dC4,dC5) + all 6 param grads. Probe
   `fpnDetectProbeModule` + exe `fpn-detect-probe` + `scripts/fpn_detect_probe_check.py`,
   run at **focal γ=0** (objectness weight becomes a true constant ⇒ whole loss exactly
   FD-able, no channel-skipping). All configs PASS: fwd ~1e-7, all 9 grads vs f64 FD ~1e-4.
   **DONE (`c9aca30`):** the `emitTrainStepBody` plumbing (forward tap, loss branch, backward
   tap-injection at the residualBlock markers, optimizer, 3-target sig) — assembled and
   iree-input-verified, then GPU-trained.

8. **Decode + train — DONE (`1eb45f1`+`d31fa49`, trained e12).** `yolo_map_visdrone.py --fpn`
   decodes all 3 scales + merges before NMS (vectorized + top-1000 cap). Spec `r34FpnDet` +
   exe `yolov1-visdrone-fpn` + `data/visdrone_fpn`. Trained 12 ep on GPU 0 / eval'd on GPU 1:
   **recall@0.5 12.38% (3123 TP) vs anchor A=6 5.08% (1281 TP) = 2.44×.** See the roadmap
   at the top for the mAP gap (objectness imbalance + class collapse) and how to close it.

## Reference (files / functions / how-to)

- Anchor loss: `emitAnchorYoloLoss` (MlirCodegen ~4870), routed when `TrainConfig.anchors`
  non-empty via `emitTrainStepBody` yolo branch (~6250, `if yoloAnchors.isEmpty then …`).
- DIoU block: `emitDiouForward`/`emitDiouBackward` (parameterized `pfx`, `anchorW/anchorH`).
- Loader: `lean_f32_load_voc_anchor` (ffi) / `F32.loadDetBinAnchor` — returns TARGET-ONLY
  (mask derived from target obj channels; NO FFI mask input needed — reuse this trick).
- `NetSpec.detStride` (grid = imageH/detStride); host anchor branch in `Train.lean` (~756).
- Data: `data/visdrone448_a6` (A=6, 14×14). Anchors `data/visdrone/anchors_a6.txt`.
- Build+train: `lake build yolov1-visdrone-anchor` then
  `IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=N ./.lake/build/bin/yolov1-visdrone-anchor data/visdrone448_a6`.
- Eval: copy `<prefix>_params_eN.bin`→`<prefix>_params.bin` (+bn_stats), run `… infer …`,
  then `python3 scripts/yolo_map_visdrone.py <logits> data/visdrone448/val.bin --grid 14
  --anchors data/visdrone/anchors_a6.txt`. GT read from the single-box 448 val.bin geometry.

## Gotchas (learned this session)

- yolo train steps DON'T CPU-compile (`vector.contract` on conv/class-reduce) — verify on
  ROCm by training; keep the FD probes CPU-compilable (two-step reduce for `[B,NC,H,W]→scalar`).
- f32 FD of a compiled vmfb has noise floor ~ε·|loss|/2h ~1e-2 — gate the emitter against an
  f64-FD-verified numpy oracle, NOT raw f32 FD.
- Detached focal weight → obj-channel grad is NOT the FD of the forward (check it vs the
  analytic detached formula; FD-check box+cls channels only).
- When looping a block per anchor/scale, parameterize its SSA prefix (`pfx`) — plain-string
  args to helper lambdas need `s!"%{pfx}_…"`, not `"%{pfx}_…"`.
- lossOut/gradOut names must not collide with a block's internal SSAs (e.g. DIoU's
  `%{pfx}_diou`).
- Train step iree-compile is slow on gfx1100 (~15 min for a big graph); the FPN concat +
  3 losses will be bigger — budget for it.
- On-disk checkpoint prefix = sanitized NetSpec `name` (+ buildTag); give the FPN spec a
  distinct name so it doesn't clobber the anchor arm.
