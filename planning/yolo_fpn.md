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
encodability ceiling, and per-class mAP@0.5 pinned at 0.0001. Two diagnosed failure modes
(both visible in the flat loss 315→304→301→300.9 converging by ~e6):

1. **Objectness pos/neg imbalance (the loss flatlines here).** The objectness focal-BCE
   sums over ALL ~12,348 anchors/img with only ~60 positive (~200:1), then `/B` (batch),
   NOT `/num_pos`. Even with focal γ=2 the real objects are swamped → "predict low
   objectness everywhere" → diffuse detections. RetinaNet is focal PLUS this balance.
2. **Class collapse (pins per-class mAP at ~0).** The softmax class head collapsed to
   "car" (9943 GT vs bus's 139) — at e12 ~every detection is labeled car (`car dets=539824`,
   most classes 0). Class-agnostic localization is 2× anchor; per-class mAP is dead.

### Tier 1 — loss rebalancing (loss-only, cheapest, DO FIRST)

Edits to `emitAnchorYoloLoss` (`MlirCodegen.lean:5146`), which `emitMultiScaleYoloLoss`
(`:5288`) calls 3×. Validate each on the eval's recall/mAP — one lever at a time.
- **[T1a] Normalize objectness by positive count / hard-negative mining.** Objectness loss
  is `Σ focal_bce / B` (`%ay{tag}_Bf`, ~`:5165`/`:5225`). Switch to `/num_pos` (count of
  obj==1 from the target mask `%{pa}_m4` already in scope), OR a fixed pos-weight, OR
  keep only the top-K≈3·num_pos hardest negatives (SSD-style). **Highest-leverage change.**
- **[T1b] Class-balanced classification.** The class term (softmax CE, `:5253`) has no
  reweighting. Add inverse-frequency class weights (VisDrone counts known) or class focal.
  FIRST check the class loss is masked to POSITIVE cells only — background leakage alone
  would drive the collapse. This lifts per-class mAP off 0.0001.

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
