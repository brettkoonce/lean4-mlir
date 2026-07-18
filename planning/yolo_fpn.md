# yolo_fpn.md — the multi-scale FPN neck (detection-infra brick #3)

Handoff doc for continuing the VisDrone detection build. Prereqs: `yolo_drone.md`
(the overall plan), `visdrone-fetch-and-wsa` memory (session log). Written
2026-07-17 after the anchor detector landed; the FPN neck is the last big brick.

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

**Status: every FD-checkable piece of the detector is verified. What's left is
non-FD-validatable plumbing (conv train step is ROCm-only) — do it + compile + train as
ONE build-loop session. START HERE ↓ (also mirrored in the `yolo-fpn-thread` memory):**

### Next-session ordered checklist (bite 7 wiring → bite 8 train)

1. **MlirCodegen** — the codegen half. Get it build-green, then `lake exe` the FPN train
   spec to emit + eyeball the MLIR *before* the ~15-min ROCm compile. Sites (all lockstep):
   - `FwdRec`: add `isFpnDetect : Bool`, `fpnTapGrad : String` (on residualBlock stage
     markers), `fpnC3SSA/fpnC4SSA/fpnC5SSA : String`, `fpnOc/fpnC3ch/fpnC4ch/fpnC5ch/fpnG5/fpnA : Nat`.
   - `emitTrainStepSig` (the 3 param walks + the m_/v_ walks, ~8476/~8925): add a `.fpnDetect`
     case emitting **6 weight-only args** `%W{p+i}` (order Wn3,Wn4,Wn5,Wh3,Wh4,Wh5), pidx += 6.
     Target arg: emit ONE flat `%y_fpn : [B, ΣA·15·g_s²]` (not the anchor's `%y_yolo`+`%m_yolo`)
     when `fpnScales` set.
   - Forward walk (emitTrainStepBody ~5348+): after each `.residualBlock` push
     `(curSSA, curShape, records.size-1)` to a `fpnStages` accumulator. Add a `.fpnDetect` case:
     take last-3 stages = C3/C4/C5, `let base := pidx`, read the 6 weights `%W{base+i}`, call
     `emitFpnDetectForward`, set curSSA=`%fpn_out` curShape=`[B,Ntot]`, push an `isFpnDetect`
     record (store C3/C4/C5 SSAs + dims + `pidx := base`), and set `fpnTapGrad` on the C3-stage
     marker = `"%fpn_dc3_dc"` and C4-stage = `"%fpn_dc4_dc"` (the deterministic dc names). pidx += 6.
   - Loss branch (~6382, alongside the yolo/anchor branch): if `fpnScales` non-empty, slice
     `%y_fpn` into 3 `[B,A·15,g_s,g_s]` targets, `emitMultiScaleYoloLoss B fpnScales tgts
     "%fpn_out" γ 5.0 "%loss" "%fpn_grad"`, seed gradSSA=`%fpn_grad` gradShape=`[B,Ntot]`.
   - Backward reverse-walk: `.fpnDetect` record → `emitFpnDetectBackward "%fpn_grad" …`
     (produces dc3/dc4/dc5 named `%fpn_dc{3,4,5}_dc` + 6 param grads `%d_W{base+i}`); set
     gradSSA=`%fpn_dc5_dc`, gradShape=`[B,c5,g5,g5]`. At the residualBlock **marker** backward
     (~6968) add, at the very start: `if r.fpnTapGrad != "" then gradSSA = add(gradSSA, r.fpnTapGrad)`.
   - Optimizer (~8128): `.fpnDetect` case → 6× `emitUpdate %W{base+i} %d_W{base+i} …` (weight-only,
     wdActive), push to paramRet/mRet/vRet. Gradient-clip list (~8062): add the 6 `%d_W{base+i}` if used.
   - Thread `fpnScales` through `generateTrainStep` (~9312) + `compileVmfbs` (Train.lean ~197)
     → `emitTrainStepBody` + `emitTrainStepSig`. Route loss on `!fpnScales.isEmpty`.
2. **Host / loader / FFI** (Train.lean + f32_helpers.c): DatasetIO override (~451) loads image +
   flat target (Σ·4 bytes); reuse `trainStepAdamF32Yolov1` with the flat target + a dummy mask
   (byte-count matches → no new train-step FFI); add a small `lean_f32_load_voc_fpn` loader (mirror
   `lean_f32_load_voc_anchor` f32_helpers.c:472, one flat target block) + `F32.loadDetBinFpn` decl.
3. **Data**: `process_split_fpn` in preprocess_visdrone.py (mirror `process_split_anchor`; write
   image u8 + the flat 3-block target [P3|P4|P5 order] + raw boxes) → write `data/visdrone_fpn`.
4. **Spec + exe**: `r34FpnDet` NetSpec (R34 backbone tapped, then `.fpnDetect 256 128 256 512 14 3`;
   **distinct `name`** so the checkpoint doesn't clobber the anchor arm; bootstrap prefix 21284672)
   + `MainYolov1VisdroneFpn` + lakefile exe. `fpnScales := [(56,a3),(28,a4),(14,a5)]` from
   `data/visdrone/anchors_fpn_*.txt`.
5. **Train + eval**: compile (~15 min, budget for the big graph) → train on the 2 GPUs → decode
   all 3 scales + merge before NMS in `yolo_map_visdrone.py` → recall/mAP vs anchor A=6 (1281 TP / 5.08%).

Everything with a gradient is finite-difference scaffolded. FD probes/exes:
`scripts/{diou_probe_check,anchor_loss_probe_check,fpn_neck_probe_check,fpn_loss_probe_check,fpn_detect_probe_check}.py`
and exes `diou-loss-probe`, `anchor-loss-probe`, `fpn-neck-probe`, `fpn-loss-probe`, `fpn-detect-probe`.

## The result that motivates the neck (A/B, VisDrone val, class-agnostic localization AP@0.5)

| arm (448 input, 14×14 grid) | epoch | true positives | recall@0.5 |
|---|---|---|---|
| single-grid √-MSE | e12 (converged) | 0 | 0% |
| DIoU box loss | e12 (converged) | 805 | 3.2% |
| anchor A=6 detector | e12 (converged) | 1281 | **5.08%** (mAP@0.5 0.0001) |

The ladder is clean and monotonic — every FD-verified brick moves the number. But the
anchor detector **plateaus at ~5% recall**: only ~61% of GT boxes are encodable at a
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

## Remaining bites (large but bounded, mostly mechanical hand-emission)

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
   masks, no FFI mask). **Left:** `process_split_fpn` + CLI + the `data/visdrone_fpn` 4 GB
   write — hold until the on-disk record layout is co-designed with the FPN loader (bite 7);
   writing 4 GB before the consumer exists risks a format redo.

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
   **Left = the emitTrainStepBody plumbing** (param alloc/init/SGD for the 6 weights, backbone
   tap-injection at the residualBlock markers, 3-target sig) — assembly of this verified block,
   validated only by a ROCm train run.

8. **Decode + train.** `yolo_map_visdrone.py`: decode all 3 scales (per scale: anchor
   decode as now, at that grid) and merge before NMS. New spec `r34FpnDet` + demo/exe +
   `data/visdrone_fpn`. Train on freed GPU; eval vs the anchor A=6 baseline (1281 TP / 5%).

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
