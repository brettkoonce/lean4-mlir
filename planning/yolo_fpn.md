# yolo_fpn.md — the multi-scale FPN neck (detection-infra brick #3)

Handoff doc for continuing the VisDrone detection build. Prereqs: `yolo_drone.md`
(the overall plan), `visdrone-fetch-and-wsa` memory (session log). Written
2026-07-17 after the anchor detector landed; the FPN neck is the last big brick.

## Where we are (what's built + committed)

The detection stack was rebuilt from "single 7×7 grid, mAP 0.0000" into a working
anchor detector, one FD-verified brick at a time. Four commits on `main`:

- `31ad181` — VisDrone data pipeline (Ultralytics mirror) + the **DIoU box loss**
  (math/forward/backward/integration, all FD-verified) + anchor priors + target encoding.
- `ca68a3e` — the **anchor-YOLO loss codegen** (`emitAnchorYoloLoss`): per-anchor DIoU
  (with prior) + focal objectness + softmax class + gradient assembly. FD-verified.
- `e4bc585` — the anchor detector **wired end-to-end** (loader → codegen routing → host
  → spec/demo → decode). Builds, trains, exe `yolov1-visdrone-anchor`.
- `26ed613` — **FPN neck topology + DAG backward FD-verified** in numpy
  (`scripts/fpn_neck_check.py`). Bite 1 of this doc.

Everything with a gradient is finite-difference scaffolded. The FD probes:
`scripts/{diou_grad_check,diou_probe_check,anchor_loss_probe_check,fpn_neck_check}.py`
and exes `diou-loss-probe`, `anchor-loss-probe`.

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

2. **Emit `emitFpnNeck` (StableHLO) + `fpn-neck-probe`.** Translate the verified numpy:
   1×1 conv = `stablehlo.dot_general` (contract channel dim 1) + transpose to NCHW; reuse
   `emitBilinearUpsample`; add. Backward mirrors `fpn_neck_check.py`'s `fpn_grad`. Probe:
   `@main(C3,C4,C5,W3,W4,W5) -> (P3,P4,P5)` + backward; check fwd vs numpy, dC vs FD (CPU;
   watch the `[B,NC,H,W]→scalar` reduce → use two-step reduce like the anchor cls loss).

3. **Multi-tap R34 backbone.** The R34 residualBlocks at strides 8/16/32 (after the
   `64→128`, `128→256`, `256→512` stages) must expose their outputs C3/C4/C5. Use the
   `unetDown` named-SSA save pattern: record the stage output SSA; on backward, ADD the
   FPN-routed gradient at that point (like `addSkipGrad`). Backbone = R34-ImageNet bootstrap
   (prefix 21,284,672) so keep the residualBlock structure.

4. **3 heads → flatten → concat.** Each head = `conv2d 256→256 3×3 relu → 256→A·15 1×1`
   on Pn → flatten `[B, A·15·gn²]` → concatenate the three → single `[B, Ntot]` output.

5. **Multi-scale target encoding** (`preprocess_visdrone.py`): emit targets at all 3 grids
   (56/28/14). Assign each GT to the scale whose stride best fits its size (e.g. by
   `max(w,h)·448`: <~24 px → P3, ~24–64 → P4, >~64 → P5), then to the best anchor within
   that scale. New data dir `data/visdrone_fpn` (image + 3 target blocks + boxes). Recompute
   coverage — should be far above 61% (56×56 = 3136 cells cuts collisions hugely). Recompute
   per-scale k-means anchors (`visdrone_anchors.py`, 3 anchors/scale is standard).

6. **Multi-scale loss.** In `emitTrainStepBody`: after the concat output, split into the 3
   scales and call `emitAnchorYoloLoss` 3× (each with its grid + that scale's anchors +
   its target block), sum the losses, concat the 3 grads. Route on a new config field
   (e.g. `TrainConfig.fpnScales : List (Nat × List (Float×Float))` = per-scale (grid, anchors)).

7. **DAG backward wiring.** concat grad → un-concat to 3 head grads → each head's conv
   backward → 3 Pn grads → `emitFpnNeck` backward → C3/C4/C5 grads → seed the backbone
   backward at the tap points (accumulate via the unet skip-grad named-SSA mechanism).

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
