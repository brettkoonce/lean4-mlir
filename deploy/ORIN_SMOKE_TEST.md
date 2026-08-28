# Orin smoke test — brief for a Claude running on the device

> # ⛔ STOP — the ONNX route ships a DIFFERENT MODEL. Measured 2026-08-28.
>
> The 229 fps result is real and worth having. **The detector it measured is not
> the one this repo trained.**
>
> Run on the training box, replica vs the Lean eval graph on `testdata/frame.png`,
> identical weights, identical input:
>
> | | value |
> |---|---|
> | max abs difference | **16.5** (logits span ±16) |
> | correlation, all channels | 0.86 / 0.75 / 0.86 at P3 / P4 / P5 |
> | correlation, objectness | 0.90 / 0.80 / **0.62** |
> | objectness mean | −3.577 Lean vs −3.501 replica |
>
> Correlated but not equal: a *similar* model, not the same function. Ruled out —
> BN stats layout (verified per-layer interleaved, 0 negative variances), the BN
> walk (36/36 layers, exact channel sequence, none missed), BN epsilon (1e-5 both
> sides), `pool="lean"` vs `"torchvision"` (both wrong), and channel grouping
> ([A,15] vs [15,A] transposes are worse, so it is not a permutation).
>
> **Root cause is the validation, not the export.** `bespoke/diff_lean.py` never
> compares elementwise. It prints the replica's scalar loss beside *hardcoded*
> Lean numbers in a string, plus objectness mean/std against another hardcoded
> pair — and those references date from the pre-shuffle-fix era. Two different
> architectures with similar loss and similar logit statistics pass that check
> trivially. The replica was never a verified oracle, and building the deployment
> path on it was a mistake on the training side, not the device side.
>
> **What still stands from the device run:** the TensorRT toolchain, fp16 at
> 2.15×, the pinned-buffer fix, the three-stage timing split, and the finding that
> CPU decode at 57 ms is 6× the 4.4 ms network. All of that transfers unchanged to
> a correct model.
>
> **Do not** wire up the camera or chase accuracy against this engine. The export
> needs a source that is gated elementwise first — most likely a minimal PyTorch
> module mirroring the Lean spec directly rather than torchvision's ResNet, loaded
> in Lean parameter order and gated against `testdata/frame_logits.bin`. The gate
> is objective, so the task is bounded.

**Goal: get one real frames-per-second number for this detector on the Orin via
TensorRT, and confirm the exported model is still the model we measured.**

Nothing here is expected to work first try. Report what breaks; do not paper over
a failure by loosening a check.

## Context you need

A ResNet-34 + FPN multi-scale detector trained on VisDrone. Scores mAP@0.5 =
0.1526 and runs at 65 fps on an RTX 4060 Ti under XLA.

**IREE on this Orin was already tried and runs at ~0.5 fps.** It is correct, just
slow, because its CUDA backend generates its own convolution kernels instead of
calling cuDNN. That is why this is the TensorRT route. Do not spend time trying
to make IREE fast.

⚠ **Do not trust throughput projections in any doc here.** An earlier estimate
scaled the 4060 Ti number by float32 throughput and predicted ~16 fps; reality
under IREE was 32× worse. The whole point of this exercise is to replace estimates
with a measurement.

## What is in the branch, and what is not

Branch `visdrone-detector-orin`. It has the export script, the runner, the decode,
and a reference frame.

⛔ **The trained weights are NOT in git** (86 MB, under `.lake/build/`). Get them
from the training box first:

```bash
scp trainingbox:'~/lean/klawd_max_power/lean4-jax-mlir/.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__ctrl12_params.bin' \
    trainingbox:'~/lean/klawd_max_power/lean4-jax-mlir/.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__ctrl12_bn_stats.bin' \
    ~/ckpt/
```

Expected sizes, check them: `params.bin` **86,194,972** bytes,
`bn_stats.bin` **68,096** bytes. A wrong size means the wrong arm.

## Step 1 — export to ONNX

Needs `torch` (JetPack usually ships it; otherwise NVIDIA's Jetson wheel, not
pip's x86 build) plus `onnxruntime` for the verify gate.

```bash
cd deploy
python3 export_onnx.py \
  --ckpt ~/ckpt/resnet_34___fpn_detector_448_wcls_pb__visdrone__ctrl12_params.bin \
  --bn   ~/ckpt/resnet_34___fpn_detector_448_wcls_pb__visdrone__ctrl12_bn_stats.bin \
  --out  build/detector.onnx
```

**GATE:** it must print `wrote build/detector.onnx`. The export loads Lean's flat
checkpoint into a PyTorch replica of the same architecture; a shape mismatch there
means the replica has drifted from the Lean spec and is a real bug worth reporting
rather than working around.

⚠ If you can get a Lean logits dump onto the device, run with
`--verify <logits.bin>` — it re-scores the export against it and refuses on
disagreement. Without that the replica is trusted rather than checked. Say clearly
in your report which of the two you did.

## Step 2 — build the TensorRT engine

```bash
trtexec --onnx=build/detector.onnx --saveEngine=build/detector.plan --fp16
```

**GATE:** an engine file appears. Note the build time and any layer that TensorRT
says it could not accelerate. If fp16 fails, retry without `--fp16` and report
that, since it halves the expected speed.

Likely snag: opset. The export defaults to 17; if TensorRT complains, re-export
with `--opset 13` or `--opset 16`.

## Step 3 — run the reference frame

```bash
python3 orin_detect.py --backend trt --plan build/detector.plan \
  --image testdata/frame.png --out out.png --bench 50
```

`testdata/frame.png` is one of the densest VisDrone validation frames. On the
training box, under float32, that exact frame decodes to:

| quantity | expected |
|---|---|
| detections | **238** |
| top score | **0.7109** |
| pedestrian / car / people | 138 / 60 / 13 |
| bus / motor / bicycle / van | 7 / 7 / 6 / 5 |
| truck / tricycle | 1 / 1 |

**GATE:** fp16 will not reproduce these exactly, and it does not need to. What
matters is the shape: a few hundred detections, pedestrian and car dominating, top
score near 0.7. **If you get single-digit detections, or a top score near 0.001,
or a uniform class spread, the export is wrong — stop and report rather than
tuning the confidence threshold to make the number look right.** That failure mode
has bitten this project repeatedly and always looked like a mediocre model.

## Step 4 — the number

`--bench 50` reports steady-state forward milliseconds and frames per second after
a warm-up, and reports decode-plus-suppression separately since that runs on the
CPU in numpy.

**Report all three: forward fps, decode ms, and total.** The decode is O(n²) per
class and at ~240 detections it may well be the bottleneck rather than the
network. If so, say so — that is a useful finding and a different fix.

## What to send back

1. fps: forward, decode, total. This is the deliverable.
2. Whether fp16 worked, and the trtexec build time.
3. The detection count and class spread on the reference frame, against the table.
4. Whether you ran `--verify` or trusted the replica.
5. Anything that needed changing to work, precisely — those fixes need to come
   back to the branch.

## Known unverified pieces

- `TrtDetector` in `orin_detect.py` was written blind, with no Orin on the build
  box. It uses the TensorRT 8.x/10 `execute_async_v3` shape; an older runtime may
  need `execute_async_v2` with a bindings list. Fixing this is expected work.
- Camera capture is a deliberate stub. The pipeline shape to start from is in the
  comment in `main()`. Do the still-image path first — a camera is pointless until
  the frames-per-second number is known.
