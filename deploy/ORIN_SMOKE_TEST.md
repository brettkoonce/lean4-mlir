# Orin smoke test — brief for a Claude running on the device

> # ✅ RESOLVED — the export was one missing argument. Fixed 2026-08-28.
>
> **The ONNX route shipped a different model, and the cause was `pad="lean"`.**
> `export_onnx.py` built the PyTorch replica without it, so the replica ran
> torchvision's SYMMETRIC convolution padding against a Lean model that emits
> `MlirCodegen.samePad` — TF-style ASYMMETRIC SAME, odd pixel on the high side.
>
> | replica setting | max rel diff vs Lean | objectness r, P3/P4/P5 | decoded |
> |---|---|---|---|
> | `pool=lean pad=torchvision` (shipped) | **1.00** | 0.903 / 0.802 / 0.622 | 279 dets, top 0.7656 |
> | `pool=lean pad=lean` (fixed) | **5.0e-4** | 1.0000 / 1.0000 / 1.0000 | **238 dets, top 0.7108** |
>
> The Lean stack's own numbers for that frame are **238 dets, top 0.7109**. Both
> device-side figures in the original report are reproduced exactly on the
> training box, which is what confirms the diagnosis rather than merely fitting it.
>
> **Why it hid for so long, and why every "ruled out" was empty.** Output shapes,
> parameter counts, and `iree-compile` are identical under either convention, so
> nothing structural can catch it. The sampling grid shifts half an output pixel
> at each stride-2 convolution, and the shift COMPOUNDS through the 3 / 4 / 5
> downsampling stages feeding C3 / C4 / C5 — which is exactly why correlation
> fell with tap depth (0.90 → 0.80 → 0.62) instead of being uniform. And the
> probes that eliminated BN layout, the BN walk, BN eps, `pool`, and channel
> grouping all constructed the replica the same wrong way, so each one was
> comparing two wrong models and finding no improvement.
>
> ⚠ Note where the omission was: `grad_dump.py`, `validate_oracle.py` and
> `layout_hunt.py` all take `--pad` and all default it to `lean`. The only two
> places that hardcoded it away were `export_onnx.py` — which shipped — and
> `bespoke/diff_lean.py` — which was cited as the validation.
>
> **The residual 5.0e-4 is TF32, and that is measured, not assumed.** The
> reference dump runs XLA convolutions with no `precision_config`, so Ada uses
> TF32. Re-running the identical Lean graph under `NVIDIA_TF32_OVERRIDE=0` moves
> it to 1.6e-4, and the Lean graph disagrees with ITSELF by 7.8e-3 absolute
> across that switch — i.e. essentially the whole remaining gap is the reference
> side's own fp32 rounding.
>
> **Both gates now run and both reject the old model:**
> - `export_onnx.py --verify-frame` — self-contained, one committed frame.
> - `bespoke/diff_lean.py --lean-logits <infer dump> --eval-mode` — elementwise
>   over as many val records as you like. Verified over 64: 1.4e-3 relative,
>   objectness r = 1.0000 at all three scales.
>
> Tolerances are RELATIVE now. Logit magnitude varies by two orders of magnitude
> across records — the reference frame spans ±16, val record 40 spans −976 ..
> +1287 — so an absolute threshold is either flaky or vacuous depending on which
> record it was tuned on.
>
> ⚠ **Before anyone tries int8:** those ±1287 values are CLASS logits on
> background cells, and they are unconstrained by design — the class head is a
> softmax masked to positives, so it is never trained on background, and the
> decode discards those cells on objectness anyway. Harmless in fp32 and fp16
> (max 65504), but that dynamic range will wreck an int8 calibration.
>
> **What still stands from the device run:** the TensorRT toolchain, fp16 at
> 2.15×, the pinned-buffer fix, the three-stage timing split, and the finding that
> CPU decode at 57 ms is 6× the 4.4 ms network. All of it transfers unchanged —
> **but the 229 fps was measured on the wrong graph and should be re-measured**,
> even though the fix only changes padding and should not move throughput.

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
