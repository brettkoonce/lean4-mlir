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

Branch `visdrone-detector-orin`. It has the runner, the decode, and the reference
frame. **`git pull` before anything else** — the padding fix and the export gate
landed 2026-08-28 and the model you may already have on the device predates them.

⛔ **The trained weights are NOT in git.** But you no longer export on the device:
the training box now ships a **verified** ONNX with the weights baked in, so the
replica-vs-Lean gate has already run somewhere it can run properly.

```bash
scp trainingbox:'~/lean/klawd_max_power/lean4-jax-mlir/deploy/build/detector_ctrl12_padfix.onnx' ~/ckpt/
```

## Step 0 — check the file that arrived

```bash
ls -l ~/ckpt/detector_ctrl12_padfix.onnx     # expect 86,359,705 bytes
md5sum ~/ckpt/detector_ctrl12_padfix.onnx    # expect aa80648978a87d1cd18dc0376d994147
```

**GATE:** both must match. A ~200 KB file means you have the stub from before the
external-data fold and the weights are missing — re-copy. A different md5 means a
different arm or a stale export; stop rather than guessing.

This file is `ctrl12` (mAP@0.5 0.1526), deliberately — it is the arm the 229 fps
was measured on, so throughput stays comparable. Better checkpoints exist; weights
do not change throughput.

## Step 1 — build the TensorRT engine

```bash
cd ~/lean4-jax-mlir/deploy     # wherever the checkout lives
mkdir -p build
trtexec --onnx=$HOME/ckpt/detector_ctrl12_padfix.onnx \
        --saveEngine=build/detector_padfix.plan --fp16 2>&1 | tee ~/trt_build.log
```

⚠ **Write to a NEW engine filename.** The old `detector.plan` is the wrong model;
if a stale engine gets reused the run will happily reproduce the old numbers and
look like a successful re-measure. Deleting it outright is fine too.

**GATE:** an engine file appears. Note the build time and any layer TensorRT says
it could not accelerate. If fp16 fails, retry without `--fp16` and report that,
since it costs the measured 2.15×.

The model is **opset 18** — torch silently declined to down-convert to 17, and the
export now reports what is actually in the file rather than what was requested. If
TensorRT rejects opset 18, say so; re-exporting lower is a training-box job.

## Step 2 — run the reference frame

```bash
python3 orin_detect.py --backend trt --plan build/detector_padfix.plan \
  --image testdata/frame.png --out out.png --bench 50
```

`testdata/frame.png` is one of the densest VisDrone validation frames. The Lean
stack decodes it, in float32, to:

| quantity | expected |
|---|---|
| detections | **238** |
| top score | **0.7109** |
| pedestrian / car / people | 138 / 60 / 13 |
| bus / motor / bicycle / van | 7 / 7 / 6 / 5 |
| truck / tricycle | 1 / 1 |

**GATE, and this is the whole point of the re-measure:**

- **238 detections, top ≈0.71** ⇒ the padding fix is live and you are finally
  measuring the model this repo trained.
- ⛔ **279 detections, top 0.7656** ⇒ that is the OLD, wrong model. Either a stale
  engine got reused or the old ONNX did. Go back to Step 0.
- ⛔ Single-digit detections, top near 0.001, or a uniform class spread ⇒ the
  export is broken in a new way. **Report it; do not tune `--conf-thresh` until
  the number looks better.** That failure mode has bitten this project repeatedly
  and always looked like a merely mediocre model.

fp16 will not reproduce 238/0.7109 exactly and does not need to — the previous run
found fp16 and fp32 gave identical detections, so a small drift is fine and a jump
to 279 is not.

## Step 3 — the number

`--bench 50` reports steady-state forward milliseconds and fps after a warm-up,
and reports decode-plus-suppression separately since that runs on the CPU in numpy.

**Report all three: forward fps, decode ms, total.** Last time: 229 fps forward
(4.4 ms; trtexec GPU compute 4.16 ms) against **57 ms of CPU decode**, 43 ms of
which was pure-Python per-class NMS over 300 boxes. The decode is 6× the network,
so it is the real bottleneck and the next thing worth fixing — but confirm that
rather than assuming it, since it is a different machine state now.

⚠ **The device's own fixes never came back to this branch.** `git log` on
`orin_detect.py` shows only the two original commits, so the pinned-buffer fix
from the last run is not here. If the device still has a locally-patched copy that
worked, **use that one** and send the diff back this time — the version in git will
run, but its host-to-device copy is unpinned and slower.

## What to send back

1. **The detection count and top score first.** 238 / ≈0.71 means the right model;
   279 / 0.7656 means the old one and the rest of the numbers are void.
2. fps: forward, decode, total. This is the deliverable.
3. Whether fp16 worked, and the trtexec build time.
4. The class spread on the reference frame, against the table.
5. Anything that needed changing to work, precisely, **as a diff** — the last
   round's device-side fixes never made it back and are now lost.

## Known unverified pieces

- `TrtDetector` in `orin_detect.py` was written blind, with no Orin on the build
  box. It uses the TensorRT 8.x/10 `execute_async_v3` shape; an older runtime may
  need `execute_async_v2` with a bindings list. ⚠ It DID run on the device last
  time, at 229 fps — but whatever was changed to make that happen was never sent
  back, so the file here is still the blind version. Expect to redo that work, and
  send the diff this time.
- Camera capture is a deliberate stub. The pipeline shape to start from is in the
  comment in `main()`. Do the still-image path first — a camera is pointless until
  the frames-per-second number is known.
