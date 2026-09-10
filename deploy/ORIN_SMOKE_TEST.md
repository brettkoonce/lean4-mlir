# Orin re-measure — brief for a Claude running on the device

**Goal: one frames-per-second number, split three ways, for the detector this
repo actually trained (`aff30e28`, mAP@0.5 0.2363) on the Orin via TensorRT
fp16 — and the same split for the uint8-input variant, which is the one open
speed lever.** The previous device number (35.7 fps, 2026-08-28) was measured on
placeholder weights on purpose; this run replaces it.

Nothing here is expected to work first try. Report what breaks; do not paper
over a failure by loosening a check.

## Context you need

ResNet-34 + FPN multi-scale detector on VisDrone, 448×448 input, 185,220 output
logits over three grids (56/28/14, 3 anchors, 15 slots). Runs at 65 fps on an
RTX 4060 Ti under XLA. IREE on this Orin was ~0.5 fps (its CUDA backend writes
its own conv kernels); TensorRT is the route and the toolchain is proven:
JetPack 6.2 / TensorRT 10.3 / fp16 / `pycuda` worked on 2026-08-28 and gave
11.9 ms host preprocess + 6.3 ms forward + 10.0 ms decode = 28.0 ms a frame.

The device decode is argmax over the class logits; the repo's mAP headlines use
a multilabel decode. **Quote fps next to the device number, not mAP.**

## What is in the branch, and what is not

Branch **`orin/aff30e28-remeasure`**. `git pull` it before anything else — the
runner was rewritten for this run:

- `deploy/orin_detect.py`: `TrtDetector` now reads the input's name / shape /
  dtype and the output size **off the engine**, page-locks both host buffers,
  falls back to `execute_async_v2` on a TensorRT 8 runtime, and `--bench` times
  preprocess / forward / decode+NMS on separate clocks. New `--backend ort`
  runs the same pipeline on the ONNX through onnxruntime (no engine, slow, for
  when TensorRT's python binding is the problem).
- `deploy/testdata/frame_logits.bin` is now **aff30e28's** golden, so
  `--gate-decode` prints this arm's acceptance numbers.
- `deploy/export_onnx.py` is training-box only (needs torch). Nothing to run.

⛔ **The ONNX files are NOT in git** (86 MB each, `deploy/build/` is ignored).
Fetch them from the training box, where they are already gated against the Lean
stack (4.7e-4 relative over val records, objectness correlation 1.0000 at every
scale — `runs/2026-09-09-orin-remeasure/README.md` has the full table):

```bash
# host/user/checkout of this Orin live outside the repo — fill in before use:
#   device: <orin-host>      checkout: ~/lean4-jax-mlir      files: ~/ckpt/
B='~/lean/klawd_max_power/lean4-jax-mlir/deploy/build'
scp trainingbox:"$B/detector_aff30e28.onnx" trainingbox:"$B/detector_aff30e28_u8.onnx" ~/ckpt/
```

## Step 0 — check what arrived

```bash
ls -l ~/ckpt/detector_aff30e28*.onnx
md5sum ~/ckpt/detector_aff30e28*.onnx
```

| file | bytes | md5 | input tensor |
|---|---|---|---|
| `detector_aff30e28.onnx` | 86,180,752 | `0a5f8b0a0b4b4398b7bc7fbd3424159d` | `image` [1,3,448,448] float32 |
| `detector_aff30e28_u8.onnx` | 86,183,250 | `08f6df63e793d09db60965c6c8447620` | `image_u8` [1,448,448,3] uint8 (re-export 2026-09-09, cast-first) |

**GATE:** both match. A ~200 KB file is a weightless stub — re-copy. A different
md5 is a different export; stop rather than guess. Both are opset 18, batch 1.

## Step 1 — the decode, no GPU needed

```bash
cd ~/lean4-jax-mlir/deploy
python3 orin_detect.py --gate-decode
```

Asserts the fast numpy decode equals the straight-line reference on the golden
and prints what the Lean stack itself decodes for `testdata/frame.png` under
these weights:

| quantity | expected |
|---|---|
| detections | **232** |
| top score | **0.6175** |
| pedestrian / car / people | 133 / 83 / 11 |
| bus / bicycle / motor / van | 2 / 1 / 1 / 1 |

## Step 2 — build both engines

```bash
mkdir -p build
trtexec --onnx=$HOME/ckpt/detector_aff30e28.onnx \
        --saveEngine=build/detector_aff30e28.plan --fp16 2>&1 | tee ~/trt_aff30e28.log
trtexec --onnx=$HOME/ckpt/detector_aff30e28_u8.onnx \
        --saveEngine=build/detector_aff30e28_u8.plan --fp16 2>&1 | tee ~/trt_aff30e28_u8.log
```

⚠ **New engine filenames, every time.** Any `detector*.plan` already in
`build/` is a ctrl12 engine; a stale one reused will happily reproduce the old
numbers and look like a successful re-measure. Deleting them is fine.

**GATE:** two engine files. Note each build time and any layer TensorRT says it
could not run in fp16. The first u8 export was refused with `legalUINT8:
TensorRT does not support UINT8 types for intermediate tensors` at node 0 — a
uint8 Transpose ahead of the Cast; the export now casts first (md5 above is
the re-export). If a u8 build is refused again, say so and carry on with the
plain engine; the fix is a training-box export, not something to work around
here.

⚠ Before any `--bench`: `sudo jetson_clocks` (needs the human's password) and
record `nvpmodel -q`. Without it the governor drops the GPU to its 306 MHz
floor during the CPU stages and the forward column roughly triples (measured
15.5 ms governed against 4.3 ms under trtexec's continuous driving, same
engine). The bench prints a back-to-back forward number too, so both regimes
are captured either way.

## Step 3 — the reference frame, plain engine

```bash
python3 orin_detect.py --backend trt --plan build/detector_aff30e28.plan \
    --image testdata/frame.png --out out_aff30e28.png
```

The loader line prints what it read off the engine — expect
`input image (1, 3, 448, 448) float32 -> preprocessing mode 'none'` and
`execute_async_v3`. Then:

- **232 detections, top ≈ 0.6175** ⇒ the right model. fp16 will drift a little
  (last time fp16 and fp32 gave identical detection sets; a count off by one or
  two near the 0.05 threshold is fine).
- ⛔ **238 / 0.7109** ⇒ that is the **ctrl12** model: a stale engine or the old
  ONNX. Back to Step 0.
- ⛔ **279 / 0.7656** ⇒ the pre-padding-fix export from the first session.
  Same remedy.
- ⛔ Single-digit detections, top near 0.001, or a uniform class spread ⇒
  broken in a new way. **Report it; do not tune `--conf-thresh`.** That failure
  mode has bitten this project repeatedly and always looked like a mediocre model.

## Step 4 — the number

```bash
python3 orin_detect.py --backend trt --plan build/detector_aff30e28.plan \
    --image testdata/frame.png --bench 50
```

`--bench` warms up on three frames, then reports means over 50 of

```
  preprocess  xx.xx ms | forward  xx.xx ms | decode+nms  xx.xx ms | total  xx.xx ms
  forward-only  xxx.x fps | end-to-end  xx.x fps
  forward back-to-back  xx.xx ms = xxx.x fps (GPU clock not throttled by the CPU stages)
```

**All of it is the deliverable, verbatim.** Measured 2026-09-09 on this arm
with `jetson_clocks`: plain engine 7.11 / 5.23 / 9.76 / 22.11 ms → 45.2 fps
end-to-end; u8 engine 1.55 / 5.10 / 11.26 / 17.90 ms → **55.9 fps**. Governor-
managed the plain engine read 11.14 / 15.48 / 13.07 / 39.69 → 25.2 fps, with
trtexec at 4.12 ms GPU compute on the same engine — the forward column is a
clock reading, not a model one. The ctrl12 figure of 6.3 ms forward was a
back-to-back loop; compare it to the back-to-back line, not the interleaved
column.

## Step 5 — the u8 engine

```bash
python3 orin_detect.py --backend trt --plan build/detector_aff30e28_u8.plan \
    --image testdata/frame.png --out out_aff30e28_u8.png --bench 50
```

Expect the loader to say `input image_u8 (1, 448, 448, 3) uint8 ->
preprocessing mode 'u8'`. The detections must match Step 3's (the two graphs
are the same function; on the training box both give 232 / 0.6176 through
onnxruntime). The **preprocess** column is the question: on the training box it
went 3.30 → 0.39 ms, and on this Orin the plain path's 11.9 ms was 8.5 ms of
numpy arithmetic that the u8 graph now does on the GPU. Projected 28 → ~17 ms.

## What to send back

1. **Detection count and top score first**, both engines. 232 / ≈0.6175 means
   the right model; 238 / 0.7109 means ctrl12 and the rest is void.
2. The `--bench` split for both engines, verbatim.
3. Both `trtexec` build times, fp16 fallbacks, whether u8 was accepted.
4. Anything that needed changing to work, **as a diff, committed to the
   branch** — the device-side fixes were lost twice before this runner was
   rewritten; the pinned buffer and the three-stage timing you see are the
   reconstruction. If the rewrite is wrong, fix it in place and push.
5. Put 1–3 into the device table at the end of
   `runs/2026-09-09-orin-remeasure/README.md` and commit that too.

## Known unverified pieces

- `TrtDetector` was rewritten on the build box and verified on the device
  2026-09-09 (TensorRT 10.3, `execute_async_v3` path). The `execute_async_v2`
  fallback for TensorRT 8 is still untested.
- On this Orin `pycuda` is NOT in the system python3: use
  `~/orinvenv/bin/python`. `trtexec` is `/usr/src/tensorrt/bin/trtexec`, not
  on PATH. Verified 2026-09-09 on L4T R36.4.7 (JetPack 6.2.x) / TensorRT
  10.3.0 / pycuda 2026.1: `TrtDetector` ran as written, first try.
  `pip install onnxruntime` and `--backend ort --onnx ~/ckpt/…` gets detections
  (slowly, CPU) if TensorRT's binding is ever the problem, and proves the file.
- Camera capture is still a stub — the GStreamer pipeline shape is in the
  comment in `main()`. Still-image path first; the camera is pointless until
  the frame time is known.

## Traps

- **Never loosen a tolerance or a threshold to get past a gate.** The
  falling-correlation diagnostic in `export_onnx.py` and the 232/0.6175
  acceptance exist because a plausible-looking wrong model shipped twice.
- **INT8 is off the table** without new work: class logits on background cells
  are unconstrained (±1287, ±3460 seen) because the class softmax is masked to
  positives — fine in fp16, fatal to a calibration.
- The four `deploy/build/*.onnx` on the training box from before today are all
  ctrl12; only the two named above are this arm.
