# deploy/ — the VisDrone detector on a Jetson Orin

Two routes, and the measurement that decided between them.

> ## ⚠ IREE runs at ~0.5 fps on an Orin. Go through TensorRT.
>
> IREE compiles this graph for sm_87 without complaint and produces correct
> output — it is simply slow, because its CUDA backend generates its own
> convolution kernels where TensorRT dispatches to cuDNN and tensor cores and
> does fp16 natively. **That is a compiler gap, not a hardware one**, and it is
> worth stating because the obvious scaling argument gets it wrong: this detector
> measures 65 fps on an RTX 4060 Ti *under XLA*, which lowers through cuDNN.
> Projecting that to an Orin by float32 throughput predicted ~16 fps and reality
> was 32× worse. Never project across compilers.

| route | what | speed |
|---|---|---|
| **TensorRT** (`export_onnx.py` → `trtexec`) | Lean ckpt → PyTorch replica → ONNX → engine | the usable one |
| IREE (`build_orin.sh`) | Lean ckpt → batch-1 StableHLO → vmfb | ~0.5 fps, kept for portability |

The camera capture is a deliberate stub in both — on Orin the IMX path is a
sensor- and JetPack-specific GStreamer pipeline, and guessing it from here would
be worse than leaving it marked.

## TensorRT route — measured 2026-09-09 on the trained arm

Orin Nano 8 GB (25 W, JetPack 6.2, TensorRT 10.3, fp16, `jetson_clocks`
applied), the `aff30e28` weights (mAP@0.5 0.2363), uint8-input engine:
**17.90 ms a frame = 55.9 fps** end-to-end, split 1.55 ms host preprocess +
5.10 ms forward + 11.26 ms decode+NMS. The float-input engine on the same
board is 22.11 ms = 45.2 fps; engine-level GPU compute is 4.1 ms for either.
Full table, both passes, and the clock analysis:
`runs/2026-09-09-orin-remeasure/README.md`. `ORIN_SMOKE_TEST.md` is the
device brief.

⚠ Any Orin number must say whether `jetson_clocks` was applied. Governor-managed,
the same u8 pipeline reads ~1.8× slower, because the GPU idles through the CPU
stages between forwards and devfreq drops it to its 306 MHz floor.

It first shipped a **different model**: `export_onnx.py` built the PyTorch
replica without `pad="lean"`, so it ran torchvision's symmetric convolution
padding against a Lean spec that emits TF-style asymmetric SAME. Max relative
difference 1.00, objectness correlation falling 0.90 / 0.80 / 0.62 with tap
depth, 279 decoded detections against Lean's 238. Fixed; the replica now
reproduces the Lean logits to ~5e-4 relative. The 229 fps forward-only figure
from that first session was measured on the wrong graph and is not quoted.

```bash
# 1. on the training box, in the CPU-only .venv-timm (torch + onnx +
#    onnxruntime live there). ⚠ never the pinned .venv — torch drags its own
#    CUDA wheels over the pinned cuDNN and kills every JAX/XLA convolution.
P=../.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__aff30e28
../.venv-timm/bin/python export_onnx.py --regen-golden ../runs/<dump>/logits.bin   # once per arm
../.venv-timm/bin/python export_onnx.py --ckpt ${P}_params.bin --bn ${P}_bn_stats.bin \
    --out build/detector_aff30e28.onnx --opset 18 --verify-frame        # ⚠ do not skip
../.venv-timm/bin/python orin_detect.py --backend ort --onnx build/detector_aff30e28.onnx \
    --image testdata/frame.png --bench 5                               # the whole runner, no engine

# 2. on the device
trtexec --onnx=detector_aff30e28.onnx --saveEngine=detector_aff30e28.plan --fp16
python3 orin_detect.py --backend trt --plan detector_aff30e28.plan \
    --image testdata/frame.png --bench 50
```

The frame golden `testdata/frame_logits.bin` is the Lean stack's logits for
val record 374 **under one checkpoint**. Every arm that ships regenerates it
(`--regen-golden`), or `--verify-frame` rejects a correct export with the same
signature as a wrong one.

`--fold-preprocess u8` moves the permute, `/255` and normalize into the graph
and makes the input `[1,448,448,3]` uint8 — what `np.asarray(pil)` already is.
On the Orin the host preprocess was 11.9 ms of the 28.0, so this is the biggest
lever left; the input tensor's name and dtype tell `orin_detect.py` which
contract the graph wants, so the two cannot be mismatched.

⚠ **Why the gate is not optional.** The ONNX comes from the PyTorch replica in
`demos/visdrone/bespoke/`, which was only ever a validation oracle. Exporting
through it makes it load-bearing for deployment — and it was wrong. `--verify-frame`
is self-contained (`testdata/frame.png` + `testdata/frame_logits.bin`, both in git),
compares **elementwise**, and refuses on disagreement. For a broader check, the same
comparison over as many val records as you like:

```bash
FPN_TAG=<tag> ./.lake/build/bin/yolov1-visdrone-fpn infer data/visdrone_fpn runs/<out>
python3 -m bespoke.diff_lean --ckpt ..._params.bin --bn-stats ..._bn_stats.bin \
    --data data/visdrone_fpn/val.bin --lean-logits runs/<out>/logits.bin \
    --eval-mode --n 64
```

Both tolerances are RELATIVE: logit magnitude varies by two orders of magnitude
across records (the reference frame spans ±16, val record 40 spans −976 .. +1287),
so an absolute threshold is either flaky or vacuous depending on where it was tuned.
A correct export sits at 5e-4, the broken one sat at ~1.0.

⚠ **Before trying int8:** those ±1287 values are class logits on background cells.
The class head is a softmax masked to positives, so background is never trained and
those logits are unconstrained; the decode discards those cells on objectness. Fine
in fp32 and fp16 (max 65504), fatal to an int8 calibration.

## IREE route (portable, slow)

## Build (on the training box, not the device)

```bash
cd deploy
FPN_TAG=ctrl12 ./build_orin.sh          # cuda sm_87
FPN_TAG=ctrl12 ./build_orin.sh cpu      # llvm-cpu aarch64
```

Produces `build/`:

| file | what |
|---|---|
| `detector_fwd_eval_b1.mlir` | batch-1 forward graph, emitted by `yolov1-visdrone-fpn emit-deploy` |
| `detector.vmfb` | compiled module |
| `params.bin` | 21,548,743 float32 |
| `bn_stats.bin` | 17,024 float32 |

## Run (on the device)

```bash
scp -r deploy/ orin:~/detector/
ssh orin
pip install iree-base-runtime --find-links https://iree.dev/pip-release-links.html
cd ~/detector
python3 orin_detect.py --image frame.jpg --out out.png --bench 50
```

Prints per-class detection counts, writes a rendered image, and with `--bench`
reports steady-state forward milliseconds and frames per second after a warm-up.

## Why batch 1

Training artifacts are emitted at the training batch size (8). A camera runs one
frame at a time, and shipping the batch-8 graph would divide the frame rate by
eight. `emit-deploy` re-emits the same spec at batch 1 — same weights, same
190-argument calling convention, no retraining.

## The two contracts that are easy to break silently

**Preprocessing.** Normalization happens in the C loader (`ffi/f32_helpers.c`),
*not* in the graph, so the device has to reproduce it: resize to 448×448 by
**squash, not letterbox**, then uint8 → `/255` → subtract ImageNet mean →
divide by ImageNet std → CHW → flatten. Squash is not an accident — measured
23% better than letterbox on this data, because letterboxing a wide frame spends
~44% of the canvas on grey padding and shrinks objects that are already a few
pixels across. Get this wrong and the detector still runs, just worse.

**Anchors.** The head regresses residuals off per-scale k-means priors. They are
duplicated in `orin_detect.py` from `demos/MainYolov1VisdroneFpn.lean`. A
different set produces plausible-looking wrong boxes, not an error.

The 190-argument weight split is *not* in that category: `orin_detect.py` parses
the signature out of the MLIR rather than hardcoding it, and hard-fails on a size
mismatch, so a retrained or reshaped model cannot silently misload.

## Performance, and one wrong projection kept as a warning

| where | measured |
|---|---|
| RTX 4060 Ti, XLA (cuDNN) | **65 fps** end-to-end, 548 images in 8.34 s |
| Orin, IREE | **~0.5 fps** |
| Orin Nano, TensorRT fp16, u8 engine, clocks pinned | **55.9 fps** end-to-end (17.9 ms = 1.55 preprocess + 5.10 forward + 11.26 decode), `aff30e28`, 2026-09-09 |
| Orin Nano, TensorRT fp16, f32 engine, clocks pinned | 45.2 fps end-to-end (22.1 ms = 7.11 + 5.23 + 9.76) |
| Orin Nano, TensorRT fp16, `ctrl12`, 2026-08-28, governor | 35.7 fps end-to-end (28.0 ms = 11.9 + 6.3 + 10.0) — superseded |

The middle row is 130× below the top one on hardware perhaps 4× slower, which is
the whole argument for TensorRT. An earlier version of this file projected ~16 fps
for the Orin by scaling float32 throughput; that was wrong by 32×, because it
scaled across two compilers with completely different convolution strategies.
**Get the TensorRT number with `--bench` before believing any estimate here.**

A Nano would rather have MnV4 than R34 regardless.

`--bench` times the three stages separately — host preprocess, forward, and the
numpy decode+NMS — because on the Orin the network is the smallest of the three.
The decode was 49 ms before it was vectorized (`decode_reference` is kept as its
oracle, `--gate-decode` asserts they agree) and is 10–11 ms now, which makes it
the largest stage of the u8 pipeline and the next thing worth attention.

## What to fill in on the device

1. **Camera capture** — `main()` carries the `nvarguscamerasrc` pipeline shape to
   start from. A global-shutter IMX at 1456×1088 squashes to 448 at ~3×, which is
   *kinder* than VisDrone's own ~4.5× training squash, so the domain transfers
   without tiling.
2. **The decode**: at 11 ms of a 17.9 ms frame it is now 63% of the pipeline.
3. Optionally a self-contained artifact — weights baked in as constants rather
   than passed as 189 inputs — if process startup ever matters more than
   flexibility.
