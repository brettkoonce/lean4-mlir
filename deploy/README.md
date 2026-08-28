# deploy/ — the VisDrone detector on a Jetson Orin

**Status: skeleton.** Two routes, and the measurement that decides between them.

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

## ⛔ TensorRT route — BLOCKED, see ORIN_SMOKE_TEST.md

The toolchain works and hits 229 fps. But the ONNX comes from
`demos/visdrone/bespoke/`, and that replica does **not** compute the same
function as the Lean model: max abs difference 16.5 on logits spanning ±16,
objectness correlation as low as 0.62. Its only validation compared a scalar
loss and two summary statistics against hardcoded numbers, which cannot
distinguish architectures. Fix the source before shipping anything from here.

```bash
# 1. anywhere torch exists (NOT the training box — no torch there)
python3 export_onnx.py \
    --ckpt ../.lake/build/..._ctrl12_params.bin \
    --bn   ../.lake/build/..._ctrl12_bn_stats.bin \
    --out  build/detector.onnx \
    --verify ../runs/fpn_ctrl12_final/logits.bin      # ⚠ do not skip

# 2. on the device
trtexec --onnx=detector.onnx --saveEngine=detector.plan --fp16
python3 orin_detect.py --backend trt --plan detector.plan --image frame.jpg --bench 50
```

⚠ **Why `--verify` is not optional.** The ONNX comes from the PyTorch replica in
`demos/visdrone/bespoke/`, which until now was only a validation oracle. Exporting
through it makes it load-bearing for deployment, so the export is gated on
reproducing a Lean logits dump and refuses on disagreement. A silent drift between
replica and spec would ship a detector that is subtly not the one that was measured.

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
| Orin, TensorRT fp16 | not yet measured — this is the number to get |

The middle row is 130× below the top one on hardware perhaps 4× slower, which is
the whole argument for TensorRT. An earlier version of this file projected ~16 fps
for the Orin by scaling float32 throughput; that was wrong by 32×, because it
scaled across two compilers with completely different convolution strategies.
**Get the TensorRT number with `--bench` before believing any estimate here.**

A Nano would rather have MnV4 than R34 regardless.

Note `decode()` runs on the CPU in numpy and its cost is reported separately by
`--bench`. At ~70 objects a frame the per-class NMS is the part that will want
attention first if the pipeline is tight; it is O(n²) per class.

## What to fill in on the device

1. **Camera capture** — `main()` carries the `nvarguscamerasrc` pipeline shape to
   start from. A global-shutter IMX at 1456×1088 squashes to 448 at ~3×, which is
   *kinder* than VisDrone's own ~4.5× training squash, so the domain transfers
   without tiling.
2. **A real `--bench` number**, replacing the projections above.
3. Optionally a self-contained artifact — weights baked in as constants rather
   than passed as 189 inputs — if process startup ever matters more than
   flexibility.
