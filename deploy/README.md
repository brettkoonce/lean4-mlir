# deploy/ — the VisDrone detector on a Jetson Orin

**Status: skeleton.** The graph, the weights, the compile line, the decode and
the benchmark are all real and testable. The camera capture is a deliberate stub
— on Orin the IMX path is a sensor- and JetPack-specific GStreamer pipeline, and
guessing it from here would be worse than leaving it marked.

The back half of this path was already validated on real hardware: stack MLIR →
`iree-compile` (cuda sm_87 / llvm-cpu) → `iree-run-module` → exact output, both
backends. What was missing was a batch-1 graph and something to drive it. That is
what this directory is.

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

## Expected performance

Measured 65 fps end-to-end on one RTX 4060 Ti (548 images in 8.34 s, including
process start, runtime init, a 625 MB read and a 406 MB write — so forward alone
is faster). Scaling by fp32 throughput, which is arithmetic and wants replacing
with a real measurement from `--bench`:

| target | projected fp32 | with bf16 |
|---|---|---|
| AGX Orin | ~16 fps | ~30 fps |
| Orin Nano | ~4 fps | ~7 fps |

The bf16 column is not a guess about the multiplier — R34 is measured at 1.92×
whole-step in this repo. A Nano would rather have MnV4 than R34.

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
