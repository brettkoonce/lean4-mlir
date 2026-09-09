# Orin re-measure on the trained detector (aff30e28)

Host-side half done 2026-09-09 on the training box; device-side half is the
Orin Claude's, following `deploy/ORIN_SMOKE_TEST.md`. Plan: `planning/orin_rerun.md`.

## Model

| | |
|---|---|
| arm | `aff30e28` — ResNet-34 + FPN, VisDrone 448, affine p=0.50, 30 ep; mAP@0.5 0.2363 (multilabel decode) |
| weights | `.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__aff30e28_{params,bn_stats}.bin` (86,194,972 + 68,096 B) |
| binary | `lake build yolov1-visdrone-fpn` on `main` (the Sep-2 binary was the `yolo-v5-assignment` branch's) |

## Artifacts (not in git — `deploy/build/` is ignored; fetch them from this box)

| file | bytes | md5 | input |
|---|---|---|---|
| `deploy/build/detector_aff30e28.onnx` | 86,180,752 | `0a5f8b0a0b4b4398b7bc7fbd3424159d` | `image` [1,3,448,448] f32, host-normalized |
| `deploy/build/detector_aff30e28_u8.onnx` | 86,183,250 | `08f6df63e793d09db60965c6c8447620` | `image_u8` [1,448,448,3] uint8, graph does the rest (cast-first re-export; the first one, `ab574e0f…`, put a uint8 Transpose at node 0 and TensorRT refused it) |

Both: opset 18, batch 1, 4 asymmetric-pad convs, 0 Pad ops (torch 2.13's
TorchScript exporter leaves the four `F.pad`s as Pad ops with a computed
`pads` input; `export_onnx.py` now folds them into the Conv `pads` so the
graph has the shape that was gated and measured on 2026-08-28).

Export environment: `.venv-timm` (torch 2.13.0+cpu, onnx 1.22.0,
onnxruntime 1.23.2, onnxscript 0.7.2 — the last three installed today).

## Gates, all on this box

| gate | result |
|---|---|
| replica vs Lean, 64 val records, eval-mode BN (`bespoke.diff_lean --lean-logits`) | rel 4.453e-3 (max abs 1.417 on a record with logits to ±3460), obj r 1.0000 / 1.0000 / 1.0000 |
| ONNX vs Lean, 8 val records (`export_onnx.py --verify`, now relative) | rel 4.737e-4 (max abs 9.98e-3), obj r 1.0000 ×3 — identical for both files |
| ONNX vs Lean, reference frame (`--verify-frame`, regenerated golden) | rel 2.163e-4 (max abs 5.06e-3), obj r 1.0000 ×3 — identical for both files |
| fast decode == reference decode on the new golden (`--gate-decode`) | ✅ 232 detections, top 0.6175; pedestrian 133 / car 83 / people 11 / bus 2 / bicycle 1 / motor 1 / van 1 |
| whole runner via onnxruntime (`--backend ort`), both files | 232 detections, top 0.6176, same class table for both |

The Lean dump used for the row gates is `runs/2026-09-01-aff30e28-dump/logits.bin`
(548 × 185220 f32, XLA, TF32 convs). `lean_dump_main/` is the same dump re-made
today from the `main` binary (the Sep-1 one came from the `yolo-v5-assignment`
build): **byte-identical**, all 548 rows. The frame golden
`deploy/testdata/frame_logits.bin` is its row 374.

## Training-box dry run (onnxruntime, CPU provider, one Zen core — NOT the deliverable)

| file | preprocess | forward | decode+nms | total |
|---|---|---|---|---|
| plain | 3.30 ms | 30.2 ms | 3.8 ms | 37.3 ms |
| u8 fold | 0.39 ms | 30.9 ms | 3.8 ms | 35.1 ms |

Only the preprocess column transfers: it is the host work the u8 fold removes,
and on the Orin Nano that column was 11.9 ms of a 28.0 ms frame.

## Device-side (Orin Nano 8 GB, 2026-09-09)

Orin Nano 8 GB Engineering Reference Dev Kit (Super) · L4T R36.4.7 (JetPack 6.2.x) ·
TensorRT 10.3.0 · pycuda 2026.1 · numpy 1.26.4 · pillow 9.0.1 ·
power mode `pmode:0001` = **25 W** · 46-47 °C, no thermal throttle.

Two passes. Pass 1 ran with the DVFS governor free and the FIRST u8 export
(which would not build). Pass 2 ran with `jetson_clocks` applied (GPU pinned
918 MHz, `min_freq == max_freq`) and the CORRECTED u8 export
(md5 `08f6df63e793d09db60965c6c8447620`). **Pass 2 is the deliverable.**

| | plain, governor | plain, **pinned** | u8, **pinned** |
|---|---|---|---|
| preprocess | 11.14 ms | 7.11 ms | **1.55 ms** |
| forward | 15.48 ms | 5.23 ms | **5.10 ms** |
| decode+nms | 13.07 ms | 9.76 ms | 11.26 ms |
| **total** | 39.69 ms | 22.11 ms | **17.90 ms** |
| forward-only fps | 64.6 | 191.1 | 196.2 |
| **end-to-end fps** | 25.2 | 45.2 | **55.9** |

Previous device number for reference: `ctrl12`, 2026-08-28, same board/25 W/fp16 —
11.9 + 6.3 + 10.0 = 28.0 ms = 35.7 fps. **55.9 fps is a 1.57x improvement on
that**, from two independent wins that compose: pinning the clock and folding
the preprocess into the graph.

Detections, identical on both engines and both passes: **232, top 0.6180**;
pedestrian 132 / car 83 / people 12 / bus 2 / bicycle 1 / motor 1 / van 1.
Against the gate's 232 / 0.6175, fp16 moves exactly one box across the
pedestrian/people boundary (133/11 -> 132/12). Not ctrl12 (238 / 0.7109) and
not the pre-padding-fix export (279 / 0.7656).

Loader lines, both as the brief predicted:

    engine input image    (1, 3, 448, 448) float32 -> preprocessing mode 'none'; execute_async_v3
    engine input image_u8 (1, 448, 448, 3) uint8   -> preprocessing mode 'u8';   execute_async_v3

Builds: plain 66.2 s, u8 49.3 s. **Neither engine had a single fp16 fallback** —
zero layer-precision warnings in either log.

### The u8 export fix worked, and it bought exactly what it promised

The first u8 export failed to build: `export_onnx.py:374` did
`x.permute(0,3,1,2).to(torch.float32)`, exporting `image_u8 -> Transpose -> Cast`,
so node 0 was a uint8->uint8 Transpose. TensorRT accepts UINT8 only as a network
I/O tensor, never as an intermediate:

    [8] Assertion failed: legalUINT8: TensorRT does not support UINT8 types for
                          intermediate tensors!

Re-exported cast-first on the training box, the engine builds clean and the
**preprocess column drops 7.11 -> 1.55 ms, a 4.6x cut** — the host arithmetic
moved onto the GPU and the H2D copy shrank 4x (602 KB against 2.4 MB). That is
the u8 fold's entire thesis, and it is confirmed. The GPU pays almost nothing
for it: engine-level compute rises only 4.108 -> 4.153 ms.

### The forward column never regressed; the governor was hiding the clock

Pass 1's 15.48 ms against ctrl12's 6.3 ms was a bench-shape artifact, now
proven twice over. Engine-level, pinned, `trtexec --loadEngine`:

| engine | GPU compute (median) | throughput |
|---|---|---|
| `detector_aff30e28.plan` | 4.108 ms | 241.9 qps |
| `detector_aff30e28_u8.plan` | 4.153 ms | 240.3 qps |
| `detector_padfix.plan` (ctrl12, pass 1) | 4.126 ms | 241.9 qps |

All three identical — the architecture reproduced exactly, as predicted. The
Python forward number tracks the GPU clock and nothing else: `--bench`
interleaves the three stages, so between forwards the GPU idles through the CPU
preprocess+decode and devfreq falls to its 306 MHz floor. Measured:

| loop shape | GPU clock | forward |
|---|---|---|
| `trtexec`, continuous | 918 MHz | 4.32 ms |
| back-to-back `det.forward(x)` | 612 MHz | 6.82 ms |
| `--bench`, governor free | 306 MHz | 12.69 ms |
| `--bench`, `jetson_clocks` pinned | 918 MHz | **5.23 ms** |

4.32 x 918/306 = 12.9, and pinning the clock recovered 15.48 -> 5.23 ms with no
code change at all. ctrl12's 6.3 ms came from the older back-to-back bench
(`98d50fbe`) = the 612 MHz row, so 6.3 vs 15.48 compared two loop shapes, not
two models. **Any future device number must state whether `jetson_clocks` was
applied; without it the frame time is ~1.8x pessimistic.**

One honest wart: decode+nms is 11.26 ms on the u8 run against 9.76 ms on the
plain run, though it is byte-identical CPU work on identical logits. Most
likely cache/scheduling state behind a 4.6x shorter preprocess. It is not
understood, and it is why u8's end-to-end win (17.90 vs 22.11 ms) is smaller
than its preprocess win alone would imply.

### Reproducing

    sudo jetson_clocks                      # REQUIRED, or every number is ~1.8x slow
    cd deploy && python3 orin_detect.py --backend trt \
        --plan build/detector_aff30e28_u8.plan \
        --image testdata/frame.png --out out.png --bench 50

`pycuda` is NOT in system python3 — use `/home/skoonce/orinvenv/bin/python`.
`trtexec` is at `/usr/src/tensorrt/bin/trtexec`, not on PATH.
