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
| `deploy/build/detector_aff30e28_u8.onnx` | 86,183,250 | `ab574e0f67edf33a375e15f0cee73e50` | `image_u8` [1,448,448,3] uint8, graph does the rest |

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

## Device-side (to be filled in by the Orin run)

| | plain fp16 | u8 fp16 |
|---|---|---|
| trtexec build time / warnings | | |
| detections / top score on `testdata/frame.png` | | |
| preprocess / forward / decode ms | | |
| end-to-end fps | | |

Previous device number, for reference: `ctrl12` weights, 2026-08-28, Orin Nano
8 GB / 25 W / JetPack 6.2 / TensorRT 10.3 / fp16 — 11.9 + 6.3 + 10.0 = 28.0 ms
= 35.7 fps. Same architecture, so the forward column is expected to reproduce.
