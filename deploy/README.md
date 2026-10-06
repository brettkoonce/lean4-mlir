# deploy/ — the platforms beyond the training boxes

One page per platform, and the scripts that page runs. The support matrix — which platform
has run which rung of the ladder, suite → mnist → cifar → imagenette → imagenet — is
[`PLATFORMS.md`](../PLATFORMS.md) at the repo root, generated from
`scripts/platform/support.tsv`; the one profile that says what each platform runs with
(backend, plugin, device variable, a Jetson's allocator defaults) is
[`scripts/platform/env.sh`](../scripts/platform/env.sh); the suite that says whether a
platform runs a build at all is [`scripts/platform/check.sh`](../scripts/platform/check.sh).
A platform that needs nothing beyond the environment — a desktop CUDA or ROCm card — has no
page here: it runs `lake run mnist | cifar | imagenette` as the README's tour has them.

| page | platform | what it covers |
|---|---|---|
| [`ORIN.md`](ORIN.md) | Jetson Orin Nano 8 GB (sm_87, unified memory) | the PJRT plugin built on x86 for the board (§1), device setup and the allocator recipe (§2), the suite (§3), the verified trainers and `lake run imagenette-orin` (§4), the detector on TensorRT (§5), bringing results back (§6), the camera demos (§7) |
| [`DETECTOR.md`](DETECTOR.md) | the VisDrone detector on the Orin | the TensorRT route (`export_onnx.py` → `trtexec`, 55.9 fps) and the IREE route (`build_orin.sh`, ~0.5 fps), the frame golden gate, the two contracts a port breaks silently |
| [`ORIN_SMOKE_TEST.md`](ORIN_SMOKE_TEST.md) | the Orin | the device brief for the detector's first measurement |

ROCm and Intel Arc get a page here when they run; until then their rows in `PLATFORMS.md`
point at `historical/ROCM.md` and `planning/platform_integration.md` §6.

## Scripts

| script | runs on | does |
|---|---|---|
| [`build_orin_pjrt_plugin.sh`](build_orin_pjrt_plugin.sh) | the training box | builds the jax PJRT CUDA plugin for sm_87 in a container (ORIN.md §1) |
| [`orin_setup.sh`](orin_setup.sh) | the board | stages that plugin and the Tegra cuDNN under `~/pjrt`, writes `~/pjrt/orin_env.sh` (ORIN.md §2) |
| [`orin_imagenette.sh`](orin_imagenette.sh) | the board | one Imagenette net on the board's recipe, under a memory cap, relaunching from the checkpoint — the engine behind `lake run imagenette-orin` (ORIN.md §4) |
| [`export_onnx.py`](export_onnx.py) | the training box | the detector checkpoint → PyTorch replica → ONNX, with the frame-golden gate (DETECTOR.md) |
| [`orin_detect.py`](orin_detect.py) | the board | the detector runner: TensorRT, ONNX Runtime or IREE backend, `--bench` (DETECTOR.md) |
| [`build_orin.sh`](build_orin.sh) | the training box | the IREE route: batch-1 StableHLO → vmfb (DETECTOR.md) |

`testdata/` holds the detector's reference frame and its logits under one checkpoint;
`build/` is where the exports land and is not committed.
