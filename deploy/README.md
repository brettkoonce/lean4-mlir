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
A new platform is a row in `support.tsv`, an arm in `env.sh`, an expected file under
`scripts/platform/expected/` and a page here; `check.sh` regenerates `PLATFORMS.md` and the
book's platforms table (Appendix A) from the row, so the book's prose never counts platforms.

| page | platform | what it covers |
|---|---|---|
| [`ORIN.md`](ORIN.md) | Jetson Orin Nano 8 GB (sm_87, unified memory) | the PJRT plugin built on x86 for the board (§1), device setup and the allocator recipe (§2), the suite (§3), the verified trainers and `lake run imagenette-orin` (§4), the detector on TensorRT (§5), bringing results back (§6), the camera demos (§7) |
| [`DETECTOR.md`](DETECTOR.md) | the VisDrone detector on the Orin | the TensorRT route (`export_onnx.py` → `trtexec`, 55.9 fps) and the IREE route (`build_orin.sh`, ~0.5 fps), the frame golden gate, the two contracts a port breaks silently |
| [`ORIN_SMOKE_TEST.md`](ORIN_SMOKE_TEST.md) | the Orin | the device brief for the detector's first measurement |
| [`MAC.md`](MAC.md) | Apple Silicon Mac, macOS 26 (Metal through jax-graft's PJRT plugin) | the MNIST tier on the GPU, rung by rung: the box, the venv, JAX on `mtl`, the shim's smoke test, the data, the Lean build, `lake run mnist`. Works on an M1 Air, MNIST matching CUDA; TODO: full runs |
| [`ARC_B50.md`](ARC_B50.md) | Intel Arc Pro B50 16 GB (Battlemage, oneAPI through Intel's OpenXLA PJRT plugin, kernel 7.0) | the MNIST and CIFAR tiers on the GPU, rung by rung, and the bring-up in the order it cost time (§1: kernel 7.0, a full-VRAM BAR, Intel's runtime), the suite (§2), the SYCL copy races and the shim's serial uploads and readbacks (§3), the bus drop under load at Gen4 and the slot at Gen3 that cured it (§4). Works on minerva, MNIST and CIFAR matching CUDA, bf16 included, all seven Imagenette nets' 80 epochs inside CUDA's five-seed ranges or within two images of them (~9 h at 62–69 W); one process on the card at a time, and automatic upgrades off for overnight runs |
| [`TRAINIUM.md`](TRAINIUM.md) | AWS Trainium, trn1.2xlarge (one Trainium1 chip, two NeuronCores; Neuron SDK 2.33.0 through libneuronxla's PJRT plugin) | the rungs before any Trainium hour — `neuronx-cc` compiling the renders on any x86 box (§2), and the compile cache warmed through the plugin under Neuron's fake device (§3) — then the box (§1), the suite and the MNIST and CIFAR tiers on a NeuronCore. In preparation: every MNIST and CIFAR graph compiles for trn1, four of the suite's ten Imagenette train steps do (the depthwise nets and ConvNeXt are over the compiler's instruction cap at batch 32); nothing has run on the device yet |

ROCm gets a page here when it runs again; until then its row in `PLATFORMS.md` points at
`historical/ROCM.md`.

## Scripts

| script | runs on | does |
|---|---|---|
| [`build_orin_pjrt_plugin.sh`](build_orin_pjrt_plugin.sh) | the training box | builds the jax PJRT CUDA plugin for sm_87 in a container (ORIN.md §1) |
| [`orin_setup.sh`](orin_setup.sh) | the board | stages that plugin and the Tegra cuDNN under `~/pjrt`, writes `~/pjrt/orin_env.sh` (ORIN.md §2) |
| [`orin_imagenette.sh`](orin_imagenette.sh) | the board | one Imagenette net on the board's recipe, under a memory cap, relaunching from the checkpoint — the engine behind `lake run imagenette-orin` (ORIN.md §4) |
| [`mac_setup.sh`](mac_setup.sh) | the Mac | MAC.md's rungs as one resumable script, logged to `runs/<date>-mac-setup/` |
| [`arc_setup.sh`](arc_setup.sh) | the Arc box | ARC_B50.md's rungs as one resumable script (`check venv jax smoke suite data lean mnist cifar`), logged to `runs/<date>-arc-setup/` with the box's facts in `manifest.txt` |
| [`trainium_setup.sh`](trainium_setup.sh) | the Trainium instance (`venv` and `aot` on any x86 box) | TRAINIUM.md's rungs as one resumable script (`check venv jax smoke suite data lean mnist cifar`, and `aot` by name), logged to `runs/<date>-trainium-setup/` with the box's facts in `manifest.txt`; `scripts/platform/neuron_warm.sh` fills the instance's compile cache from an x86 box (§3) |
| [`export_onnx.py`](export_onnx.py) | the training box | the detector checkpoint → PyTorch replica → ONNX, with the frame-golden gate (DETECTOR.md) |
| [`orin_detect.py`](orin_detect.py) | the board | the detector runner: TensorRT, ONNX Runtime or IREE backend, `--bench` (DETECTOR.md) |
| [`build_orin.sh`](build_orin.sh) | the training box | the IREE route: batch-1 StableHLO → vmfb (DETECTOR.md) |

`testdata/` holds the detector's reference frame and its logits under one checkpoint;
`build/` is where the exports land and is not committed.
