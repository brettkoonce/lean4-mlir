# Apple Silicon Mac — the verified MNIST tier on Metal

**Status: works — every rung green on an M1 MacBook Air (2026-10-13). TODO: full runs.** The
MNIST tier trains end to end and matches CUDA; CIFAR-10 and ResNet-34 on Imagenette train, but
only partial runs have been done so far (§ Measured).

The trainers reach a GPU through a PJRT plugin that the shim (`ffi/pjrt_ffi.c`) `dlopen`s, so a
Mac needs a Metal plugin and nothing else. The plugin is
[jax-graft](https://github.com/dfm/jax-graft) (formerly metal-pjrt-plugin). It is XLA's GPU
compiler with a Metal backend, built from the XLA in jaxlib 0.11.2, and the JAX platform it
registers is named `mtl`. Its compiled library is the `metal-pjrt-core` wheel, which exports
the standard `GetPjrtApi` the shim already calls on CUDA. Apple's own `jax-metal` is not used:
JAX dropped it from its install docs in January 2026.

## Requirements

| | |
|---|---|
| Mac | Apple Silicon (any M-series). jax-graft's published numbers are from an 8 GB M3 Air, and an M1 is untested upstream |
| macOS | **26 or later**: jax-graft's wheel is `macosx_26_0_arm64`. macOS 26 runs on every Apple Silicon Mac, the 2020 M1 Air included |
| tools | Xcode command-line tools (`xcode-select --install`), Python 3.12+ (Homebrew, python.org or uv), elan |
| disk | ~15 GB: Mathlib's prebuilt cache plus the build |

## Run it

```bash
git clone https://github.com/brettkoonce/lean4-mlir && cd lean4-mlir
bash deploy/mac_setup.sh                 # every rung, in order
bash deploy/mac_setup.sh smoke           # resume from a rung after fixing it
```

If `uname -m` prints `x86_64`, the terminal itself runs under Rosetta. The script re-execs
itself as arm64, but a bare `lake build` typed in that terminal compiles the C helpers as
x86_64, and the arm64 link then skips them (`undefined symbol: lean_f32_*`). The `lean` rung
rebuilds any such objects. To build by hand, open a native shell first: `arch -arm64 zsh`.

| rung | proves | when it fails, the problem is in |
|---|---|---|
| `check` | arm64, macOS 26+, compiler, Python, elan, disk | the box |
| `venv` | `.venv` with `jax-graft==0.0.5`, which pulls in jax, jaxlib and `metal-pjrt-core` | pip / the wheel |
| `jax` | JAX computes on the `mtl` device, with `JAX_PLATFORMS=mtl` so it cannot fall back to the CPU quietly | the plugin itself, independent of this repo |
| `smoke` | `ffi/test_pjrt_smoke.c` through the real shim: create the Metal client, run `verified_mlir/linear_fwd.mlir` against the host's answer, compile the other five MNIST renders | the shim ↔ plugin seam: client options, compile options, the StableHLO text |
| `data` | MNIST into `data/` | the network |
| `lean` | elan installs the toolchain, `lake exe cache get` fetches Mathlib, and the three MNIST trainers build | the Lean build on macOS |
| `mnist` | `lake run mnist`: linear, MLP and CNN, trained on the GPU | the trainers on Metal |

Everything is teed to `runs/<date>-mac-setup/setup.log`, with the box's chip, memory, GPU cores,
macOS and package versions in `manifest.txt` beside it. Each trainer's own log lands under
`runs/<date>-mnist-*/`, as on the other platforms.

## A second Mac

[`MAC_HANDOFF.md`](MAC_HANDOFF.md) is the list for someone else's Mac: `mac_setup.sh` as the
smoke test, `deploy/mac_bench.sh` for a time estimate (the eight train steps timed on synthetic
data, ~15 minutes, then the ladder's and each Imagenette run's hours), the suite, the ladder,
Imagenette, the platforms row, the PR. The M1 Air's own numbers are § Measured below; the first
Mac to run the whole list, an M5 Pro Mac mini, is § Measured on M5 Pro.

## The ladder: suite through CIFAR

Once `mac_setup.sh` is green, `deploy/mac_ladder.sh` runs the rest of the 7900 XTX's ladder at
CIFAR and below (`deploy/ROCM_7900XTX.md` § Measured on ares, on the `rocm10-7900xtx` branch):

```bash
bash deploy/mac_ladder.sh                # every stage, in order
bash deploy/mac_ladder.sh cifar          # resume from a stage
bash deploy/mac_ladder.sh seeds grid     # only the stages named
```

| stage | runs | the 7900 XTX's result on ares |
|---|---|---|
| `build` | `lake build` the eight binaries below | — |
| `data` | MNIST and CIFAR-10 into `data/` | — |
| `suite` | `scripts/platform/check.sh`, tiers 0–2, into `runs/platform/<date>-<host>-metal/` | 21 pass, 3 skips (two-device tests; a Mac has one GPU, so they skip here too) |
| `mnist` | `lake run mnist`, 12 epochs | 92.10 / 97.80 / 98.60 % |
| `gradcheck` | `cifar8-bbn-gradcheck`: the batch-BN step's backward, both tiers | passes |
| `cifar` | `lake run cifar`: no BN, per-image BN, batch BN, 40 epochs | best 77.13 % |
| `seeds` | `cifar8wb-bbn-ablation` at seeds 1–5, six arms each | per-arm means 74.37–77.39 % |
| `grid` | `cifar8wb-bbn-grid` at ten head widths, 25 epochs | 76.23–78.52 % |

The seeds and grid are the Linux sweeps (`scripts/sweeps/seed_sweep.sh SUITE=bbn`,
`run_cifar8wb_bbn_grid_sweep.sh`) redone for a stock Mac, one job after another on the one GPU,
with the same log names and result columns. Those scripts need bash 4, gawk, `flock` and a
`journalctl` watchdog, none of which a stock Mac has. Every stage leaves a done-marker, so a
second run finishes the first. Checkpoints are tagged `mac-ladder`, so a stale checkpoint from
an earlier partial run is never resumed. `caffeinate` keeps the Mac awake while the ladder runs.
`sensors.tsv` logs the thermal speed limit and free memory once a minute, which shows when a
fanless Air throttles. `cifar` and `seeds` are the long stages (twelve and thirty 40-epoch
arms): 19 and 49 minutes on an M5 Pro, about four times that on an M1 Air.

For `check.sh` to run on a Mac it has a `metal` backend: the chip name stands in for a vendor
SMI, perl's `alarm` stands in for `timeout`, the tests link the shim by path (macOS's linker
never resolves `-lpjrt_ffi` to a `.so`), and the manifest runs under the venv's python.
`scripts/platform/probe.c` creates its client with the shim's `LEAN_MLIR_PJRT_PLATFORM` /
`ALLOCATOR` / `PREALLOCATE` options. Without `platform_name=METAL` the plugin looks for CUDA.
None of these options is set off a Mac.

Send back `runs/<date>-mac-ladder/`, `runs/platform/<date>-<host>-metal/` and the
`runs/<date>-{mnist,cifar8w}*/` trainer logs. The results fill the Mac row's `suite` and `cifar`
cells in `scripts/platform/support.tsv`, then `scripts/platform/platforms_md.py` regenerates
`PLATFORMS.md` and the book's table (`blueprint/src/figures/platforms.tex`).

## What is Mac-specific

- **`scripts/platform/env.sh`** has a `metal` backend, chosen when `uname -s` is Darwin. It
  points `PJRT_PLUGIN` at the venv's `metal_pjrt_core/pjrt_c_api_mtl_plugin.dylib` and, unless
  they are already set, exports the client options that `jax_graft/__init__.py` passes:

  | variable | value | becomes the client option |
  |---|---|---|
  | `LEAN_MLIR_PJRT_PLATFORM` | `METAL` | `platform_name`: the GPU client's StreamExecutor platform. Without it the client looks for CUDA |
  | `LEAN_MLIR_ALLOCATOR` | `platform` | `allocator`: XLA's pass-through allocator. The plugin caches and budgets buffers itself |
  | `LEAN_MLIR_PREALLOCATE` | `0` | `preallocate` |
  | `LEAN_MLIR_MEM_FRACTION` | `1.0` | `memory_fraction`: the real budget is the plugin's, scaled by `METAL_PJRT_MEMORY_FRACTION` (default: half of RAM) |
  | `LEAN_MLIR_STAGE_H2D` | `0` | `should_stage_host_to_device_transfers`: unified memory, nothing to stage through |

- **`ffi/pjrt_ffi.c`** reads `LEAN_MLIR_PJRT_PLATFORM`, which is unset (and the option absent)
  everywhere else. On macOS only, it calls `PJRT_Plugin_Initialize` before creating the client,
  as JAX does, and searches the venv's plugin path by default. The Linux build is unchanged.
- **`run.sh`** accepts the `metal` backend. There is one GPU, so the index argument is ignored.

## Measured

M1 MacBook Air, 8-core GPU, 16 GB, macOS 26.3.1; jax-graft 0.0.5 (jax 0.11.2); one run at a
time (two at once share the GPU and roughly halve each).

| | per epoch | result | CUDA (README) |
|---|---|---|---|
| `smoke`: `linear_fwd` vs the host | — | max error 6.4e-06; all six MNIST renders compile | — |
| MNIST linear, 12 ep | ~0.35–0.5 s | **92.10 %** | 92.10 % |
| MNIST MLP, 12 ep | ~1.4 s | **97.79 %** | 97.78 % (median of 30) |
| MNIST CNN, 12 ep | ~13.4 s | **98.71 %** | 98.78 % (median of 30) |
| CIFAR-10 `lake run cifar` | — | trains; TODO: full run | 76.3 % |
| ResNet-34, Imagenette 224² | ~5.7 min | 46.0 % val at epoch 3 (stopped); ~7.5–8 h for 80 ep. TODO: full run | 89.99 % |

The seven Imagenette nets (ResNet-34/50, MobileNetV2/V4, EfficientNet-B0, ConvNeXt-T, ViT-Tiny)
all compile on Metal, ConvNeXt's and ViT's ~550–600-output train steps in ~1.4 s. On a 16 GB
machine train with `LEAN_MLIR_IMAGENETTE_STREAM=1`: the resident f32 train split alone is
~7.4 GB of the memory the GPU shares.

The full runs the Air is too slow for are § Measured on M5 Pro, below.

## Measured on M5 Pro

Mac mini, Apple M5 Pro (18-core CPU, 20-core GPU), 64 GB, macOS 27.2, host `zs-mac-mini`;
jax-graft 0.0.5 (jax 0.12.0.1, which jax-graft does not pin — the Air had 0.11.2), Lean
v4.35.0; branch `mac-ladder` at `a42a6af1`. Steps 1–5 of [`MAC_HANDOFF.md`](MAC_HANDOFF.md),
2026-11-07 13:07 to 2026-11-08 07:58 PST, one job at a time; `sensors.tsv` shows no thermal
throttling. The ladder's stages ran one at a time by hand, the Imagenette nets each from scratch
in one piece (a run resumed from a checkpoint continues but does not reproduce the uninterrupted
one — the shuffle state is not in the checkpoint). Logs with a timestamp per line sit beside
each Imagenette log.

| | M5 Pro | 7900 XTX (ares) | CUDA |
|---|---|---|---|
| suite, tiers 0–2 | 21 pass, 3 skips (two-device) ([`runs/platform/2026-11-07-zs-mac-mini-metal/`](../runs/platform/2026-11-07-zs-mac-mini-metal/)) | 21 + 3 | — |
| MNIST linear / MLP / CNN, 12 ep | **92.10 / 97.82 / 98.56 %** | 92.10 / 97.80 / 98.60 | 92.10 / 97.78 / 98.78 (medians of 30) |
| CIFAR-10 `lake run cifar`, 12 × 40 ep | no BN 68.29 / 72.74 / 73.47 %, per-image BN 73.89 / 75.96 / 75.04 %, batch-BN f32 74.68 / 75.35 / 75.64 %, bf16 **77.11** / 76.99 / 76.56 % | best 77.13 | 76.3 best |
| batch-BN gradient check | passes both tiers (worst rel 0.0050) | passes | passes |
| batch-BN, 5 seeds × 6 arms | per-arm means 75.94–77.55 %, each range overlapping CUDA's | 74.37–77.39 | 75.29–78.01 |
| batch-BN head-width grid, 10 widths × 25 ep | 75.77–78.51 %, each width within 0.98 of CUDA's | 76.23–78.52 | 75.98–77.85 |

Wall clock: the suite 76 s, MNIST ~4 min, the ladder's build 2 min plus the trainers' compile,
`lake run cifar` 18.8 min, the grid 11.6 min, the seeds 48.6 min (3 s per CIFAR epoch at batch
128; the Air takes 12). Run directories: `runs/2026-11-07-mnist-*`, `runs/2026-11-07-cifar8w*-ablation/`,
[`runs/2026-11-07-mac-ladder/`](../runs/2026-11-07-mac-ladder/) (`gradcheck.out`, `bbn-seeds/`,
`cifar8wb-bbn-grid-xla/`, `sensors.tsv`). Seeds, per arm, mean (range): SGD 76.01 (74.47–77.50),
Nesterov 76.49 (74.95–79.00), AdamW 76.41 (75.18–77.36), bf16 SGD 75.94 (74.84–77.11), bf16
Nesterov 77.55 (75.79–78.72), bf16 AdamW 76.15 (75.09–76.56); CUDA's five seeds
([`runs/2026-10-24-bbn-seeds/`](../runs/2026-10-24-bbn-seeds/RESULTS.md)) span 74.93–77.89,
75.73–77.91, 75.00–77.25, 76.29–78.08, 77.76–78.37 and 72.22–77.47 on the same arms.

Imagenette, 80 epochs, one run each, from scratch, `LEAN_MLIR_IMAGENETTE_STREAM` off (64 GB):

| net | top-1 (95 % CI) | 7900 XTX | B50 | CUDA, 5 seeds | wall | s / epoch |
|---|---|---|---|---|---|---|
| ResNet-34 | **89.66 %** (88.66–90.57) | 90.47 | 89.68 | 89.68–90.37 | 79 min | 59 |
| ResNet-50 | **89.81 %** (88.82–90.72) | 90.34 | 89.61 | 89.58–89.96 | 114 min | 86 |
| MobileNetV2 | **89.20 %** (88.19–90.13) | 89.30 | 88.71 | 88.74–89.32 | 69 min | 52 |
| MobileNetV4-Conv-M | **85.17 %** (84.03–86.25) | 86.39 | 86.06 | 85.15–86.90 | 136 min | 102 |
| EfficientNet-B0 | **89.71 %** (88.72–90.62) | 89.45 | 90.55 | 89.45–90.50 | 128 min | 96 |
| ConvNeXt-T | **81.86 %** (80.62–83.03) | 82.60 | 82.65 | 82.27 (one run) | 223 min | 167 |
| ViT-Tiny | **68.56 %** (67.09–69.99) | 69.35 | 68.46 | 67.82–69.17 | 33 min | 24 |

Five of the seven are inside CUDA's five-seed range; ResNet-34 is one image (of 3,925) under
its low end and ConvNeXt-T 16 images under CUDA's single run. MobileNetV4 is inside the range
but at its floor, 35–48 images under the B50, Orin and 7900 XTX runs of the same net, from one
run; its final train loss (0.5072) is in their band. The step-0 loss differs from the B50's and
the XTX's (which agree with each other to six digits) by 1e-5 to 2e-2 depending on the net, and
every Mac run is deterministic against its own 1-epoch rerun; whether the drift is Metal's
numerics or this box's Imagenette preprocessing (Pillow 13.0.0, numpy 2.5.4) is open — the
reference `train.bin` / `val.bin` hash to `845fe8a4…7410` / `14ee8cb6…f19f` (sha256).
All seven together took 12.7 h; the 7900 XTX does them in 6.7. ViT ran the GPU at 34–84 %
busy with the trainer the top CPU process, so it is partly host-bound here; ResNet-34 ran it at
95–98 %. Logs: `runs/2026-11-07-{resnet34,vit,mobilenetv2,resnet50,efficientnet}-verified-adam/`,
`runs/2026-11-08-{mobilenetv4,convnext}-verified-adam/`. The bench
([`runs/2026-11-07-mac-bench/`](../runs/2026-11-07-mac-bench/)) had put ResNet-34 at 61 s an epoch
and the ladder at 1.9 h; the runs came in at 59 s and 1.65 h.

## What to expect

- **Float32 only.** Apple GPUs have no float64, and the MNIST renders use none. jax-graft
  refuses an unsupported op at compile time, naming it, rather than computing it wrongly, so
  op-coverage trouble surfaces in the `smoke` rung.
- **Small steps are dispatch-bound.** A round trip to the GPU costs ~175 µs on the plugin's M3
  measurements, so MNIST's small steps will not show the GPU off. The run is the point, not the
  time.
- **The GPU also drives the display.** jax-graft splits work to stay clear of macOS's GPU
  watchdog. Stop a run with Ctrl-C, never `kill -9`.
- **8 GB machines.** MNIST itself fits easily. The Lean build of the trainers is the heavier
  part, and has not been measured on an 8 GB Mac.

## After it runs

Send back `runs/<date>-mac-setup/` (the log and manifest) and the `runs/<date>-mnist-*/` logs.
A green `mnist` rung earns the Mac a row of numbers in `scripts/platform/support.tsv`, and with
it `PLATFORMS.md`.
