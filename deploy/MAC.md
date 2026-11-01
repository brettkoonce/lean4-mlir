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
fanless Air throttles. The wall time on an M1 has not been measured: `cifar` and `seeds` are
the long stages (twelve and thirty 40-epoch arms).

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

**TODO: full runs.** `lake run cifar` and an 80-epoch ResNet-34 to completion, then the other
Imagenette nets; a newer Mac (M4/M5) for a second row.

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
