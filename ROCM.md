# ROCm Bootstrap: Running on AMD GPUs (7900 XTX)

Steps to run the Lean → MLIR → IREE training pipeline on an AMD GPU.
The MLIR codegen is backend-agnostic — only the IREE compile flag and
runtime library change.

## Prerequisites

- AMD GPU with ROCm support (7900 XTX = gfx1100; the reference box has two)
- ROCm 7.2.0 installed (`/opt/rocm-7.2.0`, with `/opt/rocm` symlinked to it)
- Lean 4 toolchain via elan (pinned to **v4.31.0** by `lean-toolchain`)
- Python 3.10+ with pip

## Step 1: Clone and build Lean project

```bash
git clone <repo> lean4-mlir
cd lean4-mlir
lake build LeanMlir
```

## Step 2: Install IREE compiler (pip)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -f https://iree.dev/pip-release-links.html --pre \
    'iree-base-compiler>=3.12.0rc20260428'
```

The `>=3.12.0rc20260428` pin is the first nightly that includes the two
ROCm/HIP `Distribute`-pass fixes we hit while landing ConvNeXt
([iree-org/iree#24282](https://github.com/iree-org/iree/issues/24282) and
[#24283](https://github.com/iree-org/iree/issues/24283)). Earlier 3.11.0
nightlies fail to compile the LN-NCHW + stacked-`[0,2,3]` reduction
patterns the codegen emits.

Verify:
```bash
.venv/bin/iree-compile --version
```

## Step 3: Generate and compile MLIR for ROCm

The MLIR is identical — only the compile target changes:

```bash
# Build the test binary (generates MLIR + compiles)
lake build test-resnet-residual
```

The backend is now picked via the `IREE_BACKEND` env var — no source edits
needed. Set `IREE_BACKEND=rocm` (default is `cuda`) and everything
downstream (iree-compile flags, target-chip selection, HAL device setup)
routes through that single env var. See `LeanMlir/Train.lean:44`.

Or compile manually:
```bash
# Generate MLIR from Lean
.lake/build/bin/test-resnet-residual  # generates .lake/build/resnet34_train_step.mlir

# Compile for ROCm
.venv/bin/iree-compile \
  .lake/build/resnet34_train_step.mlir \
  --iree-hal-target-backends=rocm \
  --iree-rocm-target-chip=gfx1100 \
  -o .lake/build/resnet34_train_step.vmfb
```

## Step 4: Build IREE runtime with HIP support

The pip package only ships the compiler, not the runtime library. We need
`libiree_ffi.so` built from source with the HIP HAL driver.

```bash
# Clone IREE (if not already)
git clone https://github.com/iree-org/iree.git
cd iree
git submodule update --init

# Build runtime only (no compiler — we use pip for that)
mkdir -p ../iree-build && cd ../iree-build
cmake ../iree \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_HIP=ON \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DBUILD_SHARED_LIBS=OFF

ninja
```

This builds static `.a` libraries in `iree-build/runtime/src/iree/`.

## Step 5: Build libiree_ffi.so

The FFI wrapper links our C shim against the IREE static libraries:

```bash
cd lean4-mlir/ffi

# Compile the FFI C files
gcc -fPIC -O2 -c iree_ffi.c \
  -I../../iree/runtime/src \
  -I../../iree-build/runtime/src

# Link into shared library with IREE runtime
gcc -shared -o libiree_ffi.so iree_ffi.o \
  -Wl,--whole-archive \
  ../../iree-build/runtime/src/iree/runtime/libiree_runtime_unified.a \
  -Wl,--no-whole-archive \
  -lm -lpthread -ldl
```

Verify it has HIP symbols:
```bash
nm libiree_ffi.so | grep hip_driver
# Should show: t iree_hal_hip_driver_module_register
```

**Important:** The FFI C code references `iree_hal_cuda_driver_module_register`.
For HIP, change this to `iree_hal_hip_driver_module_register` in `iree_ffi.c`:

```c
// Line ~37 in iree_ffi.c — change:
#include "iree/hal/drivers/cuda/registration/driver_module.h"
// to:
#include "iree/hal/drivers/hip/registration/driver_module.h"

// And change the registration call:
iree_hal_cuda_driver_module_register(...)
// to:
iree_hal_hip_driver_module_register(...)

// And the device string (line ~53):
iree_make_cstring_view("cuda")
// to:
iree_make_cstring_view("hip")
```

Or better: use `#ifdef` to support both:
```c
#ifdef USE_HIP
  #include "iree/hal/drivers/hip/registration/driver_module.h"
  #define IREE_REGISTER_DRIVER iree_hal_hip_driver_module_register
  #define IREE_DEVICE_NAME "hip"
#else
  #include "iree/hal/drivers/cuda/registration/driver_module.h"
  #define IREE_REGISTER_DRIVER iree_hal_cuda_driver_module_register
  #define IREE_DEVICE_NAME "cuda"
#endif
```

Then compile with `-DUSE_HIP` for AMD.

## Step 6: Prepare Imagenette data

```bash
# Download
mkdir -p data/imagenette
python3 -c "
import urllib.request, tarfile
urllib.request.urlretrieve(
    'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
    'data/imagenette/imagenette2-320.tgz')
import tarfile
tarfile.open('data/imagenette/imagenette2-320.tgz').extractall('data/imagenette/')
"

# Preprocess to binary format
pip install Pillow numpy
python3 preprocess_imagenette.py data/imagenette/imagenette2-320 data/imagenette
```

## Step 7: Build and run

The trainer shells out to `iree-compile` at startup and loads
`ffi/libiree_ffi.so` via a relative `./ffi/` path, so: `iree-compile`
must be on `PATH` (use the `.venv`), the ROCm libraries must be on
`LD_LIBRARY_PATH`, and you must **run from the repo root**.

```bash
# Build a trainer (e.g. the verified-codegen ResNet-34)
lake build resnet34-verified

# Current run recipe (gfx1100):
export PATH="$PWD/.venv/bin:$PATH"           # iree-compile (pip wrapper)
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib   # HIP runtime libs
export IREE_BACKEND=rocm                      # target ROCm...
export IREE_CHIP=gfx1100                       # ...on the 7900 XTX
.lake/build/bin/resnet34-verified data         # data dir as argv[0]
```

`IREE_BACKEND` routes everything downstream (iree-compile flags,
target chip, HAL device); `gfx1100` is the default chip when
`IREE_BACKEND=rocm`, so `IREE_CHIP` is only needed to override it.

**Two GPUs.** The IREE runtime is single-device per process; use both
cards by launching two trainers pinned with `HIP_VISIBLE_DEVICES=0` and
`=1` (verified — both gfx1100s sit at ~80% concurrently, no contention).
The JAX-ROCm multi-GPU `Mesh` hang that once forced single-GPU
(`upstream-issues/2026-04-jax-rocm-multigpu-mesh-hang/`, ROCm/jax#746) is
**fixed as of jax 0.10.0**, so the JAX comparator's `jax.sharding` path
runs data-parallel across cards too (see `jax/README.md`).

The `*-verified` exes (`mnist-{linear,mlp,cnn}-verified`,
`cifar8{,w}{,-bn}{,-verified,-ablation}`, `resnet34-verified`,
`mobilenetv2-verified`, …) train on the proof-rendered StableHLO. Data:
the MNIST/CIFAR loaders read the dir passed as `argv[0]` (CIFAR under
`<dir>/cifar-10/`); Imagenette nets read `<dir>/imagenette/`.

The first run compiles the vmfbs (seconds for the small CIFAR/MNIST
nets, ~10–15 min for ResNet-sized models). Subsequent runs reuse the
cached vmfb in `.lake/build/` unless it is cleared or the MLIR changes.

## Expected performance

The 7900 XTX has ~61 TFLOPS f32 vs ~22 TFLOPS on the 4060 Ti.
With the same batch size 16, expect roughly 2-3x faster per step
(~1s vs ~2.5s), depending on memory bandwidth and IREE's ROCm codegen
maturity.

The 24GB VRAM (vs 16GB) also allows larger batch sizes (32 or 64),
which would further improve throughput and training stability.

## Troubleshooting

**`iree-compile` fails with ROCm target:**
- Ensure ROCm is installed: `rocminfo` should list your GPU
- Check chip name: `rocminfo | grep gfx` → should show `gfx1100`
- Try `--iree-rocm-target-chip=gfx1100` (not gfx11)

**`libiree_ffi.so` segfaults on session create:**
- Verify HIP symbols: `nm libiree_ffi.so | grep hip`
- Check ROCm runtime: `hipInfo` or `rocminfo`
- Ensure `iree_ffi.c` uses `hip` not `cuda` for driver registration

**Performance much slower than expected:**
- Check `rocm-smi` for GPU utilization
- The first step is slow (JIT compilation) — subsequent steps are cached
- Our FFI pushes 220 tensors one-by-one; this is the main overhead

## Architecture notes

The entire pipeline is backend-agnostic up to the IREE compile step:

```
Lean NetSpec → MlirCodegen.generateTrainStep → StableHLO MLIR (same for all backends)
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ▼                                   ▼
                          iree-compile --cuda              iree-compile --rocm
                                    │                                   │
                                    ▼                                   ▼
                              .vmfb (CUDA)                       .vmfb (ROCm)
                                    │                                   │
                              libiree_ffi.so                    libiree_ffi.so
                              (CUDA HAL)                         (HIP HAL)
                                    │                                   │
                                    └───────── same Lean code ──────────┘
```

Zero changes to the Lean code, MLIR codegen, or training loop.
Only the compile flags and runtime library differ.
