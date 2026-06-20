# CUDA Bootstrap: Running on NVIDIA GPUs (RTX 4060 Ti)

Steps to run the Lean → MLIR → IREE training pipeline on an NVIDIA GPU.
The MLIR codegen is backend-agnostic — only the IREE compile flag and
runtime library change. This file is the CUDA mirror of [`ROCM.md`](ROCM.md);
the consolidated `libiree_ffi.so` recipe lives in [`IREE_BUILD.md`](IREE_BUILD.md).

CUDA is the **default** backend (`IREE_BACKEND=cuda`), so most of this is
"do nothing special." The reference CUDA box is **ares** (6× RTX 4060 Ti,
Ada = `sm_89`).

## Prerequisites

- NVIDIA GPU with a recent driver + CUDA toolkit (`libcudart` on the loader path)
- Lean 4 toolchain via elan (pinned to **v4.31.0** by `lean-toolchain`;
  elan auto-installs it on the first `lake` call)
- Python 3.10+ with pip

## Step 0: Pull + Lean toolchain

```bash
git pull           # picks up the 4.31.0 lean-toolchain + refreshed docs
```

elan auto-installs Lean 4.31.0 on the first `lake` invocation. Nothing
GPU-specific here.

## Step 1: Proofs (no GPU needed — instant parity)

The Lean/MLIR/proofs side is fully backend-agnostic. The CUDA box reaches
parity with the ROCm box here without touching any GPU runtime:

```bash
lake exe cache get        # ~5 GB precompiled Mathlib
lake build ProofsMinimal  # the minimum-working-set on-ramp
lake build Proofs         # whole VJP suite — backend-independent
```

## Step 2: Install IREE compiler (pip) — identical to ROCm

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -f https://iree.dev/pip-release-links.html --pre \
    'iree-base-compiler>=3.12.0rc20260428'
iree-compile --version
```

The compiler package is backend-agnostic — same `>=3.12.0rc20260428` pin
as ROCm (first nightly with the two distribute-pass fixes from landing
ConvNeXt: [iree#24282](https://github.com/iree-org/iree/issues/24282),
[#24283](https://github.com/iree-org/iree/issues/24283)).

## Step 3: CUDA runtime FFI (`ffi/libiree_ffi.so`)

You need `ffi/libiree_ffi.so` built with the **CUDA HAL**. Good news: CUDA
is the default FFI path — `ffi/iree_ffi.c` registers
`iree_hal_cuda_driver_module_register` and device `"cuda"` *unless*
`-DUSE_HIP` is passed (that flag is the ROCm override).

So, following [`IREE_BUILD.md`](IREE_BUILD.md) §2 and §4:

- Build the IREE runtime with `-DIREE_HAL_DRIVER_CUDA=ON` (instead of
  `-DIREE_HAL_DRIVER_HIP=ON`).
- Build the FFI **without** `-DUSE_HIP`.

Verify:
```bash
nm ffi/libiree_ffi.so | grep cuda_driver
# Should show: t iree_hal_cuda_driver_module_register
```

> If ares already ran the CUDA comparator traces, this `.so` is probably
> already built — just confirm the `nm` check above and skip to Step 4.

**Stale `.so` gotcha (seen on ares, 2026-06-20).** A `.so` left over from an
earlier bring-up can predate FFI entry points the Lean shim now references,
so `lake build` fails at *link* time (not compile) with e.g.:

```
ld.lld: error: undefined symbol: iree_ffi_train_step_adam_softlabel
```

`ffi/iree_ffi.c` defines these (`grep iree_ffi_train_step_adam_ ffi/iree_ffi.c`),
but the prebuilt `.so` doesn't export them (`nm ffi/libiree_ffi.so | grep
train_step_adam_softlabel` → empty). Fix: rebuild the `.so` from current
source via [`IREE_BUILD.md`](IREE_BUILD.md) §4 (recompile `iree_ffi.o`,
relink against the runtime + flatcc archives). Quick check that a `.so` is
current: `nm ffi/libiree_ffi.so | grep -c iree_ffi_train_step_adam_seg`
should be `1`.

## Step 4: Run recipe (CUDA)

```bash
export PATH="$PWD/.venv/bin:$PATH"            # iree-compile (pip wrapper)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64  # wherever libcudart lives
export IREE_BACKEND=cuda                        # (this is the code default anyway)
# Do NOT set IREE_CHIP — see the sm_89 note below.
.lake/build/bin/cifar8w-ablation data           # run from repo root; data = argv[0]
```

`IREE_BACKEND=cuda` is the default in `LeanMlir/Types.lean:484`, so the
`export` is belt-and-suspenders; it routes iree-compile flags, target chip,
and HAL device through one var.

### ⚠️ `sm_89` (Ada): leave `IREE_CHIP` unset — do not set `sm_89`

The default cuda target is `sm_86` (`Types.lean:489`), and that is
**intentional**. The IREE compiler has no GPU target metadata for `sm_89`
and later ([iree#21122](https://github.com/iree-org/iree/issues/21122),
[#22147](https://github.com/iree-org/iree/issues/22147)), so we compile to
`sm_86` PTX and let the CUDA driver **JIT-forward to Ada at load time**.
Setting `IREE_CHIP=sm_89` trips the missing-metadata bug and fails the
compile. Leave it unset.

(Reference for other cards, in case the metadata gap closes: A100 = `sm_80`,
RTX 30xx/A-series = `sm_86`, 4060 Ti/Ada = `sm_89`, H100 = `sm_90`. Check
with `nvidia-smi --query-gpu=name,compute_cap --format=csv`.)

### Multi-GPU

The IREE runtime is single-device. Pin one process per card with
`CUDA_VISIBLE_DEVICES=0` / `=1` (the CUDA analog of `HIP_VISIBLE_DEVICES`).

**ares-specific:** the 6×4060 Ti box hard-resets under load — PCIe AER
`BadTLP` storms on the cards at bus02 (idx 1) and bus62 (idx 5); it's a
link/riser fault, not power. Mask those out:

```bash
export CUDA_VISIBLE_DEVICES=0,2,3,4    # avoid idx1 (bus02) and idx5 (bus62)
```

### Data layout

Same as ROCm: CIFAR under `data/cifar-10/`, Imagenette under
`data/imagenette/`. Symlink the preprocessed bins as on the ROCm box.

## Step 5: Smoke test

```bash
lake build cifar8-verified
.lake/build/bin/cifar8-verified data    # compiles a CUDA vmfb and trains
```

First run compiles the vmfbs (seconds for the small CIFAR/MNIST nets,
~10–15 min for ResNet-sized models); subsequent runs reuse the cached vmfb
in `.lake/build/` unless it is cleared or the MLIR changes.

## Architecture notes

Identical to the ROCm path up to the IREE compile step — zero changes to
Lean code, MLIR codegen, or the training loop. Only the compile flag
(`--iree-hal-target-backends=cuda`) and the runtime library (CUDA HAL in
`libiree_ffi.so`) differ. See the diagram at the bottom of [`ROCM.md`](ROCM.md).
