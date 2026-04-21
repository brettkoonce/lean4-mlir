# SIGSEGV in `miopen::kernels[abi:cxx11]()` on first `conv_general_dilated` (gfx1100 / ROCm 7.2)

## Summary

JAX 0.9.2 with `jax-rocm7-plugin 0.9.1.post3` segfaults on any call to
`jax.lax.conv_general_dilated` on gfx1100 / ROCm 7.2. The crash is deep in
MIOpen — XLA calls `miopenFindConvolutionForwardAlgorithm`, MIOpen tries to
load a precompiled kernel source, and `miopen::kernels()` (abi:cxx11 return)
dereferences a bad pointer.

Non-conv ops (arange, elementwise, reductions, matmul) work correctly.
Simple `jax.jit(lambda x: x + 1)` works. The problem is conv-specific.

## Minimal reproducer (6 lines)

```python
import jax, jax.numpy as jnp
@jax.jit
def step(x, w):
    return jax.lax.conv_general_dilated(x, w, (1, 1), "SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"))
x = jnp.ones((1, 1, 4, 4))
w = jnp.ones((1, 1, 3, 3))
print(step(x, w))
```

Smallest possible conv (1×1×4×4 input, 1×1×3×3 kernel) — crashes before any
training, any autotune DB population, any batching. Pure compile-path SEGV.

## Expected / actual

Expected: shape `(1, 1, 4, 4)` filled with 9s (or similar sum-of-ones conv).
Actual: `Segmentation fault (core dumped)` (exit code 139).

## Environment

| | |
|---|---|
| **OS** | Ubuntu 24.04.2 LTS (kernel 6.8.0-110-generic) |
| **GPU** | 2× AMD Radeon RX 7900 XTX (gfx1100) |
| **Python** | 3.12.3 |
| **jax** | 0.9.2 |
| **jaxlib** | 0.9.2 |
| **jax-rocm7-pjrt** | 0.9.1.post3 |
| **jax-rocm7-plugin** | 0.9.1.post3 |
| **ROCm** | 7.2.0.70200-43~24.04 |
| **MIOpen** | 3.5.1.70200-43~24.04 (miopen-hip) |
| **rocprofiler-sdk** | 1.1.0-43~24.04 |
| **hip-runtime-amd** | 7.2.26015.70200-43~24.04 |

Possibly relevant: jax and jaxlib are 0.9.2 but the ROCm PJRT plugin is
0.9.1.post3 — one minor version behind. The backtrace lands inside MIOpen
though, not the PJRT boundary, so I don't think this is PJRT ABI skew.

## Backtrace (gdb, Thread 126)

```
#0  miopen::kernels[abi:cxx11]()                              /opt/rocm/lib/libMIOpen.so.1
#1  miopen::GetKernelSrc(path)
#2  miopen::HIPOCProgramImpl::BuildCodeObject(...)
#3  miopen::HIPOCProgramImpl::HIPOCProgramImpl(...)
#4  std::_Construct<miopen::HIPOCProgramImpl, ...>(...)
#5  miopen::HIPOCProgram::HIPOCProgram(...)
#6  miopen::Handle::LoadProgram(...)
#7  (internal)                                                /opt/rocm/lib/libMIOpen.so.1
#8  miopen::solver::PrecompileKernels(...)
#9  miopen::solver::PrecompileSolutions(...)
#10 miopen::FindCore(...)
...
#13 miopen::FindConvolution(...)
#14 miopen::ConvolutionDescriptor::FindConvFwdAlgorithm(...)
#15 miopenFindConvolutionForwardAlgorithm
#16 stream_executor::gpu::MIOpenSupport::PopulateMIOpenFindDb(...)
      at xla_rocm_plugin.so
#17 stream_executor::gpu::MIOpenSupport::GetMIOpenConvolveAlgorithms(...)
#18 stream_executor::gpu::MIOpenSupport::GetConvolveRunners(...)
#19 xla::gpu::GetConvolutionCustomCallConfigs(...)
#20 xla::gpu::MIOpenBackend::GetSupportedConfigs(...)
#21 xla::Autotuner::GetSupportedConfigs(...)
#22 xla::Autotuner::TuneBestConfig(...)
#23 xla::Autotuner::GetConfig(...)
#24 xla::Autotuner::Autotune(...)
#25 xla::gpu::AutotunerPass::RunImpl(...)
...
#32 xla::gpu::AMDGPUCompiler::OptimizeHloPostLayoutAssignment(...)
...
#39 xla::PjRtStreamExecutorClient::Compile(...)
```

Top frame is `miopen::kernels[abi:cxx11]()` — a name-mangled function
returning a cxx11-ABI object (likely `std::string` or `std::vector<std::string>`).
`GetKernelSrc(path)` at #1 suggests MIOpen is trying to look up a precompiled
kernel source by filesystem path and getting a dangling reference — possible
static-initializer order issue or a kernel-db lookup that can't find a gfx1100
entry.

## Not the trigger

- Multi-GPU setup (`HIP_VISIBLE_DEVICES=0` / `=1` / `ROCR_VISIBLE_DEVICES=0` all crash)
- Batch size (1×1×4×4 crashes same as 128×32×28×28)
- `JAX_PLATFORMS=cpu` correctly routes around it (CPU jax works fine)
- Pure arange / JIT'd elementwise ops on ROCm work fine

## Impact

Blocks all convolution-using JAX workloads on gfx1100 / ROCm 7.2. MLP-style
dense-only networks work; any CNN (MNIST CNN, ResNet, any
torchvision-equivalent) SIGSEGV at JAX compile time.

Workaround: `JAX_PLATFORMS=cpu` (slow but correct).

## Files

All reproducers in this report:
- `repro_06_tiny_conv.py` — the 6-line minimum (1×1×4×4)
- `repro_04_conv.py` — larger conv (128×1×28×28, matches MNIST)
- `repro_05_bn.py` — conv + BN (matches MNIST-CNN first layer)
- `repro_01_import.py`, `repro_02_arange.py`, `repro_03_jit.py` — controls (pass)
