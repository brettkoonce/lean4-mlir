# SIGSEGV in `RocmCommandBuffer::LaunchGraph` during steady-state training (gfx1100, ROCm 7.2)

**Status: open, no workaround found.** Not our code: reproduces with a
30-line pure-JAX script (`repro.py`) that touches nothing from this
project. Filed here because it blocks the phase-4 (PJRT) trainers on
ROCm, and because this repo's own crash reports kept pointing at our FFI
shim until the control below ruled it out.

## Summary

A jitted training loop segfaults after a nondeterministic number of
steps. It is not a compile-time failure and not a first-iteration
failure. Compilation succeeds, training runs correctly for anywhere from
a few hundred to several thousand dispatches, and then the process dies
with SIGSEGV inside AMD's HSA runtime, reached through XLA's ROCm
command-buffer (HIP graph) path.

The probability of surviving to the end scales inversely with how much
work the graph does. A single dense layer (2 outputs) usually finishes;
a 3-layer MLP (6 outputs) essentially never does.

## Environment

- GPU: 2x AMD Radeon RX 7900 XTX (gfx1100, RDNA 3)
- ROCm: 7.2.0 (driver/runtime/toolkit 7.2.26015)
- OS: Linux 7.0.0-28-generic
- Python: 3.12.3
- jax / jaxlib: 0.10.2
- jax-rocm7-pjrt / jax-rocm7-plugin: 0.10.2

Note this is *newer* than the stack in
`2026-04-jax-jit-conv-backward-segv/`, which 0.10.0 fixed. This is a
different crash: that one was at compile time in the backward pass, this
one is at run time in steady state.

## Reproducer

```
HIP_VISIBLE_DEVICES=0 python repro.py
```

784-512-512-10 MLP, SGD, batch 128, 468 steps/epoch, 12 epochs, jitted
eval between epochs, random data. Prints the epoch it reached.

**11 of 11 runs died**, at epochs 1, 1, 1, 2, 2, 3, 3, 3, 6, 7, 11. One
earlier run of an identical script did complete 12/12, so the failure is
probabilistic rather than certain.

## Backtrace

Thread is the main dispatch thread; the same signature appears on a
`py_xla_callback` thread when the crash lands in the D2H path instead.

```
Thread 3 "mnist-mlp-verif" received signal SIGSEGV.
#0  ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#1  ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#2  ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#3  ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#4  ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
#5  ?? () from /opt/rocm/lib/libamdhip64.so.7
...
#11 ?? () from /opt/rocm/lib/libamdhip64.so.7
#12 ?? () from /opt/rocm/lib/librocprofiler-sdk.so.1
#13 ?? () from /opt/rocm/lib/librocprofiler-sdk.so.1
#14 stream_executor::gpu::RocmCommandBuffer::LaunchGraph(stream_executor::Stream*)
#15 stream_executor::gpu::GpuCommandBuffer::Submit(stream_executor::Stream*)
#16 xla::gpu::CommandBufferThunk::ExecuteOnStream(...)
#17 xla::gpu::ThunkExecutor::ExecuteOnStream(...)
#18 xla::gpu::GpuExecutable::ExecuteThunksImpl(...)
...
#24 xla::CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice(...)
```

`librocprofiler-sdk` sits between HIP and XLA at frames 12-13, i.e. it
has hooked the HIP graph launch. It is **not** the cause: see below.

A second signature, seen on longer ImageNet runs, is a glibc
`free(): invalid next size (normal)` abort (exit 134) rather than a
SIGSEGV. Both appear from the same workloads, which is consistent with
one memory-corruption bug surfacing at whichever point next touches the
damaged region.

## What rules out our code

`repro.py` is pure JAX. No Lean, no `ffi/pjrt_ffi.c`, no `verified_mlir/`,
nothing from this project on the path. Measured back to back on the same
box, same GPU, same session:

| binary | complete runs |
|---|---|
| `repro.py` (pure JAX, no repo code) | 0 of 6 |
| `mnist-mlp-verified` (repo, via our PJRT shim) | 0 of 6 |

Same failure, same rate, with and without our code in the process. This
supersedes the earlier note in this repo that "not repo code" was
unproven.

## Things tried that do not fix it

| knob | result |
|---|---|
| `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (disable HIP graphs) | Survives longer (reached epochs 2, 3, 5, 10) but still dies |
| `HSA_TOOLS_LIB=` (unhook rocprofiler) | No effect; rocprofiler is a passenger, not the cause |
| `HSA_ENABLE_SDMA=0` | No effect. An early 2-of-3 result looked promising and did not survive 5 more trials |
| `GPU_MAX_HW_QUEUES=1` | No effect |
| `AMD_SERIALIZE_KERNEL=3` | No effect |
| Second GPU (`HIP_VISIBLE_DEVICES=1`) | No effect |

Disabling command buffers is the only knob that measurably moves the
survival point, which is weak evidence that the HIP-graph path is where
the corruption starts rather than merely where it is noticed.

## Impact here

Blocks the phase-4 (PJRT) path on ROCm for anything larger than the
Chapter 1 linear model. `lake run mnist` cannot complete: the linear net
usually survives, the MLP and CNN do not.

## Not yet tried

- The same repro on CUDA. Everything in this project was shaken out on
  NVIDIA first, so a clean CUDA run would localise this to the ROCm
  backend and make the upstream report much sharper.
- An older `jax-rocm7-*` (0.10.0, 0.10.1) to find where it entered.
- `AMD_LOG_LEVEL=4` around the failing dispatch.
