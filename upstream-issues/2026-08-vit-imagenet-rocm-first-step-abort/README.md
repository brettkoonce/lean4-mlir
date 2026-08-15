# SIGSEGV at the first jitted step on gfx1100 — XLA's ROCm command-buffer path

**Status: OPEN, REPRODUCED IN 60 LINES OF PURE JAX, READY TO FILE.**
`XLA_FLAGS=--xla_gpu_enable_command_buffer=` is a complete workaround.

```
python repro.py 8 1 f32 noconv noattn                       -> SIGSEGV, 3 of 3
XLA_FLAGS=--xla_gpu_enable_command_buffer= python repro.py … -> survives, 3 of 3
```

Batch **8**. One transformer block. **No convolution and no attention.** It dies
on the first execution of the jitted train step, before any output.

## Two corrections to the earlier version of this page

This folder previously said the trigger was MIOpen's `GemmFwdRest` under-sizing
its workspace for the ViT patch-embed convolution. **That was wrong**, and the
way it was wrong is worth keeping:

1. **It is not the convolution.** MIOpen logs that conv immediately before the
   abort, which made it the obvious suspect. But `repro.py noconv noattn`
   removes every convolution *and* all attention and still crashes, and
   `ruled_out.py` issues that exact conv four ways — bare, tokenised, under
   grad, and with the driver's flat input — and all four pass. The MIOpen line
   is the last thing logged before the graph launch, not the thing that fails.
2. **It is not ViT-specific.** Nothing in the minimum is characteristic of a
   transformer beyond "several matmuls, LayerNorm, GELU and a softmax CE, under
   `grad`, in one jitted step."

Both errors came from the same move: treating the last line in a log as the
cause. The command-buffer knob is what actually separates crash from survive.

## Relationship to the LaunchGraph report

Same subsystem, and the sibling report
[`../2026-08-jax-rocm-command-buffer-launchgraph-segv/`](../2026-08-jax-rocm-command-buffer-launchgraph-segv/)
is genuinely resolved (it was a stale `/opt/rocm`; 7.2.4 fixes it, verified over
five ImageNet nets and thousands of steps). **This one survives that fix.**

So the honest summary of the ROCm command-buffer path on this box: fixed for
the workloads that used to break it, still broken for a graph shape that a
one-block transformer reaches immediately. Whatever 7.2.4 repaired, it was not
all of it.

Note the difference in *when*: the LaunchGraph bug needed hundreds to thousands
of dispatches. This one is the **first** execution, every time.

## What is measured

All on ROCm **7.2.4**, jax/jaxlib **0.11.0** + `jax-rocm7-{plugin,pjrt}` 0.11.0,
gfx1100.

| variant | command buffers ON | OFF |
|---|---|---|
| `repro.py 8 1 f32 noconv noattn` | SIGSEGV 3/3 | survives 3/3 |
| `repro.py 128 12 f32` (full ViT-Tiny) | SIGSEGV | survives |
| `repro.py 512 12 bf16` | SIGSEGV | — |
| this repo's JAX ViT-Tiny ImageNet reference (1899 lines) | SIGSEGV at first `pjit` | — |
| `../2026-08-…-launchgraph-segv/repro.py` (MLP) | survives 12/12 | — |

Not batch size (8 through 512 all crash), not depth (1 through 12), not dtype
(f32 and bf16), not autotuning (`--xla_gpu_autotune_level=0` does not help),
not the GPU (the MLP control passes on the same device immediately after).

Individually, every ingredient passes under `grad` + jit — 3D matmul, LayerNorm
over a 3D tensor, GELU, the 4D patchify transpose, a 1000-class softmax head.
Only the combination in one jitted step fails, which is consistent with this
being about the shape of the emitted command buffer rather than any one kernel.

Python-level fault location, identical for the minimum and the 1899-line
reference:

```
Fatal Python error: Segmentation fault
  File ".../jax/_src/interpreters/pxla.py", line 420 in __call__
  File ".../jax/_src/pjit.py", line 1222 in _pjit_call_impl_python
```

## Still open here

**The repo's own `vit-imagenet-verified` trainer is NOT fixed by the
workaround** — it still aborts with `free(): invalid next size` at
`--xla_gpu_enable_command_buffer=`, where pure-JAX ViT survives. The shim sets
no command-buffer compile options, and the flag demonstrably changes behaviour
in-process, so this looks like a second, distinct failure in the verified path
rather than the flag being ignored. Unresolved; do not assume the workaround
unblocks ViT on the verified path until that is chased.

## Filing

Belongs with **openxla/xla** (ROCm command buffers), not MIOpen. `repro.py` has
no repo code, no Lean, no FFI shim, and the workaround identifies the
subsystem. Worth stating in the report that the same box runs ResNet-34,
ResNet-50, MobileNetV2 and ConvNeXt on ImageNet for thousands of steps with
command buffers enabled — this is a graph-shape trigger, not "ROCm is broken."

## Environment

- 2× AMD Radeon RX 7900 XTX (gfx1100, RDNA 3), Linux 7.0.0-28-generic
- ROCm 7.2.4 (7.2.53211; MIOpen 3.5.1.70204, rocBLAS 5.2.0.70204)
- jax/jaxlib 0.11.0 + `jax-rocm7-{plugin,pjrt}` 0.11.0, Python 3.12.3
