# MIOpen bug: `MIOpenIm2d2Col.cpp` uses OpenCL builtins, fails to build as HIP on gfx1100

**Status: reproduced, minimal repro in `scripts/miopen_im2col_repro.py`. NOT YET FILED.**

## Symptom

Any convolution XLA lowers to MIOpen's **im2col** path fails at execution with
`miopenStatusUnknownError`. Through the PJRT shim this surfaces once per output buffer
(603 identical `d2h: Failed to enqueue convolution on stream` lines for our ViT graph),
because PJRT's `Execute` is async and the error is only observed when awaiting the
device-to-host copies. `Execute` itself returns success and compilation is clean.

## Root cause (from the MIOpen log)

```
MIOpen(HIP): Warning [BuildHip] .../MIOpenIm2d2Col.cpp:298:19:
    error: use of undeclared identifier 'get_global_id'
  298 |     index_t tid = get_global_id(0);
.../MIOpenIm2d2Col.cpp:326:16:
    error: use of undeclared identifier 'get_global_size'
2 errors generated when compiling for gfx1100.
MIOpen Error: hipoc_program.cpp:299: Code object build failed. Source: MIOpenIm2d2Col.cpp
```

`get_global_id` / `get_global_size` are **OpenCL** builtins. The kernel is being compiled
through the HIP path (`BuildHip`), where they are undeclared. So the runtime kernel build
fails and the convolution can never be enqueued. This looks like a kernel that was only ever
exercised on the OpenCL backend, or a missing HIP compatibility shim.

## Reproducer

`scripts/miopen_im2col_repro.py` — ~20 lines of JAX, no ViT and no framework specifics:

```python
u  = lax.pad(dy, 0.0, ((0,0,0),(0,0,0),(0,0,15),(0,0,15)))   # interior dilation
xt = jnp.transpose(x, (1,0,2,3))
dt = jnp.transpose(u, (1,0,2,3))
lax.conv_general_dilated(xt, dt, (1,1), ((0,0),(0,0)), (1,1), (1,1), dn)
```

with `x : f32[32,3,224,224]`, `dy : f32[32,192,14,14]`. This is a patch-embed
convolution's weight gradient (16x16/s16 patchify, 224², ViT-Tiny).

**The pad and the conv must be in the SAME jit.** Materialise the dilated 209x209 filter
first and pass it to a standalone conv (`scripts/miopen_conv_probe.py`) and it runs fine —
XLA picks a different algorithm. Fusing them selects the im2col path, which is the broken one.

## Ruled out by measurement

* **Not the filter shape.** The same `(3,32,224,224) x (192,32,209,209) -> (3,192,16,16)`
  convolution succeeds standalone (`scripts/miopen_conv_probe.py`).
* **Not memory.** It succeeds holding 12 GiB of ballast on top of its own 1.02 GiB
  (`scripts/miopen_mem_probe.py`), and a genuine OOM reports a clean
  `RESOURCE_EXHAUSTED: Out of memory while trying to allocate ...` instead.

## Environment

* GPU: 2x Radeon RX 7900 XTX, gfx1100
* ROCm: /opt/rocm; driver/runtime 7.2.26015 (per the PJRT device string)
* jax 0.10.0 + rocm plugin 0.9.1.post4, PJRT 0.96

## Impact here

Blocks the ViT-Tiny AdamW graph on the XLA/PJRT backend entirely (IREE runs the identical
graph fine), which in turn leaves `vit-dp-check` — the data-parallel gate — unrunnable on
this box. See `planning/xla_pjrt_handoff.md`.

## Where to file

MIOpen (`ROCm/MIOpen`) is the primary owner — the broken source file is theirs. Mention the
XLA/ROCm lowering only as the trigger.
