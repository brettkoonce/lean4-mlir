# ViT-Tiny/ImageNet aborts at the FIRST training step on gfx1100

**Status: OPEN, NOT MINIMALLY REPRODUCIBLE, NOT FILED.** Narrowed to one
convolution and four dead ends closed, but the isolated conv passes, so there
is no upstream report to write yet. Everything below is what is measured;
the one thing this page does *not* claim is a root cause.

**This is not the LaunchGraph bug.** That one was a stale ROCm userspace and
is fixed at 7.2.4
([`../2026-08-jax-rocm-command-buffer-launchgraph-segv/`](../2026-08-jax-rocm-command-buffer-launchgraph-segv/)).
This one survives that fix and is older. It shares a crash *signature*, which
is exactly why it needs its own page — otherwise "7.2.4 fixed it" reads as
covering ViT, and it does not.

## Symptom

`vit-imagenet-verified` dies at the **first invoke**, after the residency line
and before any step timing. No error message — the process is simply gone.

```
[pjrt_ffi] RESIDENT: @vitin_adam128wxclip_train_step holds 600 parameter tensors (65.4 MB)
free(): invalid next size (normal)          # exit 134
```
or, at batch 32 and on the 2-replica render, exit 139 (SIGSEGV). Same
corruption, different thing next to the damaged region.

## What is ruled out (measured 2026-08-13, all on ROCm 7.2.4)

| suspect | control | result |
|---|---|---|
| data parallelism / collectives | 1 replica (`adam128wxclip`) vs 2 (`adamdp256x2wxclipdrop`) | **both die** |
| plugin version | 0.9.1.post4 vs 0.11.0 | **both die**, identical message |
| stale ROCm userspace | 7.2.4, where every other net is clean | **still dies** |
| batch size | 32 and 128 | **both die** |
| MIOpen's GEMM conv path | `MIOPEN_DEBUG_CONV_GEMM=0` | **still dies** |
| the patch-embed conv in isolation | `ruled_out.py {bare,tokens,grad,flat}` | **all four PASS** |

And it is not "ROCm can't do this repo": R34, R50, MNv2 and ConvNeXt all train
on this box on the same day, on the same stack, for thousands of steps.

## The fingerprint

With `MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=5`, the last MIOpen call logged
before the abort is the ViT patch-embed convolution, forward:

```
wDesc     = {192, 3, 16, 16}, packed          # 16x16 patches, 3 -> 192
xDesc     = {128, 3, 224, 224}, packed
convDesc  = conv2d, padding {0,0}, stride {16,16}, dilation {1,1}
yDesc     = {128, 192, 14, 14}, packed
solution_id = 91  ->  solver GemmFwdRest ("convolution, non 1x1")
workSpace = 602112 bytes
[KernDb] database not present
free(): invalid next size (normal)
```

⚠ **The forward, not the weight-grad.** `lakefile.lean` describes this net as
dying "in the patch-embed weight-grad convolution with
`miopenStatusUnknownError`"; neither half matches what is logged here. No
`miopenStatus` error is returned at all — the heap is simply corrupt.

**An observation, explicitly NOT a proven cause:** `workSpace` is **602112
bytes at batch 32 and at batch 128** — identical. That is `(3·16·16)·(14·14)`
floats, the im2col buffer for exactly *one* image, and it does not scale with
N. That looks like an under-sized workspace for a batched im2col. It is not
sufficient on its own, because `ruled_out.py` issues the same conv at the same
shapes through the same solver and passes. Either the trigger needs the
surrounding graph, or the workspace is a red herring.

## Where to look next

The sibling issue
[`../2026-06-jax-rocm-miopen-im2col-hiprtc/`](../2026-06-jax-rocm-miopen-im2col-hiprtc/)
went the same way — standalone conv fine, and the trigger turned out to be an
interior-dilated `pad` **fused into** the conv. It also notes "a latent
*no-workspace* limitation in MIOpen's GEMM solver," which may be this. The
obvious next step is to bisect the ViT graph rather than the conv: emit the
train step, cut it down, and find the smallest fused neighbourhood that still
aborts.

## Reproducing

Needs this repo (no standalone reproducer exists yet — that is the open work):

```
lake build vit-imagenet-verified
HIP_VISIBLE_DEVICES=0 LEAN_MLIR_VARIANT=adam128wxclip LEAN_MLIR_BATCH=128 \
  PJRT_FFI_RESIDENT=1 LEAN_MLIR_SKIP_EVAL=1 \
  .lake/build/bin/vit-imagenet-verified data
```

Add `MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=5` for the fingerprint above.
`ruled_out.py` needs only JAX.

## Impact

ViT-Tiny is the one net in the verified ImageNet set that cannot run on this
box at all. It is unaffected on CUDA, where the same graph trains end to end,
so ViT work belongs on the NVIDIA box until this is understood.

## Environment

- 2× AMD Radeon RX 7900 XTX (gfx1100, RDNA 3), Linux 7.0.0-28-generic
- ROCm **7.2.4** (7.2.53211; MIOpen 3.5.1.70204, rocBLAS 5.2.0.70204)
- jax/jaxlib 0.11.0 + `jax-rocm7-{plugin,pjrt}` 0.11.0 — and reproduced on
  0.9.1.post4 before it was removed
