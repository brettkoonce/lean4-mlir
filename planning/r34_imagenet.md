# r34_imagenet.md — ResNet-34 on full 1000-class ImageNet

Goal: a credible 1000-class ImageNet ResNet-34 training run inside this
project. Not a frontier paper-style result — paper recipes get ~74% top-1
at 90 epochs, we target ~70% at 30 (the demo-grade backbone for YOLOv1
detection bootstrap, see `planning/yolo_demo_v3.md` Phase 4).

The dual goals here are (1) producing the backbone, and (2) proving that
"Lean + raw data + GPU time" — no torchvision pretrained checkpoint, no
huggingface download — produces real ML at modern scale.

## Status quo (2026-05-28)

**Phase 2 (Lean → JAX) end-to-end working.** Kicked off the actual
30-epoch run tonight at ~7:30 PM on mars (2× 7900 XTX, ROCm 7.2). ETA
~12.5 hr based on the measured 295 ms/step steady-state on real TFDS
data; demo backbone ready ~12 PM Saturday.

What landed today (committed in `2b1aacc`):

- `jax/MainResnetImagenet.lean` — spec + paper-flavored config, epochs
  reduced to 30
- `jax/Jax/Codegen.lean::emitMainImagenet` — multi-GPU `Mesh +
  NamedSharding(P('batch'))` trainer with TFDS streaming, cosine LR +
  5-epoch warmup, per-step trace emission
- `jax/Jax/Codegen.lean::emitParamsToFile` — save-side mirror of
  `init_params_from_file`. Writes a `.bin` in the EXACT byte order
  `LeanMlir.SpecHelpers.paramShapes` uses, so a JAX-saved checkpoint
  drops straight into the phase-3 Lean trainer via
  `TrainConfig.bootstrapBackbone := some (path, prefixFloats)`.
- Per-N-epoch checkpoint via `LEAN_MLIR_PARAMS_OUT` env var (default
  every 10 epochs)
- `scripts/jax_multigpu_probe.py` — diagnostic that confirmed
  jax 0.10.0 + jaxlib 0.10.0 + RCCL distributes work across both GPUs
  (10 AllReduce ops in compiled HLO; both GPUs ~95% utilization)

## Operational gotchas (worth remembering)

These ate hours today. Memorialize so future runs go straight through.

### RCCL must be `LD_PRELOAD`ed

`/opt/rocm/lib/librccl.so.1` is installed on mars (Ubuntu package
`rccl 2.27.7.70200-43~24.04`), but JAX's RCCL resolver doesn't find it
via `LD_LIBRARY_PATH=/opt/rocm/lib` alone — the runtime calls
`dlopen("librccl.so")` via a hardcoded path resolver that
`LD_LIBRARY_PATH` doesn't reach for collectives. `LD_PRELOAD` does, by
putting the library in the process's global symbol table before JAX
loads. The launch line that works:

```bash
LD_PRELOAD=/opt/rocm/lib/librccl.so.1 \
TFDS_DATA_DIR=$HOME/tensorflow_datasets \
LEAN_MLIR_PARAMS_OUT=.lake/build/jax_r34_imagenet \
LEAN_MLIR_CKPT_EVERY=10 \
PYTHONUNBUFFERED=1 \
.venv/bin/python .lake/build/generated_resnet34_imagenet.py
```

Small workloads (small MLPs, 50K params) silently work without RCCL —
XLA falls back to in-process AllReduce. ResNet-34 sized gradients hit
the NCCL path and fail with `RCCL operation ncclGetUniqueId(&id)
failed: Unable to load NCCL library`. The error message is misleading
(says NCCL even on AMD); the fix is RCCL.

### TFDS data layout

After rsync of `~/tensorflow_datasets/imagenet2012/` (153.7 GB across
1093 files: train shards `*_train.tfrecord-NNNNN-of-01024`, val shards
`*_validation.tfrecord-NNNNN-of-00064`, plus metadata), the JAX
`build_imagenet_iter` codegen finds it via `TFDS_DATA_DIR` env var —
no path arg needed.

### Step time gotcha

The first 100 steps include JIT compile amortized in the avg → reports
~430ms/step. After step ~500 the JIT cost is washed out and steady
state lands ~280-300 ms/step on this 2-GPU setup at batch 256
(128/device). Don't extrapolate ETAs from the early reported number.

## Path to phase-3 (Lean → MLIR → IREE) on full ImageNet

What's missing in our IREE pipeline, ordered by how blocking each is:

### 1. C-side streaming data loader (~3-5 days)

Documented in `LeanMlir/Train.lean:276` — phase-3 doesn't yet support
full ImageNet because every loader reads the whole `.bin` into one
ByteArray (`loadImagenette` → 1.4 GB; `loadVoc` → 750 MB). ImageNet at
150 GB doesn't fit.

Required:
- C reader for TFRecord chunks (or per-shard `.bin`), one shard at a
  time
- JPEG decode + center-crop + ImageNet-normalize on CPU per batch
- Async prefetch via pthread or io_uring so I/O overlaps GPU compute
- Extend `DatasetIO` so the dataloader exposes `nextBatch : IO (Image,
  Label, IsLastBatch)` rather than returning the whole train/val
  buffer up front
- Update `runTraining`'s outer loop to iterate the streaming source

Mostly C work + a `DatasetIO` API extension. Independent of multi-GPU
— useful even for single-GPU ImageNet (which is still impractical at
~28+ hr / 30 epochs, but unblocks the structural integration).

### 2. IREE multi-GPU plumbing (~1-2 weeks)

Today every opaque in `LeanMlir/IreeRuntime.lean`
(`trainStepAdamF32`, `trainStepAdamF32Yolov1`, etc.) assumes a single
`IreeSession` bound to one device. To get the same multi-GPU speedup
JAX gets on mars, we need:

- **MLIR codegen extension** in `LeanMlir/MlirCodegen.lean`: insert
  `stablehlo.all_reduce` ops at the gradient-aggregation points (after
  loss backward, before optimizer update). Per-param-tensor: ~20 SSA
  ops. The hardest part. Roughly mirrors what XLA emits when JAX sees
  the sharded gradients pattern.
- **IREE multi-device target**: `--iree-hal-target-backends=rocm` with
  the device mesh declared. IREE's HAL has the abstraction; the
  codegen just emits the right annotations.
- **C-side multi-device session**: new `IreeSession.createMulti` that
  holds N devices, splits incoming batch tensors via the same
  `librccl.so.1` we just learned about, invokes per-device, gathers.
- **New FFI entry**: `trainStepAdamF32MultiDevice` that wraps the
  multi-session invocation.
- **Linkage**: rebuild `libiree_ffi.so` with `-lrccl` and the right
  include paths.
- **A smoke test** mirroring `scripts/jax_multigpu_probe.py` to
  confirm both GPUs hit ~95% utilization on a tiny train step.

### 3. Performance gap closure (open-ended)

Our current Lean→IREE R34 throughput at batch 32 on a single 7900 XTX
is **23 img/sec** (1.4 s/step, measured earlier today). Tonight's JAX
run sustains **~918 img/sec** at batch 256 on 2 GPUs (279 ms/step).

That's **40× delta**. Even after the 2× multi-GPU win, per-device we'd
be ~20× slower than JAX. Likely causes:
- IREE's ROCm conv kernels are less mature than XLA's MIOpen
  integration
- Our codegen emits conv + BN + ReLU as three separate StableHLO ops;
  XLA/MIOpen fuses aggressively
- We've never tested batch > 32 — bigger batches amortize per-op
  overhead

Mitigations to explore (no commitment to fully closing the gap):
- `iree-benchmark-module` per-layer profile to find hot kernels
- Emit fused `convBnRelu` MLIR sequences instead of three ops
- Tune at batch 64+
- Possibly need to land MIOpen integration in IREE upstream

Could be 1 week to halve the gap, could be months to fully close it.
**Not blocking for any demo** — single-GPU phase-3 ImageNet at 100
img/sec (a generous mid-step number) gets 30 epochs in ~107 hours
which isn't practical anyway. So perf only matters once multi-GPU is
done, and the marginal payoff has to be weighed against wall-clock.

## What we DON'T need

Things people might expect we'd need that actually already work:

- **Per-N-epoch checkpointing**: landed today as
  `TrainConfig.checkpointEveryNEpochs` (default 10).
- **1000-class spec**: existing codegen handles any `.dense fanIn fanOut`
  head. The `MainResnetImagenet` shape compiles today — it just won't
  have data to feed it until item 1 above lands.
- **Bootstrap path**: `TrainConfig.bootstrapBackbone` reads `.bin`
  bytes regardless of who wrote them. The phase-2 JAX trainer's
  output drops straight in. No converter script needed because the
  byte order is the spec-canonical one (see
  `Jax/Codegen.lean::emitParamsToFile`).

## Three concrete paths

| Path | Bootstrap source | "Pure phase-3"? | Effort | Backbone quality |
|---|---|---|---|---|
| **A. Imagenette today** | `resnet34-train` (existing) on 9.5K-image / 10-class Imagenette | YES — every byte through Lean→MLIR→IREE | 0 (works) | 10 classes, weak transfer |
| **B. JAX ImageNet (running tonight)** | `jax/resnet34-imagenet` (phase-2 Lean→JAX) | Mixed — phase-2 trains the weights, phase-3 runs the detection demo | 0 (running) | 1000 classes, ~70% top-1 expected |
| **C. Phase-3 ImageNet** | `LeanMlir/Train.lean` end-to-end on ImageNet | YES, end-to-end | 3-5 weeks (items 1 + 2) | 1000 classes, eventually strong |

For the YOLOv1 demo this weekend, **(B) is the path**. The detection
half of the demo (forward + NMS + visualization) is all phase-3 with
proven gradients; only the bootstrap weights' producer differs from a
hypothetical (C). The argument "every line of code in the deliverable
was emitted by our system" still holds — phase-2 codegen we wrote
produced the trainer, not someone else's framework.

(C) is the right multi-week follow-on project after the demo lands.
Suggested ordering: streaming loader (item 1) first as a standalone
piece — useful for other things too — then IREE multi-GPU (item 2) as
the bigger lift, then perf tuning (item 3) as ongoing work.

## Out of scope (for this doc)

- COCO or other detection datasets
- Multi-machine training (single-machine multi-GPU only)
- INT8 / FP16 mixed-precision training (FP32 throughout for now)
- Distillation from a stronger teacher

## See also

- `planning/yolo_demo_v3.md` — the downstream consumer of whatever
  backbone this run produces
- `upstream-issues/2026-04-jax-rocm-multigpu-mesh-hang/` — the prior
  multi-GPU JAX issue (closed; fixed in jax 0.10.0)
- `LeanMlir/Train.lean` and `jax/Jax/Codegen.lean` — the two trainer
  codepaths
- `~/.claude/projects/.../memory/project_gpu_plan.md` — the
  mars-specific GPU notes (updated today with the RCCL LD_PRELOAD
  finding)
