# Verified-IREE train-step memory scaling — findings + optimization plan

**Status:** investigation notes from the 2026-07-08 A100 pod session (remote-controlled
probe), **resolved by the same-day GPU verification session on ares (§GPU verification)**:
H1 is refuted — IREE already folds all splat constants — and the real bottleneck is
**conv/matmul codegen quality** (0.1–2.5 % of fp32 peak per kernel). Companion to
`mi300x_rental_program.md` (§Run-0a Session 2).

## TL;DR

- On a real 224² net (ResNet-34, verified-IREE/CUDA path), throughput **scales with
  batch but with sharply diminishing returns**, and the GPU shows a **memory-bandwidth-
  bound signature** (100 % util at only ~100 W, A100 max ~400 W) well before it runs
  out of memory.
- **`iree-compile` flags do nothing** — ptxas, product-target (`a100`), and opt-levels
  are all no-ops on the conv step, and `--iree-opt-level=O3` actively *regressed* it.
  So the lever is **the emitted MLIR (the renderer), not the compiler config.**
- The renderer emits **~190 full-size splat constants per train step** (a single scalar
  materialised as a full `[B,C,H,W]` tensor). At bs512 each is up to ~205 MB of HBM for
  one repeated number. This is the leading *suspect* for the memory-bound behaviour —
  but see the caveat: IREE may already splat-fold some of these, so **profile first.**
- **2026-07-08 verification verdict (see §GPU verification below): H1 is DEAD.** IREE
  folds every one of the splat constants at the flow level; an A/B rewrite of all 181
  of them to scalar+broadcast measured a perfect wash on GPU. The step is
  **conv-codegen-bound**: nsys shows ~88 % of GPU time in convolution dispatches running
  at 0.2–1.7 % (fwd) / 0.5 % (bwd) of fp32 peak. The one flag that helps is
  `--iree-global-opt-use-im2col-for-convs` (+5 %, numerically equivalent, zero proof
  surface — landable via `IREE_EXTRA_FLAGS` today).

## Empirical evidence (measured, reproducible)

### Batch scaling — ResNet-34, real Imagenette 224², steady-state (median steps 9–20)

| batch | ms/step | img/s | avg GPU util | peak mem |
|------:|--------:|------:|-------------:|---------:|
| 32    | 8,117   | 3.9   | ~25 %        | 3.3 GiB  |
| 128   | 11,738  | 10.9  | 58 %         | 8.4 GiB  |
| 256   | 20,600  | 12.4  | 76 %         | 15.4 GiB |

4× batch (32→128) bought +180 % throughput; the next 2× (128→256) only +14 %. The curve
is **plateauing around ~12–15 img/s**, not cliffing. Utilisation climbs (25→76 %) but
sustained power stays low (~100 W of ~400 W) — the tell that the step is increasingly
**memory-bandwidth-bound**, not compute-bound. (bs512/bs1024 were queued but not
completed this session; expected to extend the plateau and eventually approach the
80 GB memory wall around bs1024–2048.)

Reference points: the JAX/XLA path (non-verified) hits **2,110 img/s** on R50 at bs512
with the GPU saturated — i.e. the verified path is ~200× slower, and this memory
inefficiency is a large part of why.

### `iree-compile` flag sweep — cifar8-bn conv step (baseline 67,708 ms/epoch)

| variant | ms/epoch | vs baseline |
|---|---:|---|
| baseline (sm_80) | 67,708 | — |
| `--iree-cuda-use-ptxas` | 68,160 | flat |
| ptxas + `-O3` | 67,609 | flat |
| `--iree-cuda-target=a100` | 68,728 | flat |
| a100 + ptxas | 69,385 | flat |
| `--iree-opt-level=O3` | 100,825 | **+49 % (regression)** |
| a100 + ptxas + O3 | 68,361 | flat |

Conclusion: **no standard compile flag helps**; one hurts. The inefficiency is baked
into the emitted StableHLO, so it must be fixed in the renderer.

## Root-cause hypotheses

### H1 — full-size splat constants (leading suspect, NEEDS PROFILING)

The renderers emit a scalar value as a materialised full-size tensor. Grep counts **190
such constants** in one ResNet-34 train step. Examples at bs512:

```
stablehlo.constant dense<6422528.0> : tensor<512x64x112x112xf32>   # BN nf = BS·H·W  → ~205 MB
stablehlo.constant dense<1.0e-5>    : tensor<512x64x112x112xf32>   # BN epsilon      → ~205 MB
stablehlo.constant dense<0.0>       : tensor<512x64x112x112xf32>   # relu zero       → ~205 MB
```

Emitting sites (pattern is systemic, not r34-specific):
- `tests/TestResnet34Train.lean:49` — `bnPC`: `%nf = dense<{BS*m}.0> : {ty [BS,oc,Hh,Ww]}`
  (also `%ep` epsilon at :50; `relu` zero at :68).
- `LeanMlir/Proofs/CnnRender.lean:54,255` — relu `dense<0.0> : {ty [B,C,Hh,Ww]}`.
- `LeanMlir/Proofs/MlpRender.lean:58,62` — bias-zero full-size.
- `LeanMlir/MlirCodegen.lean` — 227 `stablehlo.constant dense<…>` (SE-block sigmoids,
  hardswish consts at :4506–4517, :5607–5647, etc.).
- `LeanMlir/Proofs/StableHLO.lean` — 179 (shared helper library).
- Every `tests/Test*Train.lean` offline emitter (ConvNeXt 29, EfficientNet 25, r34 23,
  MobileNetV2 20, cifar8 20 …).

**Fix pattern:** emit the scalar as `tensor<f32>` (a true splat) and `broadcast_in_dim`
into the op that needs it — or restructure so the constant divides/compares against a
scalar and lets shape inference broadcast. IREE treats a genuine splat + broadcast as
zero-materialisation; a `dense<v> : tensor<BxCxHxW>` *may or may not* be folded.

> ⚠️ **CAVEAT — confirm before mass-editing.** Modern IREE often *does* splat-fold
> constant tensors of a single value, in which case these cost nothing at runtime and
> H1 is a red herring. Before touching 190 call sites, PROFILE (see below) to prove
> these constants actually generate HBM traffic. The empirical memory-bound behaviour
> is real; which cause dominates is not yet proven.

### H2 — all forward activations held live for the backward (structural)

The train step is **one fused `func.func`** (585 KB of MLIR, a single function) doing
forward → full reverse pass → Adam update. Every forward activation is kept resident for
its backward consumer — there is no rematerialisation/checkpointing — so **peak working
set scales linearly with batch** and stops fitting in cache, then in HBM. This is the
more fundamental scaling limit and is independent of H1. Fix is heavier: activation
checkpointing (recompute cheap ops in the backward instead of storing), or splitting the
monolith so IREE can free intermediates earlier.

### H3 — BN backward materialises several full-size intermediates

`bnBackPC` (e.g. `tests/TestResnet34Train.lean:93–108`) computes a batch-coupled 3-term
gradient, materialising `dxh`, `t1`, `i1`, `xs`, `i2`, `sN` — each a full `[B,C,H,W]`
tensor — before reducing. Some are fusable/avoidable. Lower priority than H1/H2.

## How to verify (do this BEFORE editing renderers)

1. **Dispatch-level memory trace.** We already built `iree-benchmark-module` +
   `iree-run-module` (in `/root/src/iree-build/tools`). Compile one train-step MLIR and
   inspect per-dispatch bytes / whether the splat constants become their own dispatches:
   `iree-compile … --iree-flow-trace-dispatch-tensors` / `--iree-hal-dump-executable-*`.
2. **A/B one constant.** Hand-edit a single `dense<v>:[B,C,H,W]` → scalar+broadcast in a
   copy of `verified_mlir/resnet34_adam_train_step.mlir`, recompile, benchmark. If ms/step
   doesn't move, IREE was already folding it → H1 dead, focus on H2.
3. **`memory_stats()`** peak vs theoretical activation footprint — how much of peak is
   activations (H2) vs constants (H1).

## Reproducing the batch sweep (reusable method)

The batch size is a single constant per net, baked into the committed MLIR:
- emitter: `tests/TestResnet34Train.lean` `private def BS : Nat := N`
- trainer host loop: `apps/imagenette/MainResnet34VerifiedAdam.lean` `batchSize := N`

Change both, regenerate + rebuild + run:

```bash
# 1. edit BS in the emitter and batchSize in the trainer to N
lake env lean tests/TestResnet34Train.lean        # re-emits verified_mlir/resnet34_*.mlir at N
lake build resnet34-verified-adam                 # ~22 jobs (proof cone cached)
IREE_BACKEND=cuda IREE_CHIP=sm_80 LEAN_MLIR_MAX_STEPS=20 \
  .lake/build/bin/resnet34-verified-adam data     # PROBE: median ms/step
# sample `nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used` alongside
```

Watch: **power.draw** (low + 100 % util = memory-bound) and **median steps 9–20**
(steps 0–2 are compile/autotune warmup — do NOT time them; that misled us mid-session).

A one-shot `IREE_EXTRA_FLAGS` env hook in `LeanMlir/Types.lean` `ireeCompileArgs` (append
space-split env flags to the `iree-compile` args) makes flag sweeps rebuild-free — worth
landing as a small permanent probe affordance.

## Suggested priority (SUPERSEDED — see §GPU verification "Revised levers")

Step 1 was executed 2026-07-08 and killed H1; steps 2–4 below are obsolete as written.
Kept for the record:

1. **Profile** (H1 vs H2) — cheap, decides everything below.
2. If H1 real: scalar-splat the BN `nf`/`eps` + relu/activation zeros in the shared
   renderers (`StableHLO.lean`, `CnnRender.lean`, `MlirCodegen.lean`) — one pattern,
   ~hundreds of sites, likely a big win and low risk to correctness (same math).
3. Then H2: activation checkpointing for the deepest conv activations — bigger lift,
   raises the batch ceiling so larger batches actually pay off.
4. Re-run the batch sweep; expect the plateau to lift and img/s to keep climbing.

## GPU verification session (2026-07-08, ares 4060 Ti) — H1 refuted, real cause found

Ran the doc's §How-to-verify plan locally on ares (RTX 4060 Ti 16 GB, sm_89, compiled
as sm_86 — this IREE build rejects `sm_89`). Compiler = venv `iree-compile` 3.12.0rc20260428
@ af030e43. **Gotcha:** the standalone runtime tools at `~/src/iree-build/tools` are the
stale pre-af030e43 build (vmfb ABI mismatch: `hal.command_buffer.dispatch` signature) —
use the venv's own `iree-benchmark-module`/`iree-run-module`, which match the compiler.
Method: `iree-benchmark-module` directly on the committed bs32
`verified_mlir/resnet34_adam_train_step.mlir` (515 synthetic inputs generated from the
signature), 5 repetitions, medians; no trainer/host loop in the measurement.

### 1. H1 A/B — full splat-constant rewrite is a perfect wash

Rewrote **all 181** shaped `stablehlo.constant dense<v> : tensor<…>` sites to
`tensor<f32>` + `broadcast_in_dim` (script: one regex pass) and benchmarked both:

| variant | median ms/step | mean | avg util | avg power | peak mem |
|---|---:|---:|---:|---:|---:|
| baseline (committed MLIR) | 3,173 | 3,176 | 89 % | 97 W / 165 W | 4.2 GiB |
| all-181-splats → broadcast | 3,148 | 3,176 | 89 % | 97 W | 4.2 GiB |

Identical to the milliwatt. Confirmed structurally: at `--compile-to=flow` the baseline
has **zero** surviving full-size dense constants — IREE folds them all into consuming
dispatches as scalars. The only `flow.tensor.splat` ops (17) are genuine conv
padding/layout workspaces (e.g. `32x230x230x3`), not the BN/relu constants.
**Do not spend renderer/proof work on splat constants.**

### 2. nsys kernel profile — it's the convs

Top-20 kernels = ~88 % of GPU time, essentially all convolution dispatches (BN backward
reductions ≈ 1 %, H3 negligible). Per-instance efficiency vs the 4060 Ti's ~22 fp32
TFLOPS:

| dispatch (bs32) | ms/inst | GFLOPS | % of peak |
|---|---:|---:|---:|
| bwd-layout conv 256×256 k3 @14² | 66.2 | 112 | 0.5 % |
| bwd-layout conv 512→256 k3 @14² | 130.5 | 113 | 0.5 % |
| fwd conv 256×256 k3 @14² | 21.8 | 339 | 1.5 % |
| fwd conv 128×128 k3 @28² | 20.3 | 365 | 1.7 % |
| stem conv 3→64 k7 @224² | 183.6 | 41 | 0.2 % |

This *is* the memory-bound signature from the A100 session: naive direct-conv kernels
stream activations from HBM with almost no reuse, so util pins at 100 % while power and
FLOPs stay low. The "~200× vs XLA" gap is conv codegen quality, full stop.

### 3. Flag probes on this axis (all rebuild-free, zero proof surface)

| variant | median ms/step | verdict |
|---|---:|---|
| baseline | 3,173 | — |
| `--iree-codegen-llvmgpu-use-igemm` | 3,189 | no-op |
| `--iree-global-opt-use-im2col-for-convs` | **3,016** | **−5 %, only winner** |
| im2col + `--iree-dispatch-creation-split-matmul-reduction=8` | 3,015 | no further gain |
| im2col + `--iree-codegen-llvmgpu-use-tile-and-fuse-matmul` | 3,015 | no further gain |

im2col numerics: all 513 printed result tensors match baseline within **7.7e-5 rel**
(reduction-reorder noise; the verified MLIR is byte-identical, so the proof artifact is
untouched). But im2col only converts the *form* — the resulting `matmul_like` kernels
are still 0.14–2.5 % of peak. Worst single kernel after im2col: the stem's weight
gradient `64x147x1605632` (K = 1.6 M skinny reduction) at **963 ms for one launch**
(31 GFLOPS). Split-K via the available flag did not touch it.

### Revised conclusions

1. ~~H1~~ dead (proven, both structurally and by A/B).
2. **H4 (new, confirmed): IREE's LLVMGPU fp32 codegen is the bottleneck** — a few % of
   peak on every conv/matmul shape in the step, worst on AD-generated backward-layout
   convs and skinny weight-gradient GEMMs. No available compile flag fixes it (5 % from
   im2col is the ceiling of the flag space in rc20260428).
3. H2 (activation liveness) still governs the *memory ceiling* and batch scaling, but is
   not what caps throughput at bs32.
4. Cross-check oddity: pure-device bs32 step here is 3.17 s (10.1 img/s) vs the A100
   trainer's 8.1 s (3.9 img/s) — the trainer number includes host loop/data prep, so
   pod-session ms/step overstates device time. Re-baseline on-device before optimizing.

### Revised levers, in order of expected value

1. **Land `--iree-global-opt-use-im2col-for-convs`** (via `IREE_EXTRA_FLAGS` or
   `ireeCompileArgs`): free 5 %, verified equivalent. CUDA-only until checked on ROCm.
2. **bf16 train step**: IREE's CUDA codegen only has a strong path (tensor-core
   mma.sync) for f16/bf16 matmul. The bf16 trainer work from the pod branch is the
   likeliest order-of-magnitude lever — emit a bf16 r34 step and re-profile.
3. **Newer IREE**: re-run this benchmark on a current nightly before investing —
   conv codegen (esp. igemm-on-CUDA) is an active upstream area. Keep the CUDA/ROCm
   lock rule in mind; this is a probe, not a migration.
4. **Custom dispatch / preprocessing for the pathological shapes** (backward convs,
   skinny-K weight gradients) only if 1–3 stall.

## XLA reference on the same silicon (2026-07-08, ares 4060 Ti)

To calibrate the gap without renting anything: `scripts/jax_r34_imagenet_bench.py`
(synthetic data, ImageNet-shaped R34, 1000-class) run on the same GPU 0 that produced
the verified-IREE numbers above. **Dataset is irrelevant for kernel profiling** — kernel
shapes are fixed by (model, batch, resolution) — so this needs neither ImageNet nor
Imagenette; a parametrized copy of the script (PER_DEV/STEPS via env) does everything.

| path | bs | ms/step | img/s | vs verified-IREE |
|---|---:|---:|---:|---:|
| verified-IREE (measured above) | 32 | 3,173 | 10.1 | 1× |
| JAX/XLA, default (TF32 convs) | 32 | 91.9 | 348 | **34.5×** |
| JAX/XLA, `JAX_DEFAULT_MATMUL_PRECISION=highest` (true fp32) | 32 | 103.1 | 310 | **30.8×** |
| JAX/XLA, default | 128 | 318.4 | 402 | — |

nsys on the XLA step: everything goes through **cuDNN implicit-GEMM / Winograd**
kernels (TF32 tensor-core variants by default on Ampere+ — worth remembering when
quoting "fp32" XLA numbers, though on this GPU TF32 is only +12 %). Overheads visible:
~8 % NCHW↔NHWC transposes (model is NCHW, cuDNN wants NHWC) and ~20 %
redzone-allocator/autotune noise in the early profiled steps.

Takeaways for the local→cloud program:
- The honest same-shape same-silicon gap is **~31× fp32-vs-fp32** (not "~200×", which
  compared different models/batches/hosts). cuDNN's implicit-GEMM is the existence
  proof for what these exact shapes can do; IREE's direct-conv codegen is what's
  missing.
- XLA nearly saturates the 4060 Ti already at bs32 (348→402 img/s from bs32→128), so
  small-batch local profiling is representative; big-batch behaviour is the only thing
  that genuinely needs A100/MI300X memory.
- Anything kernel-shaped — flag sweeps, im2col-style rewrites, bf16 step comparisons,
  XLA-vs-IREE ratios — can and should be measured here first; pod time is for capacity
  (true-2048 batches) and final anchors only.

## Repo-side addendum (local review, 2026-07-08)

Citations spot-checked against the tree: `bnPC` full-size `nf`/`ep` at
`tests/TestResnet34Train.lean:49-50`, the 227 (`MlirCodegen.lean`) / 179
(`Proofs/StableHLO.lean`) constant counts, and the `BS := 32` / `batchSize := 32`
pair all verified. One correction of emphasis on priority-2's "low risk to
correctness": **the emitted MLIR is the proof-carrying artifact.** The renderers
under `LeanMlir/Proofs/` are certified emitters — a `dense<v>:[B,C,H,W]` →
scalar-splat + `broadcast_in_dim` rewrite changes the StableHLO the tie/faithfulness
proofs are stated against, so each edited site drags its proof cone with it (same
math ≠ same discharged obligations; see the rfl-kink and Eq.mpr-cast lessons). Cost
it as renderer-plus-proofs work, not a mechanical sed — one more reason step 1
(profile; maybe H1 is already folded and free) comes first. The `IREE_EXTRA_FLAGS`
hook, by contrast, touches only `ireeCompileArgs` (no proof surface) and is safe to
land immediately.
