# cifar_lowprec_stability.md — CIFAR in bf16 and fp8, to show low precision TRAINS

Scoped 2026-08-25 on ares (6× RTX 4060 Ti, sm_89, CUDA 12.9, jax 0.11.0).
Successor to `planning/fp8_lowering.md`, whose blocking gate this document opens.

---

## 0. ⭐⭐ THE ONE-PARAGRAPH VERSION

`bf16_renderer.md` ends with seven nets, 27 bf16 ops, measured speedups on real ImageNet — and
**nothing trained to convergence in any of them.** `fp8_lowering.md` ends with fp8 proven, emulated
and *not lowered*, blocked on its own §4 step 1 ("does the backend lower an f8 `dot_general` to the
Ada fp8 MMA, and is it faster than fp32? If the f8 path is immature, that gates everything").
**That gate is now open** — measured 2026-08-25, §2: XLA lowers f8E4M3FN to `__cublas$lt$matmul$f8`
and `__cudnn$convForwardGraph`, at **2.71× f32** on a 4096³ gemm and **3.43×** on ConvNeXt's 1×1.
▶ This document proposes CIFAR as the vehicle for closing BOTH gaps at once, because it is the only
net in the repo where a convergence answer costs **hours, not weeks** — and because the fp8 accuracy
targets to reproduce **already exist** (§3).

---

## 1. Why CIFAR, and why now

| | ImageNet nets | CIFAR |
|---|---|---|
| a convergence run | 18.6 h (MNv4) to 15.6 d (ConvNeXt-B) — `bf16_renderer.md` §21 | **hours** |
| f32 baseline to compare against | one (R34 30-ep, 70.71 %) | ✅ three optimizers × two precisions, §3 |
| ops needed for bf16 | already built (27) | already built — cifar8 is 23 convs + 9 dots |
| ops needed for fp8 | none exist | none exist — same work either way |

▶ **The branch's single largest risk is that none of it has been trained**, and CIFAR is where that
risk can be retired cheaply. A bf16 CIFAR number and an fp8 CIFAR number, both against existing f32
baselines, is a *stability* result that costs a day and licenses everything above it.

⚠ It does NOT license an accuracy claim about the ImageNet nets. CIFAR's cifar8 is 8 convs at
batch 128; ConvNeXt-T is 173 convs at 32. Stability at depth 8 is evidence, not proof, at depth 53 —
and `bf16_renderer.md` §11.3 is explicit that the composed bound is vacuous in absolute terms
regardless. ▶ Say "trains stably on CIFAR", never "bf16 is safe".

---

## 2. ⭐⭐⭐ THE BLOCKING GATE IS OPEN — fp8 LOWERS ON XLA, MEASURED

`fp8_lowering.md` §4 step 1 asked one question and made everything conditional on it. Answered on
this box, 2026-08-25, with the same standalone-first method `bf16_renderer.md` §15.3 uses:

### 2.1 It compiles, and it reaches real fp8 silicon

| op form | lowers to | fp8 in the optimized HLO? |
|---|---|---|
| `dot_general` f8E4M3FN → f32 | **`__cublas$lt$matmul$f8`** | ✅ 8 f8e4m3fn values |
| `convolution` f8E4M3FN → f8E4M3FN | **`__cudnn$convForwardGraph`** | ✅ 13 |
| `convolution` f8E4M3FN → f32 | `__cudnn$convForwardGraph` | ✅ 8 |

⭐ **Both of the IREE issue's two gaps are gone.** `upstream-issues/2026-06-iree-cuda-fp8-nvptx-lowering/`
records (1) the f8 type not surviving NVPTX translation and (2) *"no fp8 tensor-core path — even a
successful compile would `arith.extf` to f32 and do a fp32 matmul, so there would be no fp8
speedup."* Neither holds on XLA. ▶ **That issue should be updated or closed**: it is an IREE bug,
still true of IREE, and no longer a statement about this repo's engine.

### 2.2 And it is fast — gemm 4096³, one 4060 Ti

| operands | ms | vs f32 | kernel |
|---|---|---|---|
| f32 | 6.507 | 1.00× | `__cublas$lt$matmul` |
| bf16 | 4.411 | 1.48× | `__cublas$lt$matmul` |
| **f8E4M3FN** | **2.404** | **2.71×** | `__cublas$lt$matmul$f8` |
| i8 | 2.725 | 2.39× | `__cublas$lt$matmul` |
| i4 | **2388.766** | ⛔ **0.003×** | (fused/triton — no tensor core) |
| f4E2M1FN | — | ⛔ | won't compile: *"GEMM is not supported by cublasLt"* |

**Convolution** — ConvNeXt-T stage-1 1×1 expand ×6 (c=96→384, 56², B=32):

| operands → result | ms | vs f32 |
|---|---|---|
| f32 | 6.50 | 1.00× |
| bf16 → bf16 | 3.54 | 1.84× |
| **f8E4M3FN → f8E4M3FN** | **1.90** | **3.43×** |
| f8E4M3FN → **f32** | 5.54 | ⛔ 1.17× |

### 2.3 ⚠⚠ THREE TRAPS, and two are new spellings of ones this repo already knows

1. **The RESULT-TYPE rule recurs at fp8, and harder.** f8 conv with an f32 result is **1.17×**;
   with an f8 result **3.43×**. That is `bf16_renderer.md` §9.2 (conv folds) and §20.1 (a bf16 dot's
   f32 result costs ~1.2×) at a third precision. ▶ Assume the emit shape is load-bearing and
   *measure it per op form*, exactly as the bf16 work had to.
2. ⛔ **f8E5M2 SILENTLY WIDENS.** It compiles, lowers to plain `__cublas$lt$matmul`, and the
   optimized HLO contains **zero** `f8e5m2` values — the type is gone. Only E4M3 reaches the fp8
   units here. This is §9.2's fold in its purest form and nothing structural sees it.
3. ⛔ **int4 is not a path, at all.** 367× SLOWER than f32 — the INT4 MMA was deprecated after
   Ampere, so XLA emulates element-wise. And 4-bit float (`f4E2M1FN`, i.e. NVFP4/MXFP4) does not
   compile on sm_89 — it needs Blackwell. ▶ "Emit int4 and carry the scale ourselves" is closed on
   this hardware. §6 has the accuracy reason it would be closed anyway.

---

## 3. ⭐ THE TARGETS ALREADY EXIST — and this is what makes CIFAR cheap

`fp8_lowering.md` §2 measured the **emulated** fp8 path (host-side E4M3 rounding, fp32 graph, fp32
master, fp32 accumulate) on the verified cifar8 CNN. Those numbers are the *true fp8 numerics*; only
the speed was missing. **They are therefore exactly the numbers real fp8 silicon must reproduce:**

| optimizer | fp8 @20 | fp32 @20 | fp32 final @40 | fp8 penalty @20 |
|---|---|---|---|---|
| plain SGD (const lr) | 63.5 % | 65.7 % | 66.7 % | −2.2 pt |
| AdamW (cosine) | 71.4 % | 72.1 % | 74.0 % | −0.7 pt |
| Nesterov-mom (cosine) | **75.4 %** | 75.1 % | 76.8 % | **≈0 (+0.3)** |

⭐⭐ **The stability question is, for fp8, ALREADY ANSWERED — numerically.** The optimizer ranking is
identical in fp8 and fp32, and the penalty is optimizer-dependent: plain SGD eats the per-step
rounding (~2 pt), Nesterov's fp32-master velocity averages it away (~0). ▶ So the fp8 half of this
project is **a hardware-faithfulness check, not an accuracy experiment** — the same shape as the JAX
`vjp_oracle`. If real fp8 silicon reproduces this table, the lowering is correct.

⚠ Measured on gfx1100 (RDNA3) under IREE, which has no fp8 units — hence emulation. Re-running the
**fp32 arm** on this box is part of step 1, because a cross-machine, cross-backend comparison is not
a comparison.

⭐ There is a second, independent precedent: `floatbridge_quantization.md` §3a, MNIST at 20 epochs,
**fp32 92.25 % → E4M3 92.30 %** (+0.05 pt, i.e. inside noise).

▶▶ **bf16 has NO CIFAR baseline of any kind.** Not emulated, not lowered, not measured. That half is
genuinely new work and is the cheaper half (§4).

---

## 4. What each half actually costs

### 4.1 bf16 CIFAR — the cheap half, and no new ops

cifar8's train step is **23 `stablehlo.convolution` + 9 `stablehlo.dot_general`** at batch 128.
Every one of those kinds already has a bf16 twin among the 27 ops (`bf16_renderer.md` §STATUS).

* **Ops:** none new. Possibly one, if cifar8 uses a conv geometry no ImageNet net does — check
  before assuming.
* **Render:** thread `bf16 : Bool := false` through `CnnRender.lean`, the `wx`/`clip`/`sd` idiom.
  Gate 1 is then free.
* ⚠ **Give the dots bf16-TYPED results** (§20.1), not `dotInBf16`'s f32-result shape, which is a
  PoC artefact and ~1.2× slower.
* **Proof:** nothing new. `conv_close_mixed` is stated over arbitrary `ic`/`kH`/`kW`.

### 4.2 fp8 CIFAR — the real work, and it is NOT the ops

* **Link mode, and this is a one-line-per-exe change rather than a port.** The CIFAR exes are
  `moreLinkArgs := ireeLink`, which puts `libiree_ffi.so` on the link line. Its successor
  `lowererLink` (`#["-ldl"]`) makes `ffi/lowerer.c` dlopen whichever of `libpjrt_ffi.so` /
  `libiree_ffi.so` `$LEAN_MLIR_LOWERER` names, so **one executable serves both backends**. ▶ Switch
  the CIFAR exes to `lowererLink` FIRST — until that lands, none of them can reach the fp8 path,
  because IREE is where fp8 does not lower.
* **Emit f8 types.** `fp8_lowering.md` §3's table is still the design and is unchanged by this
  document, except that its "lowering" column now targets XLA rather than IREE.
* ⚠⚠ **SCALING IS THE ENGINEERING, not the emit.** E4M3's max is 448 and its min normal is ~2⁻⁶.
  XLA's fp8 gemm takes scale operands; asked for none, it synthesized `%constant_1` (scale = 1.0),
  which will overflow real activations. The emulated path already carries per-tensor `sx` and
  per-column `sW` — and `e4m3_render_faithful` is proven for **any** `q` and any scales, so the
  *static* scaled form is done. What does not exist anywhere is the **dynamic** part: amax history /
  delayed scaling, which is stateful and which the renderer has no concept of.
  ▶ CIFAR is small enough that a **fixed, calibrated scale per tensor** may suffice — that is the
  cheapest thing that could work and it should be tried before building amax machinery.
* **Proof:** ≈ free, and `fp8_lowering.md` §3 already argues why — the render-tie is about the ∑ and
  the scale factoring, not the storage type. It carries under one documented HW-semantics assumption
  (the f8 dot computes the exact ∑ with fp32 accumulate), which is the same trust tier as "the
  lowering is faithful". ⭐ `fp8E4M3` is already a `FloatModel` instance at `u = 2⁻⁴`
  (`Binary32Instance.lean`), so the accuracy bounds instantiate rather than needing restatement.

---

## 5. ▶ STEP ORDER, by risk

1. **Re-baseline fp32 CIFAR on THIS box, on XLA.** §3's table is gfx1100-under-IREE. Without a
   same-box, same-backend fp32 arm there is nothing to compare to. Cheapest step, and it also
   proves the link-mode switch works.
2. **bf16 CIFAR** — no new ops, no new proof. Expect it to be undramatic; that is the point. Gate:
   final test accuracy within noise of step 1's fp32, across all three optimizers.
3. **fp8 CIFAR, fixed scales.** Emit f8 types on the dense path first (§3b's own scope), then conv.
   Gate: reproduce §3's emulated table to within rounding-convention noise.
4. **Only then, amax/delayed scaling** — and only if step 3's fixed scales are shown insufficient.
5. ⛔ **Do NOT scale to ImageNet in fp8 off a CIFAR result.** §6.

⚠ **Profile every new op standalone against its f32 peer at CIFAR's own shapes before wiring it**
(`bf16_renderer.md` §19.4). ViT's stem weight gradient is 0.19× its f32 peer with every gate green;
fp8 conv already shows a 3× swing on result type alone. CIFAR's convs are small (16×3×3×3), which is
*exactly* the regime where kernel selection is least likely to have a good fp8 path.

---

## 6. ⚠⚠ WHAT THIS CANNOT SHOW, AND THE NUMBER THAT SAYS SO

Per-element unit roundoff, and the composed certificate factor `(1+u)^depth`:

| | u | R34 (36 convs) | R50 (53 convs) |
|---|---|---|---|
| bf16 | 0.39 % | 1.2× | 1.2× |
| fp8 E4M3 | 6.25 % | 8.9× | **24.9×** |
| fp4 E2M1 | 25 % | 3,081× | **136,846×** |

* **bf16's certificate story survives depth** — `bf16_renderer.md` §11.3's "1.86× the f32 bound at
  R50's 53 layers" is the meaningful claim, and it is under a factor of two.
* **fp8's does not.** 24.9× at R50. A CIFAR stability result at depth 8 says nothing about that, and
  the honest framing for fp8 at ImageNet depth is empirical-only.
* **fp4's is hopeless** and that is *independent* of hardware — which is the real reason "int4 +
  carry the exponent ourselves" is not a direction, over and above §2.2's 367× measurement. NVFP4's
  block scale recovers dynamic range, not per-element relative precision.

⚠ And note the whole table is a worst-case forward-error analysis composed depth-first: §11.3 shows
even the f32 absolute bound is vacuous. **Only the ratios mean anything.**

---

## 7. File index

* `planning/fp8_lowering.md` — the design (§3's table is still correct), the emulated numbers (§2),
  and the step order this document supersedes at step 1.
* `planning/floatbridge_quantization.md` — §3a MNIST E4M3 demo, §3c the accuracy bound, the `u` table.
* `LeanMlir/Proofs/Float/E4M3FaithfulPoC.lean` — `e4m3_render_faithful`, for any `q`/`sx`/`sW`.
* `LeanMlir/Proofs/Float/Binary32Instance.lean` — `fp8E4M3 : FloatModel` at `u_e4m3 = 2⁻⁴`.
* `LeanMlir/E4M3Quant.lean` — the host-side rounding used by the emulated trainers.
* `upstream-issues/2026-06-iree-cuda-fp8-nvptx-lowering/` — ⚠ still an open IREE bug; no longer a
  statement about this repo's engine (§2.1).
* `scripts/bf16_device_step.py`, `scripts/bf16_peak_memory.py`, `scripts/bf16_gate2.py --dots` —
  the measurement tools, all precision-agnostic.
