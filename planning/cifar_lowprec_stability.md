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

* ⛔⛔ **"Ops: none new" was WRONG — and the guessed cause was wrong too.** Checked 2026-08-25.
  cifar8's BACKWARD needs `convBackBf16` and `dotOutBf16`, and **neither exists**. The cause is
  not conv geometry, it is **op family**:

  | | `convBack` | `convBackBatched` | `dotOut` |
  |---|---|---|---|
  | `CnnRender.lean` (cifar8 + mnist-cnn) | **28** | 0 | **18** |
  | all 7 ImageNet `*RenderB.lean` | **0** | 3–18 each | 0–1 |

  CIFAR renders the **per-example** backward ops; every ImageNet net renders the **batched**
  ones. The 27 bf16 ops were built for ImageNet, so bf16 twins exist ONLY for the batched
  family (`convBackBatchedBf16`, `denseRowBackBf16` — but no `convBackBf16`, no `dotOutBf16`).
  ▶ Writing the two missing ops would be **CIFAR-only work rehearsing a technique ImageNet
  never runs**, which defeats the point of §1.

* ⭐⭐ **THE FIX IS UNIFICATION, AND IT IS SEMANTICALLY FREE.** Migrate `CnnRender` to the
  batched family instead. Both denote the SAME proven VJP (`StableHLO.lean` l.2016 vs l.2200):

      convBack:         (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).backward v (den e)
      convBackBatched:  batchMap N (… (conv2d_has_vjp3 W b)).backward (fun _ => 0) …

  The only difference is the primal argument (`v` vs zero), and l.2990 states why that is free:
  *"conv is linear, so this is a global VJP"* — the input-VJP ignores the primal. l.2205 shows
  the bf16 twin is that same VJP with `rnd` on the weight, so bf16 drops in once CIFAR is on the
  batched op. ▶ Cost: **0 new verified ops**, and CIFAR ends up on the ops ImageNet actually
  runs — a real rehearsal. Risk: re-gate the cifar8 §1a tie. Chosen 2026-08-25.
* **Render:** thread `bf16 : Bool := false` through `CnnRender.lean`, the `wx`/`clip`/`sd` idiom.
  Gate 1 is then free.
* ⚠ **Give the dots bf16-TYPED results** (§20.1), not `dotInBf16`'s f32-result shape, which is a
  PoC artefact and ~1.2× slower.
* **Proof:** nothing new. `conv_close_mixed` is stated over arbitrary `ic`/`kH`/`kW`.

### 4.2 fp8 CIFAR — the real work, and it is NOT the ops

* ✅ **Link mode — DONE 2026-08-25, and the reason this document gave for it was wrong.**
  BOTH shim link modes are retired: `ireeLink` (95 sites) and `xlaLink` (32 sites) are gone, defs
  and all, and the lakefile now has **exactly one** link mode across all 167 executables —
  `lowererLink = #["-ldl"]`. `ffi/lowerer.c` dlopens whichever of `libpjrt_ffi.so` /
  `libiree_ffi.so` `$LEAN_MLIR_LOWERER` names, so **one executable serves both backends**.
  ⚠ **But this was never the fp8 blocker.** This bullet read "until that lands, none of them can
  reach the fp8 path, because IREE is where fp8 does not lower" — that is the SAME stale premise
  the lakefile's three "is an IREE binary" docstrings carried (`vit-adam-tie`, `convnext-adam-tie`,
  `mobilenetv2-adam-tie`), and it is wrong for the same reason: **`-liree_ffi` never selected a
  backend.** `ffi/lowerer.h` (≈l.137) `#define`s every `iree_ffi_*` entry point to the corresponding
  dlopen'd `lowerer_*` pointer, so nothing in the executable ever resolved against that library —
  `nm -D --undefined-only` on the ireeLink-built `cifar8-verified` listed no `iree_ffi_*` symbol at
  all. The ireeLink CIFAR exes were **already defaulting to XLA**, and therefore already had the
  fp8 path open. Verified by running the migrated `cifar8-verified`: it comes up on
  `PJRT CUDA / 6× RTX 4060 Ti` by default and still selects the IREE shim under
  `LEAN_MLIR_LOWERER=iree`.
  ▶ What the flag actually did was stamp `DT_NEEDED: libiree_ffi.so` + `RUNPATH ./ffi`, so the
  binary refused to *start* on a box without the IREE shim while doing all its work through XLA.
  That is the whole delta the migration buys, and it is worth having — but **no step below was
  gated on it**, and step 1 should not be sequenced as if it were.
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

0. ✅ **Link mode — DONE 2026-08-25.** `ireeLink` (95 sites) and `xlaLink` (32) both retired →
   one `lowererLink` across all 167 exes. The migrated `cifar8-verified` comes up on PJRT CUDA by
   default and still takes the IREE shim under `LEAN_MLIR_LOWERER=iree`; `cifar8-dp-check` — the
   gate that leans on the *optional* `pjrt_ffi_invoke_f32_dp` dlsym — passes at norm-rel 7e-6.
   ⚠ Numbered 0, not 1, because §4.2 shows it gated nothing: the ireeLink exes were already running
   on XLA. It is hygiene (a binary no longer needs a shim *present* to start), not a prerequisite.
1. ✅ **Re-baseline fp32 CIFAR on THIS box, on XLA — DONE 2026-08-25.** §5.2 has the table.
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

### 5.2 ⭐⭐⭐ MEASURED — THE 3×3 SWEEP: the optimizer ordering is INVARIANT under precision

ares, 1× RTX 4060 Ti, PJRT 0.114, `LEAN_MLIR_LOWERER=xla`, seed 1, bs 128, same net and same
init in every cell (the bf16 arm is a SLUG change only, §4.1).

**Test accuracy @20 / @40 epochs:**

| optimizer | fp32 | bf16 (fwd convs) | fp8 E4M3 (emulated) |
|---|---|---|---|
| plain SGD (const lr) | 60.26 / 62.86 | 63.16 / 66.29 | 63.47–64.07 / — |
| AdamW (cosine) | 71.92 / **75.50** | 71.79 / 74.86 | 71.87 / 74.74 |
| Nesterov-mom (cosine) | **74.25** / **76.91** | **73.72** / **77.61** | **74.28** / 76.09–76.82 |

⭐⭐ **THE RESULT: `SGD < AdamW < Nesterov` holds in ALL THREE precisions, at both 20 and 40
epochs.** That is the CIFAR chapter's claim — optimizers scale — extended to say the **math scales
with them**. The precision changes the arithmetic, not the ranking.

⭐ And the two arms with an fp32 master agree ACROSS precisions to within the noise floor: AdamW
spreads 0.13 pt @20 (71.92 / 71.79 / 71.87) and Nesterov 0.56 pt @20 — both at or under the
measured ~0.6–0.7 pt run-to-run spread (§5.2a). Three different arithmetics, one curve.

#### 5.2a ⚠ The one arm that does not reconcile, and the noise floor

⚠⚠ **plain SGD's fp32 cell is an OUTLIER — do not quote "low precision beats fp32".** bf16
(63.16) and fp8 (63.47) agree with each other and with §3's fp8 (63.5); it is **fp32** (60.26)
that sits ~3 pt below all of them and ~5 pt below §3's fp32 (65.7). Two independent low-precision
arms agreeing against one fp32 arm points at the fp32 arm. ▶ Repeat it before anyone uses it.
▶ Note this dents an ABSOLUTE-accuracy claim only — plain SGD is last in every column, so the
ordering result above is untouched by it.

⚠ **Noise floor ~0.6–0.7 pt**: two from-scratch runs of the SAME momentum binary at the SAME seed
gave 76.82 % / 76.09 % @40. Treat any single delta of that size as noise.
⚠ The fp8 trainers CHECKPOINT AND RESUME — a re-run silently no-ops ("resuming from fp8
checkpoint at epoch 40"). Delete `.lake/build/<slug>_<variant>_e4m3_ckpt.bin*` between configs.

#### 5.2b The three precisions are NOT the same kind of object

| | where the precision lives | speedup |
|---|---|---|
| fp32 | — | baseline |
| **bf16** | **IN the graph** — `flatConvFBf16`, bf16-typed conv result | ⛔ none, §5.3 |
| **fp8** | **host-side** — E4M3 rounding into an fp32 graph | ⛔ none (emulated) |

⚠ Neither low-precision arm is faster, for different reasons: bf16 because cifar8's convs are
launch-bound (§5.3), fp8 because no f8 type reaches the StableHLO at all (step 3, unbuilt). The
sweep is a statement about NUMERICS, not throughput.

### 5.2c fp32 vs fp8 against §3's cross-machine table

ares, 1× RTX 4060 Ti, PJRT 0.114, `LEAN_MLIR_LOWERER=xla`, seed 1, bs 128. This is the first
comparison of these two arms on ONE machine and ONE backend; §3's table is gfx1100-under-IREE and
the two are not comparable.

| optimizer | fp32 @20 | fp8 E4M3 @20 | penalty | §3 said (gfx1100/IREE) |
|---|---|---|---|---|
| plain SGD (const lr) | 60.26 % | 63.47 / 64.07 % | ⚠ **+3.2 to +3.8** | fp8 63.5 / fp32 65.7 (−2.2) |
| AdamW (cosine) | 71.92 % | 71.87 % | **−0.05** | fp8 71.4 / fp32 72.1 (−0.7) |
| Nesterov-mom (cosine) | 74.25 % | 74.28 % | **+0.03** | fp8 75.4 / fp32 75.1 (≈0) |

⭐ **AdamW and Nesterov reproduce §3's story**: once the optimizer carries an fp32 master, the fp8
penalty is ~zero. Both are *smaller* here than §3 measured.

⚠⚠ **The plain-SGD arm does NOT reconcile and must not be quoted.** The fp8 arm matches §3 almost
exactly (63.47 vs 63.5); the **fp32** arm is 5.4 pt BELOW §3's 65.7. So the fp32 plain-SGD arm is
the outlier, not the fp8 one, and "fp8 beats fp32" is not a supportable claim. ▶ Repeat it before
anyone uses it.

⚠ **Run-to-run spread on this path is ~0.6–0.7 pt**: two from-scratch runs of the SAME momentum
binary at the SAME seed gave 76.82 % / 76.09 % @40. Treat any single delta of that size as noise.
⚠ The fp8 trainers CHECKPOINT AND RESUME — a re-run silently no-ops ("resuming from fp8 checkpoint
at epoch 40"). Delete `.lake/build/<slug>_<variant>_e4m3_ckpt.bin*` between configs.

⚠ These are the EMULATED fp8 numerics (host-side E4M3 rounding into an fp32 graph). No f8 type
reaches the StableHLO and there is no speedup — that is step 3, still unbuilt.

MNIST fp8, same box/backend, for the record: linear **92.14 %** @12, MLP **97.84 %** @12, CNN
**98.73 %** @10. ⚠ The MLP and CNN required a fix — `trainE4M3` supplied N destinations for a graph
returning N+1 on the two nets with `lossSlot := true`, which the PJRT shim's G4 arity gate caught
and IREE never would have.

### 5.3 ⛔ bf16 BUYS NO SPEED AT CIFAR'S SHAPES — build the row for stability only

Standalone f32-vs-bf16 at cifar8's own eight conv shapes (B=128, 3×3, [16,16,32,32], 32→16→8→4→2):

| | f32 ms | bf16 ms | speedup |
|---|---|---|---|
| stem 3→16 32² | 0.060 | 0.099 | **0.60×** |
| **whole conv stack** | **0.497** | **0.574** | ⛔ **0.87×** |

⛔ **bf16 is SLOWER than f32 across the entire cifar8 conv stack.** At ≤32 channels and 4²–32²
spatial these convs are launch-bound, not compute-bound: the tensor cores buy nothing and the
converts cost. This is §5's own warning, confirmed.

⭐ **The harness was controlled** — the same script gives **1.81×** on a large 3×3 (256→256, 28²,
B=32), so it does find bf16 wins where they exist. ⚠ It gave 0.52× on ConvNeXt's 1×1 where §2.2
measured 1.84×, so its 1×1 path is an artifact; trust it for 3×3, which is all CIFAR uses.

▶ **Consequence:** the bf16 CIFAR row is a **stability / technique-rehearsal** result and must
never be quoted as a speed one, nor used to estimate ImageNet bf16 throughput
(`bf16_renderer.md` has the real per-net numbers).

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
