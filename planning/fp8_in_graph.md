# fp8_in_graph.md — from EMULATED fp8 to fp8 IN THE GRAPH: CIFAR first, then R34/ImageNet

Scoped 2026-08-25 on ares (6× RTX 4060 Ti, sm_89, CUDA 12.9, PJRT 0.114, jax 0.11.0).
Successor to `planning/fp8_lowering.md` (the design) and `planning/cifar_lowprec_stability.md`
(which measured the emulated arm). This is the EXECUTION plan for the rung neither covers.

---

## 0. ⭐⭐ THE ONE-PARAGRAPH VERSION

Every fp8 number this repo has is **emulated**: host-side E4M3 rounding into an **fp32 graph**,
fp32 accumulate, fp32 master. **No `f8` type has ever reached the StableHLO** — measured, §2.1:
zero fp8 emit ops exist and no artifact in `verified_mlir/` contains an f8 type. So the fp8 CIFAR
results (§5.2 of the stability doc) validate the *numerics* and say **nothing** about the lowering.
▶ The missing rung is fp8 IN THE GRAPH, and this document argues it should be built **at CIFAR
first**, because (a) it is now measured to lower there (§1), (b) the emulated arm is already a
per-cell **oracle** to check it against (§3.2), and (c) the same op set then transfers to R34
unchanged, since CIFAR is now on ImageNet's batched family.

---

## 1. ⭐ MEASURED: fp8 LOWERS AT CIFAR'S OWN SHAPES (2026-08-25)

`cifar_lowprec_stability.md` §2 proved f8E4M3FN lowers at ConvNeXt's 1×1 (3.43×). That does **not**
transfer automatically: §5.3 found bf16 is **0.87×** at CIFAR's shapes, because these convs are
launch-bound. So it had to be measured at cifar8's own geometry:

| layer | shape | f32 ms | fp8 ms | vs f32 | lowers to |
|---|---|---|---|---|---|
| s1c1 | B128 3→16 32² | 0.060 | 0.074 | 0.82× | `__cudnn$convForwardGraph` (21 f8 vals) |
| s1c2 | B128 16→16 32² | 0.140 | 0.062 | 2.27× | ″ |
| s2c1 | B128 16→16 16² | 0.065 | 0.062 | 1.06× | ″ |
| s3c1 | B128 16→32 8² | 0.049 | 0.064 | 0.77× | ″ |
| s3c2 | B128 32→32 8² | 0.055 | 0.071 | 0.78× | ″ |
| s4c2 | B128 32→32 4² | 0.062 | 0.069 | 0.90× | ″ |

⭐ **It lowers — every layer reaches real cuDNN fp8, with f8 values surviving into the optimized
HLO.** There is no silent fold at these shapes, which was the live risk (`bf16_renderer.md` §9.2's
conv fold, and E5M2's silent widening in §2.3, are both exactly that failure).

⛔ **And it buys no speed**, 0.77–1.06× on five of six layers. (The 2.27× is one f32 outlier at
0.140 ms against its own 0.060 ms peer — treat it as noise, not a win.) Same cause as bf16: at
≤32 channels and 4²–32² spatial these convs are launch-bound, so the tensor cores idle.
▶ **fp8 at CIFAR is a LOWERING-CORRECTNESS vehicle, not a performance one.** Say so explicitly;
the speed case lives at ImageNet shapes (§5).

⚠ The probe used all-ones tensors, so it tests the *lowering*, not the *range*. Range is §4.

---

## 2. WHAT EXISTS, AND WHAT DOES NOT

### 2.1 ⛔ There are ZERO fp8 emit ops

Measured 2026-08-25: `grep -ohE '[A-Za-z]*(Fp8|F8|E4M3)[A-Za-z]*' LeanMlir/Proofs/Codegen/` returns
**0** constructors, and no `verified_mlir/*.mlir` contains an f8 type. Compare **27 bf16 ops**.

▶ This is the whole difference in difficulty between the bf16 work and this one. bf16 CIFAR was
"thread a `Bool` through an op family that already existed". fp8 is "build the op family".

### 2.2 ⭐ But the bf16 ops are a working template, and the denotation is already right

An f8 op is *structurally* a copy of its bf16 peer:

| piece | what changes |
|---|---|
| constructor | nothing — `convBf16` already carries `(rnd : ℝ → ℝ)` |
| **denotation** | **nothing** — it is already parameterised by `rnd`; pass E4M3 rounding |
| `pretty` (emit) | `tyBf16` → a new `tyF8` (a ONE-LINE function: append `"f8E4M3FN"`) |
| parse round-trip | one new token |

⭐ The denotation being `rnd`-parametric is the load-bearing fact: **the accuracy bounds
instantiate rather than needing restatement**, and `fp8E4M3 : FloatModel` at `u = 2⁻⁴` already
exists (`Binary32Instance.lean`). `e4m3_render_faithful` is proven for **any** `q` and any scales.

### 2.3 The op set R34 needs is exactly 8

R34's batched render uses 18 ops; only these **8 carry matmuls** and therefore need f8 twins:

`conv`, `convStrided`, `convBackBatched`, `convStridedBackBatched`, `convWeightGradB`,
`convStridedWeightGradB`, `denseRowBack`, `denseWeightGradB`

⭐ **All 8 already have a bf16 twin**, so the effort is precisely "the bf16 op-building exercise,
once more". ⚠ Everything else stays f32 — `bnBatchF`/`bnBatchBack`/`bnBatchMeanB`/`bnBatchVarB`,
`selectPosB`, `gapBackBatched`, `maxPool3s2BackB`, the bias grads. **BN in fp8 is a bad idea on
purpose**: a variance computed at 6.25 % unit roundoff is not a normalisation.

⭐⭐ **cifar8 needs a SUBSET of that same 8.** Since the batched migration (2026-08-25) CIFAR is on
`conv` / `convBackBatched` / `convWeightGradB` / `denseRowBack` / `denseWeightGradB` — five of the
eight, no strided variants. ▶ **Building fp8 at CIFAR builds 5/8 of R34's fp8 set**, which is the
argument for doing CIFAR first rather than as a detour.

---

## 3. CIFAR WITH ACTUAL fp8 OPS — the plan

### 3.1 Scope

Add `tyF8` + f8 peers of the five ops above, mirroring the bf16 constructors exactly, and a
`fp8 : Bool := false` flag through `cifar8AdamTrainStepFaithfulB` beside the existing `bf16` one.
▶ Emit shape is the one §1 measured: **f8 operands, f8-TYPED result, convert back** — not an f32
result. `cifar_lowprec_stability.md` §2.3 measured that exact choice as a **3× swing** (1.17× vs
3.43×), and it is the third time this repo has hit the rule (bf16 conv fold §9.2, bf16 dot §20.1).

### 3.2 ⭐⭐ THE ORACLE ALREADY EXISTS — this is what makes it cheap

The emulated arm is the *true fp8 numerics* (host-side E4M3 rounding is exact E4M3); only the
lowering was missing. So it is **precisely the number the lowered path must reproduce**. Measured
on the wide head (d1=512), n=1, 2026-08-25:

| optimizer | fp8 emulated |
|---|---|
| SGD (lr 0.1) | 71.74 % |
| AdamW (lr 1e-3) | 73.20 % |
| Nesterov (μ0.9, lr 0.02) | 75.38 % |

▶ **Gate: the in-graph fp8 arm must reproduce this column to within rounding-convention noise.**
That makes the project a *hardware-faithfulness check* with a known answer — the same shape as the
JAX `vjp_oracle` — rather than an open-ended accuracy experiment. ⚠ Use the n=5 medians once the
Lever 3 sweep lands, not these n=1 values.

⚠ One asymmetry to keep honest: the emulated path quantises **weights and input** and accumulates
in fp32. An in-graph f8 conv rounds the same operands but the *accumulate* is whatever cuDNN does
(fp32 for the f8 path). If the two disagree, suspect the accumulate before suspecting the emit.

---

## 3.3 ⭐⭐ MEASURED 2026-08-25: what actually lowers, and where scaling becomes mandatory

`convF8`, `convBackBatchedF8` and `convWeightGradBF8` built (the clone surface of §2.2 was exactly
as predicted — denotations byte-identical to their bf16 peers). Then measured with
`cifar8-opt-tie` against the f32 batched render on one shared θ:

| fp8 coverage | f8-typed convs | recovered-gradient norm-rel vs f32 | `%loss` |
|---|---|---|---|
| forward only | 8/23 | **0.101** | 7.606 vs 7.834 |
| forward + input-VJP | 15/23 | ⚠ **0.756** | 7.606 vs 7.833 |
| + weight-gradient | 23/23 | ⛔ **does not compile** | — |

### ⛔⛔ THERE IS NO fp8 `convBackwardFilter` ON sm_89

    __cudnn$convBackwardFilter ... f8e4m3fn[16,3,3,3] ...
    No supported configs found for this instruction.
    Client_Compile: Failed to get configs for: 6 out of 88 instructions

The weight gradient has **no fp8 cuDNN config at all** — it is a library/hardware limit, not an
emit bug (the f32 control compiles; only the f8 bw-filter instructions fail). ▶ §2 of
`cifar_lowprec_stability.md` gated **forward conv and gemm only**, so this was never covered.
`convBackwardData` (the input-VJP) *does* lower. ▶ Any fp8 R34 plan must budget the weight
gradient in f32 or bf16, or re-express it as a `dot_general` (cublasLt f8 works where cuDNN
does not) — that is a design question, not a detail.

#### ⭐⭐ …BUT THE GEMM FORM OF THE SAME GRADIENT DOES WORK, ON THIS HARDWARE

Measured immediately after, at cifar8's s3c2 weight-grad shape (`dL/dW = dYᵀ·X`, i.e. the im2col
`dot_general` M=B·H·W=8192, K=ic·kH·kW=288, N=oc=32):

| operands | kernel |
|---|---|
| f32 | fusion/triton |
| bf16 | fusion/triton |
| **f8E4M3** | **`__cublas$lt$matmul$f8`** |

⭐ **So the fp8 weight gradient is NOT blocked by the silicon — it is blocked by
`__cudnn$convBackwardFilter` specifically.** cublasLt's f8 path is present on the same sm_89 card
and takes this shape. ▶ The fix is to emit the weight gradient as a `dot_general` rather than a
`convolution`, which is a design choice available TODAY and does not wait on new hardware.

⚠ It is not free: an im2col form is a **different op** (same mathematics, different accumulation
order and a materialised `[B·H·W, ic·kH·kW]` matrix), so it needs its own verified constructor and
its own faithfulness argument — it cannot be a retyping of `convWeightGradB`. Budget it as a new
op, not a flag.

⚠ Whether a Blackwell part (sm_120, e.g. a 5060) has the cuDNN conv path is **unknown and not
worth assuming**: the failure is XLA's autotuner finding no cuDNN engine config, so it needs both
cuDNN to ship bw-filter fp8 engines for that arch AND XLA to plumb them. ▶ It is a measurement, not
an inference — and one worth making, since §2.2's `f4E2M1FN` result ("needs Blackwell") is waiting
on the same box.

### ⚠⚠ THE BACKWARD NEEDS SCALES; THE FORWARD MAY NOT

Unscaled, the forward alone costs **0.101** relative on the gradient — the order E4M3's 6.25 %
per-element roundoff predicts, i.e. working as designed. Adding the unscaled input-VJP takes it
to **0.756**, which is broken.

▶ The asymmetry is the finding, and it is the expected one: **cotangents are small**. E4M3's min
normal is ~2⁻⁶ ≈ 0.0156, so gradient values below that underflow, while forward activations are
O(1) and sit comfortably inside the range. This is why real fp8 training loss-scales the
backward. ▶ It means §4's "try fixed calibrated scales first" should be read as **mandatory on
the backward, optional on the forward** — and it is now an empirical claim on this net, not a
guess.

---

## 4. ⚠⚠ SCALING IS THE PROJECT, NOT THE EMIT

E4M3's max is **448** and its min normal is ~2⁻⁶. XLA's fp8 gemm takes scale operands; asked for
none it synthesised `%constant_1` (scale = 1.0), which **will** overflow real activations.

* ✅ **The static scaled form is DONE on paper.** The emulated path already carries per-tensor `sx`
  and per-column `sW`, and `e4m3_render_faithful` is proven for **any** `q` and any scales.
* ⛔ **The dynamic part does not exist anywhere**: amax history / delayed scaling is **stateful**,
  and the renderer has no concept of state. This is the single largest unknown in the document.
* ▶ **Try fixed, calibrated per-tensor scales first.** CIFAR is small enough that one calibration
  pass may suffice, and it is the cheapest thing that could work.
* ⚠⚠ **Do not assume that transfers to R34.** Activation ranges vary far more across 36 layers and
  across ImageNet's diversity than across 8 layers of CIFAR. Fixed scales working at CIFAR is
  **weak** evidence for R34 — much weaker than the op set transferring, which is strong.

---

## 5. R34 + IMAGENET — what it looks like, and what changes

* **The speed prize is real here and only here.** 2.71× gemm / 3.43× conv (§2.2 of the stability
  doc) versus bf16's 1.48×/1.84×, and R34's ImageNet shapes are the compute-bound regime where
  that lands — the opposite of CIFAR (§1) and of bf16 at CIFAR (§5.3).
* **There is a reference to check against**: R34 A3 reached **77.43 %** verified
  (`a3_paper_fidelity.md`). ⚠ Read that doc's own warning first — every ImageNet top-1 quoted
  before its §195 is over 49,920 images, not 50,000.
* **Ops**: the 8 of §2.3, of which 5 come free from the CIFAR work.
* **BN stays f32**, and R34 is BN-dense — so the fp8 fraction of an R34 step is lower than the
  conv count suggests. Budget the win accordingly rather than assuming 3.43× end-to-end.

### 5.1 ⛔⛔ THE CERTIFICATE DOES NOT SURVIVE, AND THAT IS A CLAIM CHANGE

| | u | R34 (36 convs) | R50 (53 convs) |
|---|---|---|---|
| bf16 | 0.39 % | 1.2× | 1.2× |
| **fp8 E4M3** | **6.25 %** | **8.9×** | **24.9×** |

bf16's composed certificate survives depth (under 2× at R50). **fp8's does not.** ▶ The honest
framing for fp8 at ImageNet depth is **empirical-only**: "it trains to the reference accuracy",
never "it is certified to". That is a genuine reduction in what the chapter can assert, and it
should be stated in the chapter rather than discovered by a reader. ⚠ Only the RATIOS in that
table mean anything — `bf16_renderer.md` §11.3 shows even the f32 absolute bound is vacuous.

---

## 6. ▶ STEP ORDER, by risk

1. **`tyF8` + the 5 CIFAR f8 ops**, mirroring their bf16 peers. Gate: the f32 artifacts stay
   byte-identical and `cifar8-opt-tie` still ties (both are free — the flag defaults false).
2. **Profile each op standalone at cifar8's shapes** before wiring (§1 did the forward conv; the
   backward and weight-grad shapes are NOT covered by it). ⚠ `bf16_renderer.md` §19.4's rule, and
   ViT's 0.19× stem wgrad is why it exists.
3. **Fixed calibrated scales**, then the in-graph CIFAR arm. Gate: reproduce §3.2's column.
4. **Only then** amax/delayed scaling, and only if step 3 is shown insufficient.
5. **Then R34**: the 3 strided ops, then the net. Gate against 77.43 %.
6. ⛔ **Do not quote an fp8 speedup from CIFAR** (§1: 0.77–1.06×) and ⛔ **do not make a certified
   claim at ImageNet depth** (§5.1).

---

## 7. File index

* `planning/fp8_lowering.md` — the original design; §3's emit table is still the shape.
* `planning/cifar_lowprec_stability.md` — §2 the lowering gate, §5.2 the measured 3×3 sweep,
  §5.3 why bf16 buys no speed at these shapes, §4.1 the batched-family unification.
* `planning/a3_paper_fidelity.md` — R34's 77.43 %, and the 49,920-vs-50,000 caveat.
* `LeanMlir/Proofs/Float/E4M3FaithfulPoC.lean` — `e4m3_render_faithful`, any `q`/`sx`/`sW`.
* `LeanMlir/Proofs/Float/Binary32Instance.lean` — `fp8E4M3 : FloatModel` at `u = 2⁻⁴`.
* `LeanMlir/E4M3Quant.lean` — the host-side rounding the emulated arms use, i.e. the oracle.
* `LeanMlir/Proofs/Codegen/StableHLO.lean` — `tyBf16` (l.4385) is the template for `tyF8`;
  `convBf16` (l.156) is the constructor template.
