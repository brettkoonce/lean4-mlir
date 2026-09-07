import LeanMlir.Proofs.Foundation.ResNet34FaithfulPoCB
import LeanMlir.Proofs.Architectures.EfficientNetFaithfulPoCG
import LeanMlir.Proofs.Architectures.ConvNeXtFaithfulPoCGB
import LeanMlir.Proofs.Architectures.MobileNetV4FullBVJP

/-! # T3 §1 fold for MobileNetV4-Conv-M — every parameter GRADIENT node, by block profile

`MobileNetV4FullB.lean` gives MNv4 its ℝ forward and typed graph (T1, T2). This is the first half
of T3: every parameter gradient node the batched train step emits denotes the certified gradient,
for an arbitrary cotangent.

⚠⚠ **No accuracy is quoted for this net.** Conv-M has no Imagenette run and no verified ImageNet
run; what pins these tiers to the reference's function is the 2026-09-07 tie pair (forward
`max |Δ| = 3.770e-06`, gradient inside the reference's own fp32 floor).

## ⭐⭐ Zero new fp32 op-kind lemmas — MNv4's nine kinds are three other nets', verbatim

4b's lesson was that op kinds are shared far more than the per-net file names suggest. For MNv4 it
is **nine of nine**, drawn from three files and not one:

| op kind | sites | certificate |
|---|---|---|
| `bnGammaGradB` / `bnBetaGradB` | 77 BN layers | `ResNet34PoCB.bnGammaGradB_den` / `bnBetaGradB_den` |
| `convWeightGradB` | expands, projects, both head convs, the fused project | `ResNet34PoCB.convWGradB_den` |
| `convStridedWeightGradB` (SYMMETRIC) | the fused stage's 3×3/s2 | `ResNet34PoCB.convStridedWGradB_den` |
| `convStridedXlaWeightGradB` (XLA-`SAME`) | the stem, and only the stem | `EnetPoCG.convStridedXlaWGradB_den` |
| `depthwiseWeightGradB` | every stride-1 depthwise | `EnetPoCG.depthwiseWGradB_den` |
| `depthwiseStridedWeightGradB` | rows 1, 3, 11's leading depthwise | `EnetPoCG.depthwiseStridedWGradB_den` |
| `denseWeightGradB` / `denseBiasGradB` | the classifier | `ResNet34PoCB.denseWGradB_den` / `denseBGradB_den` |

⚠⚠ **TWO padding phases, and the two strided conv kinds are NOT interchangeable.** The stem is
XLA-`SAME` (`flatConvStride2Xla`, EfficientNet-B0's op) and the fused stage is SYMMETRIC
(`flatConvStride2`, ResNet's). Identical types, identical emitted shapes, different certificates —
`scripts/convention_audit.py` is what reads them apart, and swapping one for the other is the
6.16e-2-vs-1.79e-6 forward-tie defect `planning/mnv4_verified.md` §3b measured.

⛔ **MNv4 emits no conv BIAS gradient at all.** `MobileNetV4RenderB` has no `convBias` flag — every
bias is folded into its BatchNorm and bound to `%zb{c}` — so `convBiasGradB` and its strided peers
are never emitted and there is nothing to state. Same situation as ResNet-50.

## The op table, per profile

| profile | rows | conv-weight nodes | BN pairs | slots |
|---|---|---|---|---|
| stem | — | `convStridedXlaWeightGradB` | 1 | 3 |
| fused stage | — | `convStridedWeightGradB` + `convWeightGradB` | 2 | 6 |
| ExtraDW UIB | 10 stride-1 | dw, 1×1, dw, 1×1 | 4 | 12 |
| pre-strided UIB | 1, 3, 11 | **dw-STRIDED**, 1×1, dw, 1×1 | 4 | 12 |
| ConvNeXt-like UIB | 8, 10, 16, 21 | dw, 1×1, 1×1 — **no post-DW** | 3 | 9 |
| FFN UIB | 9, 15, 19, 20 | 1×1, 1×1 — **neither depthwise** | 2 | 6 |
| head | — | two `convWeightGradB`, then `denseWeightGradB` + `denseBiasGradB` | 2 | 8 |

3 + 6 + 13×12 + 4×9 + 4×6 + 8 = **233**, which is the 234-argument signature of
`verified_mlir/mnv4_fwd.mlir` minus `%x`, and the 233 the train step's 858 inputs carry three
copies of. ⭐ Every one of the 233 is exercised: MNv4 is bias-free by construction, so unlike
ResNet-34 (146 stated / 110 exercised) there is no `convBias := true` census to over-count.

⛔⛔ **An absent depthwise gets NO conjunct.** The ConvNeXt and FFN profiles below are not the
ExtraDW one with a spare slot: `UibParams` still carries a degenerate `DepthwiseKernel _ 0 0` in
the empty slot, the render emits no token for it, and a conjunct stated there would be the `den` of
a node the artifact does not have. This file's profiles branch exactly as `mnv4PreDWSlot` does.

⛔ **Zero post-strided rows.** Conv-M has none (Conv-S had one), so there is no profile for that
arm here — its `CertLayer` remains certified and unexercised one tier down.

## Honest residual

* The cotangents are free variables — every conjunct is `∀ cot`, so each holds at the actual
  backward-chain cotangent without naming it. Pinning them to the emitted backward subgraph is the
  §1a tie (`MobileNetV4TiePoCB.lean`).
* ⛔ **One replica.** Under `mnv4in_adamdp64*` each node is followed by the all-reduce mean, which
  `Foundation/DataParallel.lean` handles as its own tier (4d); this statement is at the
  per-replica gradient.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.Mnv4PoCB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § BatchNorm — one statement for all 77 sites
-- ════════════════════════════════════════════════════════════════

/-- **Every BatchNorm γ and β gradient node denotes the certified per-channel gradient**, at the
    merged batch+spatial width `m = N·(h·w)`. Width-generic, so all 77 of MNv4's sites — up to four
    per UIB block, two in the fused stage, one in the stem and two in the head — are instances of
    these two. ⭐ No bf16 twin exists or should: every bf16 net keeps BatchNorm in f32. -/
theorem mnv4BnGradsCertified {N oc h w : Nat} :
    (∀ (vN epsStr cotN : String) (ε : ℝ) (γ β : Vec oc)
       (v : Vec (N * (oc * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (c : Fin oc),
        den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN cot)) c
          = ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun γ' : Vec oc =>
                      bnPerChannelFlat oc (N * (h * w)) ε γ' β (bnchwFwd N oc h w v))
                   γ c j * bnchwFwd N oc h w cot j) ∧
    (∀ (cotN : String) (ε : ℝ) (γ β : Vec oc)
       (v : Vec (oc * (N * (h * w)))) (cot : Vec (N * (oc * (h * w)))) (c : Fin oc),
        den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN cot)) c
          = ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) ε γ β' v)
                   β c j * bnchwFwd N oc h w cot j) :=
  ⟨fun vN epsStr cotN ε γ β v cot c => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN ε γ β v cot c,
   fun cotN ε γ β v cot c => ResNet34PoCB.bnBetaGradB_den cotN ε γ β v cot c⟩

-- ════════════════════════════════════════════════════════════════
-- § The stem — 3x3/s2 at the XLA-`SAME` phase, and the ONLY such site
-- ════════════════════════════════════════════════════════════════

/-- **The stem's weight gradient node denotes the certified `Σ_n` weight gradient.**

    ⚠⚠ `convStridedXlaWeightGradB`, whose `den` runs `flatConvStride2Xla` — EfficientNet-B0's op,
    not ResNet's. XLA `'SAME'` on a 3×3/s2 at 224 pads (0,1), the symmetric token pads (1,1), both
    give 112×112, and no shape check, `#guard`, op count or arity audit can tell them apart. The
    forward tie is the only thing that can. ⛔ Every OTHER stride-2 site in this net is genuinely
    symmetric; do not "tidy" one to match the other. -/
theorem mnv4StemGradsCertified {N ic oc h w kH kW : Nat} :
    ∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
      (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)),
      den (SHlo.convStridedXlaWeightGradB xN b x W (.operand cotN cot)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                    flatConvStride2Xla (Kernel4.unflatten v') b
                      (batchSlice N (ic * (2 * h) * (2 * w)) x n))
                 (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j :=
  fun xN cotN b x W cot idx => EnetPoCG.convStridedXlaWGradB_den xN cotN b x W cot idx

-- ════════════════════════════════════════════════════════════════
-- § The fused stage (stage 0) — SYMMETRIC strided conv, then a 1x1 project
-- ════════════════════════════════════════════════════════════════

/-- **The fused stage's two weight gradient nodes.** ⚠ The 3×3/s2 is `convStridedWeightGradB` —
    SYMMETRIC padding, `flatConvStride2` — where the stem three lines up is XLA-`SAME`. One net,
    two phases, both correct. -/
theorem mnv4FusedGradsCertified {N ic mid oc h w kH kW : Nat} :
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (ic * (2 * h) * (2 * w))))
       (W : Kernel4 mid ic kH kW) (cot : Vec (N * (mid * h * w)))
       (idx : Fin (mid * ic * kH * kW)),
        den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * ic * kH * kW) =>
                      flatConvStride2 (Kernel4.unflatten v') b
                        (batchSlice N (ic * (2 * h) * (2 * w)) x n))
                   (Kernel4.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (mid * h * w))) (W : Kernel4 oc mid 1 1)
       (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * mid * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j) :=
  ⟨fun xN cotN b x W cot idx => ResNet34PoCB.convStridedWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx⟩

-- ════════════════════════════════════════════════════════════════
-- § The four UIB profiles — and the absent depthwise gets NO conjunct
-- ════════════════════════════════════════════════════════════════

/-- **The ExtraDW block's four weight gradient nodes** — pre-depthwise at `ic`, the 1×1 expand
    `ic → mid`, the post-depthwise at `mid`, the 1×1 project `mid → oc`. Thirteen of Conv-M's 21
    rows have this profile (ten at stride 1, three strided — those use
    `mnv4PreStridedGradsCertified` for the leading node). With `mnv4BnGradsCertified` at the four
    BatchNorm sites that is the block's twelve parameters. -/
theorem mnv4ExtraDWGradsCertified {N ic mid oc h w kq kd : Nat} :
    (∀ (xN cotN : String) (b : Vec ic) (x : Vec (N * (ic * h * w)))
       (W : DepthwiseKernel ic kq kq) (cot : Vec (N * (ic * h * w))) (idx : Fin (ic * kq * kq)),
        den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (ic * h * w),
              pdiv (fun v' : Vec (ic * kq * kq) =>
                      depthwiseFlat (Tensor3.unflatten v') b (batchSlice N (ic * h * w) x n))
                   (Tensor3.flatten W) idx j * batchSlice N (ic * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (ic * h * w))) (W : Kernel4 mid ic 1 1)
       (cot : Vec (N * (mid * h * w))) (idx : Fin (mid * ic * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (mid * h * w)))
       (W : DepthwiseKernel mid kd kd) (cot : Vec (N * (mid * h * w)))
       (idx : Fin (mid * kd * kd)),
        den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * kd * kd) =>
                      depthwiseFlat (Tensor3.unflatten v') b (batchSlice N (mid * h * w) x n))
                   (Tensor3.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (mid * h * w))) (W : Kernel4 oc mid 1 1)
       (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * mid * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j) :=
  ⟨fun xN cotN b x W cot idx => EnetPoCG.depthwiseWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => EnetPoCG.depthwiseWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx⟩

/-- ⭐ **The ConvNeXt-like block's THREE weight gradient nodes** — pre-depthwise, expand, project.
    Rows 8, 10, 16 and 21.

    ⛔ This is not `mnv4ExtraDWGradsCertified` with a conjunct dropped for tidiness: `postDWk = 0`
    means the render emits **no post-depthwise token**, so a fourth conjunct would be the `den` of
    a node the artifact does not contain. The record still carries a degenerate
    `DepthwiseKernel mid 0 0` in that slot and nothing reads it — the same `k = 0` rule
    `mnv4PostDWSlot` dispatches on, one tier up. -/
theorem mnv4ConvNeXtGradsCertified {N ic mid oc h w kq : Nat} :
    (∀ (xN cotN : String) (b : Vec ic) (x : Vec (N * (ic * h * w)))
       (W : DepthwiseKernel ic kq kq) (cot : Vec (N * (ic * h * w))) (idx : Fin (ic * kq * kq)),
        den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (ic * h * w),
              pdiv (fun v' : Vec (ic * kq * kq) =>
                      depthwiseFlat (Tensor3.unflatten v') b (batchSlice N (ic * h * w) x n))
                   (Tensor3.flatten W) idx j * batchSlice N (ic * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (ic * h * w))) (W : Kernel4 mid ic 1 1)
       (cot : Vec (N * (mid * h * w))) (idx : Fin (mid * ic * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (mid * h * w))) (W : Kernel4 oc mid 1 1)
       (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * mid * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j) :=
  ⟨fun xN cotN b x W cot idx => EnetPoCG.depthwiseWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx⟩

/-- ⭐ **The FFN block's TWO weight gradient nodes** — expand and project, and nothing else. Rows
    9, 15, 19 and 20: `preDWk = postDWk = 0`, so both depthwise slots are absent from the render
    and from this statement. Six parameters, the smallest profile in the net. -/
theorem mnv4FfnGradsCertified {N ic mid oc h w : Nat} :
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (ic * h * w))) (W : Kernel4 mid ic 1 1)
       (cot : Vec (N * (mid * h * w))) (idx : Fin (mid * ic * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (mid * h * w))) (W : Kernel4 oc mid 1 1)
       (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * mid * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j) :=
  ⟨fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx⟩

/-- ⭐ **The pre-strided block's leading node** — rows 1, 3 and 11, the only stride-2 rows Conv-M
    has, and all three PRE-strided. `depthwiseStridedWeightGradB`, whose `den` runs
    `depthwiseStride2Flat` at SYMMETRIC padding (EfficientNet-B0's op, and the same phase every
    non-stem stride-2 site in this net uses).

    ⚠ The block's other three nodes are `mnv4ExtraDWGradsCertified`'s — all three strided rows have
    `postDWk > 0`, so the profile below the leading depthwise is ExtraDW's, at the reduced
    resolution. -/
theorem mnv4PreStridedGradsCertified {N c h w kH kW : Nat} :
    ∀ (xN cotN : String) (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w))))
      (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)),
      den (SHlo.depthwiseStridedWeightGradB xN b x W (.operand cotN cot)) idx
        = ∑ n : Fin N, ∑ j : Fin (c * h * w),
            pdiv (fun v' : Vec (c * kH * kW) =>
                    depthwiseStride2Flat (Tensor3.unflatten v') b
                      (batchSlice N (c * (2 * h) * (2 * w)) x n))
                 (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j :=
  fun xN cotN b x W cot idx => EnetPoCG.depthwiseStridedWGradB_den xN cotN b x W cot idx

-- ════════════════════════════════════════════════════════════════
-- § The head — TWO 1x1 convs, then the classifier
-- ════════════════════════════════════════════════════════════════

/-- **The head's four weight/bias gradient nodes.** ⚠ Conv-M's head has **two** convolutions
    (`%h1W` 256 → 960, then `%hW` 960 → 1280) where `mnv4Head` models one; both are ordinary
    `convWeightGradB` at 7×7, so the first two conjuncts differ only in their widths. Then the
    classifier's weight and bias.

    ⚠ The dense conjuncts are stated in `ResNet34PoCB`'s own shape, which is not the shape the two
    conv conjuncts take: the weight index is `finProdFinEquiv (i, j)` rather than a flat `idx`, and
    the bias node's `den` does not read `W` or `x` at all — the emitted `denseBiasGradB` carries
    only the cotangent, so its certificate quantifies over a `W`/`x` the node never sees. Restating
    them in a tidier shape would be a second reading of the same fact; these are the lemmas as
    proven. -/
theorem mnv4HeadGradsCertified {N c mid oc h w nCls : Nat} :
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (c * h * w))) (W : Kernel4 mid c 1 1)
       (cot : Vec (N * (mid * h * w))) (idx : Fin (mid * c * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * c * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (c * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (mid * h * w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (mid * h * w))) (W : Kernel4 oc mid 1 1)
       (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * mid * 1 * 1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid * h * w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) cot n j) ∧
    (∀ (xN cotN : String) (x : Vec (N * oc)) (Wd : Mat oc nCls) (bd : Vec nCls)
       (cot : Vec (N * nCls)) (i : Fin oc) (j : Fin nCls),
        den (SHlo.denseWeightGradB (c := nCls) xN x (.operand cotN cot))
              (finProdFinEquiv (i, j))
          = ∑ n : Fin N, ∑ k : Fin nCls,
              pdiv (fun v : Vec (oc * nCls) =>
                      dense (Mat.unflatten v) bd (batchSlice N oc x n))
                   (Mat.flatten Wd) (finProdFinEquiv (i, j)) k
                * batchSlice N nCls cot n k) ∧
    (∀ (cotN : String) (Wd : Mat nCls nCls) (x : Vec nCls) (bd : Vec nCls)
       (cot : Vec (N * nCls)) (j : Fin nCls),
        den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j
          = ∑ n : Fin N, ∑ k : Fin nCls,
              pdiv (fun b' : Vec nCls => dense Wd b' x) bd j k
                * batchSlice N nCls cot n k) :=
  ⟨fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN x Wd bd cot i j => ResNet34PoCB.denseWGradB_den xN cotN x Wd bd cot i j,
   fun cotN Wd x bd cot j => ResNet34PoCB.denseBGradB_den cotN Wd x bd cot j⟩

-- ════════════════════════════════════════════════════════════════
-- § bf16 — the five nodes `mnv4in_adam64bf16` / `adamdp64bf16` emit
-- ════════════════════════════════════════════════════════════════

/-! ⛔⛔ **"the bf16 twins consume the same node" is FALSE, and this section exists because three
other nets' fold headers say it.** A bf16 render emits its OWN `*GradBBf16` constructor with its
own `den`: the operands are rounded before the contraction and the result is rounded ONCE, outside
`Σ_n`. That is a different real number from the f32 node's, so it needs its own certificate.

MNv4's bf16 artifacts emit **five** weight-gradient kinds. Three are already proven generically in
`ConvNeXtFaithfulPoCGB.lean` and are cited; two did not exist and are proven here. ⭐ Both are four
lines on the same template as the three — the rounding is outside the sum, so `congr 1` peels it
and the inner equality is the f32 certificate at rounded operands.

⚠ **BatchNorm has no bf16 twin here or anywhere**, by design: every bf16 net in the suite keeps its
BatchNorm in f32, so all 77 of MNv4's γ/β nodes are `mnv4BnGradsCertified`'s in both worlds. -/

/-- **bf16 STRIDED-DEPTHWISE weight gradient denotes the certified `Σ_n` gradient at the rounded
    operands, rounded once.** New here: rows 1, 3 and 11's leading depthwise on the bf16 path, and
    no other net in the suite emits this node. -/
theorem depthwiseStridedWGradBBf16_den {N c h w kH kW : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
      = rnd (∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  depthwiseStride2Flat (Tensor3.unflatten v') b
                    (fun j => rnd (batchSlice N (c * (2 * h) * (2 * w)) x n j)))
               (Tensor3.flatten W) idx j * rnd (batchSlice N (c * h * w) cot n j)) := by
  simp only [den]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2_weight_grad_has_vjp b
    (fun j => rnd (batchSlice N (c * (2 * h) * (2 * w)) x n j))).correct
    (Tensor3.flatten W) (fun j => rnd (batchSlice N (c * h * w) cot n j)) idx

/-- **bf16 XLA-`SAME` STRIDED-CONV weight gradient denotes the certified `Σ_n` gradient at the
    rounded operands, rounded once.** New here: MNv4's stem on the bf16 path. ⚠ `flatConvStride2Xla`,
    not `flatConvStride2` — the same two-phase distinction the f32 stem carries, and equally
    invisible to every shape check. -/
theorem convStridedXlaWGradBBf16_den {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedXlaWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
      = rnd (∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  flatConvStride2Xla (Kernel4.unflatten v') b
                    (fun j => rnd (batchSlice N (ic * (2 * h) * (2 * w)) x n j)))
               (Kernel4.flatten W) idx j * rnd (batchSlice N (oc * h * w) cot n j)) := by
  simp only [den]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2Xla_weight_grad_has_vjp b
    (fun j => rnd (batchSlice N (ic * (2 * h) * (2 * w)) x n j))).correct
    (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc * h * w) cot n j)) idx

/-- ⭐⭐ **All five bf16 weight-gradient kinds MNv4's bf16 artifacts emit, certified together.**
    In render order: the stem (XLA-`SAME` strided conv), the fused stage's 3×3/s2 (symmetric
    strided conv), every 1×1, every stride-1 depthwise, and rows 1/3/11's strided depthwise.

    ⭐ Three of the five are `CnxPoCGB`'s, generic and reused verbatim; the last two are this
    file's. ⚠ Every one rounds ONCE, outside `Σ_n` — the operands go in rounded and the batch sum
    is contracted in the accumulate type. -/
theorem mnv4Bf16GradsCertified {N ic oc c h w kH kW : Nat} (rnd : ℝ → ℝ) :
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
       (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w)))
       (idx : Fin (oc * ic * kH * kW)),
        den (SHlo.convStridedXlaWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
          = rnd (∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                      flatConvStride2Xla (Kernel4.unflatten v') b
                        (fun j => rnd (batchSlice N (ic * (2 * h) * (2 * w)) x n j)))
                   (Kernel4.flatten W) idx j * rnd (batchSlice N (oc * h * w) cot n j))) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
       (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w)))
       (idx : Fin (oc * ic * kH * kW)),
        den (SHlo.convStridedWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
          = rnd (∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                      flatConvStride2 (Kernel4.unflatten v') b
                        (fun j => rnd (batchSlice N (ic * (2 * h) * (2 * w)) x n j)))
                   (Kernel4.flatten W) idx j * rnd (batchSlice N (oc * h * w) cot n j))) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * h * w)))
       (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w)))
       (idx : Fin (oc * ic * kH * kW)),
        den (SHlo.convWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
          = rnd (∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (fun j => rnd (batchSlice N (ic * h * w) x n j)))))
                   (Kernel4.flatten W) idx j * rnd (batchSlice N (oc * h * w) cot n j))) ∧
    (∀ (xN cotN : String) (b : Vec c) (x : Vec (N * (c * h * w)))
       (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)),
        den (SHlo.depthwiseWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
          = rnd (∑ n : Fin N, ∑ j : Fin (c * h * w),
              pdiv (fun v' : Vec (c * kH * kW) =>
                      depthwiseFlat (Tensor3.unflatten v') b
                        (fun j => rnd (batchSlice N (c * h * w) x n j)))
                   (Tensor3.flatten W) idx j * rnd (batchSlice N (c * h * w) cot n j))) ∧
    (∀ (xN cotN : String) (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w))))
       (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)),
        den (SHlo.depthwiseStridedWeightGradBBf16 rnd xN b x W (.operand cotN cot)) idx
          = rnd (∑ n : Fin N, ∑ j : Fin (c * h * w),
              pdiv (fun v' : Vec (c * kH * kW) =>
                      depthwiseStride2Flat (Tensor3.unflatten v') b
                        (fun j => rnd (batchSlice N (c * (2 * h) * (2 * w)) x n j)))
                   (Tensor3.flatten W) idx j * rnd (batchSlice N (c * h * w) cot n j))) :=
  ⟨fun xN cotN b x W cot idx => convStridedXlaWGradBBf16_den rnd xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => CnxPoCGB.convStridedWGradBBf16_den rnd xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => CnxPoCGB.convWGradBBf16_den rnd xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => CnxPoCGB.depthwiseWGradBBf16_den rnd xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => depthwiseStridedWGradBBf16_den rnd xN cotN b x W cot idx⟩

end Proofs.Mnv4PoCB
