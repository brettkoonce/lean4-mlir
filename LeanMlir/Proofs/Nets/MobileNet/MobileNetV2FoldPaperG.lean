import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFoldG
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FoldPaper

/-! # T3 §1 fold for MobileNetV2 at 17 blocks, UN-FUSED and BATCHED — the Adam artifact's op set

`MobileNetV2FoldPaper.lean` makes every parameter output of the SGD-inline
`mobilenetv2_train_step.mlir` `den`-faithful at the fused `θ − lr·g` ops of the PER-EXAMPLE render.
This is its peer for every other train step this net ships — and, since 4c leg 2 retired that
renderer and that artifact on 2026-09-06, for every train step it ships at all.

⛔ **MobileNetV2's two renders did not overlap the way the other four nets' do.** ConvNeXt, ViT and
EfficientNet each render one traversal with two endings under an `adam : Bool`. MobileNetV2 did
not: `MobileNetV2Render` was SGD-inline only (no `adam` flag anywhere in it) and
`MobileNetV2RenderB` is AdamW/RMSProp-only, at the batched index and at batch BatchNorm. So
`mobilenetv2_adam_train_step`, `mobilenetv2_rms_train_step`, `mobilenetv2_adamdp_train_step` and
every ImageNet artifact — including `mobilenetv2in_rmsdp64`, whose accuracy the book quotes — are on
a chain that has **no fused op to un-fuse**: they emit `*GradB` from the start. This file is
therefore at the BATCHED gradient nodes, which is the "or" branch section 4b.4 allowed and, since
these are the artifacts, the right one. ⭐ It was also a down-payment on 4c, and it paid: the op
table below is the one the converged render kept.

⭐ **One lemma per op kind certifies every optimizer tail at once** — AdamW, RMSProp and the
data-parallel twins all consume the same node. ⚠ The bf16 twins do NOT: they emit `*GradBBf16`,
folded in [`Foundation/Bf16GradNodes.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Foundation/Bf16GradNodes.lean).

## ⭐ Eight of the twelve kinds were already proven, and none of them here

`ResNet34FoldB.lean` and `EfficientNetFoldG.lean` state their folds at op kinds, not
at their nets, and between them they already cover `convWeightGradB`, `convBiasGradB`,
`bnGammaGradB`, `bnBetaGradB`, `denseWeightGradB`, `denseBiasGradB`,
`convStridedXlaWeightGradB` and `depthwiseWeightGradB` — every one at the generality MobileNetV2
needs. The four genuinely new ones are all XLA-`SAME`-or-depthwise BIAS shapes that no other net
emits: `convStridedXlaBiasGradB`, `depthwiseBiasGradB`, `depthwiseStridedXlaWeightGradB`,
`depthwiseStridedXlaBiasGradB`.

## ⛔ Two corrections to `MobileNetV2FoldPaper.lean`'s header, found here

1. **The artifact it names does not exist.** That file says it writes
   `verified_mlir/mobilenetv2_paper_train_step.mlir`. `mnv2TrainStepFaithfulVPaper`'s `funcName`
   DEFAULT was `"mobilenetv2_paper_train_step"`, but its one call site passed
   `"mobilenetv2_train_step"`, and that was the committed 17-block artifact
   (`mobilenetv2_reduced_train_step.mlir` was the 6-block one). ⛔ Both, and the writer, are retired
   as of 2026-09-06 — 4c leg 2.
2. **The shipped parameter count is 158, not 210.** 210 is the census at `convBias := true`; both
   that writer and `MobileNetV2RenderB` default to `convBias := false` — the conv, depthwise and
   project biases are folded into the BatchNorm that follows each of them — and
   `mobilenetv2_train_step.mlir` returned exactly 158 updated tensors, as does the parameter
   half of `mobilenetv2_adam_train_step.mlir`. 158 = stem 3 + b1 6 + 16 blocks × 9 + head 3 +
   dense 2. Neither correction touches a theorem: every fold in that file is `∀`-quantified over
   op instances and true at both flag settings.

## The op table of `mobilenetv2_adam_train_step.mlir`

| emitted node | count at `convBias := false` | lemma |
|---|---|---|
| `convStridedXlaWeightGradB` (3×3/s2 stem) | 1 | `EnetPoCG.convStridedXlaWGradB_den` |
| `convWeightGradB` (expand / project / head 1×1) | 33 | `ResNet34PoCB.convWGradB_den` |
| `depthwiseWeightGradB` (stride-1 3×3) | 13 | `EnetPoCG.depthwiseWGradB_den` |
| `depthwiseStridedXlaWeightGradB` (stride-2 3×3) | 4 | `depthwiseStridedXlaWGradB_den` ⭐ new |
| `bnGammaGradB` / `bnBetaGradB` (52 sites) | 104 | `ResNet34PoCB.bn{Gamma,Beta}GradB_den` |
| `denseWeightGradB` / `denseBiasGradB` | 2 | `ResNet34PoCB.dense{W,B}GradB_den` |

and the four bias kinds the render emits at `convBias := true`, which no committed artifact
contains but which the flag can turn on: `convBiasGradB` (`ResNet34PoCB.convBGradB_den`),
`convStridedXlaBiasGradB`, `depthwiseBiasGradB`, `depthwiseStridedXlaBiasGradB` (the three ⭐ new
ones below).

⚠ **XLA-`SAME` padding at all five stride-2 sites** — the stem and the four stride-2 depthwises.
These are the `convStridedXla*` / `depthwiseStridedXla*` ops, not r34's symmetric peers; the two
families have identical types and identical emitted shapes, so only the certificate distinguishes
them, and MobileNetV2 is the TF-origin net.

## Honest residual
* Every lemma is `∀ cot`: each holds at the actual backward-chain cotangent without naming it.
  Pinning them is the §1a tie, and it landed 2026-09-06 as [`Nets/MobileNet/MobileNetV2StepTieB.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Nets/MobileNet/MobileNetV2StepTieB.lean)
  (§4.2c). The per-example tie (`MobileNetV2TiePoCPaper.lean`, deleted 2026-09-08) was at the fused
  ops and at per-example BatchNorm, so nothing transferred.
* ⚠ **The forward these nodes differentiate is batch BatchNorm** (`bnBatchLA`), which every
  MobileNetV2 statement in `Proofs/` was NOT when this file was written. `MobileNetV2FullB.lean`
  and `MobileNetV2FullBVJP.lean` (§4.2b) closed that the same day. Either way a
  `den = certified gradient` fold is about one op and its free cotangent, and says nothing about
  which whole-net forward produced that cotangent.
* `mobilenetv2in_*dp*` is four replicas: the all-reduce is emitted text outside the AST, so these
  lemmas are about the per-replica gradient node.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.Mnv2PaperPoCG

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The four new op kinds
--   ⚠ XLA-`SAME` at every strided site.
-- ════════════════════════════════════════════════════════════════

/-- **Batched XLA-`SAME` strided conv bias GRADIENT denotes the certified `Σ_n` bias gradient.**
    The stem's bias slot, at `convBias := true`. Same `reduce` text as the stride-1 bias grad; the
    `den` is the odd-phase bias VJP. -/
theorem convStridedXlaBGradB_den {N ic oc h w kH kW : Nat} (cotN : String)
    (W : Kernel4 oc ic kH kW) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (b : Vec oc)
    (cot : Vec (N * (oc * h * w))) (o : Fin oc) :
    den (SHlo.convStridedXlaBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc =>
                  flatConvStride2Xla W b' (batchSlice N (ic * (2 * h) * (2 * w)) x n))
               b o j * batchSlice N (oc * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2Xla_bias_grad_has_vjp W
    (batchSlice N (ic * (2 * h) * (2 * w)) x n)).correct b (batchSlice N (oc * h * w) cot n) o

/-- **Batched stride-1 depthwise bias GRADIENT denotes the certified `Σ_n` bias gradient.** The
    depthwise bias slots at `convBias := true`. EfficientNet has no instance of this op — its
    depthwise convs are followed by BatchNorm, so their bias is always folded. -/
theorem depthwiseBGradB_den {N c h w kH kW : Nat} (cotN : String)
    (W : DepthwiseKernel c kH kW) (x : Vec (N * (c * h * w))) (b : Vec c)
    (cot : Vec (N * (c * h * w))) (o : Fin c) :
    den (SHlo.depthwiseBiasGradB W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c =>
                  Tensor3.flatten (depthwiseConv2d W b'
                    (Tensor3.unflatten (batchSlice N (c * h * w) x n))))
               b o j * batchSlice N (c * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwise_bias_grad_has_vjp W
    (Tensor3.unflatten (batchSlice N (c * h * w) x n))).correct b
    (batchSlice N (c * h * w) cot n) o

/-- **Batched XLA-`SAME` strided depthwise weight GRADIENT denotes the certified `Σ_n` weight
    gradient.** The four stride-2 depthwises (b2/b4/b7/b14). ⚠ This is the `Xla` op — its
    weight-grad correlation keeps the `[p−1, p+1]` pad, the opposite asymmetry from the input-grad,
    and that asymmetry is the whole content of the variant. B0's strided depthwise is the
    SYMMETRIC op, so the two nets do not share this certificate. -/
theorem depthwiseStridedXlaWGradB_den {N c h w kH kW : Nat} (xN cotN : String)
    (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedXlaWeightGradB xN b x W (.operand cotN cot)) idx
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  depthwiseStride2FlatXla (Tensor3.unflatten v') b
                    (batchSlice N (c * (2 * h) * (2 * w)) x n))
               (Tensor3.flatten W) idx j * batchSlice N (c * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2Xla_weight_grad_has_vjp b
    (batchSlice N (c * (2 * h) * (2 * w)) x n)).correct
    (Tensor3.flatten W) (batchSlice N (c * h * w) cot n) idx

/-- **Batched XLA-`SAME` strided depthwise bias GRADIENT denotes the certified `Σ_n` bias
    gradient.** At `convBias := true`. -/
theorem depthwiseStridedXlaBGradB_den {N c h w kH kW : Nat} (cotN : String)
    (W : DepthwiseKernel c kH kW) (x : Vec (N * (c * (2 * h) * (2 * w)))) (b : Vec c)
    (cot : Vec (N * (c * h * w))) (o : Fin c) :
    den (SHlo.depthwiseStridedXlaBiasGradB (h := h) (w := w) W x b (.operand cotN cot)) o
      = ∑ n : Fin N, ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c =>
                  depthwiseStride2FlatXla W b' (batchSlice N (c * (2 * h) * (2 * w)) x n))
               b o j * batchSlice N (c * h * w) cot n j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2Xla_bias_grad_has_vjp W
    (batchSlice N (c * (2 * h) * (2 * w)) x n)).correct b
    (batchSlice N (c * h * w) cot n) o

-- ════════════════════════════════════════════════════════════════
-- § The per-block-profile capstones — the 158-parameter accounting, checked
--   Each is GENERIC in the block dims, so one theorem covers every block of that profile.
--   The conjuncts are exactly the parameter slots `mnv2SigList` emits at `convBias := false`.
-- ════════════════════════════════════════════════════════════════

/-! ## Stem — 3×3/s2 XLA conv (3→32) → batch BN.  3 params: `%sW`, `%sg`, `%sbt`. -/

/-- **Stem gradient nodes all denote the certified gradient.** Paper net: `ic=3, oc=32, 112×112`. -/
theorem mnv2StemGradsCertified {N ic oc h w : Nat} :
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2*h) * (2*w))))
       (W : Kernel4 oc ic 3 3) (cot : Vec (N * (oc*h*w))) (idx : Fin (oc*ic*3*3)),
        den (SHlo.convStridedXlaWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*3*3) =>
                      flatConvStride2Xla (Kernel4.unflatten v') b
                        (batchSlice N (ic*(2*h)*(2*w)) x n))
                   (Kernel4.flatten W) idx j * batchSlice N (oc*h*w) cot n j) ∧
    (∀ (vN epsStr cotN : String) (ε : ℝ) (γ β : Vec oc)
       (v cot : Vec (N * (oc * (h*w)))) (c : Fin oc),
        den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN cot)) c
          = ∑ j : Fin (oc * (N * (h*w))),
              pdiv (fun γ' : Vec oc =>
                      bnPerChannelFlat oc (N*(h*w)) ε γ' β (bnchwFwd N oc h w v))
                   γ c j * bnchwFwd N oc h w cot j) ∧
    (∀ (cotN : String) (ε : ℝ) (γ β : Vec oc) (v : Vec (oc * (N * (h*w))))
       (cot : Vec (N * (oc * (h*w)))) (c : Fin oc),
        den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN cot)) c
          = ∑ j : Fin (oc * (N * (h*w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N*(h*w)) ε γ β' v)
                   β c j * bnchwFwd N oc h w cot j) :=
  ⟨fun xN cotN b x W cot idx => EnetPoCG.convStridedXlaWGradB_den xN cotN b x W cot idx,
   fun vN epsStr cotN ε γ β v cot c => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN ε γ β v cot c,
   fun cotN ε γ β v cot c => ResNet34PoCB.bnBetaGradB_den cotN ε γ β v cot c⟩

/-! ## No-expand block (b1, t=1): depthwise(s1) → BN → relu6 → project 1×1 → BN.  6 params. -/

/-- **No-expand block gradient nodes all denote the certified gradient.** Paper net:
    `ic=32, oc=16, 112×112`. The two BN pairs are the stem capstone's second and third conjuncts
    at `oc := ic` and `oc := oc`, so only the two weight slots are spelled here. -/
theorem mnv2NoExpGradsCertified {N ic oc h w : Nat} :
    (∀ (xN cotN : String) (b : Vec ic) (x : Vec (N * (ic*h*w))) (W : DepthwiseKernel ic 3 3)
       (cot : Vec (N * (ic*h*w))) (idx : Fin (ic*3*3)),
        den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (ic*h*w),
              pdiv (fun v' : Vec (ic*3*3) =>
                      Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic*h*w) x n))))
                   (Tensor3.flatten W) idx j * batchSlice N (ic*h*w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic*h*w))) (W : Kernel4 oc ic 1 1)
       (cot : Vec (N * (oc*h*w))) (idx : Fin (oc*ic*1*1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*1*1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic*h*w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc*h*w) cot n j) :=
  ⟨fun xN cotN b x W cot idx => EnetPoCG.depthwiseWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx⟩

/-! ## Stride-1 inverted-residual block: expand 1×1 → BN → relu6 → depthwise(s1) → BN → relu6 →
    project 1×1 → BN.  9 params.  Covers all twelve stride-1 blocks including the no-skip
    widenings b11/b17 — the skip difference is in the `dx` fan-in, not in the parameter ops. -/

/-- **Stride-1 inverted-residual gradient nodes all denote the certified gradient.** Generic in
    `{ic mid oc h w}`. -/
theorem mnv2Stride1GradsCertified {N ic mid oc h w : Nat} :
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (ic*h*w))) (W : Kernel4 mid ic 1 1)
       (cot : Vec (N * (mid*h*w))) (idx : Fin (mid*ic*1*1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid*h*w),
              pdiv (fun v' : Vec (mid*ic*1*1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic*h*w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (mid*h*w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (mid*h*w))) (W : DepthwiseKernel mid 3 3)
       (cot : Vec (N * (mid*h*w))) (idx : Fin (mid*3*3)),
        den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid*h*w),
              pdiv (fun v' : Vec (mid*3*3) =>
                      Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid*h*w) x n))))
                   (Tensor3.flatten W) idx j * batchSlice N (mid*h*w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (mid*h*w))) (W : Kernel4 oc mid 1 1)
       (cot : Vec (N * (oc*h*w))) (idx : Fin (oc*mid*1*1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*mid*1*1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid*h*w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc*h*w) cot n j) :=
  ⟨fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => EnetPoCG.depthwiseWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx⟩

/-! ## Stride-2 inverted-residual block (b2/b4/b7/b14): identical to the stride-1 profile except
    the depthwise is the XLA-`SAME` strided one, at the `2h×2w` input grid.  9 params. -/

/-- **Stride-2 inverted-residual gradient nodes all denote the certified gradient.** The one
    conjunct that differs from `mnv2Stride1GradsCertified` is the depthwise. -/
theorem mnv2Stride2GradsCertified {N ic mid oc h w : Nat} :
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (ic*(2*h)*(2*w)))) (W : Kernel4 mid ic 1 1)
       (cot : Vec (N * (mid*(2*h)*(2*w)))) (idx : Fin (mid*ic*1*1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid*(2*h)*(2*w)),
              pdiv (fun v' : Vec (mid*ic*1*1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic*(2*h)*(2*w)) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (mid*(2*h)*(2*w)) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec mid) (x : Vec (N * (mid*(2*h)*(2*w))))
       (W : DepthwiseKernel mid 3 3) (cot : Vec (N * (mid*h*w))) (idx : Fin (mid*3*3)),
        den (SHlo.depthwiseStridedXlaWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (mid*h*w),
              pdiv (fun v' : Vec (mid*3*3) =>
                      depthwiseStride2FlatXla (Tensor3.unflatten v') b
                        (batchSlice N (mid*(2*h)*(2*w)) x n))
                   (Tensor3.flatten W) idx j * batchSlice N (mid*h*w) cot n j) ∧
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (mid*h*w))) (W : Kernel4 oc mid 1 1)
       (cot : Vec (N * (oc*h*w))) (idx : Fin (oc*mid*1*1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*mid*1*1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (mid*h*w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc*h*w) cot n j) :=
  ⟨fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => depthwiseStridedXlaWGradB_den xN cotN b x W cot idx,
   fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx⟩

/-! ## Head — 1×1 conv (320→1280) → BN (3 params) — and the dense classifier (2 params). -/

/-- **Head and classifier gradient nodes all denote the certified gradient.** Head: `ic=320,
    oc=1280, 7×7`; classifier: `a=1280, c=nClasses`. The head's BN pair is the stem capstone's
    second and third conjuncts at the head width. ⭐ Generic in the class count, so one statement
    covers the 10-class Imagenette artifacts and the 1000-class `mobilenetv2in` ones. -/
theorem mnv2HeadDenseGradsCertified {N ic oc h w a c : Nat} :
    (∀ (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic*h*w))) (W : Kernel4 oc ic 1 1)
       (cot : Vec (N * (oc*h*w))) (idx : Fin (oc*ic*1*1)),
        den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx
          = ∑ n : Fin N, ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*1*1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                        (Tensor3.unflatten (batchSlice N (ic*h*w) x n))))
                   (Kernel4.flatten W) idx j * batchSlice N (oc*h*w) cot n j) ∧
    (∀ (xN cotN : String) (x : Vec (N * a)) (W : Mat a c) (b : Vec c) (cot : Vec (N * c))
       (i : Fin a) (j : Fin c),
        den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN cot)) (finProdFinEquiv (i, j))
          = ∑ n : Fin N, ∑ k : Fin c,
              pdiv (fun v : Vec (a*c) => dense (Mat.unflatten v) b (batchSlice N a x n))
                   (Mat.flatten W) (finProdFinEquiv (i, j)) k * batchSlice N c cot n k) ∧
    (∀ (cotN : String) (W : Mat a c) (x : Vec a) (b : Vec c) (cot : Vec (N * c)) (j : Fin c),
        den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j
          = ∑ n : Fin N, ∑ k : Fin c,
              pdiv (fun b' : Vec c => dense W b' x) b j k * batchSlice N c cot n k) :=
  ⟨fun xN cotN b x W cot idx => ResNet34PoCB.convWGradB_den xN cotN b x W cot idx,
   fun xN cotN x W b cot i j => ResNet34PoCB.denseWGradB_den xN cotN x W b cot i j,
   fun cotN W x b cot j => EnetPoCG.denseBGradB_den cotN W x b cot j⟩

end Proofs.Mnv2PaperPoCG
