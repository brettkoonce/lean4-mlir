import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFoldG
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2Fold
import LeanMlir.Proofs.Nets.ResNet.ResNet34Fold
import LeanMlir.Proofs.Nets.Small.CifarFold
import LeanMlir.Proofs.Nets.Small.CifarBnFold

/-! # T3 §1 fold for MobileNetV2 at 17 blocks, UN-FUSED and BATCHED — the Adam artifact's op set

This file makes every parameter gradient node of MobileNetV2's batched train steps `den`-faithful
— every train step the net ships, since 4c leg 2 retired the per-example SGD-inline renderer and
its artifact on 2026-09-06.

⛔ **MobileNetV2's two renders did not overlap the way the other four nets' do.** ConvNeXt, ViT and
EfficientNet each render one traversal with two endings under an `adam : Bool`. MobileNetV2 did
not: `MobileNetV2Render.lean` was SGD-inline only (no `adam` flag anywhere in it) and
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

⚠ **The shipped parameter count is 158, not 210.** 210 is the census at `convBias := true`;
`MobileNetV2RenderB` defaults to `convBias := false` — the conv, depthwise and project biases are
folded into the BatchNorm that follows each of them — and the parameter half of
`mobilenetv2_adam_train_step.mlir` returns exactly 158 updated tensors: stem 3 + b1 6 + 16 blocks
× 9 + head 3 + dense 2.

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
  simp only [denStep, denStepApp]
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
  simp only [denStep, denStepApp]
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
  simp only [denStep, denStepApp]
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
  simp only [denStep, denStepApp]
  apply Finset.sum_congr rfl
  intro n _
  exact (depthwiseStride2Xla_bias_grad_has_vjp W
    (batchSlice N (c * (2 * h) * (2 * w)) x n)).correct b
    (batchSlice N (c * h * w) cot n) o

end Proofs.Mnv2PaperPoCG
