import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFaithfulPoC
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FaithfulPoC
import LeanMlir.Proofs.Nets.ViT.ViTFaithfulPoC
import LeanMlir.Proofs.Nets.ResNet.ResNet34FaithfulPoC

/-! # T3 §1 fold for ConvNeXt-T at the UN-FUSED gradient — the Adam artifact's op set

`ConvNeXtFaithfulPoC.lean` (with the shared per-example folds it leans on) makes every parameter
output of the SGD-inline `convnext_train_step.mlir` `den`-faithful at the fused `θ − lr·g` ops.
`ConvNeXtRender.lean` renders that traversal TWICE, under one `adam : Bool`: the `false` branch
emits the fused `*Sgd` ops, and the `true` branch emits the RAW gradient and hands it to the AdamW
triple. Every non-SGD artifact of this net — `convnext_adam_train_step.mlir`, the ImageNet
`convnextin_adamdpwxclipdrop` whose accuracy the book quotes, and the clip/weight-decay/drop-path
variants — is on the `true` branch. This file is the fold at those nodes.

⭐ **One lemma per op kind certifies every optimizer tail at once**, because AdamW, the clipped and
weight-decayed variants and plain SGD all consume the same gradient node. The fusion itself is
`rfl` (`StableHLO.lean`'s `*Sgd_eq_grad` family), so nothing here is new mathematics: each proof is
the fused lemma's proof with the `congr 1` / `congrArg (lr * ·)` wrapper peeling dropped, which is
`ResNet34FaithfulPoCB.lean`'s recipe at the per-example index.

## ⭐ ConvNeXt was already half-way there, and the reason is worth recording

`psW`, the 4×4/s4 patchify stem weight, has **no fused peer at all** — `convStride4WeightGrad` is a
gradient op in both renders, because the SGD path wraps it in hand-written text (a declared §5
carve-out). That carve-out was the exception; 4b makes it the norm, and `psWGrad_den` below is the
only lemma in this file whose statement is unchanged from what the SGD render already needed.

## The op table of `convnext_adam_train_step.mlir`

| emitted node | lemma | fused peer it un-fuses |
|---|---|---|
| `layerScaleChGammaGrad` (18 block γ) | `layerScaleChGammaGrad_den` | `ConvNeXtFaithfulPoC.layerScaleChGammaSgd_den` |
| `convWeightGrad` / `convBiasGrad` (18 expand + 18 project 1×1, + the stem bias) | `convWGrad_den` / `convBGrad_den` | `CifarPoC.convW_den` / `convB_den` |
| `depthwiseWeightGrad` / `depthwiseBiasGrad` (18 × 7×7) | `depthwiseWGrad_den` / `depthwiseBGrad_den` | `Mnv2PoC.depthwiseW_den` / `depthwiseB_den` |
| `convStridedWeightGrad` / `convStridedBiasGrad` (3 × 2×2/s2 downsample) | `convStridedWGrad_den` / `convStridedBGrad_den` | `ResNet34PoC.convStridedW_den` / `convStridedB_den` |
| `convStride4WeightGrad` (patchify stem) | `psWGrad_den` | none — already un-fused |
| `veclnGammaGrad` / `rowDenseBiasGrad` at `N = h·w` (22 spatial LN sites) | `chanLnGammaGrad_den` / `chanLnBetaGrad_den` | `ConvNeXtFaithfulPoC.chanLnGammaSgd_den` / `chanLnBetaSgd_den` |
| `veclnGammaGrad` / `rowDenseBiasGrad` at `N = 1` (the head LN, after GAP) | `headLnGammaGrad_den` / `headLnBetaGrad_den` | `ViTPoC.veclnGammaSgd_den` / `rowDenseBiasSgd_den_lnbeta` |
| `weightGrad` / `biasGrad` (the classifier) | `headWGrad_den` / `headBGrad_den` | `Cifar8PoC.denseW_den` / `denseB_den` |

⚠ **Symmetric padding at the downsamples.** These are `convStridedWeightGrad` /
`convStridedBiasGrad`, whose `den` is `flatConvStride2_*`; MobileNetV2's and B0's stems are the
`convStridedXla*` ops. ConvNeXt is PyTorch-origin, and at an even 2×2 kernel the two phases are
genuinely different functions, not a cosmetic choice.

## Honest residual
* Every lemma is `∀ cot`, so each holds at the actual backward-chain cotangent without naming it.
  Pinning them is the §1a tie (`ConvNeXtTiePoC.lean`); re-pointing that capstone at these nodes
  needs the smoothed-target loss cotangent, scoped with r34's batched tie.
* The all-reduce in `convnextin_adamdp*` is emitted text outside the AST, so these lemmas are about
  the per-replica gradient node.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxPoCG

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-channel layer scale — the one op kind unique to this net
-- ════════════════════════════════════════════════════════════════

/-- **Per-channel layer-scale γ GRADIENT denotes the certified gradient.** The emitted `lsGradCh`
    reduce `dγ_c = Σ_{k : chanIdx k = c} x_k·dy_k` IS the certified Jacobian of `layerScaleChF`
    as a function of `γ : Vec c`, contracted with the cotangent. -/
theorem layerScaleChGammaGrad_den {c h w : Nat} (xN cotN : String)
    (x : Vec (c * h * w)) (γ : Vec c) (dy : Vec (c * h * w)) (cc : Fin c) :
    den (SHlo.layerScaleChGammaGrad xN x (.operand cotN dy)) cc
      = ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => layerScale (fun k => γ' (chanIdx c h w k)) x) γ cc j * dy j := by
  simp only [den]
  apply Finset.sum_congr rfl
  intro j _
  rw [Proofs.CnxPoC.pdiv_layerScaleCh_gamma]
  by_cases hcc : chanIdx c h w j = cc
  · rw [if_pos hcc, if_pos hcc.symm]
  · rw [if_neg hcc, if_neg (fun h => hcc h.symm)]; ring

-- ════════════════════════════════════════════════════════════════
-- § The 1×1 convolutions (expand / project) and the stem bias
-- ════════════════════════════════════════════════════════════════

/-- **Conv weight GRADIENT denotes the certified weight gradient.** Kernel-generic, so the one
    lemma covers every 1×1 expand and project AND — at 4×4 — the stem's bias-side twin. -/
theorem convWGrad_den {ic oc h w kH kW : Nat} (xN cotN : String)
    (b : Vec oc) (x : Tensor3 ic h w) (W : Kernel4 oc ic kH kW) (cot : Vec (oc * h * w))
    (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convWeightGrad xN b x W (.operand cotN cot)) idx
      = ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
               (Kernel4.flatten W) idx j * cot j :=
  conv_weight_grad_bridge b x (Kernel4.flatten W) cot idx

/-- **Conv bias GRADIENT denotes the certified bias gradient** (the channel sum). Covers the
    expand/project biases and the patchify stem's `psb`. -/
theorem convBGrad_den {ic oc h w kH kW : Nat} (cotN : String)
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b : Vec oc) (cot : Vec (oc * h * w))
    (o : Fin oc) :
    den (SHlo.convBiasGrad W x b (.operand cotN cot)) o
      = ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * cot j :=
  conv_bias_grad_bridge W x b cot o

-- ════════════════════════════════════════════════════════════════
-- § The 7×7 depthwise
-- ════════════════════════════════════════════════════════════════

/-- **Depthwise weight GRADIENT denotes the certified weight gradient.** All 18 blocks; the
    kernel size is a variable, so 7×7 is an instance. -/
theorem depthwiseWGrad_den {c h w kH kW : Nat} (xN cotN : String)
    (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c kH kW) (cot : Vec (c * h * w))
    (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseWeightGrad xN b x W (.operand cotN cot)) idx
      = ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
                  Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b x))
               (Tensor3.flatten W) idx j * cot j := by
  simp only [den]
  rw [← (hasVJP3_to_hasVJP (depthwise_weight_grad_has_vjp3 b x)).correct
      (Tensor3.flatten W) cot idx]
  simp only [hasVJP3_to_hasVJP, Tensor3.flatten, Tensor3.unflatten_flatten]

/-- **Depthwise bias GRADIENT denotes the certified bias gradient.** -/
theorem depthwiseBGrad_den {c h w kH kW : Nat} (cotN : String)
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w) (b : Vec c) (cot : Vec (c * h * w))
    (o : Fin c) :
    den (SHlo.depthwiseBiasGrad W x b (.operand cotN cot)) o
      = ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b o j * cot j :=
  (depthwise_bias_grad_has_vjp W x).correct b cot o

-- ════════════════════════════════════════════════════════════════
-- § The 2×2/s2 downsamples and the 4×4/s4 patchify stem
--   ⚠ SYMMETRIC padding — `flatConvStride2`, not the XLA-`SAME` twin B0/MobileNetV2 use.
-- ════════════════════════════════════════════════════════════════

/-- **Strided conv weight GRADIENT denotes the certified weight gradient.** The three 2×2/s2
    downsamples. Kernel-generic: `sWGradGeom`'s odd/even split is what makes the even kernel a
    call site of the same certificate rather than a hand-written emit. -/
theorem convStridedWGrad_den {ic oc h w kH kW : Nat} (xN cotN : String)
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (oc * h * w)) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedWeightGrad xN b x W (.operand cotN cot)) idx
      = ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) => flatConvStride2 (Kernel4.unflatten v') b x)
               (Kernel4.flatten W) idx j * cot j :=
  (flatConvStride2_weight_grad_has_vjp b x).correct (Kernel4.flatten W) cot idx

/-- **Strided conv bias GRADIENT denotes the certified bias gradient.** -/
theorem convStridedBGrad_den {ic oc h w kH kW : Nat} (cotN : String)
    (W : Kernel4 oc ic kH kW) (x : Vec (ic * (2 * h) * (2 * w))) (b : Vec oc)
    (cot : Vec (oc * h * w)) (o : Fin oc) :
    den (SHlo.convStridedBiasGrad W x b (.operand cotN cot)) o
      = ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * cot j :=
  (flatConvStride2_bias_grad_has_vjp W x).correct b cot o

/-- **Patchify-stem weight GRADIENT denotes the certified weight gradient.** ⭐ The one op in this
    net that was ALREADY un-fused: `convStride4WeightGrad` has no `*Sgd` peer, because the SGD
    render wraps it in hand-written text. What 4b changes is that this shape is no longer the
    exception. -/
theorem psWGrad_den {ic oc h w kH kW : Nat} (xN cotN : String)
    (b : Vec oc) (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (oc * h * w)) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStride4WeightGrad xN b x W (.operand cotN cot)) idx
      = ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) => flatConvStride4 (Kernel4.unflatten v') b x)
               (Kernel4.flatten W) idx j * cot j :=
  (flatConvStride4_weight_grad_has_vjp b x).correct (Kernel4.flatten W) cot idx

-- ════════════════════════════════════════════════════════════════
-- § The 22 spatial LayerNorm sites — the CHANNEL-LN form the render actually emits
--   The op operands are the `[h·w, c]` transposed views the render re-emits; the certified
--   Jacobian is `chanLNTensor3`'s in the `c·h·w` activation layout, and `ConvNeXtChannelLN`'s
--   permutation argument is what lets one op serve both.
-- ════════════════════════════════════════════════════════════════

/-- **Channel-LN γ GRADIENT denotes the certified γ gradient.** All 22 spatial sites (1 stem +
    18 block + 3 downsample). -/
theorem chanLnGammaGrad_den {c h w : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (β : Vec c) (x : Vec (c * h * w)) (γ : Vec c) (cot : Vec (c * h * w)) (k : Fin c) :
    den (SHlo.veclnGammaGrad (N := h * w) (D := c) xN epsStr ε
          (chanLNRows c h w x) (.operand cotN (chanLNRows c h w cot))) k
      = ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β x) γ k j * cot j := by
  simp only [den]
  rw [chanLN_gamma_contract ε β γ x cot k]
  exact vit_veclnGamma_grad_bridge ε β γ (Mat.unflatten (chanLNRows c h w x))
    (chanLNRows c h w cot) k

/-- **Channel-LN β GRADIENT denotes the certified β gradient.** The β gradient is the plain row
    reduce, so the render uses the same `rowDenseBiasGrad` op ViT's LN β does. -/
theorem chanLnBetaGrad_den {c h w : Nat} (cotN : String)
    (ε : ℝ) (γ : Vec c) (x : Vec (c * h * w)) (β : Vec c) (cot : Vec (c * h * w)) (k : Fin c) :
    den (SHlo.rowDenseBiasGrad (N := h * w) (c := c)
          (.operand cotN (chanLNRows c h w cot))) k
      = ∑ j : Fin (c * h * w),
          pdiv (fun β' : Vec c => chanLNTensor3 c h w ε γ β' x) β k j * cot j := by
  simp only [den]
  rw [chanLN_beta_contract ε γ β x cot k]
  exact vit_veclnBeta_grad_bridge ε γ β (Mat.unflatten (chanLNRows c h w x))
    (chanLNRows c h w cot) k

-- ════════════════════════════════════════════════════════════════
-- § The head — the post-GAP LayerNorm at one row, and the classifier
-- ════════════════════════════════════════════════════════════════

/-- **Head-LN γ GRADIENT denotes the certified γ gradient.** ViT's vector LayerNorm at `N = 1`:
    the head LN runs after GAP, on a single `[1, d]` row. -/
theorem headLnGammaGrad_den {N D : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (βv : Vec D) (x : Vec (N * D)) (γ : Vec D) (dy : Vec (N * D)) (k : Fin D) :
    den (SHlo.veclnGammaGrad xN epsStr ε x (.operand cotN dy)) k
      = ∑ o : Fin (N * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε gv βv (Mat.unflatten x r))) γ k o * dy o := by
  simp only [den]
  exact vit_veclnGamma_grad_bridge ε βv γ (Mat.unflatten x) dy k

/-- **Head-LN β GRADIENT denotes the certified β gradient** (`Σ_rows dy`). -/
theorem headLnBetaGrad_den {N D : Nat} (cotN : String)
    (ε : ℝ) (γv : Vec D) (X : Mat N D) (β : Vec D) (dy : Vec (N * D)) (k : Fin D) :
    den (SHlo.rowDenseBiasGrad (N := N) (c := D) (.operand cotN dy)) k
      = ∑ o : Fin (N * D),
          pdiv (fun bv : Vec D => Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β k o * dy o := by
  simp only [den]
  exact vit_veclnBeta_grad_bridge ε γv β X dy k

/-- **Classifier weight GRADIENT denotes the certified outer product.** The head runs on the
    single GAP+LN vector, so this is the plain `weightGrad`, not the row-lifted one. -/
theorem headWGrad_den {m n : Nat} (aN cotN : String)
    (a : Vec m) (W : Mat m n) (b : Vec n) (cot : Vec n) (i : Fin m) (j : Fin n) :
    den (SHlo.weightGrad aN a (.operand cotN cot)) (finProdFinEquiv (i, j))
      = ∑ k : Fin n,
          pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b a) (Mat.flatten W)
               (finProdFinEquiv (i, j)) k * cot k := by
  simp only [den, Mat.flatten, Equiv.symm_apply_apply]
  exact dense_weight_grad_correct W b a cot i j

/-- **Classifier bias GRADIENT denotes the certified cotangent.** -/
theorem headBGrad_den {m n : Nat} (cotN : String)
    (W : Mat m n) (a : Vec m) (b : Vec n) (cot : Vec n) (i : Fin n) :
    den (SHlo.biasGrad (.operand cotN cot)) i
      = ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' a) b i j * cot j := by
  simp only [den]
  exact dense_bias_grad_correct W b a cot i

end Proofs.CnxPoCG
