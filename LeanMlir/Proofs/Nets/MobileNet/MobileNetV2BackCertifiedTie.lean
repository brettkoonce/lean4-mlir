import LeanMlir.Proofs.Nets.MobileNet.MobileNetBackChains
import LeanMlir.Proofs.Codegen.MobileNetV2RenderPC
import LeanMlir.Proofs.Architectures.DepthwiseBackCertifiedTie
import LeanMlir.Proofs.Architectures.ConvBackCertifiedTie

/-! # The certified per-channel-BN MobileNetV2 inverted-residual body VJPs

The repo's `invresBody_has_vjp_at` (`MobileNetV2.lean`) is for the *global*-`bnForward` body, NOT
the per-channel one the per-example renders use — so, as the per-example r34 tier once built its
own block VJPs, this file builds the certified per-channel body VJPs `invresBodyPC_has_vjp_at` (stride-1)
and `invresBodyStridedPC_has_vjp_at` (downsample) from per-channel stage VJPs
(`bnPerChannelTensor3`). The forward body is `project ∘ depthwise ∘ expand`, each stage
`(relu6) ∘ bnPC ∘ conv`, so the VJP applies `projectBack → depthwiseBack → expandBack`.
`MobileNetV2FullVJP` composes them into the per-example whole-net VJP. 3-axiom-clean.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Per-channel-BN stage VJPs (the b1-free vocabulary; mirror the global ones)
-- ════════════════════════════════════════════════════════════════

/-- Expand / stem stage VJP, per-channel BN: `relu6 ∘ bnPC ∘ conv`. Mirror of `convBnRelu6_has_vjp_at`
    with `bnPerChannelTensor3` for `bnForward`. -/
noncomputable def convBnRelu6PC_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) k ≠ 6)) :
    HasVJPAt (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β ∘ flatConv W b) v :=
  stage_has_vjp_at (flatConv W b) (bnPerChannelTensor3 oc h w ε γ β) (relu6 (oc * h * w)) v
    (flatConv_differentiable W b) (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b))
    (bnPerChannelTensor3_differentiable oc h w ε hε γ β)
    (bnPerChannelTensor3_has_vjp oc h w ε hε γ β)
    (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth)
    (relu6_has_vjp_at (oc * h * w) _ h_smooth)

theorem convBnRelu6PC_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) k ≠ 6)) :
    DifferentiableAt ℝ (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β ∘ flatConv W b) v := by
  fun_prop (disch := assumption)

/-- Depthwise stage VJP (stride-1), per-channel BN: `relu6 ∘ bnPC ∘ depthwise`. -/
noncomputable def dwBnRelu6PC_has_vjp_at {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε)
    (v : Vec (c * h * w))
    (h_smooth : ∀ k, (bnPerChannelTensor3 c h w ε γ β (depthwiseFlat W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 c h w ε γ β (depthwiseFlat W b v) k ≠ 6)) :
    HasVJPAt (relu6 (c * h * w) ∘ bnPerChannelTensor3 c h w ε γ β ∘ depthwiseFlat W b) v :=
  stage_has_vjp_at (depthwiseFlat W b) (bnPerChannelTensor3 c h w ε γ β) (relu6 (c * h * w)) v
    (depthwiseFlat_differentiable W b) (depthwiseFlat_has_vjp W b)
    (bnPerChannelTensor3_differentiable c h w ε hε γ β) (bnPerChannelTensor3_has_vjp c h w ε hε γ β)
    (relu6_differentiableAt_of_smooth (c * h * w) _ h_smooth)
    (relu6_has_vjp_at (c * h * w) _ h_smooth)

theorem dwBnRelu6PC_differentiableAt {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε)
    (v : Vec (c * h * w))
    (h_smooth : ∀ k, (bnPerChannelTensor3 c h w ε γ β (depthwiseFlat W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 c h w ε γ β (depthwiseFlat W b v) k ≠ 6)) :
    DifferentiableAt ℝ (relu6 (c * h * w) ∘ bnPerChannelTensor3 c h w ε γ β ∘ depthwiseFlat W b) v := by
  fun_prop (disch := assumption)

/-- Project (linear bottleneck) stage VJP, per-channel BN: `bnPC ∘ conv` (no relu6, global `HasVJP`). -/
noncomputable def convBnPC'_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε) :
    HasVJP (bnPerChannelTensor3 oc h w ε γ β ∘ flatConv W b
      : Vec (ic * h * w) → Vec (oc * h * w)) :=
  vjp_comp (flatConv W b) (bnPerChannelTensor3 oc h w ε γ β)
    (flatConv_differentiable W b) (bnPerChannelTensor3_differentiable oc h w ε hε γ β)
    (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)) (bnPerChannelTensor3_has_vjp oc h w ε hε γ β)

theorem convBnPC'_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε) :
    Differentiable ℝ (bnPerChannelTensor3 oc h w ε γ β ∘ flatConv W b
      : Vec (ic * h * w) → Vec (oc * h * w)) :=
  (bnPerChannelTensor3_differentiable oc h w ε hε γ β).comp (flatConv_differentiable W b)

-- ════════════════════════════════════════════════════════════════
-- § The certified per-channel inverted-residual body VJP (b1-free target)
-- ════════════════════════════════════════════════════════════════

/-- **Certified VJP of the per-channel-BN inverted-residual body `invresBodyPC`** (stride-1,
    non-batched). `project ∘ depthwise ∘ expand`, mirroring the global `invresBody_has_vjp_at` with
    `bnPerChannelTensor3` — no batched/`batchMap` reconciliation. -/
noncomputable def invresBodyPC_has_vjp_at {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid) (hεe : 0 < εe)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid) (hεd : 0 < εd)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) (hεp : 0 < εp)
    (v : Vec (ic * h * w))
    (h_se : ∀ k, (bnPerChannelTensor3 mid h w εe γe βe (flatConv We be v) k ≠ 0 ∧
                   bnPerChannelTensor3 mid h w εe γe βe (flatConv We be v) k ≠ 6))
    (h_sd : ∀ k, (bnPerChannelTensor3 mid h w εd γd βd
                    (depthwiseFlat Wd bd (ivExpandPC (h := h) (w := w) We be εe γe βe v)) k ≠ 0 ∧
                   bnPerChannelTensor3 mid h w εd γd βd
                    (depthwiseFlat Wd bd (ivExpandPC (h := h) (w := w) We be εe γe βe v)) k ≠ 6)) :
    HasVJPAt (invresBodyPC (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v := by
  have hexp_vjp : HasVJPAt (ivExpandPC (h := h) (w := w) We be εe γe βe) v :=
    convBnRelu6PC_has_vjp_at We be εe γe βe hεe v h_se
  have hexp_diff : DifferentiableAt ℝ (ivExpandPC (h := h) (w := w) We be εe γe βe) v :=
    convBnRelu6PC_differentiableAt We be εe γe βe hεe v h_se
  have hdw_vjp : HasVJPAt (ivDepthwisePC (h := h) (w := w) Wd bd εd γd βd)
      (ivExpandPC (h := h) (w := w) We be εe γe βe v) :=
    dwBnRelu6PC_has_vjp_at Wd bd εd γd βd hεd _ h_sd
  have hdw_diff : DifferentiableAt ℝ (ivDepthwisePC (h := h) (w := w) Wd bd εd γd βd)
      (ivExpandPC (h := h) (w := w) We be εe γe βe v) :=
    dwBnRelu6PC_differentiableAt Wd bd εd γd βd hεd _ h_sd
  have hde_vjp : HasVJPAt
      (ivDepthwisePC (h := h) (w := w) Wd bd εd γd βd ∘
        ivExpandPC (h := h) (w := w) We be εe γe βe) v :=
    vjp_comp_at _ _ v hexp_diff hdw_diff hexp_vjp hdw_vjp
  have hde_diff : DifferentiableAt ℝ
      (ivDepthwisePC (h := h) (w := w) Wd bd εd γd βd ∘
        ivExpandPC (h := h) (w := w) We be εe γe βe) v :=
    hdw_diff.comp v hexp_diff
  exact vjp_comp_at _ (ivProjectPC (h := h) (w := w) Wp bp εp γp βp) v
    hde_diff ((convBnPC'_differentiable Wp bp εp γp βp hεp) _) hde_vjp
    ((convBnPC'_has_vjp Wp bp εp γp βp hεp).toHasVJPAt _)

-- ════════════════════════════════════════════════════════════════
-- § The strided (downsample) body — strided depthwise stage + body VJP
-- ════════════════════════════════════════════════════════════════

/-- Strided depthwise stage VJP, per-channel BN: `relu6 ∘ bnPC ∘ depthwiseStride2FlatXla`. -/
noncomputable def dwStridedBnRelu6PC_has_vjp_at {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε)
    (v : Vec (c * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 c h w ε γ β (depthwiseStride2FlatXla W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 c h w ε γ β (depthwiseStride2FlatXla W b v) k ≠ 6)) :
    HasVJPAt (relu6 (c * h * w) ∘ bnPerChannelTensor3 c h w ε γ β ∘ depthwiseStride2FlatXla W b) v :=
  stage_has_vjp_at (depthwiseStride2FlatXla W b) (bnPerChannelTensor3 c h w ε γ β)
    (relu6 (c * h * w)) v
    (depthwiseStride2FlatXla_differentiable W b) (depthwiseStride2FlatXla_has_vjp W b)
    (bnPerChannelTensor3_differentiable c h w ε hε γ β) (bnPerChannelTensor3_has_vjp c h w ε hε γ β)
    (relu6_differentiableAt_of_smooth (c * h * w) _ h_smooth)
    (relu6_has_vjp_at (c * h * w) _ h_smooth)

theorem dwStridedBnRelu6PC_differentiableAt {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε)
    (v : Vec (c * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 c h w ε γ β (depthwiseStride2FlatXla W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 c h w ε γ β (depthwiseStride2FlatXla W b v) k ≠ 6)) :
    DifferentiableAt ℝ
      (relu6 (c * h * w) ∘ bnPerChannelTensor3 c h w ε γ β ∘ depthwiseStride2FlatXla W b) v := by
  fun_prop (disch := assumption)

/-- **Certified VJP of the per-channel-BN strided inverted-residual body `invresBodyStridedPC`**
    (downsample, non-batched). `project ∘ depthwiseStrided ∘ expand(2h×2w)` — the strided twin of
    `invresBodyPC_has_vjp_at`. -/
noncomputable def invresBodyStridedPC_has_vjp_at {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid) (hεe : 0 < εe)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid) (hεd : 0 < εd)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_se : ∀ k, (bnPerChannelTensor3 mid (2 * h) (2 * w) εe γe βe (flatConv We be v) k ≠ 0 ∧
                   bnPerChannelTensor3 mid (2 * h) (2 * w) εe γe βe (flatConv We be v) k ≠ 6))
    (h_sd : ∀ k, (bnPerChannelTensor3 mid h w εd γd βd
                    (depthwiseStride2FlatXla Wd bd
                      (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v)) k ≠ 0 ∧
                   bnPerChannelTensor3 mid h w εd γd βd
                    (depthwiseStride2FlatXla Wd bd
                      (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v)) k ≠ 6)) :
    HasVJPAt (invresBodyStridedPC (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v := by
  have hexp_vjp : HasVJPAt (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe) v :=
    convBnRelu6PC_has_vjp_at We be εe γe βe hεe v h_se
  have hexp_diff : DifferentiableAt ℝ (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe) v :=
    convBnRelu6PC_differentiableAt We be εe γe βe hεe v h_se
  have hdw_vjp : HasVJPAt (ivDepthwiseStridedPC (h := h) (w := w) Wd bd εd γd βd)
      (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v) :=
    dwStridedBnRelu6PC_has_vjp_at Wd bd εd γd βd hεd _ h_sd
  have hdw_diff : DifferentiableAt ℝ (ivDepthwiseStridedPC (h := h) (w := w) Wd bd εd γd βd)
      (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v) :=
    dwStridedBnRelu6PC_differentiableAt Wd bd εd γd βd hεd _ h_sd
  have hde_vjp : HasVJPAt
      (ivDepthwiseStridedPC (h := h) (w := w) Wd bd εd γd βd ∘
        ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe) v :=
    vjp_comp_at _ _ v hexp_diff hdw_diff hexp_vjp hdw_vjp
  have hde_diff : DifferentiableAt ℝ
      (ivDepthwiseStridedPC (h := h) (w := w) Wd bd εd γd βd ∘
        ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe) v :=
    hdw_diff.comp v hexp_diff
  exact vjp_comp_at _ (ivProjectPC (h := h) (w := w) Wp bp εp γp βp) v
    hde_diff ((convBnPC'_differentiable Wp bp εp γp βp hεp) _) hde_vjp
    ((convBnPC'_has_vjp Wp bp εp γp βp hεp).toHasVJPAt _)

end Proofs
