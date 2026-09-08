import LeanMlir.Proofs.Foundation.MobileNetBackChains
import LeanMlir.Proofs.Codegen.MobileNetV2RenderPC
import LeanMlir.Proofs.Architectures.DepthwiseBackCertifiedTie
import LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie

/-! # §B: the MobileNetV2 inverted-residual body backward targets the CERTIFIED VJP

`invresBodyBackPC` (`MobileNetBackChains.lean`; until 2026-09-08 a float bridge that also budgeted it) proves
**deployed-float ≈ a hand-assembled reverse-mode transcription** of the inverted-residual body. This
file closes §B for that body: the transcription IS the certified input-gradient VJP, in the SAME
**non-batched per-channel-BN** vocabulary the deployed net renders (`invresBodyPC`, `MobileNetV2RenderPC`).

The repo's `invresBody_has_vjp_at` (`MobileNetV2.lean`) is for the *global*-`bnForward` body, NOT the
deployed per-channel one — so (exactly as r34 built `rblkPC_has_vjp_at` fresh) we build the certified
per-channel body VJP `invresBodyPC_has_vjp_at` here (per-channel stage VJPs via `bnPerChannelTensor3`),
then tie. b1-free: the per-example per-channel body is the non-batched object the float reverses, no
`batchMap` reconciliation.

The forward body is `invresBodyPC = project ∘ depthwise ∘ expand`, each stage `(relu6) ∘ bnPC ∘ conv`,
so the certified VJP applies `projectBack → depthwiseBack → expandBack`. The float `invresBodyBackPC`
is the peer chain `(convFlatBack We ∘ bnBe ∘ reluMaskBack m_e) ∘ (depthwiseFlatBack Wd ∘ bnBd ∘
reluMaskBack m_d) ∘ (convFlatBack Wp ∘ bnBp)`. The tie pins the per-channel BN backs (`bnBe/bnBd/bnBp`)
to `bnPerChannelTensor3_has_vjp.backward` at the saved activations and the relu6 masks (`m_e/m_d`) to
the actual `0 < preact < 6` clamp-window signs (relu6's certified backward), and ties the two 1×1 convs
+ the depthwise via the leaf gates (`convFlatBack_eq_vjp_backward`, `depthwiseFlatBack_eq_vjp_backward`).
The conv/depthwise backwards ignore their (linear) primal, the pinned backs/masks carry the certified
saved activations, so after rewriting the three convolution leaves everything matches definitionally.
3-axiom-clean.
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
    HasVJPAt (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β ∘ flatConv W b) v := by
  have hconv_diff : Differentiable ℝ (flatConv W b : Vec (ic * h * w) → Vec (oc * h * w)) :=
    flatConv_differentiable W b
  have hbn_diff : Differentiable ℝ (bnPerChannelTensor3 oc h w ε γ β) :=
    bnPerChannelTensor3_differentiable oc h w ε hε γ β
  have step1 : HasVJPAt (bnPerChannelTensor3 oc h w ε γ β ∘ flatConv W b) v :=
    vjp_comp_at (flatConv W b) (bnPerChannelTensor3 oc h w ε γ β) v
      (hconv_diff v) (hbn_diff _)
      ((hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).toHasVJPAt v)
      ((bnPerChannelTensor3_has_vjp oc h w ε hε γ β).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ (bnPerChannelTensor3 oc h w ε γ β ∘ flatConv W b) v :=
    DifferentiableAt.comp v (hbn_diff (flatConv W b v)) (hconv_diff v)
  exact vjp_comp_at (bnPerChannelTensor3 oc h w ε γ β ∘ flatConv W b) (relu6 (oc * h * w)) v
    step1_diff (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth) step1
    (relu6_has_vjp_at (oc * h * w) _ h_smooth)

theorem convBnRelu6PC_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) k ≠ 6)) :
    DifferentiableAt ℝ (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β ∘ flatConv W b) v := by
  have hinner : DifferentiableAt ℝ (bnPerChannelTensor3 oc h w ε γ β ∘ flatConv W b) v :=
    ((bnPerChannelTensor3_differentiable oc h w ε hε γ β).comp (flatConv_differentiable W b)) v
  exact (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth).comp v hinner

/-- Depthwise stage VJP (stride-1), per-channel BN: `relu6 ∘ bnPC ∘ depthwise`. -/
noncomputable def dwBnRelu6PC_has_vjp_at {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε)
    (v : Vec (c * h * w))
    (h_smooth : ∀ k, (bnPerChannelTensor3 c h w ε γ β (depthwiseFlat W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 c h w ε γ β (depthwiseFlat W b v) k ≠ 6)) :
    HasVJPAt (relu6 (c * h * w) ∘ bnPerChannelTensor3 c h w ε γ β ∘ depthwiseFlat W b) v := by
  have hdw_diff : Differentiable ℝ (depthwiseFlat W b : Vec (c * h * w) → Vec (c * h * w)) :=
    depthwiseFlat_differentiable W b
  have hbn_diff : Differentiable ℝ (bnPerChannelTensor3 c h w ε γ β) :=
    bnPerChannelTensor3_differentiable c h w ε hε γ β
  have step1 : HasVJPAt (bnPerChannelTensor3 c h w ε γ β ∘ depthwiseFlat W b) v :=
    vjp_comp_at (depthwiseFlat W b) (bnPerChannelTensor3 c h w ε γ β) v
      (hdw_diff v) (hbn_diff _)
      ((depthwiseFlat_has_vjp W b).toHasVJPAt v)
      ((bnPerChannelTensor3_has_vjp c h w ε hε γ β).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ (bnPerChannelTensor3 c h w ε γ β ∘ depthwiseFlat W b) v :=
    DifferentiableAt.comp v (hbn_diff (depthwiseFlat W b v)) (hdw_diff v)
  exact vjp_comp_at (bnPerChannelTensor3 c h w ε γ β ∘ depthwiseFlat W b) (relu6 (c * h * w)) v
    step1_diff (relu6_differentiableAt_of_smooth (c * h * w) _ h_smooth) step1
    (relu6_has_vjp_at (c * h * w) _ h_smooth)

theorem dwBnRelu6PC_differentiableAt {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε)
    (v : Vec (c * h * w))
    (h_smooth : ∀ k, (bnPerChannelTensor3 c h w ε γ β (depthwiseFlat W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 c h w ε γ β (depthwiseFlat W b v) k ≠ 6)) :
    DifferentiableAt ℝ (relu6 (c * h * w) ∘ bnPerChannelTensor3 c h w ε γ β ∘ depthwiseFlat W b) v := by
  have hinner : DifferentiableAt ℝ (bnPerChannelTensor3 c h w ε γ β ∘ depthwiseFlat W b) v :=
    ((bnPerChannelTensor3_differentiable c h w ε hε γ β).comp (depthwiseFlat_differentiable W b)) v
  exact (relu6_differentiableAt_of_smooth (c * h * w) _ h_smooth).comp v hinner

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
    `bnPerChannelTensor3`. The same-vocabulary certified target for the float-bridge `invresBodyBackPC`
    — no batched/`batchMap` reconciliation. -/
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
-- § The §B tie (stride-1 body)
-- ════════════════════════════════════════════════════════════════

/-- **The §B mnv2 body tie: float-bridge backward = certified VJP.** `invresBodyBackPC`, with its
    abstract per-channel BN backs pinned to `bnPerChannelTensor3_has_vjp.backward` at the saved
    activations and its relu6 masks pinned to the actual `0 < preact < 6` clamp-window signs (relu6's
    certified backward), equals `(invresBodyPC_has_vjp_at …).backward`. The two 1×1 convs tie via
    `convFlatBack_eq_vjp_backward` (1×1 odd) and the depthwise via `depthwiseFlatBack_eq_vjp_backward`;
    conv/depthwise backwards ignore their (linear) primal, so after rewriting the three leaves
    everything matches definitionally. Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem invresBodyBackPC_eq_invresBodyPC_vjp {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (hkHe : 2 * ((kHe - 1) / 2) + 1 = kHe) (hkWe : 2 * ((kWe - 1) / 2) + 1 = kWe)
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (hkHp : 2 * ((kHp - 1) / 2) + 1 = kHp) (hkWp : 2 * ((kWp - 1) / 2) + 1 = kWp)
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
    invresBodyBackPC We Wd Wp
      ((bnPerChannelTensor3_has_vjp mid h w εe hεe γe βe).backward (flatConv We be v))
      ((bnPerChannelTensor3_has_vjp mid h w εd hεd γd βd).backward
        (depthwiseFlat Wd bd (ivExpandPC (h := h) (w := w) We be εe γe βe v)))
      ((bnPerChannelTensor3_has_vjp oc h w εp hεp γp βp).backward
        (flatConv Wp bp (ivDepthwisePC (h := h) (w := w) Wd bd εd γd βd
          (ivExpandPC (h := h) (w := w) We be εe γe βe v))))
      (fun i => 0 < bnPerChannelTensor3 mid h w εe γe βe (flatConv We be v) i ∧
                bnPerChannelTensor3 mid h w εe γe βe (flatConv We be v) i < 6)
      (fun i => 0 < bnPerChannelTensor3 mid h w εd γd βd
                  (depthwiseFlat Wd bd (ivExpandPC (h := h) (w := w) We be εe γe βe v)) i ∧
                bnPerChannelTensor3 mid h w εd γd βd
                  (depthwiseFlat Wd bd (ivExpandPC (h := h) (w := w) We be εe γe βe v)) i < 6)
      = (invresBodyPC_has_vjp_at We be εe γe βe hεe Wd bd εd γd βd hεd Wp bp εp γp βp hεp
          v h_se h_sd).backward := by
  funext dy
  unfold invresBodyBackPC
  rw [convFlatBack_eq_vjp_backward (W := Wp) (b := bp)
        (x := ivDepthwisePC (h := h) (w := w) Wd bd εd γd βd
          (ivExpandPC (h := h) (w := w) We be εe γe βe v)) hkHp hkWp,
      depthwiseFlatBack_eq_vjp_backward hkHd hkWd Wd bd
        (ivExpandPC (h := h) (w := w) We be εe γe βe v),
      convFlatBack_eq_vjp_backward (W := We) (b := be) (x := v) hkHe hkWe]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The strided (downsample) body — strided depthwise stage + tie
-- ════════════════════════════════════════════════════════════════

/-- Strided depthwise stage VJP, per-channel BN: `relu6 ∘ bnPC ∘ depthwiseStride2FlatXla`. -/
noncomputable def dwStridedBnRelu6PC_has_vjp_at {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε)
    (v : Vec (c * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 c h w ε γ β (depthwiseStride2FlatXla W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 c h w ε γ β (depthwiseStride2FlatXla W b v) k ≠ 6)) :
    HasVJPAt (relu6 (c * h * w) ∘ bnPerChannelTensor3 c h w ε γ β ∘ depthwiseStride2FlatXla W b) v := by
  have hdw_diff : Differentiable ℝ (depthwiseStride2FlatXla W b
      : Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w)) := depthwiseStride2FlatXla_differentiable W b
  have hbn_diff : Differentiable ℝ (bnPerChannelTensor3 c h w ε γ β) :=
    bnPerChannelTensor3_differentiable c h w ε hε γ β
  have step1 : HasVJPAt (bnPerChannelTensor3 c h w ε γ β ∘ depthwiseStride2FlatXla W b) v :=
    vjp_comp_at (depthwiseStride2FlatXla W b) (bnPerChannelTensor3 c h w ε γ β) v
      (hdw_diff v) (hbn_diff _)
      ((depthwiseStride2FlatXla_has_vjp W b).toHasVJPAt v)
      ((bnPerChannelTensor3_has_vjp c h w ε hε γ β).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ (bnPerChannelTensor3 c h w ε γ β ∘ depthwiseStride2FlatXla W b) v :=
    DifferentiableAt.comp v (hbn_diff (depthwiseStride2FlatXla W b v)) (hdw_diff v)
  exact vjp_comp_at (bnPerChannelTensor3 c h w ε γ β ∘ depthwiseStride2FlatXla W b) (relu6 (c * h * w)) v
    step1_diff (relu6_differentiableAt_of_smooth (c * h * w) _ h_smooth) step1
    (relu6_has_vjp_at (c * h * w) _ h_smooth)

theorem dwStridedBnRelu6PC_differentiableAt {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε)
    (v : Vec (c * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 c h w ε γ β (depthwiseStride2FlatXla W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 c h w ε γ β (depthwiseStride2FlatXla W b v) k ≠ 6)) :
    DifferentiableAt ℝ
      (relu6 (c * h * w) ∘ bnPerChannelTensor3 c h w ε γ β ∘ depthwiseStride2FlatXla W b) v := by
  have hinner : DifferentiableAt ℝ (bnPerChannelTensor3 c h w ε γ β ∘ depthwiseStride2FlatXla W b) v :=
    ((bnPerChannelTensor3_differentiable c h w ε hε γ β).comp
      (depthwiseStride2FlatXla_differentiable W b)) v
  exact (relu6_differentiableAt_of_smooth (c * h * w) _ h_smooth).comp v hinner

/-- **Certified VJP of the per-channel-BN strided inverted-residual body `invresBodyStridedPC`**
    (downsample, non-batched). `project ∘ depthwiseStrided ∘ expand(2h×2w)` — the strided twin of
    `invresBodyPC_has_vjp_at`; the same-vocabulary certified target for `invresBodyStridedBackPC`. -/
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

/-- **The §B mnv2 strided body tie: float-bridge backward = certified VJP.** The downsample peer of
    `invresBodyBackPC_eq_invresBodyPC_vjp`: `invresBodyStridedBackPC` with its per-channel BN backs and
    relu6 masks pinned to the saved activations equals `(invresBodyStridedPC_has_vjp_at …).backward`.
    The strided depthwise ties via `depthwiseStride2FlatXlaBack_eq_vjp_backward`; the expand conv at the
    `2h×2w` grid and the project conv via `convFlatBack_eq_vjp_backward`. 3-axiom-clean. -/
theorem invresBodyStridedBackPC_eq_invresBodyStridedPC_vjp {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (hkHe : 2 * ((kHe - 1) / 2) + 1 = kHe) (hkWe : 2 * ((kWe - 1) / 2) + 1 = kWe)
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (hkHp : 2 * ((kHp - 1) / 2) + 1 = kHp) (hkWp : 2 * ((kWp - 1) / 2) + 1 = kWp)
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
    invresBodyStridedBackPC We Wd Wp
      ((bnPerChannelTensor3_has_vjp mid (2 * h) (2 * w) εe hεe γe βe).backward (flatConv We be v))
      ((bnPerChannelTensor3_has_vjp mid h w εd hεd γd βd).backward
        (depthwiseStride2FlatXla Wd bd (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v)))
      ((bnPerChannelTensor3_has_vjp oc h w εp hεp γp βp).backward
        (flatConv Wp bp (ivDepthwiseStridedPC (h := h) (w := w) Wd bd εd γd βd
          (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v))))
      (fun i => 0 < bnPerChannelTensor3 mid (2 * h) (2 * w) εe γe βe (flatConv We be v) i ∧
                bnPerChannelTensor3 mid (2 * h) (2 * w) εe γe βe (flatConv We be v) i < 6)
      (fun i => 0 < bnPerChannelTensor3 mid h w εd γd βd
                  (depthwiseStride2FlatXla Wd bd
                    (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v)) i ∧
                bnPerChannelTensor3 mid h w εd γd βd
                  (depthwiseStride2FlatXla Wd bd
                    (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v)) i < 6)
      = (invresBodyStridedPC_has_vjp_at We be εe γe βe hεe Wd bd εd γd βd hεd Wp bp εp γp βp hεp
          v h_se h_sd).backward := by
  funext dy
  unfold invresBodyStridedBackPC
  rw [convFlatBack_eq_vjp_backward (W := Wp) (b := bp)
        (x := ivDepthwiseStridedPC (h := h) (w := w) Wd bd εd γd βd
          (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v)) hkHp hkWp,
      depthwiseStride2FlatXlaBack_eq_vjp_backward hkHd hkWd Wd bd
        (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v),
      convFlatBack_eq_vjp_backward (W := We) (b := be) (x := v) hkHe hkWe]
  rfl

end Proofs
