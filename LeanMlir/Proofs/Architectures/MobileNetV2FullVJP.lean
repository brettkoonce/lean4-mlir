import LeanMlir.Proofs.Architectures.MobileNetV2FullPaper
import LeanMlir.Proofs.Architectures.MobileNetV2BackCertifiedTie

/-! # The PAPER-SPEC MobileNetV2 input-VJP — the fold at all seventeen bottlenecks

Closes `MobileNetV2FullPaper.lean`'s standing TODO: `Proofs.mobilenetv2_has_vjp_at`
(`MobileNetV2.lean`) folds a stem, TWO inverted-residual blocks and a head; this file folds
the whole `[t,c,n,s]` table — stem + 17 bottlenecks + head — over the packaged `IVW`/`IVWNoExp`
weight bundles of `MobileNetV2FullPaper.lean`.

⚠ **The statement is the POINTWISE `_at` form, and that is not a limitation to be lifted.**
MobileNetV2's activation is relu6, which is kinked; a *global* `HasVJP` through a kink is false.
`EfficientNetFullB0.lean`'s `efficientnetForwardB_full_has_vjp` may be global only because
EfficientNet's swish is smooth everywhere. The axis this file moves is DEPTH (2 → 17); the
`_at` form is the repo standard for relu-family nets and stays.

## What this file assembles (no new mathematics)

The per-block bodies already had certified VJPs — `invresBodyPC_has_vjp_at` and
`invresBodyStridedPC_has_vjp_at` (`MobileNetV2BackCertifiedTie.lean`), built there as the §B
tie targets. So the eight `iv*W_{has_vjp_at, differentiableAt}` bundle lemmas below are
delegations in the `EfficientNetFullB0` style, not fresh six-operation compositions. Only two
per-stage pieces were genuinely missing and are supplied here: the differentiability peers of
those two body VJPs, and the per-channel STRIDED stem stage `convBnRelu6StridedPC_*`
(`MobileNetV2.lean`'s `convBnRelu6Strided_*` is the global-`bnForward` twin, not the
per-channel one the paper-spec net renders).

## The hypothesis budget, and why the prefix defs exist

Seventeen blocks carry 33 relu6 sites plus the stem's and the head's, each of which must be
stated AT the running activation. Spelled inline that is unreadable by block 5 and quadratic
in the writing. So the running activations are named — `mnv2StemW`, then `mnv2Pre1 … mnv2Pre17`,
each one `∘` deeper — and the kink conditions are bundled per block into `IVSmoothAt` /
`IVStridedSmoothAt` / `IVNoExpSmoothAt`. The BN-epsilon positivity is bundled the same way
(`IVPos`, `IVNoExpPos`). The theorem then binds 19 smoothness hypotheses and 19 positivity
bundles rather than 35 and 52 loose ones.

The prefix defs double as the chain: `mnv2Pre17` IS the 17-block trunk, so the VJP is stated
on `mnv2HeadW w ∘ mnv2Pre17 w` and `mobilenetv2ForwardPaper_eq_chain` bridges that back to the
nested-application forward, exactly as `efficientnetForwardB_full_eq_chain` does for B0.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The two missing per-stage pieces
-- ════════════════════════════════════════════════════════════════

/-- Strided stem stage VJP, per-channel BN: `relu6 ∘ bnPC ∘ flatConvStride2`. The per-channel
    twin of `MobileNetV2.lean`'s `convBnRelu6Strided_has_vjp_at`, in the `bnPerChannelTensor3`
    vocabulary the paper-spec net renders. -/
noncomputable def convBnRelu6StridedPC_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 6)) :
    HasVJPAt (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v := by
  have hc_diff : Differentiable ℝ
      (flatConvStride2 W b : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
    flatConvStride2_differentiable W b
  have hbn_diff : Differentiable ℝ (bnPerChannelTensor3 oc h w ε γ β) :=
    bnPerChannelTensor3_differentiable oc h w ε hε γ β
  have step1 : HasVJPAt (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v :=
    vjp_comp_at (flatConvStride2 W b) (bnPerChannelTensor3 oc h w ε γ β) v
      (hc_diff v) (hbn_diff _)
      ((flatConvStride2_has_vjp W b).toHasVJPAt v)
      ((bnPerChannelTensor3_has_vjp oc h w ε hε γ β).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ
      (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v :=
    DifferentiableAt.comp v (hbn_diff (flatConvStride2 W b v)) (hc_diff v)
  exact vjp_comp_at _ (relu6 (oc * h * w)) v
    step1_diff (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth) step1
    (relu6_has_vjp_at (oc * h * w) _ h_smooth)

theorem convBnRelu6StridedPC_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 6)) :
    DifferentiableAt ℝ
      (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v := by
  have hinner : DifferentiableAt ℝ
      (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v :=
    ((bnPerChannelTensor3_differentiable oc h w ε hε γ β).comp
      (flatConvStride2_differentiable W b)) v
  exact (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth).comp v hinner

/-- Differentiability peer of `invresBodyPC_has_vjp_at` (which `MobileNetV2BackCertifiedTie`
    did not need, having no `residual` wrapper to feed). -/
theorem invresBodyPC_differentiableAt {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
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
    DifferentiableAt ℝ
      (invresBodyPC (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v := by
  have hexp_diff := convBnRelu6PC_differentiableAt We be εe γe βe hεe v h_se
  have hdw_diff := dwBnRelu6PC_differentiableAt Wd bd εd γd βd hεd _ h_sd
  exact ((convBnPC'_differentiable Wp bp εp γp βp hεp) _).comp v (hdw_diff.comp v hexp_diff)

/-- Differentiability peer of `invresBodyStridedPC_has_vjp_at`. -/
theorem invresBodyStridedPC_differentiableAt {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid) (hεe : 0 < εe)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid) (hεd : 0 < εd)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_se : ∀ k, (bnPerChannelTensor3 mid (2 * h) (2 * w) εe γe βe (flatConv We be v) k ≠ 0 ∧
                   bnPerChannelTensor3 mid (2 * h) (2 * w) εe γe βe (flatConv We be v) k ≠ 6))
    (h_sd : ∀ k, (bnPerChannelTensor3 mid h w εd γd βd
                    (depthwiseStride2Flat Wd bd
                      (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v)) k ≠ 0 ∧
                   bnPerChannelTensor3 mid h w εd γd βd
                    (depthwiseStride2Flat Wd bd
                      (ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe v)) k ≠ 6)) :
    DifferentiableAt ℝ
      (invresBodyStridedPC (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v := by
  have hexp_diff := convBnRelu6PC_differentiableAt We be εe γe βe hεe v h_se
  have hdw_diff := dwStridedBnRelu6PC_differentiableAt Wd bd εd γd βd hεd _ h_sd
  exact ((convBnPC'_differentiable Wp bp εp γp βp hεp) _).comp v (hdw_diff.comp v hexp_diff)

-- ════════════════════════════════════════════════════════════════
-- § Per-block hypothesis bundles (BN-epsilon positivity + the relu6 kink conditions)
-- ════════════════════════════════════════════════════════════════

/-- The three BN epsilons of a full bottleneck are positive. -/
structure IVPos {ic mid oc : Nat} (q : IVW ic mid oc) : Prop where
  he : 0 < q.eε
  hd : 0 < q.dε
  hp : 0 < q.pε

/-- The two BN epsilons of the t=1 (no-expand) bottleneck are positive. -/
structure IVNoExpPos {ic oc : Nat} (q : IVWNoExp ic oc) : Prop where
  hd : 0 < q.dε
  hp : 0 < q.pε

/-- Both relu6 sites of a stride-1 bottleneck are away from the kink at `v`: the expand-BN
    output and the depthwise-BN output each avoid `0` and `6` in every coordinate. -/
structure IVSmoothAt (h w : Nat) {ic mid oc : Nat} (q : IVW ic mid oc)
    (v : Vec (ic * h * w)) : Prop where
  he : ∀ k, (bnPerChannelTensor3 mid h w q.eε q.eγ q.eβ (flatConv q.eW q.eb v) k ≠ 0 ∧
              bnPerChannelTensor3 mid h w q.eε q.eγ q.eβ (flatConv q.eW q.eb v) k ≠ 6)
  hd : ∀ k, (bnPerChannelTensor3 mid h w q.dε q.dγ q.dβ
               (depthwiseFlat q.dW q.db
                 (ivExpandPC (h := h) (w := w) q.eW q.eb q.eε q.eγ q.eβ v)) k ≠ 0 ∧
              bnPerChannelTensor3 mid h w q.dε q.dγ q.dβ
               (depthwiseFlat q.dW q.db
                 (ivExpandPC (h := h) (w := w) q.eW q.eb q.eε q.eγ q.eβ v)) k ≠ 6)

/-- Both relu6 sites of a stride-2 bottleneck are away from the kink at `v` (the expand runs at
    the pre-downsample `2h×2w` grid, the depthwise at `h×w`). -/
structure IVStridedSmoothAt (h w : Nat) {ic mid oc : Nat} (q : IVW ic mid oc)
    (v : Vec (ic * (2 * h) * (2 * w))) : Prop where
  he : ∀ k, (bnPerChannelTensor3 mid (2 * h) (2 * w) q.eε q.eγ q.eβ (flatConv q.eW q.eb v) k ≠ 0 ∧
              bnPerChannelTensor3 mid (2 * h) (2 * w) q.eε q.eγ q.eβ (flatConv q.eW q.eb v) k ≠ 6)
  hd : ∀ k, (bnPerChannelTensor3 mid h w q.dε q.dγ q.dβ
               (depthwiseStride2Flat q.dW q.db
                 (ivExpandPC (h := 2 * h) (w := 2 * w) q.eW q.eb q.eε q.eγ q.eβ v)) k ≠ 0 ∧
              bnPerChannelTensor3 mid h w q.dε q.dγ q.dβ
               (depthwiseStride2Flat q.dW q.db
                 (ivExpandPC (h := 2 * h) (w := 2 * w) q.eW q.eb q.eε q.eγ q.eβ v)) k ≠ 6)

/-- The single relu6 site of the t=1 bottleneck (depthwise-BN output) is away from the kink. -/
structure IVNoExpSmoothAt (h w : Nat) {ic oc : Nat} (q : IVWNoExp ic oc)
    (v : Vec (ic * h * w)) : Prop where
  hd : ∀ k, (bnPerChannelTensor3 ic h w q.dε q.dγ q.dβ (depthwiseFlat q.dW q.db v) k ≠ 0 ∧
              bnPerChannelTensor3 ic h w q.dε q.dγ q.dβ (depthwiseFlat q.dW q.db v) k ≠ 6)

-- ════════════════════════════════════════════════════════════════
-- § The eight bundle lemmas (delegation, the `EfficientNetFullB0` shape)
-- ════════════════════════════════════════════════════════════════

/-- t=1 bottleneck VJP: `ivProjectPC ∘ ivDepthwisePC`. The one block shape with no
    `invresBody*PC` body lemma to delegate to, so it composes its two stages here. -/
noncomputable def ivNoExpW_has_vjp_at (h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (hq : IVNoExpPos p) (v : Vec (ic * h * w)) (hs : IVNoExpSmoothAt h w p v) :
    HasVJPAt (ivNoExpW h w p) v := by
  unfold ivNoExpW
  have hdw_vjp := dwBnRelu6PC_has_vjp_at p.dW p.db p.dε p.dγ p.dβ hq.hd v hs.hd
  have hdw_diff := dwBnRelu6PC_differentiableAt p.dW p.db p.dε p.dγ p.dβ hq.hd v hs.hd
  exact vjp_comp_at _ (ivProjectPC (h := h) (w := w) p.pW p.pb p.pε p.pγ p.pβ) v
    hdw_diff ((convBnPC'_differentiable p.pW p.pb p.pε p.pγ p.pβ hq.hp) _) hdw_vjp
    ((convBnPC'_has_vjp p.pW p.pb p.pε p.pγ p.pβ hq.hp).toHasVJPAt _)

theorem ivNoExpW_differentiableAt (h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (hq : IVNoExpPos p) (v : Vec (ic * h * w)) (hs : IVNoExpSmoothAt h w p v) :
    DifferentiableAt ℝ (ivNoExpW h w p) v := by
  unfold ivNoExpW
  exact ((convBnPC'_differentiable p.pW p.pb p.pε p.pγ p.pβ hq.hp) _).comp v
    (dwBnRelu6PC_differentiableAt p.dW p.db p.dε p.dγ p.dβ hq.hd v hs.hd)

/-- Stride-1 no-skip bottleneck VJP — `invresBodyPC_has_vjp_at` at the bundle's fields. -/
noncomputable def ivExpOnlyW_has_vjp_at (h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (hq : IVPos p) (v : Vec (ic * h * w)) (hs : IVSmoothAt h w p v) :
    HasVJPAt (ivExpOnlyW h w p) v := by
  unfold ivExpOnlyW
  exact invresBodyPC_has_vjp_at p.eW p.eb p.eε p.eγ p.eβ hq.he p.dW p.db p.dε p.dγ p.dβ hq.hd
    p.pW p.pb p.pε p.pγ p.pβ hq.hp v hs.he hs.hd

theorem ivExpOnlyW_differentiableAt (h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (hq : IVPos p) (v : Vec (ic * h * w)) (hs : IVSmoothAt h w p v) :
    DifferentiableAt ℝ (ivExpOnlyW h w p) v := by
  unfold ivExpOnlyW
  exact invresBodyPC_differentiableAt p.eW p.eb p.eε p.eγ p.eβ hq.he p.dW p.db p.dε p.dγ p.dβ hq.hd
    p.pW p.pb p.pε p.pγ p.pβ hq.hp v hs.he hs.hd

/-- Stride-1 skip bottleneck VJP — the body VJP under `residual_has_vjp_at`. The identity arm is
    smooth everywhere, so it adds no hypothesis. -/
noncomputable def ivResidW_has_vjp_at (h w : Nat) {c mid : Nat} (p : IVW c mid c)
    (hq : IVPos p) (v : Vec (c * h * w)) (hs : IVSmoothAt h w p v) :
    HasVJPAt (ivResidW h w p) v := by
  unfold ivResidW
  exact residual_has_vjp_at _ v
    (invresBodyPC_differentiableAt p.eW p.eb p.eε p.eγ p.eβ hq.he p.dW p.db p.dε p.dγ p.dβ hq.hd
      p.pW p.pb p.pε p.pγ p.pβ hq.hp v hs.he hs.hd)
    (invresBodyPC_has_vjp_at p.eW p.eb p.eε p.eγ p.eβ hq.he p.dW p.db p.dε p.dγ p.dβ hq.hd
      p.pW p.pb p.pε p.pγ p.pβ hq.hp v hs.he hs.hd)

theorem ivResidW_differentiableAt (h w : Nat) {c mid : Nat} (p : IVW c mid c)
    (hq : IVPos p) (v : Vec (c * h * w)) (hs : IVSmoothAt h w p v) :
    DifferentiableAt ℝ (ivResidW h w p) v := by
  have hbody := invresBodyPC_differentiableAt p.eW p.eb p.eε p.eγ p.eβ hq.he
    p.dW p.db p.dε p.dγ p.dβ hq.hd p.pW p.pb p.pε p.pγ p.pβ hq.hp v hs.he hs.hd
  show DifferentiableAt ℝ (biPath _ (fun y => y)) v
  exact DifferentiableAt.add hbody differentiable_id.differentiableAt

/-- Stride-2 downsampling bottleneck VJP — `invresBodyStridedPC_has_vjp_at` at the bundle. -/
noncomputable def ivStridedW_has_vjp_at (h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (hq : IVPos p) (v : Vec (ic * (2 * h) * (2 * w))) (hs : IVStridedSmoothAt h w p v) :
    HasVJPAt (ivStridedW h w p) v := by
  unfold ivStridedW
  exact invresBodyStridedPC_has_vjp_at p.eW p.eb p.eε p.eγ p.eβ hq.he
    p.dW p.db p.dε p.dγ p.dβ hq.hd p.pW p.pb p.pε p.pγ p.pβ hq.hp v hs.he hs.hd

theorem ivStridedW_differentiableAt (h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (hq : IVPos p) (v : Vec (ic * (2 * h) * (2 * w))) (hs : IVStridedSmoothAt h w p v) :
    DifferentiableAt ℝ (ivStridedW h w p) v := by
  unfold ivStridedW
  exact invresBodyStridedPC_differentiableAt p.eW p.eb p.eε p.eγ p.eβ hq.he
    p.dW p.db p.dε p.dγ p.dβ hq.hd p.pW p.pb p.pε p.pγ p.pβ hq.hp v hs.he hs.hd

-- ════════════════════════════════════════════════════════════════
-- § Stem and head wrappers
-- ════════════════════════════════════════════════════════════════

/-- The paper-spec stem: 3×3 stride-2 conv 3→32 at 224² → per-channel BN → relu6. -/
noncomputable def mnv2StemW (w : MNV2PaperWeights) :
    Vec (3 * 224 * 224) → Vec (32 * 112 * 112) :=
  relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 w.sε w.sγ w.sβ ∘
    flatConvStride2 (h := 112) (w := 112) w.sW w.sb

/-- The paper-spec head: 1×1 conv 320→1280 → per-channel BN → relu6 → GAP → dense 1280→10. -/
noncomputable def mnv2HeadW (w : MNV2PaperWeights) : Vec (320 * 7 * 7) → Vec 10 :=
  dense w.fcW w.fcb ∘ globalAvgPoolFlat 1280 7 7 ∘
    (relu6 (1280 * 7 * 7) ∘ bnPerChannelTensor3 1280 7 7 w.hε w.hγ w.hβ ∘
      flatConv (h := 7) (w := 7) w.hW w.hb)

/-- Stem relu6 away from the kink at the input `x`. -/
def MNV2StemSmoothAt (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) : Prop :=
  ∀ k, (bnPerChannelTensor3 32 112 112 w.sε w.sγ w.sβ (flatConvStride2 w.sW w.sb x) k ≠ 0 ∧
         bnPerChannelTensor3 32 112 112 w.sε w.sγ w.sβ (flatConvStride2 w.sW w.sb x) k ≠ 6)

/-- Head relu6 away from the kink at the trunk output `v`. -/
def MNV2HeadSmoothAt (w : MNV2PaperWeights) (v : Vec (320 * 7 * 7)) : Prop :=
  ∀ k, (bnPerChannelTensor3 1280 7 7 w.hε w.hγ w.hβ (flatConv w.hW w.hb v) k ≠ 0 ∧
         bnPerChannelTensor3 1280 7 7 w.hε w.hγ w.hβ (flatConv w.hW w.hb v) k ≠ 6)

noncomputable def mnv2StemW_has_vjp_at (w : MNV2PaperWeights) (hs : 0 < w.sε)
    (x : Vec (3 * 224 * 224)) (h_stem : MNV2StemSmoothAt w x) :
    HasVJPAt (mnv2StemW w) x :=
  convBnRelu6StridedPC_has_vjp_at (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ hs x h_stem

theorem mnv2StemW_differentiableAt (w : MNV2PaperWeights) (hs : 0 < w.sε)
    (x : Vec (3 * 224 * 224)) (h_stem : MNV2StemSmoothAt w x) :
    DifferentiableAt ℝ (mnv2StemW w) x :=
  convBnRelu6StridedPC_differentiableAt (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ hs x h_stem

noncomputable def mnv2HeadW_has_vjp_at (w : MNV2PaperWeights) (hh : 0 < w.hε)
    (v : Vec (320 * 7 * 7)) (h_head : MNV2HeadSmoothAt w v) :
    HasVJPAt (mnv2HeadW w) v := by
  unfold mnv2HeadW
  have c_vjp := convBnRelu6PC_has_vjp_at (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ hh v h_head
  have c_diff := convBnRelu6PC_differentiableAt (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ hh v h_head
  have g_vjp : HasVJPAt (globalAvgPoolFlat 1280 7 7 ∘
      (relu6 (1280 * 7 * 7) ∘ bnPerChannelTensor3 1280 7 7 w.hε w.hγ w.hβ ∘
        flatConv (h := 7) (w := 7) w.hW w.hb)) v :=
    vjp_comp_at _ (globalAvgPoolFlat 1280 7 7) v c_diff
      ((globalAvgPoolFlat_differentiable 1280 7 7) _) c_vjp
      ((globalAvgPoolFlat_has_vjp 1280 7 7).toHasVJPAt _)
  have g_diff := ((globalAvgPoolFlat_differentiable 1280 7 7) _).comp v c_diff
  exact vjp_comp_at _ (dense w.fcW w.fcb) v g_diff ((dense_differentiable w.fcW w.fcb) _) g_vjp
    ((dense_has_vjp w.fcW w.fcb).toHasVJPAt _)

theorem mnv2HeadW_differentiableAt (w : MNV2PaperWeights) (hh : 0 < w.hε)
    (v : Vec (320 * 7 * 7)) (h_head : MNV2HeadSmoothAt w v) :
    DifferentiableAt ℝ (mnv2HeadW w) v := by
  unfold mnv2HeadW
  have c_diff := convBnRelu6PC_differentiableAt (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ hh v h_head
  exact ((dense_differentiable w.fcW w.fcb) _).comp v
    (((globalAvgPoolFlat_differentiable 1280 7 7) _).comp v c_diff)

-- ════════════════════════════════════════════════════════════════
-- § The running activations — `mnv2PreK` = the net truncated after block `K`
--   Each is one `∘` deeper than the last; `mnv2Pre17` is the whole 17-block trunk.
--   These exist so the 19 kink hypotheses can be STATED (each holds at the activation
--   entering its block) without spelling a 17-deep nested application inline.
-- ════════════════════════════════════════════════════════════════

noncomputable def mnv2Pre1 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (16 * 112 * 112) :=
  ivNoExpW 112 112 w.b1 ∘ mnv2StemW w
noncomputable def mnv2Pre2 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (24 * 56 * 56) :=
  ivStridedW 56 56 w.b2 ∘ mnv2Pre1 w
noncomputable def mnv2Pre3 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (24 * 56 * 56) :=
  ivResidW 56 56 w.b3 ∘ mnv2Pre2 w
noncomputable def mnv2Pre4 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (32 * 28 * 28) :=
  ivStridedW 28 28 w.b4 ∘ mnv2Pre3 w
noncomputable def mnv2Pre5 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (32 * 28 * 28) :=
  ivResidW 28 28 w.b5 ∘ mnv2Pre4 w
noncomputable def mnv2Pre6 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (32 * 28 * 28) :=
  ivResidW 28 28 w.b6 ∘ mnv2Pre5 w
noncomputable def mnv2Pre7 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (64 * 14 * 14) :=
  ivStridedW 14 14 w.b7 ∘ mnv2Pre6 w
noncomputable def mnv2Pre8 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (64 * 14 * 14) :=
  ivResidW 14 14 w.b8 ∘ mnv2Pre7 w
noncomputable def mnv2Pre9 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (64 * 14 * 14) :=
  ivResidW 14 14 w.b9 ∘ mnv2Pre8 w
noncomputable def mnv2Pre10 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (64 * 14 * 14) :=
  ivResidW 14 14 w.b10 ∘ mnv2Pre9 w
noncomputable def mnv2Pre11 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (96 * 14 * 14) :=
  ivExpOnlyW 14 14 w.b11 ∘ mnv2Pre10 w
noncomputable def mnv2Pre12 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (96 * 14 * 14) :=
  ivResidW 14 14 w.b12 ∘ mnv2Pre11 w
noncomputable def mnv2Pre13 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (96 * 14 * 14) :=
  ivResidW 14 14 w.b13 ∘ mnv2Pre12 w
noncomputable def mnv2Pre14 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (160 * 7 * 7) :=
  ivStridedW 7 7 w.b14 ∘ mnv2Pre13 w
noncomputable def mnv2Pre15 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (160 * 7 * 7) :=
  ivResidW 7 7 w.b15 ∘ mnv2Pre14 w
noncomputable def mnv2Pre16 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (160 * 7 * 7) :=
  ivResidW 7 7 w.b16 ∘ mnv2Pre15 w
noncomputable def mnv2Pre17 (w : MNV2PaperWeights) : Vec (3 * 224 * 224) → Vec (320 * 7 * 7) :=
  ivExpOnlyW 7 7 w.b17 ∘ mnv2Pre16 w

-- ════════════════════════════════════════════════════════════════
-- § The full-depth fold
-- ════════════════════════════════════════════════════════════════

set_option maxHeartbeats 1000000 in
/-- **The paper-spec MobileNetV2 has a (correct) input-VJP at a smooth point — all seventeen
    bottlenecks.** Chains stem → the 17 blocks of the `[t,c,n,s]` table → head with
    `vjp_comp_at`, one `IVPos` bundle and one `*SmoothAt` bundle per block. The full-depth
    replacement for `mobilenetv2_has_vjp_at`'s two-block fold.

    ⚠ Pointwise (`HasVJPAt`), not global (`HasVJP`), and necessarily so: relu6 is kinked, so
    each of the 35 activation sites carries a `≠ 0 ∧ ≠ 6` side condition at its running
    activation. EfficientNet-B0's global fold is available to it only because swish is smooth. -/
noncomputable def mobilenetv2_full_has_vjp_at (w : MNV2PaperWeights)
    (hs : 0 < w.sε)
    (q1 : IVNoExpPos w.b1) (q2 : IVPos w.b2) (q3 : IVPos w.b3) (q4 : IVPos w.b4)
    (q5 : IVPos w.b5) (q6 : IVPos w.b6) (q7 : IVPos w.b7) (q8 : IVPos w.b8)
    (q9 : IVPos w.b9) (q10 : IVPos w.b10) (q11 : IVPos w.b11) (q12 : IVPos w.b12)
    (q13 : IVPos w.b13) (q14 : IVPos w.b14) (q15 : IVPos w.b15) (q16 : IVPos w.b16)
    (q17 : IVPos w.b17) (hh : 0 < w.hε)
    (x : Vec (3 * 224 * 224))
    (h_stem : MNV2StemSmoothAt w x)
    (s1 : IVNoExpSmoothAt 112 112 w.b1 (mnv2StemW w x))
    (s2 : IVStridedSmoothAt 56 56 w.b2 (mnv2Pre1 w x))
    (s3 : IVSmoothAt 56 56 w.b3 (mnv2Pre2 w x))
    (s4 : IVStridedSmoothAt 28 28 w.b4 (mnv2Pre3 w x))
    (s5 : IVSmoothAt 28 28 w.b5 (mnv2Pre4 w x))
    (s6 : IVSmoothAt 28 28 w.b6 (mnv2Pre5 w x))
    (s7 : IVStridedSmoothAt 14 14 w.b7 (mnv2Pre6 w x))
    (s8 : IVSmoothAt 14 14 w.b8 (mnv2Pre7 w x))
    (s9 : IVSmoothAt 14 14 w.b9 (mnv2Pre8 w x))
    (s10 : IVSmoothAt 14 14 w.b10 (mnv2Pre9 w x))
    (s11 : IVSmoothAt 14 14 w.b11 (mnv2Pre10 w x))
    (s12 : IVSmoothAt 14 14 w.b12 (mnv2Pre11 w x))
    (s13 : IVSmoothAt 14 14 w.b13 (mnv2Pre12 w x))
    (s14 : IVStridedSmoothAt 7 7 w.b14 (mnv2Pre13 w x))
    (s15 : IVSmoothAt 7 7 w.b15 (mnv2Pre14 w x))
    (s16 : IVSmoothAt 7 7 w.b16 (mnv2Pre15 w x))
    (s17 : IVSmoothAt 7 7 w.b17 (mnv2Pre16 w x))
    (h_head : MNV2HeadSmoothAt w (mnv2Pre17 w x)) :
    HasVJPAt (mnv2HeadW w ∘ mnv2Pre17 w) x := by
  -- stem
  have dS : DifferentiableAt ℝ (mnv2StemW w) x := mnv2StemW_differentiableAt w hs x h_stem
  have vS : HasVJPAt (mnv2StemW w) x := mnv2StemW_has_vjp_at w hs x h_stem
  -- the seventeen blocks, each folded onto the accumulated prefix
  have d1 := ivNoExpW_differentiableAt 112 112 w.b1 q1 _ s1
  have e1 : HasVJPAt (mnv2Pre1 w) x :=
    vjp_comp_at _ _ x dS d1 vS (ivNoExpW_has_vjp_at 112 112 w.b1 q1 _ s1)
  have f1 : DifferentiableAt ℝ (mnv2Pre1 w) x := d1.comp x dS
  have d2 := ivStridedW_differentiableAt 56 56 w.b2 q2 _ s2
  have e2 : HasVJPAt (mnv2Pre2 w) x :=
    vjp_comp_at _ _ x f1 d2 e1 (ivStridedW_has_vjp_at 56 56 w.b2 q2 _ s2)
  have f2 : DifferentiableAt ℝ (mnv2Pre2 w) x := d2.comp x f1
  have d3 := ivResidW_differentiableAt 56 56 w.b3 q3 _ s3
  have e3 : HasVJPAt (mnv2Pre3 w) x :=
    vjp_comp_at _ _ x f2 d3 e2 (ivResidW_has_vjp_at 56 56 w.b3 q3 _ s3)
  have f3 : DifferentiableAt ℝ (mnv2Pre3 w) x := d3.comp x f2
  have d4 := ivStridedW_differentiableAt 28 28 w.b4 q4 _ s4
  have e4 : HasVJPAt (mnv2Pre4 w) x :=
    vjp_comp_at _ _ x f3 d4 e3 (ivStridedW_has_vjp_at 28 28 w.b4 q4 _ s4)
  have f4 : DifferentiableAt ℝ (mnv2Pre4 w) x := d4.comp x f3
  have d5 := ivResidW_differentiableAt 28 28 w.b5 q5 _ s5
  have e5 : HasVJPAt (mnv2Pre5 w) x :=
    vjp_comp_at _ _ x f4 d5 e4 (ivResidW_has_vjp_at 28 28 w.b5 q5 _ s5)
  have f5 : DifferentiableAt ℝ (mnv2Pre5 w) x := d5.comp x f4
  have d6 := ivResidW_differentiableAt 28 28 w.b6 q6 _ s6
  have e6 : HasVJPAt (mnv2Pre6 w) x :=
    vjp_comp_at _ _ x f5 d6 e5 (ivResidW_has_vjp_at 28 28 w.b6 q6 _ s6)
  have f6 : DifferentiableAt ℝ (mnv2Pre6 w) x := d6.comp x f5
  have d7 := ivStridedW_differentiableAt 14 14 w.b7 q7 _ s7
  have e7 : HasVJPAt (mnv2Pre7 w) x :=
    vjp_comp_at _ _ x f6 d7 e6 (ivStridedW_has_vjp_at 14 14 w.b7 q7 _ s7)
  have f7 : DifferentiableAt ℝ (mnv2Pre7 w) x := d7.comp x f6
  have d8 := ivResidW_differentiableAt 14 14 w.b8 q8 _ s8
  have e8 : HasVJPAt (mnv2Pre8 w) x :=
    vjp_comp_at _ _ x f7 d8 e7 (ivResidW_has_vjp_at 14 14 w.b8 q8 _ s8)
  have f8 : DifferentiableAt ℝ (mnv2Pre8 w) x := d8.comp x f7
  have d9 := ivResidW_differentiableAt 14 14 w.b9 q9 _ s9
  have e9 : HasVJPAt (mnv2Pre9 w) x :=
    vjp_comp_at _ _ x f8 d9 e8 (ivResidW_has_vjp_at 14 14 w.b9 q9 _ s9)
  have f9 : DifferentiableAt ℝ (mnv2Pre9 w) x := d9.comp x f8
  have d10 := ivResidW_differentiableAt 14 14 w.b10 q10 _ s10
  have e10 : HasVJPAt (mnv2Pre10 w) x :=
    vjp_comp_at _ _ x f9 d10 e9 (ivResidW_has_vjp_at 14 14 w.b10 q10 _ s10)
  have f10 : DifferentiableAt ℝ (mnv2Pre10 w) x := d10.comp x f9
  have d11 := ivExpOnlyW_differentiableAt 14 14 w.b11 q11 _ s11
  have e11 : HasVJPAt (mnv2Pre11 w) x :=
    vjp_comp_at _ _ x f10 d11 e10 (ivExpOnlyW_has_vjp_at 14 14 w.b11 q11 _ s11)
  have f11 : DifferentiableAt ℝ (mnv2Pre11 w) x := d11.comp x f10
  have d12 := ivResidW_differentiableAt 14 14 w.b12 q12 _ s12
  have e12 : HasVJPAt (mnv2Pre12 w) x :=
    vjp_comp_at _ _ x f11 d12 e11 (ivResidW_has_vjp_at 14 14 w.b12 q12 _ s12)
  have f12 : DifferentiableAt ℝ (mnv2Pre12 w) x := d12.comp x f11
  have d13 := ivResidW_differentiableAt 14 14 w.b13 q13 _ s13
  have e13 : HasVJPAt (mnv2Pre13 w) x :=
    vjp_comp_at _ _ x f12 d13 e12 (ivResidW_has_vjp_at 14 14 w.b13 q13 _ s13)
  have f13 : DifferentiableAt ℝ (mnv2Pre13 w) x := d13.comp x f12
  have d14 := ivStridedW_differentiableAt 7 7 w.b14 q14 _ s14
  have e14 : HasVJPAt (mnv2Pre14 w) x :=
    vjp_comp_at _ _ x f13 d14 e13 (ivStridedW_has_vjp_at 7 7 w.b14 q14 _ s14)
  have f14 : DifferentiableAt ℝ (mnv2Pre14 w) x := d14.comp x f13
  have d15 := ivResidW_differentiableAt 7 7 w.b15 q15 _ s15
  have e15 : HasVJPAt (mnv2Pre15 w) x :=
    vjp_comp_at _ _ x f14 d15 e14 (ivResidW_has_vjp_at 7 7 w.b15 q15 _ s15)
  have f15 : DifferentiableAt ℝ (mnv2Pre15 w) x := d15.comp x f14
  have d16 := ivResidW_differentiableAt 7 7 w.b16 q16 _ s16
  have e16 : HasVJPAt (mnv2Pre16 w) x :=
    vjp_comp_at _ _ x f15 d16 e15 (ivResidW_has_vjp_at 7 7 w.b16 q16 _ s16)
  have f16 : DifferentiableAt ℝ (mnv2Pre16 w) x := d16.comp x f15
  have d17 := ivExpOnlyW_differentiableAt 7 7 w.b17 q17 _ s17
  have e17 : HasVJPAt (mnv2Pre17 w) x :=
    vjp_comp_at _ _ x f16 d17 e16 (ivExpOnlyW_has_vjp_at 7 7 w.b17 q17 _ s17)
  have f17 : DifferentiableAt ℝ (mnv2Pre17 w) x := d17.comp x f16
  -- head
  exact vjp_comp_at _ (mnv2HeadW w) x f17
    (mnv2HeadW_differentiableAt w hh _ h_head) e17
    (mnv2HeadW_has_vjp_at w hh _ h_head)

-- ════════════════════════════════════════════════════════════════
-- § One-layer peeling lemmas
--
--   ⚠ PROOF-SHAPE MATTERS, and these exist ONLY for that reason (the
--   `efficientnetForwardB_full_eq_chain` / ConvNeXt-T `convNextForwardTCh_eq_chain` lesson,
--   paid for again here). Proving `mobilenetv2ForwardPaper_eq_chain` below in ONE step fails
--   both ways: a bare `rfl` exhausts the elaborator's recursion depth, and a
--   `simp only [<the 19 defs>, Function.comp_apply]` elaborates (~140 s) into a term the
--   KERNEL then rejects with a deterministic timeout.
--
--   ⚠⚠ And the per-layer lemmas must be `rw`, not `rfl` — `rfl` is NOT uniformly safe here,
--   which is the part that is easy to get wrong because 16 of the 17 peels do not show it.
--   Peeling one block off another by `rfl` is instant (both sides have the same opaque block
--   wrapper at the head, so defeq matches structurally). Peeling block 1 off the STEM by `rfl`
--   kernel-times-out — measured, ~80 s to fail. The stem is the one layer whose body carries a
--   type ascription: `flatConvStride2 (h := 112) (w := 112)` has natural domain
--   `Vec (3 * (2 * 112) * (2 * 112))` and `mnv2StemW` declares `Vec (3 * 224 * 224)`, so the
--   kernel stops matching heads and descends into the block body instead. Going through
--   `rw [<the def>, Function.comp_apply]` never unfolds the inner layer at all: it closes on
--   syntactically identical terms, and the whole file drops from 144 s (failing) to ~2 s.
-- ════════════════════════════════════════════════════════════════

theorem mnv2StemW_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2StemW w x = relu6 (32 * 112 * 112) (bnPerChannelTensor3 32 112 112 w.sε w.sγ w.sβ
      (flatConvStride2 (h := 112) (w := 112) w.sW w.sb x)) := by
  rw [mnv2StemW, Function.comp_apply, Function.comp_apply]

theorem mnv2HeadW_apply (w : MNV2PaperWeights) (v : Vec (320 * 7 * 7)) :
    mnv2HeadW w v = dense w.fcW w.fcb (globalAvgPoolFlat 1280 7 7
      (relu6 (1280 * 7 * 7) (bnPerChannelTensor3 1280 7 7 w.hε w.hγ w.hβ
        (flatConv (h := 7) (w := 7) w.hW w.hb v)))) := by
  rw [mnv2HeadW, Function.comp_apply, Function.comp_apply, Function.comp_apply,
    Function.comp_apply]

theorem mnv2Pre1_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre1 w x = ivNoExpW 112 112 w.b1 (mnv2StemW w x) := by
  rw [mnv2Pre1, Function.comp_apply]
theorem mnv2Pre2_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre2 w x = ivStridedW 56 56 w.b2 (mnv2Pre1 w x) := by
  rw [mnv2Pre2, Function.comp_apply]
theorem mnv2Pre3_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre3 w x = ivResidW 56 56 w.b3 (mnv2Pre2 w x) := by
  rw [mnv2Pre3, Function.comp_apply]
theorem mnv2Pre4_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre4 w x = ivStridedW 28 28 w.b4 (mnv2Pre3 w x) := by
  rw [mnv2Pre4, Function.comp_apply]
theorem mnv2Pre5_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre5 w x = ivResidW 28 28 w.b5 (mnv2Pre4 w x) := by
  rw [mnv2Pre5, Function.comp_apply]
theorem mnv2Pre6_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre6 w x = ivResidW 28 28 w.b6 (mnv2Pre5 w x) := by
  rw [mnv2Pre6, Function.comp_apply]
theorem mnv2Pre7_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre7 w x = ivStridedW 14 14 w.b7 (mnv2Pre6 w x) := by
  rw [mnv2Pre7, Function.comp_apply]
theorem mnv2Pre8_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre8 w x = ivResidW 14 14 w.b8 (mnv2Pre7 w x) := by
  rw [mnv2Pre8, Function.comp_apply]
theorem mnv2Pre9_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre9 w x = ivResidW 14 14 w.b9 (mnv2Pre8 w x) := by
  rw [mnv2Pre9, Function.comp_apply]
theorem mnv2Pre10_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre10 w x = ivResidW 14 14 w.b10 (mnv2Pre9 w x) := by
  rw [mnv2Pre10, Function.comp_apply]
theorem mnv2Pre11_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre11 w x = ivExpOnlyW 14 14 w.b11 (mnv2Pre10 w x) := by
  rw [mnv2Pre11, Function.comp_apply]
theorem mnv2Pre12_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre12 w x = ivResidW 14 14 w.b12 (mnv2Pre11 w x) := by
  rw [mnv2Pre12, Function.comp_apply]
theorem mnv2Pre13_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre13 w x = ivResidW 14 14 w.b13 (mnv2Pre12 w x) := by
  rw [mnv2Pre13, Function.comp_apply]
theorem mnv2Pre14_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre14 w x = ivStridedW 7 7 w.b14 (mnv2Pre13 w x) := by
  rw [mnv2Pre14, Function.comp_apply]
theorem mnv2Pre15_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre15 w x = ivResidW 7 7 w.b15 (mnv2Pre14 w x) := by
  rw [mnv2Pre15, Function.comp_apply]
theorem mnv2Pre16_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre16 w x = ivResidW 7 7 w.b16 (mnv2Pre15 w x) := by
  rw [mnv2Pre16, Function.comp_apply]
theorem mnv2Pre17_apply (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mnv2Pre17 w x = ivExpOnlyW 7 7 w.b17 (mnv2Pre16 w x) := by
  rw [mnv2Pre17, Function.comp_apply]

/-- **`mobilenetv2ForwardPaper` = the `∘`-chain the VJP is stated on** — the kernel-checked
    bridge between the nested-application forward and the layered `mnv2PreK` form, closing the
    same form-gap `efficientnetForwardB_full_eq_chain` closes for B0. Peeled one layer at a
    time through the `*_apply` lemmas above; see their section header for why the one-step
    proofs do not survive (elaborator recursion depth / kernel deterministic timeout). -/
theorem mobilenetv2ForwardPaper_eq_chain (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mobilenetv2ForwardPaper w x = (mnv2HeadW w ∘ mnv2Pre17 w) x := by
  rw [mobilenetv2ForwardPaper, Function.comp_apply, mnv2HeadW_apply,
    mnv2Pre17_apply, mnv2Pre16_apply, mnv2Pre15_apply, mnv2Pre14_apply, mnv2Pre13_apply,
    mnv2Pre12_apply, mnv2Pre11_apply, mnv2Pre10_apply, mnv2Pre9_apply, mnv2Pre8_apply,
    mnv2Pre7_apply, mnv2Pre6_apply, mnv2Pre5_apply, mnv2Pre4_apply, mnv2Pre3_apply,
    mnv2Pre2_apply, mnv2Pre1_apply, mnv2StemW_apply]

set_option maxHeartbeats 1000000 in
/-- **Public correctness theorem for `mobilenetv2_full_has_vjp_at`** — the seventeen-block
    backward equals the `pdiv`-contracted Jacobian of `mobilenetv2ForwardPaper` ITSELF (not of
    the chain it is stated on), tied back through `mobilenetv2ForwardPaper_eq_chain`. The
    full-depth analogue of `mobilenetv2_has_vjp_at_correct`. -/
theorem mobilenetv2_full_has_vjp_at_correct (w : MNV2PaperWeights)
    (hs : 0 < w.sε)
    (q1 : IVNoExpPos w.b1) (q2 : IVPos w.b2) (q3 : IVPos w.b3) (q4 : IVPos w.b4)
    (q5 : IVPos w.b5) (q6 : IVPos w.b6) (q7 : IVPos w.b7) (q8 : IVPos w.b8)
    (q9 : IVPos w.b9) (q10 : IVPos w.b10) (q11 : IVPos w.b11) (q12 : IVPos w.b12)
    (q13 : IVPos w.b13) (q14 : IVPos w.b14) (q15 : IVPos w.b15) (q16 : IVPos w.b16)
    (q17 : IVPos w.b17) (hh : 0 < w.hε)
    (x : Vec (3 * 224 * 224))
    (h_stem : MNV2StemSmoothAt w x)
    (s1 : IVNoExpSmoothAt 112 112 w.b1 (mnv2StemW w x))
    (s2 : IVStridedSmoothAt 56 56 w.b2 (mnv2Pre1 w x))
    (s3 : IVSmoothAt 56 56 w.b3 (mnv2Pre2 w x))
    (s4 : IVStridedSmoothAt 28 28 w.b4 (mnv2Pre3 w x))
    (s5 : IVSmoothAt 28 28 w.b5 (mnv2Pre4 w x))
    (s6 : IVSmoothAt 28 28 w.b6 (mnv2Pre5 w x))
    (s7 : IVStridedSmoothAt 14 14 w.b7 (mnv2Pre6 w x))
    (s8 : IVSmoothAt 14 14 w.b8 (mnv2Pre7 w x))
    (s9 : IVSmoothAt 14 14 w.b9 (mnv2Pre8 w x))
    (s10 : IVSmoothAt 14 14 w.b10 (mnv2Pre9 w x))
    (s11 : IVSmoothAt 14 14 w.b11 (mnv2Pre10 w x))
    (s12 : IVSmoothAt 14 14 w.b12 (mnv2Pre11 w x))
    (s13 : IVSmoothAt 14 14 w.b13 (mnv2Pre12 w x))
    (s14 : IVStridedSmoothAt 7 7 w.b14 (mnv2Pre13 w x))
    (s15 : IVSmoothAt 7 7 w.b15 (mnv2Pre14 w x))
    (s16 : IVSmoothAt 7 7 w.b16 (mnv2Pre15 w x))
    (s17 : IVSmoothAt 7 7 w.b17 (mnv2Pre16 w x))
    (h_head : MNV2HeadSmoothAt w (mnv2Pre17 w x))
    (dy : Vec 10) (i : Fin (3 * 224 * 224)) :
    (mobilenetv2_full_has_vjp_at w hs q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 q12 q13 q14 q15 q16
        q17 hh x h_stem s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17
        h_head).backward dy i =
      ∑ j : Fin 10, pdiv (mobilenetv2ForwardPaper w) x i j * dy j := by
  have h := (mobilenetv2_full_has_vjp_at w hs q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 q12 q13 q14
      q15 q16 q17 hh x h_stem s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17
      h_head).correct dy i
  rwa [show mobilenetv2ForwardPaper w = mnv2HeadW w ∘ mnv2Pre17 w
      from funext (mobilenetv2ForwardPaper_eq_chain w)]

end Proofs
