import LeanMlir.Proofs.Nets.Small.Cifar8BnStepTieG
import LeanMlir.Proofs.Nets.Small.Cifar8ParamGrad
import LeanMlir.Proofs.Foundation.ParamGradNodes

/-! # The 8-conv CIFAR CNN with per-channel BN — every gradient node IS the loss's derivative

`cifar8Bn_train_step_tiedG` states the 38 un-fused gradient nodes the packed `cifar8w_bn_*` arms
emit, each at the cotangent the chain threads to it. `cifar8Bn_net_lossGrad` states that each node,
at the chain cotangent, is the gradient of the loss in that parameter, for any loss `L` of the
logits with gradient `g` there; `cifar8Bn_net_lossGrad_CE` instantiates it at the softmax
cross-entropy the render emits.

The net is the BN-free 8-conv net (`Cifar8TieG.cifar8_net_lossGrad`) with a per-example,
per-channel BN between each conv and its ReLU, so each pool's pre-activation is a BN output. Two
cells equal at every conv weight stay equal through BN (one affine map per channel), so the twin
relations (`Cifar8BnPoolTwin1` … `Cifar8BnPoolTwin4`, cells equal at every weight upstream of the
pool, BN `γ`/`β` included) and the selection routing carry over unchanged. Per node kind, BN adds
`bnGamma_hasGradAt`, `bnBeta_hasGradAt` and the input pull-back `hasGradAt_bnPC`.

**Hypotheses.** Odd kernels, every BN `ε > 0` (`Cifar8BnPos`), every ReLU off its kink, every pool
window dead or tied only between twins, each selection naming a maximum of every window
(`Cifar8BnLossSmoothAt`).
**Scope.** One example (the emitted module batch-contracts; `den` is per-example; BN normalises
each channel over the example's own spatial cells).
-/

open Proofs Proofs.StableHLO Proofs.IR Proofs.SmallParamGrad Proofs.CnnFold Proofs.CifarFold
  Proofs.Cifar8TieG

namespace Proofs.Cifar8BnTieG

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-channel BN: its parameter nodes and its input pull-back, per example
-- ════════════════════════════════════════════════════════════════

/-- Per-channel BN's VJP at a point: the renderable backward `bnPerChannelTensor3GradInput`. -/
noncomputable def bnPCHasVJPAt (oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v : Vec (oc * h * w)) : HasVJPAt (bnPerChannelTensor3 oc h w ε γ β) v where
  backward dy := bnPerChannelTensor3GradInput oc h w ε γ v dy
  correct dy i := bnPerChannelTensor3GradInput_correct oc h w ε hε γ β v dy i

/-- Through per-channel BN: the backward is `bnPerChannelTensor3GradInput`, the BN-back the
    render emits. -/
theorem hasGradAt_bnPC {oc h w : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (v : Vec (oc * h * w))
    {G : Vec (oc * h * w) → Vec 1} {dy : Vec (oc * h * w)}
    (hG : HasGradAt G (bnPerChannelTensor3 oc h w ε γ β v) dy) :
    HasGradAt (fun y => G (bnPerChannelTensor3 oc h w ε γ β y)) v
      (bnPerChannelTensor3GradInput oc h w ε γ v dy) :=
  hG.comp ((bnPerChannelTensor3_differentiable oc h w ε hε γ β) v) (bnPCHasVJPAt oc h w ε hε γ β v)

/-- The loss read through the layout bridge: `reassocBack`'s backward is `reassocFwd`. -/
private theorem hasGradAt_reassocBack {oc h w : Nat} {G : Vec (oc * h * w) → Vec 1}
    {u : Vec (oc * (h * w))} {c : Vec (oc * h * w)} (hG : HasGradAt G (reassocBack oc h w u) c) :
    HasGradAt (fun u' => G (reassocBack oc h w u')) u (reassocFwd oc h w c) :=
  (hG.comp (reassocBack_differentiable oc h w u) ((reassocBackHasVJP oc h w).toHasVJPAt u)).of_eq
    (reassocBackHasVJP_backward_eq oc h w u c)

/-- **BN γ node = `∇_γ G`.** -/
theorem bnGamma_hasGradAt {oc h w : Nat} (vN epsStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (oc * h * w)) {G : Vec (oc * h * w) → Vec 1} {c : Vec (oc * h * w)}
    (hG : HasGradAt G (bnPerChannelTensor3 oc h w ε γ β v) c) :
    HasGradAt (fun θ => G (bnPerChannelTensor3 oc h w ε θ β v)) γ
      (den (SHlo.bnGammaGrad vN epsStr ε v (.operand cotN c))) := by
  have hP := (hasGradAt_reassocBack hG).param
    (layer := fun θ => bnPerChannelFlat oc (h * w) ε θ β (reassocFwd oc h w v))
    ((GradNodeB.bnPerChannelFlat_gamma_differentiable _ _ _ _ _) _)
  refine ⟨hP.differentiableAt, fun i => (hP.pdiv_eq i).trans ?_⟩
  rw [GradNode.bnGammaGrad_den vN epsStr cotN ε γ β v c i]

/-- **BN β node = `∇_β G`.** -/
theorem bnBeta_hasGradAt {oc h w : Nat} (cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (oc * h * w)) {G : Vec (oc * h * w) → Vec 1} {c : Vec (oc * h * w)}
    (hG : HasGradAt G (bnPerChannelTensor3 oc h w ε γ β v) c) :
    HasGradAt (fun θ => G (bnPerChannelTensor3 oc h w ε γ θ v)) β
      (den (SHlo.bnBetaGrad (oc := oc) (h := h) (w := w) (.operand cotN c))) := by
  have hP := (hasGradAt_reassocBack hG).param
    (layer := fun θ => bnPerChannelFlat oc (h * w) ε γ θ (reassocFwd oc h w v))
    ((GradNodeB.bnPerChannelFlat_beta_differentiable _ _ _ _ _) _)
  refine ⟨hP.differentiableAt, fun i => (hP.pdiv_eq i).trans ?_⟩
  rw [GradNode.bnBetaGrad_den cotN ε γ β v c i]

theorem bnPC_gamma_continuous {oc h w : Nat} (ε : ℝ) (β : Vec oc) (v : Vec (oc * h * w)) :
    Continuous (fun θ : Vec oc => bnPerChannelTensor3 oc h w ε θ β v) :=
  (reassocBack_differentiable oc h w).continuous.comp
    (GradNodeB.bnPerChannelFlat_gamma_differentiable _ _ _ _ _).continuous

theorem bnPC_beta_continuous {oc h w : Nat} (ε : ℝ) (γ : Vec oc) (v : Vec (oc * h * w)) :
    Continuous (fun θ : Vec oc => bnPerChannelTensor3 oc h w ε γ θ v) :=
  (reassocBack_differentiable oc h w).continuous.comp
    (GradNodeB.bnPerChannelFlat_beta_differentiable _ _ _ _ _).continuous

-- ════════════════════════════════════════════════════════════════
-- § The stages: one pool's pre-activation to the next, and the twins of each pool
-- ════════════════════════════════════════════════════════════════

/-- From one pool's pre-activation to the next pool's: ReLU, pool, conv, BN, ReLU, conv, BN. -/
noncomputable def cifar8BnUp {c c' H W kH kW : Nat} (Wc : Kernel4 c' c kH kW) (bc : Vec c')
    (εc : ℝ) (γc βc : Vec c') (Wd : Kernel4 c' c' kH kW) (bd : Vec c') (εd : ℝ) (γd βd : Vec c')
    (z : Vec (c * (2 * H) * (2 * W))) : Vec (c' * H * W) :=
  bnPerChannelTensor3 c' H W εd γd βd (flatConv (h := H) (w := W) Wd bd (relu (c' * H * W)
    (bnPerChannelTensor3 c' H W εc γc βc (flatConv (h := H) (w := W) Wc bc
      (maxPoolFlat c H W (relu (c * (2 * H) * (2 * W)) z))))))

theorem cifar8BnUp_continuous {c c' H W kH kW : Nat} (Wc : Kernel4 c' c kH kW) (bc : Vec c')
    (εc : ℝ) (hεc : 0 < εc) (γc βc : Vec c') (Wd : Kernel4 c' c' kH kW) (bd : Vec c') (εd : ℝ)
    (hεd : 0 < εd) (γd βd : Vec c') :
    Continuous (cifar8BnUp (H := H) (W := W) Wc bc εc γc βc Wd bd εd γd βd) :=
  (bnPerChannelTensor3_differentiable _ _ _ εd hεd γd βd).continuous.comp
    ((flatConv_differentiable Wd bd).continuous.comp ((relu_continuous _).comp
      ((bnPerChannelTensor3_differentiable _ _ _ εc hεc γc βc).continuous.comp
        ((flatConv_differentiable Wc bc).continuous.comp
          ((maxPoolFlat_continuous _ _ _).comp (relu_continuous _))))))

section Twins
variable {ic c1 c2 c3 c4 h w kH kW : Nat}

/-- The first pool's pre-activation (BN₂'s output). -/
noncomputable def cifar8BnPre1 (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
        (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) :
    Vec (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) :=
  bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂
      (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂
    (relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))
        (bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₁ γ₁ β₁
        (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₁ b₁ x))))

/-- Pool 2's pre-activation (BN₄'s output). -/
noncomputable def cifar8BnPre2
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) : Vec
        (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) :=
  cifar8BnUp W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
    (cifar8BnPre1 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x)

/-- Pool 3's pre-activation (BN₆'s output). -/
noncomputable def cifar8BnPre3
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) : Vec
        (c3 * (2 * (2 * h)) * (2 * (2 * w))) :=
  cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆
    (cifar8BnPre2 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ x)

/-- Pool 4's pre-activation (BN₈'s output). -/
noncomputable def cifar8BnPre4
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) : Vec
        (c4 * (2 * h) * (2 * w)) :=
  cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
    (cifar8BnPre3 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅ W₆ b₆
        ε₆ γ₆ β₆ x)

/-- **Twins of pool 1**: equal at every weight (conv and BN `γ`/`β`) upstream of it, in
    every channel. -/
def Cifar8BnPoolTwin1 (c1 kH kW : Nat) (ε₁ ε₂ : ℝ)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (p q : Fin (2 * (2 * (2 * (2 * h)))) × Fin (2 * (2 * (2 * (2 * w))))) : Prop :=
  ∀ (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (γ₁ β₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW)
      (b₂ : Vec c1) (γ₂ β₂ : Vec c1)
    (ci : Fin c1),
    cifar8BnPre1 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x (t3Idx ci p.1 p.2)
      = cifar8BnPre1 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x (t3Idx ci q.1 q.2)

/-- **Twins of pool 2**: equal at every weight (conv and BN `γ`/`β`) upstream of it, in
    every channel. -/
def Cifar8BnPoolTwin2 (c1 c2 kH kW : Nat) (ε₁ ε₂ ε₃ ε₄ : ℝ)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (p q : Fin (2 * (2 * (2 * h))) × Fin (2 * (2 * (2 * w)))) : Prop :=
  ∀ (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (γ₁ β₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW)
      (b₂ : Vec c1) (γ₂ β₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (γ₃ β₃ : Vec c2)
      (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (γ₄ β₄ : Vec c2)
    (ci : Fin c2),
    cifar8BnPre2 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ x (t3Idx ci p.1 p.2)
      = cifar8BnPre2 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ x
          (t3Idx ci q.1 q.2)

/-- **Twins of pool 3**: equal at every weight (conv and BN `γ`/`β`) upstream of it, in
    every channel. -/
def Cifar8BnPoolTwin3 (c1 c2 c3 kH kW : Nat) (ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ : ℝ)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (p q : Fin (2 * (2 * h)) × Fin (2 * (2 * w))) : Prop :=
  ∀ (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (γ₁ β₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW)
      (b₂ : Vec c1) (γ₂ β₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (γ₃ β₃ : Vec c2)
      (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (γ₄ β₄ : Vec c2) (W₅ : Kernel4 c3 c2 kH kW)
      (b₅ : Vec c3) (γ₅ β₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (γ₆ β₆ : Vec c3)
    (ci : Fin c3),
    cifar8BnPre3 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆
        γ₆ β₆ x (t3Idx ci p.1 p.2)
      = cifar8BnPre3 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅ W₆
          b₆ ε₆ γ₆ β₆ x (t3Idx ci q.1 q.2)

/-- **Twins of pool 4**: equal at every weight (conv and BN `γ`/`β`) upstream of it, in
    every channel. -/
def Cifar8BnPoolTwin4 (c1 c2 c3 c4 kH kW : Nat) (ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈ : ℝ)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (p q : Fin (2 * h) × Fin (2 * w)) : Prop :=
  ∀ (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (γ₁ β₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW)
      (b₂ : Vec c1) (γ₂ β₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (γ₃ β₃ : Vec c2)
      (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (γ₄ β₄ : Vec c2) (W₅ : Kernel4 c3 c2 kH kW)
      (b₅ : Vec c3) (γ₅ β₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (γ₆ β₆ : Vec c3)
      (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (γ₇ β₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW)
      (b₈ : Vec c4) (γ₈ β₈ : Vec c4)
    (ci : Fin c4),
    cifar8BnPre4 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆
        γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ x (t3Idx ci p.1 p.2)
      = cifar8BnPre4 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅ W₆
          b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ x (t3Idx ci q.1 q.2)

end Twins

section Net
variable {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}

/-- Every BN `ε` is positive: BN is then differentiable everywhere. -/
structure Cifar8BnPos (ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈ : ℝ) : Prop where
  h1 : 0 < ε₁
  h2 : 0 < ε₂
  h3 : 0 < ε₃
  h4 : 0 < ε₄
  h5 : 0 < ε₅
  h6 : 0 < ε₆
  h7 : 0 < ε₇
  h8 : 0 < ε₈

/-- **The smooth-point bundle the loss gradient needs.** Every ReLU off its kink (at the BN
    outputs and the dense head); every window of each pool dead or tied only between that pool's
    twins; each selection naming a maximum of every window. -/
structure Cifar8BnLossSmoothAt
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (σ₁ : Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin c4 → Fin h → Fin w → Fin 2 × Fin 2) : Prop where
  z1 : ∀ k, bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₁ γ₁ β₁
      (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₁ b₁ x) k ≠ 0
  z2 : ∀ k, cifar8BnPre1 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x k ≠ 0
  pool1 : MaxPool2SmoothUpTo (Cifar8BnPoolTwin1 c1 kH kW ε₁ ε₂ x)
    (Tensor3.unflatten (cifar8BnPre1 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x) : Tensor3 c1
        (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))))
  sel1 : PoolSelDom σ₁ (relu _ (cifar8BnPre1 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x))
  z3 : ∀ k, bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₃ γ₃ β₃
      (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₃ b₃
    (maxPoolFlat c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w)))
        (relu _ (cifar8BnPre1 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x)))) k ≠ 0
  z4 : ∀ k, cifar8BnPre2 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ x k ≠ 0
  pool2 : MaxPool2SmoothUpTo (Cifar8BnPoolTwin2 c1 c2 kH kW ε₁ ε₂ ε₃ ε₄ x)
    (Tensor3.unflatten
        (cifar8BnPre2 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ x) : Tensor3 c2
        (2 * (2 * (2 * h))) (2 * (2 * (2 * w))))
  sel2 : PoolSelDom σ₂
      (relu _ (cifar8BnPre2 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ x))
  z5 : ∀ k, bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₅ γ₅ β₅
      (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₅ b₅
    (maxPoolFlat c2 (2 * (2 * h)) (2 * (2 * w))
        (relu _ (cifar8BnPre2 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ x)))) k ≠
        0
  z6 : ∀ k, cifar8BnPre3 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅
      W₆ b₆ ε₆ γ₆ β₆ x k ≠ 0
  pool3 : MaxPool2SmoothUpTo (Cifar8BnPoolTwin3 c1 c2 c3 kH kW ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ x)
    (Tensor3.unflatten
        (cifar8BnPre3 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅ W₆
        b₆ ε₆ γ₆ β₆ x) : Tensor3 c3 (2 * (2 * h)) (2 * (2 * w)))
  sel3 : PoolSelDom σ₃
      (relu _ (cifar8BnPre3 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅
      β₅ W₆ b₆ ε₆ γ₆ β₆ x))
  z7 : ∀ k, bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₇ γ₇ β₇
      (flatConv (h := 2 * h) (w := 2 * w) W₇ b₇
    (maxPoolFlat c3 (2 * h) (2 * w)
        (relu _ (cifar8BnPre3 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅
        γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ x)))) k ≠ 0
  z8 : ∀ k, cifar8BnPre4 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅
      W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ x k ≠ 0
  pool4 : MaxPool2SmoothUpTo (Cifar8BnPoolTwin4 c1 c2 c3 c4 kH kW ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈ x)
    (Tensor3.unflatten
        (cifar8BnPre4 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅ W₆
        b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ x) : Tensor3 c4 (2 * h) (2 * w))
  sel4 : PoolSelDom σ₄
      (relu _ (cifar8BnPre4 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅
      β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ x))
  z9 : ∀ k, dense W₉ b₉
      (maxPoolFlat c4 h w
      (relu _ (cifar8BnPre4 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅
      β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ x))) k ≠ 0
  za : ∀ k, dense Wa ba
      (relu d1 (dense W₉ b₉
      (maxPoolFlat c4 (h) (w)
      (relu _ (cifar8BnPre4 W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅
      β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ x))))) k ≠ 0

/-- **Every cifar8-bn gradient node is the gradient of `L`** in that parameter: the 38 un-fused
    nodes `cifar8Bn_train_step_tiedG` states, each at the cotangent the chain threads to its layer
    (each pool routed at its selection), stated against `L` of `cifarCnnBn8Forward` with that one
    parameter varied (`F` is `L` of the forward at the given weights, the `ε`s fixed). -/
def Cifar8BnNetLossTied (xN vN epsStr cotN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (σ₁ : Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin c4 → Fin h → Fin w → Fin 2 × Fin 2) (L : Vec nClasses → Vec 1) (g : Vec nClasses) :
    Prop :=
  let F := fun (W₁' : Kernel4 c1 ic kH kW) (b₁' : Vec c1) (γ₁' β₁' : Vec c1)
      (W₂' : Kernel4 c1 c1 kH kW) (b₂' : Vec c1) (γ₂' β₂' : Vec c1) (W₃' : Kernel4 c2 c1 kH kW)
      (b₃' : Vec c2) (γ₃' β₃' : Vec c2) (W₄' : Kernel4 c2 c2 kH kW) (b₄' : Vec c2)
      (γ₄' β₄' : Vec c2) (W₅' : Kernel4 c3 c2 kH kW) (b₅' : Vec c3) (γ₅' β₅' : Vec c3)
      (W₆' : Kernel4 c3 c3 kH kW) (b₆' : Vec c3) (γ₆' β₆' : Vec c3) (W₇' : Kernel4 c4 c3 kH kW)
      (b₇' : Vec c4) (γ₇' β₇' : Vec c4) (W₈' : Kernel4 c4 c4 kH kW) (b₈' : Vec c4)
      (γ₈' β₈' : Vec c4)
      (W₉' : Mat (c4 * h * w) d1) (b₉' : Vec d1) (Wa' : Mat d1 d1) (ba' : Vec d1)
      (Wb' : Mat d1 nClasses) (bb' : Vec nClasses) =>
    L (cifarCnnBn8Forward W₁' b₁' ε₁ γ₁' β₁' W₂' b₂' ε₂ γ₂' β₂' W₃' b₃' ε₃ γ₃' β₃' W₄' b₄' ε₄ γ₄'
        β₄' W₅' b₅' ε₅ γ₅' β₅' W₆' b₆' ε₆ γ₆' β₆' W₇' b₇' ε₇ γ₇' β₇' W₈' b₈' ε₈ γ₈' β₈'
      W₉' b₉' Wa' ba' Wb' bb' x)
  let cc1 := flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₁ b₁ x
  let bn1o := bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₁ γ₁ β₁
      cc1
  let a1 := relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) bn1o
  let cc2 := flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂ a1
  let bn2o := bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂
      cc2
  let pl1 := maxPoolFlat c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w)))
      (relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) bn2o)
  let cc3 := flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₃ b₃ pl1
  let bn3o := bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₃ γ₃ β₃ cc3
  let a3 := relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) bn3o
  let cc4 := flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄ a3
  let bn4o := bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄ cc4
  let pl2 := maxPoolFlat c2 (2 * (2 * h)) (2 * (2 * w))
      (relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) bn4o)
  let cc5 := flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₅ b₅ pl2
  let bn5o := bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₅ γ₅ β₅ cc5
  let a5 := relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) bn5o
  let cc6 := flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆ a5
  let bn6o := bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆ cc6
  let pl3 := maxPoolFlat c3 (2 * h) (2 * w) (relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) bn6o)
  let cc7 := flatConv (h := 2 * h) (w := 2 * w) W₇ b₇ pl3
  let bn7o := bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₇ γ₇ β₇ cc7
  let a7 := relu (c4 * (2 * h) * (2 * w)) bn7o
  let cc8 := flatConv (h := 2 * h) (w := 2 * w) W₈ b₈ a7
  let bn8o := bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈ cc8
  let pl4 := maxPoolFlat c4 h w (relu (c4 * (2 * h) * (2 * w)) bn8o)
  let h9 := dense W₉ b₉ pl4
  let ha := dense Wa ba (relu d1 h9)
  let cotHa := (mlpCotOut1 Wb ha).denote g
  let cotH9 := (mlpCotOut0 Wa Wb h9 ha).denote g
  let dyBn8 := cnnChainCotW2Sel σ₄ W₉ Wa Wb h9 ha bn8o g
  let cotC8 := bnPerChannelTensor3GradInput c4 (2 * h) (2 * w) ε₈ γ₈ cc8 dyBn8
  let dyBn7 := cnnChainCotW1 W₈ bn7o cotC8
  let cotC7 := bnPerChannelTensor3GradInput c4 (2 * h) (2 * w) ε₇ γ₇ cc7 dyBn7
  let dyBn6 := cifarChainCotW2Sel σ₃ W₇ bn6o cotC7
  let cotC6 := bnPerChannelTensor3GradInput c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ cc6 dyBn6
  let dyBn5 := cnnChainCotW1 W₆ bn5o cotC6
  let cotC5 := bnPerChannelTensor3GradInput c3 (2 * (2 * h)) (2 * (2 * w)) ε₅ γ₅ cc5 dyBn5
  let dyBn4 := cifarChainCotW2Sel σ₂ W₅ bn4o cotC5
  let cotC4 := bnPerChannelTensor3GradInput c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ cc4
      dyBn4
  let dyBn3 := cnnChainCotW1 W₄ bn3o cotC4
  let cotC3 := bnPerChannelTensor3GradInput c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₃ γ₃ cc3
      dyBn3
  let dyBn2 := cifarChainCotW2Sel σ₁ W₃ bn2o cotC3
  let cotC2 := bnPerChannelTensor3GradInput c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w))))
      ε₂ γ₂ cc2 dyBn2
  let dyBn1 := cnnChainCotW1 W₂ bn1o cotC2
  let cotC1 := bnPerChannelTensor3GradInput c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w))))
      ε₁ γ₁ cc1 dyBn1
  HasGradAt (fun θ => F (Kernel4.unflatten θ) b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅
      β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₁)
      (den (SHlo.convWeightGrad xN b₁ (Tensor3.unflatten x) W₁ (.operand cotN cotC1)))
  ∧ HasGradAt
      (fun θ => F W₁ θ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) b₁
      (den (SHlo.convBiasGrad W₁ (Tensor3.unflatten x) b₁ (.operand cotN cotC1)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ θ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) γ₁
      (den (SHlo.bnGammaGrad vN epsStr ε₁ cc1 (.operand cotN dyBn1)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ θ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) β₁
      (den (SHlo.bnBetaGrad (oc := c1) (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w))))
          (.operand cotN dyBn1)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ (Kernel4.unflatten θ) b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆
      b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₂)
      (den (SHlo.convWeightGrad xN b₂ (Tensor3.unflatten a1) W₂ (.operand cotN cotC2)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ θ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) b₂
      (den (SHlo.convBiasGrad W₂ (Tensor3.unflatten a1) b₂ (.operand cotN cotC2)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ θ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) γ₂
      (den (SHlo.bnGammaGrad vN epsStr ε₂ cc2 (.operand cotN dyBn2)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ θ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) β₂
      (den (SHlo.bnBetaGrad (oc := c1) (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w))))
          (.operand cotN dyBn2)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ (Kernel4.unflatten θ) b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆
      b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₃)
      (den (SHlo.convWeightGrad xN b₃ (Tensor3.unflatten pl1) W₃ (.operand cotN cotC3)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ θ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) b₃
      (den (SHlo.convBiasGrad W₃ (Tensor3.unflatten pl1) b₃ (.operand cotN cotC3)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ θ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) γ₃
      (den (SHlo.bnGammaGrad vN epsStr ε₃ cc3 (.operand cotN dyBn3)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ θ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) β₃
      (den (SHlo.bnBetaGrad (oc := c2) (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w)))
          (.operand cotN dyBn3)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ (Kernel4.unflatten θ) b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆
      b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₄)
      (den (SHlo.convWeightGrad xN b₄ (Tensor3.unflatten a3) W₄ (.operand cotN cotC4)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ θ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) b₄
      (den (SHlo.convBiasGrad W₄ (Tensor3.unflatten a3) b₄ (.operand cotN cotC4)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ θ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) γ₄
      (den (SHlo.bnGammaGrad vN epsStr ε₄ cc4 (.operand cotN dyBn4)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ θ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) β₄
      (den (SHlo.bnBetaGrad (oc := c2) (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w)))
          (.operand cotN dyBn4)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ (Kernel4.unflatten θ) b₅ γ₅ β₅ W₆
      b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₅)
      (den (SHlo.convWeightGrad xN b₅ (Tensor3.unflatten pl2) W₅ (.operand cotN cotC5)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ θ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) b₅
      (den (SHlo.convBiasGrad W₅ (Tensor3.unflatten pl2) b₅ (.operand cotN cotC5)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ θ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) γ₅
      (den (SHlo.bnGammaGrad vN epsStr ε₅ cc5 (.operand cotN dyBn5)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ θ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) β₅
      (den (SHlo.bnBetaGrad (oc := c3) (h := 2 * (2 * h)) (w := 2 * (2 * w)) (.operand cotN dyBn5)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ (Kernel4.unflatten θ)
      b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₆)
      (den (SHlo.convWeightGrad xN b₆ (Tensor3.unflatten a5) W₆ (.operand cotN cotC6)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ θ γ₆ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) b₆
      (den (SHlo.convBiasGrad W₆ (Tensor3.unflatten a5) b₆ (.operand cotN cotC6)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ θ β₆ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) γ₆
      (den (SHlo.bnGammaGrad vN epsStr ε₆ cc6 (.operand cotN dyBn6)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ θ W₇ b₇ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) β₆
      (den (SHlo.bnBetaGrad (oc := c3) (h := 2 * (2 * h)) (w := 2 * (2 * w)) (.operand cotN dyBn6)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆
      (Kernel4.unflatten θ) b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₇)
      (den (SHlo.convWeightGrad xN b₇ (Tensor3.unflatten pl3) W₇ (.operand cotN cotC7)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ θ γ₇ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) b₇
      (den (SHlo.convBiasGrad W₇ (Tensor3.unflatten pl3) b₇ (.operand cotN cotC7)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ θ β₇
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) γ₇
      (den (SHlo.bnGammaGrad vN epsStr ε₇ cc7 (.operand cotN dyBn7)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ θ
      W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb) β₇
      (den (SHlo.bnBetaGrad (oc := c4) (h := 2 * h) (w := 2 * w) (.operand cotN dyBn7)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇
      β₇ (Kernel4.unflatten θ) b₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₈)
      (den (SHlo.convWeightGrad xN b₈ (Tensor3.unflatten a7) W₈ (.operand cotN cotC8)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇
      β₇ W₈ θ γ₈ β₈ W₉ b₉ Wa ba Wb bb) b₈
      (den (SHlo.convBiasGrad W₈ (Tensor3.unflatten a7) b₈ (.operand cotN cotC8)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇
      β₇ W₈ b₈ θ β₈ W₉ b₉ Wa ba Wb bb) γ₈
      (den (SHlo.bnGammaGrad vN epsStr ε₈ cc8 (.operand cotN dyBn8)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇
      β₇ W₈ b₈ γ₈ θ W₉ b₉ Wa ba Wb bb) β₈
      (den (SHlo.bnBetaGrad (oc := c4) (h := 2 * h) (w := 2 * w) (.operand cotN dyBn8)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇
      β₇ W₈ b₈ γ₈ β₈ (Mat.unflatten θ) b₉ Wa ba Wb bb)
      (Mat.flatten W₉) (den (SHlo.weightGrad xN pl4 (.operand cotN cotH9)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇
      β₇ W₈ b₈ γ₈ β₈ W₉ θ Wa ba Wb bb) b₉
      (den (SHlo.biasGrad (.operand cotN cotH9)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇
      β₇ W₈ b₈ γ₈ β₈ W₉ b₉ (Mat.unflatten θ) ba Wb bb)
      (Mat.flatten Wa) (den (SHlo.weightGrad xN (relu d1 h9) (.operand cotN cotHa)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇
      β₇ W₈ b₈ γ₈ β₈ W₉ b₉ Wa θ Wb bb) ba
      (den (SHlo.biasGrad (.operand cotN cotHa)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇
      β₇ W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba (Mat.unflatten θ) bb)
      (Mat.flatten Wb) (den (SHlo.weightGrad xN (relu d1 ha) (.operand cotN g)))
  ∧ HasGradAt
      (fun θ => F W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇
      β₇ W₈ b₈ γ₈ β₈ W₉ b₉ Wa ba Wb θ) bb
      (den (SHlo.biasGrad (.operand cotN g)))

-- the 38-clause conjunction nests past the default recursion depth
set_option maxRecDepth 4000 in
/-- **Every cifar8-bn gradient node is the gradient of `L` in that parameter**, whenever `g` is
    `L`'s gradient at the logits.

    Hypotheses: odd kernels, every BN `ε` positive (`Cifar8BnPos`), and `Cifar8BnLossSmoothAt` —
    every ReLU off its kink, every window of each pool dead or tied only between cells that are
    the same function of the weights upstream of it, each selection naming a maximum of every
    window. -/
theorem cifar8Bn_net_lossGrad (xN vN epsStr cotN : String) (hkH : 2 * ((kH - 1) / 2) + 1 = kH)
    (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (hq : Cifar8BnPos ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (σ₁ : Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin c4 → Fin h → Fin w → Fin 2 × Fin 2)
    (hx : Cifar8BnLossSmoothAt W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅
        γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba x σ₁ σ₂ σ₃ σ₄)
    {L : Vec nClasses → Vec 1} {g : Vec nClasses}
    (hL : HasGradAt L
        (cifarCnnBn8Forward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅
        β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb x) g) :
    Cifar8BnNetLossTied xN vN epsStr cotN W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄
        β₄ W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb x σ₁ σ₂ σ₃
        σ₄ L g := by
  unfold Cifar8BnNetLossTied
  intro F cc1 bn1o a1 cc2 bn2o pl1 cc3 bn3o a3 cc4 bn4o pl2 cc5 bn5o a5 cc6 bn6o pl3 cc7 bn7o a7 cc8
      bn8o pl4 h9 ha cotHa cotH9 dyBn8 cotC8 dyBn7 cotC7 dyBn6 cotC6 dyBn5 cotC5 dyBn4 cotC4 dyBn3
      cotC3 dyBn2 cotC2 dyBn1 cotC1
  -- the dense head
  have hLb : HasGradAt L (dense Wb bb (relu d1 ha)) g := hL
  have hHa : HasGradAt (fun y => L (dense Wb bb (relu d1 y))) ha cotHa :=
    (hasGradAt_relu ha hx.za (hasGradAt_dense Wb bb _ hLb)).of_eq (denote_subst _ _ g).symm
  have hH9 : HasGradAt (fun y => L (dense Wb bb (relu d1 (dense Wa ba (relu d1 y))))) h9 cotH9 :=
    (hasGradAt_relu h9 hx.z9 (hasGradAt_dense Wa ba _ hHa)).of_eq
      (by simp only [cotH9, mlpCotOut0, denote_subst]; rfl)
  -- the gather model: each pool frozen at its selection
  let Gp4 : Vec (c4 * h * w) → Vec 1 := fun u =>
    L (dense Wb bb (relu d1 (dense Wa ba (relu d1 (dense W₉ b₉ u)))))
  let G8 : Vec (c4 * (2 * h) * (2 * w)) → Vec 1 := fun y =>
    Gp4 (fun k => relu (c4 * (2 * h) * (2 * w)) y (poolSelIdx σ₄ k))
  let G6 : Vec (c3 * (2 * (2 * h)) * (2 * (2 * w))) → Vec 1 := fun y =>
    G8 (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈ (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈
      (relu (c4 * (2 * h) * (2 * w))
          (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₇ γ₇ β₇ (flatConv (h := 2 * h) (w := 2 * w) W₇ b₇
        (fun k => relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) y (poolSelIdx σ₃ k)))))))
  let G4 : Vec (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) → Vec 1 := fun y =>
    G6 (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆
        (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
      (relu (c3 * (2 * (2 * h)) * (2 * (2 * w)))
          (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₅ γ₅ β₅
          (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₅ b₅
        (fun k => relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) y (poolSelIdx σ₂ k)))))))
  let G2 : Vec (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) → Vec 1 := fun y =>
    G4 (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄
        (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄
      (relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))
          (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₃ γ₃ β₃
          (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₃ b₃
        (fun k => relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) y
            (poolSelIdx σ₁ k)))))))
  have hpt4 : pl4 = fun k => relu (c4 * (2 * h) * (2 * w)) bn8o (poolSelIdx σ₄ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ₄ _ hx.sel4
  have hpt3 : pl3 = fun k => relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) bn6o (poolSelIdx σ₃ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ₃ _ hx.sel3
  have hpt2 : pl2 = fun k => relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) bn4o
      (poolSelIdx σ₂ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ₂ _ hx.sel2
  have hpt1 : pl1 = fun k => relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) bn2o
      (poolSelIdx σ₁ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ₁ _ hx.sel1
  have hB8 : HasGradAt G8 bn8o dyBn8 :=
    (hasGradAt_gatherRelu (poolSelIdx σ₄) bn8o hx.z8
      ((hasGradAt_dense W₉ b₉ pl4 hH9).congr_point hpt4)).of_eq (by
      funext i
      simp only [dyBn8, cnnChainCotW2Sel, cnnDenseHeadCot, cotH9, mlpCotOut0, mlpCotOut1,
        denote_subst]
      rfl)
  have hC8 : HasGradAt (fun z => G8 (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈ z)) cc8
      cotC8 :=
    hasGradAt_bnPC ε₈ hq.h8 γ₈ β₈ cc8 hB8
  have hB7 : HasGradAt
      (fun y => G8
      (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈ (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈
      (relu (c4 * (2 * h) * (2 * w)) y)))) bn7o dyBn7 :=
    hasGradAt_relu bn7o hx.z7 (hasGradAt_conv hkH hkW W₈ b₈ a7 hC8)
  have hC7 : HasGradAt
      (fun z => G8
      (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈ (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈
      (relu (c4 * (2 * h) * (2 * w)) (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₇ γ₇ β₇ z))))) cc7
          cotC7 :=
    hasGradAt_bnPC ε₇ hq.h7 γ₇ β₇ cc7 hB7
  have hB6 : HasGradAt G6 bn6o dyBn6 :=
    (hasGradAt_gatherRelu (poolSelIdx σ₃) bn6o hx.z6
      ((hasGradAt_conv hkH hkW W₇ b₇ pl3 hC7).congr_point hpt3)).of_eq rfl
  have hC6 : HasGradAt (fun z => G6 (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆ z))
      cc6 cotC6 :=
    hasGradAt_bnPC ε₆ hq.h6 γ₆ β₆ cc6 hB6
  have hB5 : HasGradAt
      (fun y => G6
      (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆
      (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
      (relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) y)))) bn5o dyBn5 :=
    hasGradAt_relu bn5o hx.z5 (hasGradAt_conv hkH hkW W₆ b₆ a5 hC6)
  have hC5 : HasGradAt
      (fun z => G6
      (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆
      (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
      (relu (c3 * (2 * (2 * h)) * (2 * (2 * w)))
          (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₅ γ₅ β₅ z))))) cc5 cotC5 :=
    hasGradAt_bnPC ε₅ hq.h5 γ₅ β₅ cc5 hB5
  have hB4 : HasGradAt G4 bn4o dyBn4 :=
    (hasGradAt_gatherRelu (poolSelIdx σ₂) bn4o hx.z4
      ((hasGradAt_conv hkH hkW W₅ b₅ pl2 hC5).congr_point hpt2)).of_eq rfl
  have hC4 : HasGradAt
      (fun z => G4 (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄ z)) cc4
      cotC4 :=
    hasGradAt_bnPC ε₄ hq.h4 γ₄ β₄ cc4 hB4
  have hB3 : HasGradAt
      (fun y => G4
      (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄
      (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄
      (relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) y)))) bn3o dyBn3 :=
    hasGradAt_relu bn3o hx.z3 (hasGradAt_conv hkH hkW W₄ b₄ a3 hC4)
  have hC3 : HasGradAt
      (fun z => G4
      (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄
      (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄
      (relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))
          (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₃ γ₃ β₃ z))))) cc3
          cotC3 :=
    hasGradAt_bnPC ε₃ hq.h3 γ₃ β₃ cc3 hB3
  have hB2 : HasGradAt G2 bn2o dyBn2 :=
    (hasGradAt_gatherRelu (poolSelIdx σ₁) bn2o hx.z2
      ((hasGradAt_conv hkH hkW W₃ b₃ pl1 hC3).congr_point hpt1)).of_eq rfl
  have hC2 : HasGradAt
      (fun z => G2
      (bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂ z)) cc2
      cotC2 :=
    hasGradAt_bnPC ε₂ hq.h2 γ₂ β₂ cc2 hB2
  have hB1 : HasGradAt
      (fun y => G2
      (bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂
      (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂
      (relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) y)))) bn1o dyBn1 :=
    hasGradAt_relu bn1o hx.z1 (hasGradAt_conv hkH hkW W₂ b₂ a1 hC2)
  have hC1 : HasGradAt
      (fun z => G2
      (bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂
      (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂
      (relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))
          (bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₁ γ₁ β₁
          z))))) cc1 cotC1 :=
    hasGradAt_bnPC ε₁ hq.h1 γ₁ β₁ cc1 hB1
  have germ4 : ∀ {P : Nat} (Z : Vec P → Vec (c4 * (2 * h) * (2 * w))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = bn8o →
      (∀ θ (ci : Fin c4) (p q : Fin (2 * h) × Fin (2 * w)), Cifar8BnPoolTwin4 c1 c2 c3 c4 kH kW ε₁
          ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈ x p q →
        (Z θ) (t3Idx ci p.1 p.2) = (Z θ) (t3Idx ci q.1 q.2)) →
      (fun θ => Gp4 (maxPoolFlat c4 h w (relu _ (Z θ)))) =ᶠ[nhds θ₀] fun θ => G8 (Z θ) := by
    intro P Z θ₀ hZc h0 hT4
    have hg := maxPool_relu_eventuallyEq_sel Z σ₄
        (Cifar8BnPoolTwin4 c1 c2 c3 c4 kH kW ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈ x) hT4 θ₀ hZc
      (by rw [h0]; exact hx.z8) (by rw [h0]; exact hx.pool4) (by rw [h0]; exact hx.sel4)
    filter_upwards [hg] with θ hθ
    exact congrArg Gp4 hθ
  have germ3 : ∀ {P : Nat} (Z : Vec P → Vec (c3 * (2 * (2 * h)) * (2 * (2 * w)))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = bn6o →
      (∀ θ (ci : Fin c3) (p q : Fin (2 * (2 * h)) × Fin (2 * (2 * w))), Cifar8BnPoolTwin3 c1 c2 c3
          kH kW ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ x p q →
        (Z θ) (t3Idx ci p.1 p.2) = (Z θ) (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c4) (p q : Fin (2 * h) × Fin (2 * w)), Cifar8BnPoolTwin4 c1 c2 c3 c4 kH kW ε₁
          ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈ x p q →
        (cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ (Z θ)) (t3Idx ci p.1 p.2) =
            (cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ (Z θ)) (t3Idx ci q.1 q.2)) →
      (fun θ => Gp4
          (maxPoolFlat c4 h w (relu _ (cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ (Z θ))))) =ᶠ[nhds
          θ₀] fun θ => G6 (Z θ) := by
    intro P Z θ₀ hZc h0 hT3 hT4
    refine (germ4 (fun θ => cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ (Z θ)) θ₀
      ((cifar8BnUp_continuous W₇ b₇ ε₇ hq.h7 γ₇ β₇ W₈ b₈ ε₈ hq.h8 γ₈ β₈).continuousAt.comp hZc)
      (by rw [h0]; rfl) hT4).trans ?_
    have hg := maxPool_relu_eventuallyEq_sel Z σ₃
        (Cifar8BnPoolTwin3 c1 c2 c3 kH kW ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ x) hT3 θ₀ hZc
      (by rw [h0]; exact hx.z6) (by rw [h0]; exact hx.pool3) (by rw [h0]; exact hx.sel3)
    filter_upwards [hg] with θ hθ
    exact congrArg
        (fun v => G8
        (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈ (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈
      (relu (c4 * (2 * h) * (2 * w))
          (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₇ γ₇ β₇
          (flatConv (h := 2 * h) (w := 2 * w) W₇ b₇ v)))))) hθ
  have germ2 : ∀ {P : Nat} (Z : Vec P → Vec (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))))
      (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = bn4o →
      (∀ θ (ci : Fin c2) (p q : Fin (2 * (2 * (2 * h))) × Fin (2 * (2 * (2 * w)))),
          Cifar8BnPoolTwin2 c1 c2 kH kW ε₁ ε₂ ε₃ ε₄ x p q →
        (Z θ) (t3Idx ci p.1 p.2) = (Z θ) (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c3) (p q : Fin (2 * (2 * h)) × Fin (2 * (2 * w))), Cifar8BnPoolTwin3 c1 c2 c3
          kH kW ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ x p q →
        (cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ (Z θ)) (t3Idx ci p.1 p.2) =
            (cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ (Z θ)) (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c4) (p q : Fin (2 * h) × Fin (2 * w)), Cifar8BnPoolTwin4 c1 c2 c3 c4 kH kW ε₁
          ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈ x p q →
        (cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ (cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ (Z θ)))
            (t3Idx ci p.1 p.2) =
            (cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
            (cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ (Z θ))) (t3Idx ci q.1 q.2)) →
      (fun θ => Gp4
          (maxPoolFlat c4 h w
          (relu _ (cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
          (cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ (Z θ)))))) =ᶠ[nhds θ₀] fun θ => G4 (Z θ) := by
    intro P Z θ₀ hZc h0 hT2 hT3 hT4
    refine (germ3 (fun θ => cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ (Z θ)) θ₀
      ((cifar8BnUp_continuous W₅ b₅ ε₅ hq.h5 γ₅ β₅ W₆ b₆ ε₆ hq.h6 γ₆ β₆).continuousAt.comp hZc)
      (by rw [h0]; rfl) hT3 hT4).trans ?_
    have hg := maxPool_relu_eventuallyEq_sel Z σ₂ (Cifar8BnPoolTwin2 c1 c2 kH kW ε₁ ε₂ ε₃ ε₄ x) hT2
        θ₀ hZc
      (by rw [h0]; exact hx.z4) (by rw [h0]; exact hx.pool2) (by rw [h0]; exact hx.sel2)
    filter_upwards [hg] with θ hθ
    exact congrArg
        (fun v => G6
        (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆
        (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
      (relu (c3 * (2 * (2 * h)) * (2 * (2 * w)))
          (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₅ γ₅ β₅
          (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₅ b₅ v)))))) hθ
  have germ1 : ∀ {P : Nat}
      (Z : Vec P → Vec (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = bn2o →
      (∀ θ (ci : Fin c1) (p q : Fin (2 * (2 * (2 * (2 * h)))) × Fin (2 * (2 * (2 * (2 * w))))),
          Cifar8BnPoolTwin1 c1 kH kW ε₁ ε₂ x p q →
        (Z θ) (t3Idx ci p.1 p.2) = (Z θ) (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c2) (p q : Fin (2 * (2 * (2 * h))) × Fin (2 * (2 * (2 * w)))),
          Cifar8BnPoolTwin2 c1 c2 kH kW ε₁ ε₂ ε₃ ε₄ x p q →
        (cifar8BnUp W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ (Z θ)) (t3Idx ci p.1 p.2) =
            (cifar8BnUp W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ (Z θ)) (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c3) (p q : Fin (2 * (2 * h)) × Fin (2 * (2 * w))), Cifar8BnPoolTwin3 c1 c2 c3
          kH kW ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ x p q →
        (cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ (cifar8BnUp W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ (Z θ)))
            (t3Idx ci p.1 p.2) =
            (cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆
            (cifar8BnUp W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ (Z θ))) (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c4) (p q : Fin (2 * h) × Fin (2 * w)), Cifar8BnPoolTwin4 c1 c2 c3 c4 kH kW ε₁
          ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈ x p q →
        (cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
            (cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆
            (cifar8BnUp W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ (Z θ)))) (t3Idx ci p.1 p.2) =
            (cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
            (cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆
            (cifar8BnUp W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ (Z θ)))) (t3Idx ci q.1 q.2)) →
      (fun θ => Gp4
          (maxPoolFlat c4 h w
          (relu _ (cifar8BnUp W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
          (cifar8BnUp W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆
          (cifar8BnUp W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ (Z θ))))))) =ᶠ[nhds θ₀] fun θ => G2 (Z θ) := by
    intro P Z θ₀ hZc h0 hT1 hT2 hT3 hT4
    refine (germ2 (fun θ => cifar8BnUp W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ (Z θ)) θ₀
      ((cifar8BnUp_continuous W₃ b₃ ε₃ hq.h3 γ₃ β₃ W₄ b₄ ε₄ hq.h4 γ₄ β₄).continuousAt.comp hZc)
      (by rw [h0]; rfl) hT2 hT3 hT4).trans ?_
    have hg := maxPool_relu_eventuallyEq_sel Z σ₁ (Cifar8BnPoolTwin1 c1 kH kW ε₁ ε₂ x) hT1 θ₀ hZc
      (by rw [h0]; exact hx.z2) (by rw [h0]; exact hx.pool1) (by rw [h0]; exact hx.sel1)
    filter_upwards [hg] with θ hθ
    exact congrArg
        (fun v => G4
        (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄
        (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄
      (relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))
          (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₃ γ₃ β₃
          (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₃ b₃ v)))))) hθ
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
      ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    denseW_hasGradAt xN cotN pl4 W₉ b₉ hH9, denseB_hasGradAt cotN W₉ pl4 b₉ hH9,
    denseW_hasGradAt xN cotN _ Wa ba hHa, denseB_hasGradAt cotN Wa _ ba hHa,
    denseW_hasGradAt xN cotN _ Wb bb hLb, denseB_hasGradAt cotN Wb _ bb hLb⟩
  · refine (convW_hasGradAt xN cotN b₁ x W₁ hC1).congr_of_eventuallyEq (germ1
      (fun θ => bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂
          (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂
        (relu _ (bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₁ γ₁ β₁
            (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w))))
            (Kernel4.unflatten θ) b₁ x))))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₂ hq.h2 γ₂ β₂).continuous.comp
        ((flatConv_differentiable W₂ b₂).continuous.comp ((relu_continuous _).comp
        ((bnPerChannelTensor3_differentiable _ _ _ ε₁ hq.h1 γ₁ β₁).continuous.comp
        (conv2d_weight_differentiable b₁ (Tensor3.unflatten x)).continuous)))).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq (Kernel4.unflatten θ) b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ ci)
      (fun θ ci p q hpq => hpq (Kernel4.unflatten θ) b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄
          ci)
      (fun θ ci p q hpq => hpq (Kernel4.unflatten θ) b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅
          b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ ci)
      (fun θ ci p q hpq => hpq (Kernel4.unflatten θ) b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅
          b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convB_hasGradAt cotN W₁ x b₁ hC1).congr_of_eventuallyEq (germ1
      (fun θ => bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂
          (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂
        (relu _ (bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₁ γ₁ β₁
            (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₁ θ x))))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₂ hq.h2 γ₂ β₂).continuous.comp
        ((flatConv_differentiable W₂ b₂).continuous.comp ((relu_continuous _).comp
        ((bnPerChannelTensor3_differentiable _ _ _ ε₁ hq.h1 γ₁ β₁).continuous.comp
        (conv2d_bias_differentiable W₁ (Tensor3.unflatten x)).continuous)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ θ γ₁ β₁ W₂ b₂ γ₂ β₂ ci)
      (fun θ ci p q hpq => hpq W₁ θ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ θ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ θ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnGamma_hasGradAt vN epsStr cotN ε₁ γ₁ β₁ cc1 hB1).congr_of_eventuallyEq (germ1
      (fun θ => bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂
          (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂
        (relu _ (bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₁ θ β₁
            cc1)))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₂ hq.h2 γ₂ β₂).continuous.comp
        ((flatConv_differentiable W₂ b₂).continuous.comp ((relu_continuous _).comp
        (bnPC_gamma_continuous ε₁ β₁ cc1)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ θ β₁ W₂ b₂ γ₂ β₂ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ θ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ θ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ θ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnBeta_hasGradAt cotN ε₁ γ₁ β₁ cc1 hB1).congr_of_eventuallyEq (germ1
      (fun θ => bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂
          (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂
        (relu _ (bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₁ γ₁ θ
            cc1)))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₂ hq.h2 γ₂ β₂).continuous.comp
        ((flatConv_differentiable W₂ b₂).continuous.comp ((relu_continuous _).comp
        (bnPC_beta_continuous ε₁ γ₁ cc1)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ θ W₂ b₂ γ₂ β₂ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ θ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ θ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ θ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₂ a1 W₂ hC2).congr_of_eventuallyEq (germ1
      (fun θ => bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂
          (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w))))
          (Kernel4.unflatten θ) b₂ a1)) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₂ hq.h2 γ₂ β₂).continuous.comp
        (conv2d_weight_differentiable b₂ (Tensor3.unflatten a1)).continuous).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ (Kernel4.unflatten θ) b₂ γ₂ β₂ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ (Kernel4.unflatten θ) b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄
          ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ (Kernel4.unflatten θ) b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅
          b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ (Kernel4.unflatten θ) b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅
          b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convB_hasGradAt cotN W₂ a1 b₂ hC2).congr_of_eventuallyEq (germ1
      (fun θ => bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ β₂
          (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ θ a1)) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₂ hq.h2 γ₂ β₂).continuous.comp
        (conv2d_bias_differentiable W₂ (Tensor3.unflatten a1)).continuous).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ θ γ₂ β₂ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ θ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ θ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ θ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnGamma_hasGradAt vN epsStr cotN ε₂ γ₂ β₂ cc2 hB2).congr_of_eventuallyEq (germ1
      (fun θ => bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ θ β₂
          cc2) _
      (bnPC_gamma_continuous ε₂ β₂ cc2).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ θ β₂ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ θ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ θ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ θ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnBeta_hasGradAt cotN ε₂ γ₂ β₂ cc2 hB2).congr_of_eventuallyEq (germ1
      (fun θ => bnPerChannelTensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε₂ γ₂ θ
          cc2) _
      (bnPC_beta_continuous ε₂ γ₂ cc2).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ θ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ θ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ θ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ θ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₃ pl1 W₃ hC3).congr_of_eventuallyEq (germ2
      (fun θ => bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄
          (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄
        (relu _ (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₃ γ₃ β₃
            (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) (Kernel4.unflatten θ) b₃
            pl1))))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₄ hq.h4 γ₄ β₄).continuous.comp
        ((flatConv_differentiable W₄ b₄).continuous.comp ((relu_continuous _).comp
        ((bnPerChannelTensor3_differentiable _ _ _ ε₃ hq.h3 γ₃ β₃).continuous.comp
        (conv2d_weight_differentiable b₃ (Tensor3.unflatten pl1)).continuous)))).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ (Kernel4.unflatten θ) b₃ γ₃ β₃ W₄ b₄ γ₄ β₄
          ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ (Kernel4.unflatten θ) b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅
          b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ (Kernel4.unflatten θ) b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅
          b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convB_hasGradAt cotN W₃ pl1 b₃ hC3).congr_of_eventuallyEq (germ2
      (fun θ => bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄
          (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄
        (relu _ (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₃ γ₃ β₃
            (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₃ θ pl1))))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₄ hq.h4 γ₄ β₄).continuous.comp
        ((flatConv_differentiable W₄ b₄).continuous.comp ((relu_continuous _).comp
        ((bnPerChannelTensor3_differentiable _ _ _ ε₃ hq.h3 γ₃ β₃).continuous.comp
        (conv2d_bias_differentiable W₃ (Tensor3.unflatten pl1)).continuous)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ θ γ₃ β₃ W₄ b₄ γ₄ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ θ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ θ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnGamma_hasGradAt vN epsStr cotN ε₃ γ₃ β₃ cc3 hB3).congr_of_eventuallyEq (germ2
      (fun θ => bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄
          (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄
        (relu _ (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₃ θ β₃ cc3)))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₄ hq.h4 γ₄ β₄).continuous.comp
        ((flatConv_differentiable W₄ b₄).continuous.comp ((relu_continuous _).comp
        (bnPC_gamma_continuous ε₃ β₃ cc3)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ θ β₃ W₄ b₄ γ₄ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ θ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ θ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnBeta_hasGradAt cotN ε₃ γ₃ β₃ cc3 hB3).congr_of_eventuallyEq (germ2
      (fun θ => bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄
          (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄
        (relu _ (bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₃ γ₃ θ cc3)))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₄ hq.h4 γ₄ β₄).continuous.comp
        ((flatConv_differentiable W₄ b₄).continuous.comp ((relu_continuous _).comp
        (bnPC_beta_continuous ε₃ γ₃ cc3)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ θ W₄ b₄ γ₄ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ θ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ θ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₄ a3 W₄ hC4).congr_of_eventuallyEq (germ2
      (fun θ => bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄
          (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) (Kernel4.unflatten θ) b₄ a3))
          _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₄ hq.h4 γ₄ β₄).continuous.comp
        (conv2d_weight_differentiable b₄ (Tensor3.unflatten a3)).continuous).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ (Kernel4.unflatten θ) b₄ γ₄ β₄
          ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ (Kernel4.unflatten θ) b₄ γ₄ β₄ W₅
          b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ (Kernel4.unflatten θ) b₄ γ₄ β₄ W₅
          b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convB_hasGradAt cotN W₄ a3 b₄ hC4).congr_of_eventuallyEq (germ2
      (fun θ => bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ β₄
          (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ θ a3)) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₄ hq.h4 γ₄ β₄).continuous.comp
        (conv2d_bias_differentiable W₄ (Tensor3.unflatten a3)).continuous).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ θ γ₄ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ θ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ θ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnGamma_hasGradAt vN epsStr cotN ε₄ γ₄ β₄ cc4 hB4).congr_of_eventuallyEq (germ2
      (fun θ => bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ θ β₄ cc4) _
      (bnPC_gamma_continuous ε₄ β₄ cc4).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ θ β₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ θ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ θ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnBeta_hasGradAt cotN ε₄ γ₄ β₄ cc4 hB4).congr_of_eventuallyEq (germ2
      (fun θ => bnPerChannelTensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε₄ γ₄ θ cc4) _
      (bnPC_beta_continuous ε₄ γ₄ cc4).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ θ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ θ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ θ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₅ pl2 W₅ hC5).congr_of_eventuallyEq (germ3
      (fun θ => bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆
          (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
        (relu _ (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₅ γ₅ β₅
            (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) (Kernel4.unflatten θ) b₅ pl2))))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₆ hq.h6 γ₆ β₆).continuous.comp
        ((flatConv_differentiable W₆ b₆).continuous.comp ((relu_continuous _).comp
        ((bnPerChannelTensor3_differentiable _ _ _ ε₅ hq.h5 γ₅ β₅).continuous.comp
        (conv2d_weight_differentiable b₅ (Tensor3.unflatten pl2)).continuous)))).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ (Kernel4.unflatten θ)
          b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ (Kernel4.unflatten θ)
          b₅ γ₅ β₅ W₆ b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convB_hasGradAt cotN W₅ pl2 b₅ hC5).congr_of_eventuallyEq (germ3
      (fun θ => bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆
          (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
        (relu _ (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₅ γ₅ β₅
            (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₅ θ pl2))))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₆ hq.h6 γ₆ β₆).continuous.comp
        ((flatConv_differentiable W₆ b₆).continuous.comp ((relu_continuous _).comp
        ((bnPerChannelTensor3_differentiable _ _ _ ε₅ hq.h5 γ₅ β₅).continuous.comp
        (conv2d_bias_differentiable W₅ (Tensor3.unflatten pl2)).continuous)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ θ γ₅ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ θ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnGamma_hasGradAt vN epsStr cotN ε₅ γ₅ β₅ cc5 hB5).congr_of_eventuallyEq (germ3
      (fun θ => bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆
          (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
        (relu _ (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₅ θ β₅ cc5)))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₆ hq.h6 γ₆ β₆).continuous.comp
        ((flatConv_differentiable W₆ b₆).continuous.comp ((relu_continuous _).comp
        (bnPC_gamma_continuous ε₅ β₅ cc5)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ θ β₅ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ θ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnBeta_hasGradAt cotN ε₅ γ₅ β₅ cc5 hB5).congr_of_eventuallyEq (germ3
      (fun θ => bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆
          (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
        (relu _ (bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₅ γ₅ θ cc5)))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₆ hq.h6 γ₆ β₆).continuous.comp
        ((flatConv_differentiable W₆ b₆).continuous.comp ((relu_continuous _).comp
        (bnPC_beta_continuous ε₅ γ₅ cc5)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ θ W₆ b₆ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ θ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₆ a5 W₆ hC6).congr_of_eventuallyEq (germ3
      (fun θ => bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆
          (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) (Kernel4.unflatten θ) b₆ a5)) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₆ hq.h6 γ₆ β₆).continuous.comp
        (conv2d_weight_differentiable b₆ (Tensor3.unflatten a5)).continuous).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅
          (Kernel4.unflatten θ) b₆ γ₆ β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅
          (Kernel4.unflatten θ) b₆ γ₆ β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convB_hasGradAt cotN W₆ a5 b₆ hC6).congr_of_eventuallyEq (germ3
      (fun θ => bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ β₆
          (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ θ a5)) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₆ hq.h6 γ₆ β₆).continuous.comp
        (conv2d_bias_differentiable W₆ (Tensor3.unflatten a5)).continuous).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ θ γ₆
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ θ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnGamma_hasGradAt vN epsStr cotN ε₆ γ₆ β₆ cc6 hB6).congr_of_eventuallyEq (germ3
      (fun θ => bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ θ β₆ cc6) _
      (bnPC_gamma_continuous ε₆ β₆ cc6).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ θ
          β₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ θ
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnBeta_hasGradAt cotN ε₆ γ₆ β₆ cc6 hB6).congr_of_eventuallyEq (germ3
      (fun θ => bnPerChannelTensor3 c3 (2 * (2 * h)) (2 * (2 * w)) ε₆ γ₆ θ cc6) _
      (bnPC_beta_continuous ε₆ γ₆ cc6).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          θ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          θ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₇ pl3 W₇ hC7).congr_of_eventuallyEq (germ4
      (fun θ => bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈
          (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈
        (relu _ (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₇ γ₇ β₇
            (flatConv (h := 2 * h) (w := 2 * w) (Kernel4.unflatten θ) b₇ pl3))))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₈ hq.h8 γ₈ β₈).continuous.comp
        ((flatConv_differentiable W₈ b₈).continuous.comp ((relu_continuous _).comp
        ((bnPerChannelTensor3_differentiable _ _ _ ε₇ hq.h7 γ₇ β₇).continuous.comp
        (conv2d_weight_differentiable b₇ (Tensor3.unflatten pl3)).continuous)))).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ (Kernel4.unflatten θ) b₇ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convB_hasGradAt cotN W₇ pl3 b₇ hC7).congr_of_eventuallyEq (germ4
      (fun θ => bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈
          (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈
        (relu _ (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₇ γ₇ β₇
            (flatConv (h := 2 * h) (w := 2 * w) W₇ θ pl3))))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₈ hq.h8 γ₈ β₈).continuous.comp
        ((flatConv_differentiable W₈ b₈).continuous.comp ((relu_continuous _).comp
        ((bnPerChannelTensor3_differentiable _ _ _ ε₇ hq.h7 γ₇ β₇).continuous.comp
        (conv2d_bias_differentiable W₇ (Tensor3.unflatten pl3)).continuous)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ θ γ₇ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnGamma_hasGradAt vN epsStr cotN ε₇ γ₇ β₇ cc7 hB7).congr_of_eventuallyEq (germ4
      (fun θ => bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈
          (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈
        (relu _ (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₇ θ β₇ cc7)))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₈ hq.h8 γ₈ β₈).continuous.comp
        ((flatConv_differentiable W₈ b₈).continuous.comp ((relu_continuous _).comp
        (bnPC_gamma_continuous ε₇ β₇ cc7)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ θ β₇ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (bnBeta_hasGradAt cotN ε₇ γ₇ β₇ cc7 hB7).congr_of_eventuallyEq (germ4
      (fun θ => bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈
          (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈
        (relu _ (bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₇ γ₇ θ cc7)))) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₈ hq.h8 γ₈ β₈).continuous.comp
        ((flatConv_differentiable W₈ b₈).continuous.comp ((relu_continuous _).comp
        (bnPC_beta_continuous ε₇ γ₇ cc7)))).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ θ W₈ b₈ γ₈ β₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₈ a7 W₈ hC8).congr_of_eventuallyEq (germ4
      (fun θ => bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈
          (flatConv (h := 2 * h) (w := 2 * w) (Kernel4.unflatten θ) b₈ a7)) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₈ hq.h8 γ₈ β₈).continuous.comp
        (conv2d_weight_differentiable b₈ (Tensor3.unflatten a7)).continuous).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ (Kernel4.unflatten θ) b₈ γ₈ β₈ ci)).symm
  · refine (convB_hasGradAt cotN W₈ a7 b₈ hC8).congr_of_eventuallyEq (germ4
      (fun θ => bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ β₈
          (flatConv (h := 2 * h) (w := 2 * w) W₈ θ a7)) _
      ((bnPerChannelTensor3_differentiable _ _ _ ε₈ hq.h8 γ₈ β₈).continuous.comp
        (conv2d_bias_differentiable W₈ (Tensor3.unflatten a7)).continuous).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ θ γ₈ β₈ ci)).symm
  · refine (bnGamma_hasGradAt vN epsStr cotN ε₈ γ₈ β₈ cc8 hB8).congr_of_eventuallyEq (germ4
      (fun θ => bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ θ β₈ cc8) _
      (bnPC_gamma_continuous ε₈ β₈ cc8).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ θ β₈ ci)).symm
  · refine (bnBeta_hasGradAt cotN ε₈ γ₈ β₈ cc8 hB8).congr_of_eventuallyEq (germ4
      (fun θ => bnPerChannelTensor3 c4 (2 * h) (2 * w) ε₈ γ₈ θ cc8) _
      (bnPC_beta_continuous ε₈ γ₈ cc8).continuousAt
      rfl
      (fun θ ci p q hpq => hpq W₁ b₁ γ₁ β₁ W₂ b₂ γ₂ β₂ W₃ b₃ γ₃ β₃ W₄ b₄ γ₄ β₄ W₅ b₅ γ₅ β₅ W₆ b₆ γ₆
          β₆ W₇ b₇ γ₇ β₇ W₈ b₈ γ₈ θ ci)).symm

/-- **The artifact's loss**: every node is the gradient of the softmax cross-entropy at `label`,
    `g` the emitted loss cotangent. -/
theorem cifar8Bn_net_lossGrad_CE (xN vN epsStr cotN nlogN ohN : String)
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (hq : Cifar8BnPos ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (σ₁ : Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin c4 → Fin h → Fin w → Fin 2 × Fin 2)
    (hx : Cifar8BnLossSmoothAt W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅
        γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba x σ₁ σ₂ σ₃ σ₄)
        (label : Fin nClasses) :
    Cifar8BnNetLossTied xN vN epsStr cotN W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄
        β₄ W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb x σ₁ σ₂ σ₃
        σ₄
      (fun z _ => crossEntropy nClasses z label)
      (den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN
          (cifarCnnBn8Forward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅
              γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb x))))
        (.operand ohN (oneHot nClasses label)))) := by
  rw [softmaxCELossCot_den]
  exact cifar8Bn_net_lossGrad xN vN epsStr cotN hkH hkW W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃
      W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb
      hq x σ₁ σ₂ σ₃ σ₄ hx
    (hasGradAt_crossEntropy label _)

end Net

end Proofs.Cifar8BnTieG
