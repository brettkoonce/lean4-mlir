import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Nets.Small.CifarCNN

/-! # The chapter 3–4 graphs denote the chapter nets

`StableHLO` defines the MNIST-CNN and CIFAR forward graphs (`cnnFwdGraph`, `cifarFwdGraph`,
`cifar8FwdGraph`, `cifar8BnFwdGraph`) and the MNIST-CNN backward graph (`cnnBackGraph`) as `SHlo`
terms, and writes their artifacts. This file proves each denotes its net: the forwards are
`mnistCnnNoBnForward` / `cifarCnnForward` / `cifarCnn8Forward` / `cifarCnnBn8Forward`, and the
backward is the whole-network VJP `mnistCnnNoBn_has_vjp_at.backward` at a smooth point. Kept out of
`StableHLO` so the IR imports no net. -/

open Finset BigOperators

namespace Proofs
namespace StableHLO

/-- **CNN forward faithfulness.** The forward graph denotes the proven
    `mnistCnnNoBnForward`. -/
theorem cnnFwdGraph_faithful {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) :
    den (cnnFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x)
      = mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x := by
  simp only [cnnFwdGraph, mnistCnnNoBnForward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful, den_operand]

/-- **CIFAR-CNN forward faithfulness.** The forward graph denotes the proven
    `cifarCnnForward`. -/
theorem cifarFwdGraph_faithful {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) :
    den (cifarFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x)
      = cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x := by
  simp only [cifarFwdGraph, cifarCnnForward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful, den_operand]

/-- **Deeper (8-conv) CIFAR-CNN forward faithfulness.** The forward graph denotes the
    proven `cifarCnn8Forward`. -/
theorem cifar8FwdGraph_faithful {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) :
    den (cifar8FwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
          W₉ b₉ Wa ba Wb bb x)
      = cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
          W₉ b₉ Wa ba Wb bb x := by
  simp only [cifar8FwdGraph, cifarCnn8Forward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful, den_operand]

/-- **Deeper (8-conv) BN-CIFAR forward faithfulness.** The forward graph denotes the
    proven `cifarCnnBn8Forward`. -/
theorem cifar8BnFwdGraph_faithful {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat} (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) :
    den (cifar8BnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
          W₉ b₉ Wa ba Wb bb x)
      = cifarCnnBn8Forward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
          W₉ b₉ Wa ba Wb bb x := by
  simp only [cifar8BnFwdGraph, cifarCnnBn8Forward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful,
             bnPerChannelF_faithful, den_operand]

/-- Max-pool VJP at a *raw* flattened point (no `flatten ∘ unflatten` index), so
    it composes without a transport cast; `backward` is `maxPoolBackFlat`. The
    `correct` field reuses `maxPoolFlat_has_vjp_at.correct`, aligning the point
    via `Tensor3.flatten_unflatten`. -/
noncomputable def maxPoolFlat_has_vjp_at' {c h w : Nat} (v : Vec (c*(2*h)*(2*w)))
    (hs : MaxPool2Smooth (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))) :
    HasVJPAt (maxPoolFlat c h w) v where
  backward := maxPoolBackFlat c h w v
  correct := fun dy i => by
    have hbk : maxPoolBackFlat c h w v dy i
                = (maxPoolFlat_has_vjp_at (Tensor3.unflatten v) hs).backward dy i := by
      simp only [maxPoolFlat_has_vjp_at, hasVJPAt3_to_hasVJPAt, maxPool2_has_vjp_at3, maxPoolBackFlat]
    rw [hbk, (maxPoolFlat_has_vjp_at (Tensor3.unflatten v) hs).correct dy i,
        Tensor3.flatten_unflatten]

@[simp] theorem maxPoolFlat_has_vjp_at'_backward {c h w : Nat} (v : Vec (c*(2*h)*(2*w)))
    (hs : MaxPool2Smooth (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))) :
    (maxPoolFlat_has_vjp_at' v hs).backward = maxPoolBackFlat c h w v := rfl

-- **CNN backward faithfulness (smooth point) — A2c.** The whole-chain backward
-- graph denotes the proven conditional whole-network VJP
-- `mnistCnnNoBn_has_vjp_at.backward` (the Chapter-3 peer of
-- `mlpBackGraph_faithful`). The per-op `convBack`/`selectPos`/`dotOut` ops
-- assemble through `vjp_comp_at`; the one `maxPoolBack` matches via VJP
-- uniqueness (`HasVJPAt.backward_unique`) — sidestepping the `flatten∘unflatten`
-- transport in `mnistCnnNoBn_has_vjp_at`'s maxpool step.
theorem cnnBackGraph_faithful
    {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*h) * (2*w)))
    (h1 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₁ b₁ x k ≠ 0)
    (h2 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₂ b₂
            ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁) x) k ≠ 0)
    (h_mp : MaxPool2Smooth (Tensor3.unflatten
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x)
            : Tensor3 c (2*h) (2*w)))
    (h3 : ∀ k, dense W₃ b₃ (maxPoolFlat c h w
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x)) k ≠ 0)
    (h4 : ∀ k, dense W₄ b₄ ((relu d1 ∘ dense W₃ b₃) (maxPoolFlat c h w
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x))) k ≠ 0)
    (dy : Vec nClasses) :
    den (cnnBackGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ x dy)
      = (mnistCnnNoBn_has_vjp_at W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
          hc hh hw x h1 h2 h_mp h3 h4).backward dy := by
  simp only [cnnBackGraph, denStep, denStepApp, mnistCnnNoBn_has_vjp_at, convRelu_has_vjp_at,
    denseRelu_has_vjp_at, vjp_comp_at_backward, dense_has_vjp, relu_has_vjp_at,
    hasVJP3_to_hasVJP, HasVJP.toHasVJPAt, Mat.mulVec, id, Function.comp_apply]
  rw [HasVJPAt.backward_unique _ (maxPoolFlat_has_vjp_at'
        ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
          ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁) x)) h_mp)]
  rfl

end StableHLO
end Proofs
