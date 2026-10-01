import LeanMlir.Proofs.Nets.Small.CifarFold
import LeanMlir.Proofs.Nets.Small.CnnParamGrad

/-! # The CIFAR CNN — every parameter gradient node IS the loss's derivative, up to pool twins

`cifar_train_step_tied_certified` ties each of the fourteen SGD updates to the certified per-layer
Jacobian contracted with the cotangent the emitted chain threads to it. `cifar_net_lossGrad`
states that the un-fused `*Grad` node of each layer, at the chain cotangent, is the gradient of the
loss in that parameter, for any loss `L` of the logits with gradient `g` there;
`cifar_net_lossGrad_CE` instantiates it at the softmax cross-entropy the render emits.

The two pools are handled as the MNIST CNN's one (`CnnFold.cnn_net_lossGrad`): each pool's clause
allows ties between twins, cells equal at every weight upstream of that pool (`CnnFold.CnnPoolTwin`
for the first, `CifarPoolTwin2` for the second), and each pool's backward routes a window's
cotangent to the one cell a selection names (`CnnFold.cnnChainCotW2Sel`, `cifarChainCotW2Sel`), as
the rendered `select_and_scatter` does. At the first argmax of each window, the cell that op
picks, the step tie's own chains are these (`CnnFold.cnnChainCotW2_eq_sel`, `cifarChainCotW2_eq_sel`).
A parameter of the first stage sees both pools move; its
germ rewrites the outer pool first, at the true pre-activation, then the inner one.

**Hypotheses.** Odd kernels, every ReLU off its kink, every pool window dead or tied only between
twins, each selection naming a maximum of every window (`CifarLossSmoothAt`).
**Scope.** One example (the emitted module batch-contracts; `den` is per-example).
-/

open Proofs Proofs.StableHLO Proofs.IR Proofs.SmallParamGrad Proofs.CnnFold

namespace Proofs.CifarFold

open scoped BigOperators

section Stages
variable {ic c1 c2 h w d1 nClasses kH kW : Nat}

/-- From the first pool's pre-activation to the second's: ReLU, pool, conv₃, ReLU, conv₄. -/
noncomputable def cifarUp2 (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW)
    (b₄ : Vec c2) (z : Vec (c1 * (2 * (2 * h)) * (2 * (2 * w)))) : Vec (c2 * (2 * h) * (2 * w)) :=
  flatConv (h := 2 * h) (w := 2 * w) W₄ b₄ (relu (c2 * (2 * h) * (2 * w))
    (flatConv (h := 2 * h) (w := 2 * w) W₃ b₃
      (maxPoolFlat c1 (2 * h) (2 * w) (relu (c1 * (2 * (2 * h)) * (2 * (2 * w))) z))))

theorem cifarUp2_continuous (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW)
    (b₄ : Vec c2) : Continuous (cifarUp2 (h := h) (w := w) W₃ b₃ W₄ b₄) :=
  (flatConv_differentiable W₄ b₄).continuous.comp ((relu_continuous _).comp
    ((flatConv_differentiable W₃ b₃).continuous.comp
      ((maxPoolFlat_continuous _ _ _).comp (relu_continuous _))))

/-- The second pool's pre-activation (conv₄'s output). -/
noncomputable def cifarPre2 (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW)
    (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW)
    (b₄ : Vec c2) (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w)))) : Vec (c2 * (2 * h) * (2 * w)) :=
  cifarUp2 W₃ b₃ W₄ b₄ (cnnPoolPre W₁ b₁ W₂ b₂ x)

/-- **Two cells of the second pool's input are twins**: equal at every weight of the four convs,
    in every channel. -/
def CifarPoolTwin2 (c1 c2 kH kW : Nat) (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w))))
    (p q : Fin (2 * h) × Fin (2 * w)) : Prop :=
  ∀ (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ci : Fin c2),
    cifarPre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x (t3Idx ci p.1 p.2)
      = cifarPre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x (t3Idx ci q.1 q.2)

end Stages

/-- **The conv₂-output cotangent at a first-pool selection `σ₁`**: `cifarChainCotW2` with the pool
    backward routing each window's cotangent to the ONE cell `σ₁` names, then the ReLU mask. -/
noncomputable def cifarChainCotW2Sel {c1 c2 h w kH kW : Nat}
    (σ₁ : Fin c1 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2) (W₃ : Kernel4 c2 c1 kH kW)
    (hc2 : Vec (c1 * (2 * (2 * h)) * (2 * (2 * w)))) (cotW3 : Vec (c2 * (2 * h) * (2 * w))) :
    Vec (c1 * (2 * (2 * h)) * (2 * (2 * w))) :=
  fun i => if hc2 i > 0
    then selScatter (poolSelIdx σ₁)
      ((Back3.conv (c₁ := c2) (h₁ := 2 * h) (w₁ := 2 * w) W₃ Back3.cot).flatDenote cotW3) i
    else 0

/-- **The step tie's conv₂ cotangent is the capstone's at the first argmax** of the first pool's
    windows (`maxPool2Argmax`), the cell the rendered `select_and_scatter` picks, at every point. -/
theorem cifarChainCotW2_eq_sel {c1 c2 h w kH kW : Nat} (W₃ : Kernel4 c2 c1 kH kW)
    (ac2 : Tensor3 c1 (2 * (2 * h)) (2 * (2 * w))) (hc2 : Vec (c1 * (2 * (2 * h)) * (2 * (2 * w))))
    (cotW3 : Vec (c2 * (2 * h) * (2 * w))) :
    cifarChainCotW2 W₃ ac2 hc2 cotW3 = cifarChainCotW2Sel (maxPool2Argmax ac2) W₃ hc2 cotW3 := by
  funext i
  simp only [cifarChainCotW2, cifarChainCotW2Sel, maxpool_flatDenote_eq_selScatter]

section Net
variable {ic c1 c2 h w d1 nClasses kH kW : Nat}

/-- **The smooth-point bundle the loss gradient needs.** Every ReLU off its kink; every window of
    each pool dead or tied only between that pool's twins; each selection naming a maximum of
    every window. -/
structure CifarLossSmoothAt (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW)
    (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w))))
    (σ₁ : Fin c1 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin h → Fin w → Fin 2 × Fin 2) : Prop where
  z1 : ∀ k, flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₁ b₁ x k ≠ 0
  z2 : ∀ k, cnnPoolPre W₁ b₁ W₂ b₂ x k ≠ 0
  pool1 : MaxPool2SmoothUpTo (CnnPoolTwin c1 kH kW x)
    (Tensor3.unflatten (cnnPoolPre W₁ b₁ W₂ b₂ x) : Tensor3 c1 (2 * (2 * h)) (2 * (2 * w)))
  sel1 : PoolSelDom σ₁ (relu _ (cnnPoolPre W₁ b₁ W₂ b₂ x))
  z3 : ∀ k, flatConv (h := 2 * h) (w := 2 * w) W₃ b₃
    (maxPoolFlat c1 (2 * h) (2 * w) (relu _ (cnnPoolPre W₁ b₁ W₂ b₂ x))) k ≠ 0
  z4 : ∀ k, cifarPre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x k ≠ 0
  pool2 : MaxPool2SmoothUpTo (CifarPoolTwin2 c1 c2 kH kW x)
    (Tensor3.unflatten (cifarPre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x) : Tensor3 c2 (2 * h) (2 * w))
  sel2 : PoolSelDom σ₂ (relu _ (cifarPre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x))
  z5 : ∀ k, dense W₅ b₅ (maxPoolFlat c2 h w (relu _ (cifarPre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x))) k ≠ 0
  z6 : ∀ k, dense W₆ b₆ (relu d1 (dense W₅ b₅
    (maxPoolFlat c2 h w (relu _ (cifarPre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x))))) k ≠ 0

/-- **Every CIFAR-CNN parameter node is the gradient of `L`** in that parameter: the fourteen
    un-fused nodes, each at the cotangent the chain threads to its layer (each pool routed at its
    selection), stated against `L` of `cifarCnnForward` with that one parameter varied. -/
def CifarNetLossTied (xN cotN : String) (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1) (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w))))
    (σ₁ : Fin c1 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin h → Fin w → Fin 2 × Fin 2) (L : Vec nClasses → Vec 1) (g : Vec nClasses) :
    Prop :=
  let z1 := flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₁ b₁ x
  let a1 := relu (c1 * (2 * (2 * h)) * (2 * (2 * w))) z1
  let z2 := flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₂ b₂ a1
  let pl1 := maxPoolFlat c1 (2 * h) (2 * w) (relu (c1 * (2 * (2 * h)) * (2 * (2 * w))) z2)
  let z3 := flatConv (h := 2 * h) (w := 2 * w) W₃ b₃ pl1
  let a3 := relu (c2 * (2 * h) * (2 * w)) z3
  let z4 := flatConv (h := 2 * h) (w := 2 * w) W₄ b₄ a3
  let pl2 := maxPoolFlat c2 h w (relu (c2 * (2 * h) * (2 * w)) z4)
  let h5 := dense W₅ b₅ pl2
  let h6 := dense W₆ b₆ (relu d1 h5)
  let cotH6 := (mlpCotOut1 W₇ h6).denote g
  let cotH5 := (mlpCotOut0 W₆ W₇ h5 h6).denote g
  let cotZ4 := cnnChainCotW2Sel σ₂ W₅ W₆ W₇ h5 h6 z4 g
  let cotZ3 := cnnChainCotW1 W₄ z3 cotZ4
  let cotZ2 := cifarChainCotW2Sel σ₁ W₃ z2 cotZ3
  let cotZ1 := cnnChainCotW1 W₂ z1 cotZ2
  -- stage 1
  HasGradAt (fun θ => L (cifarCnnForward (Kernel4.unflatten θ) b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x))
      (Kernel4.flatten W₁)
      (den (SHlo.convWeightGrad xN b₁ (Tensor3.unflatten x) W₁ (.operand cotN cotZ1)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ θ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x)) b₁
      (den (SHlo.convBiasGrad W₁ (Tensor3.unflatten x) b₁ (.operand cotN cotZ1)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ (Kernel4.unflatten θ) b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x))
      (Kernel4.flatten W₂)
      (den (SHlo.convWeightGrad xN b₂ (Tensor3.unflatten a1) W₂ (.operand cotN cotZ2)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ θ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x)) b₂
      (den (SHlo.convBiasGrad W₂ (Tensor3.unflatten a1) b₂ (.operand cotN cotZ2)))
  -- stage 2
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ b₂ (Kernel4.unflatten θ) b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x))
      (Kernel4.flatten W₃)
      (den (SHlo.convWeightGrad xN b₃ (Tensor3.unflatten pl1) W₃ (.operand cotN cotZ3)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ b₂ W₃ θ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x)) b₃
      (den (SHlo.convBiasGrad W₃ (Tensor3.unflatten pl1) b₃ (.operand cotN cotZ3)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ (Kernel4.unflatten θ) b₄ W₅ b₅ W₆ b₆ W₇ b₇ x))
      (Kernel4.flatten W₄)
      (den (SHlo.convWeightGrad xN b₄ (Tensor3.unflatten a3) W₄ (.operand cotN cotZ4)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ θ W₅ b₅ W₆ b₆ W₇ b₇ x)) b₄
      (den (SHlo.convBiasGrad W₄ (Tensor3.unflatten a3) b₄ (.operand cotN cotZ4)))
  -- the dense head
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ (Mat.unflatten θ) b₅ W₆ b₆ W₇ b₇ x))
      (Mat.flatten W₅) (den (SHlo.weightGrad xN pl2 (.operand cotN cotH5)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ θ W₆ b₆ W₇ b₇ x)) b₅
      (den (SHlo.biasGrad (.operand cotN cotH5)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ (Mat.unflatten θ) b₆ W₇ b₇ x))
      (Mat.flatten W₆) (den (SHlo.weightGrad xN (relu d1 h5) (.operand cotN cotH6)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ θ W₇ b₇ x)) b₆
      (den (SHlo.biasGrad (.operand cotN cotH6)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ (Mat.unflatten θ) b₇ x))
      (Mat.flatten W₇) (den (SHlo.weightGrad xN (relu d1 h6) (.operand cotN g)))
  ∧ HasGradAt (fun θ => L (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ θ x)) b₇
      (den (SHlo.biasGrad (.operand cotN g)))

/-- **Every CIFAR-CNN parameter node is the gradient of `L` in that parameter**, whenever `g` is
    `L`'s gradient at the logits.

    Hypotheses: odd kernels, and `CifarLossSmoothAt` — every ReLU off its kink, every window of
    each pool dead or tied only between cells that are the same function of the weights upstream
    of it, each selection naming a maximum of every window. -/
theorem cifar_net_lossGrad (xN cotN : String) (hkH : 2 * ((kH - 1) / 2) + 1 = kH)
    (hkW : 2 * ((kW - 1) / 2) + 1 = kW) (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1) (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w))))
    (σ₁ : Fin c1 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin h → Fin w → Fin 2 × Fin 2)
    (hx : CifarLossSmoothAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ x σ₁ σ₂)
    {L : Vec nClasses → Vec 1} {g : Vec nClasses}
    (hL : HasGradAt L (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x) g) :
    CifarNetLossTied xN cotN W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x σ₁ σ₂ L g := by
  unfold CifarNetLossTied
  intro z1 a1 z2 pl1 z3 a3 z4 pl2 h5 h6 cotH6 cotH5 cotZ4 cotZ3 cotZ2 cotZ1
  -- the dense head
  have hL₇ : HasGradAt L (dense W₇ b₇ (relu d1 h6)) g := hL
  have hH6 : HasGradAt (fun y => L (dense W₇ b₇ (relu d1 y))) h6 cotH6 :=
    (hasGradAt_relu h6 hx.z6 (hasGradAt_dense W₇ b₇ _ hL₇)).of_eq (denote_subst _ _ g).symm
  have hH5 : HasGradAt (fun y => L (dense W₇ b₇ (relu d1 (dense W₆ b₆ (relu d1 y))))) h5 cotH5 :=
    (hasGradAt_relu h5 hx.z5 (hasGradAt_dense W₆ b₆ _ hH6)).of_eq
      (by simp only [cotH5, mlpCotOut0, denote_subst]; rfl)
  -- the gather model: each pool frozen at its selection
  let Gp2 : Vec (c2 * h * w) → Vec 1 := fun u =>
    L (dense W₇ b₇ (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ u)))))
  let G4 : Vec (c2 * (2 * h) * (2 * w)) → Vec 1 := fun y =>
    Gp2 (fun k => relu (c2 * (2 * h) * (2 * w)) y (poolSelIdx σ₂ k))
  let G2 : Vec (c1 * (2 * (2 * h)) * (2 * (2 * w))) → Vec 1 := fun y =>
    G4 (flatConv (h := 2 * h) (w := 2 * w) W₄ b₄ (relu (c2 * (2 * h) * (2 * w))
      (flatConv (h := 2 * h) (w := 2 * w) W₃ b₃
        (fun k => relu (c1 * (2 * (2 * h)) * (2 * (2 * w))) y (poolSelIdx σ₁ k)))))
  have hpt2 : pl2 = fun k => relu (c2 * (2 * h) * (2 * w)) z4 (poolSelIdx σ₂ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ₂ _ hx.sel2
  have hpt1 : pl1 = fun k => relu (c1 * (2 * (2 * h)) * (2 * (2 * w))) z2 (poolSelIdx σ₁ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ₁ _ hx.sel1
  have hZ4 : HasGradAt G4 z4 cotZ4 :=
    (hasGradAt_gatherRelu (poolSelIdx σ₂) z4 hx.z4
      ((hasGradAt_dense W₅ b₅ pl2 hH5).congr_point hpt2)).of_eq (by
      funext i
      simp only [cotZ4, cnnChainCotW2Sel, cnnDenseHeadCot, cotH5, mlpCotOut0, mlpCotOut1,
        denote_subst]
      rfl)
  have hZ3 : HasGradAt (fun y => G4 (flatConv (h := 2 * h) (w := 2 * w) W₄ b₄
      (relu (c2 * (2 * h) * (2 * w)) y))) z3 cotZ3 :=
    hasGradAt_relu z3 hx.z3 (hasGradAt_conv hkH hkW W₄ b₄ a3 hZ4)
  have hZ2 : HasGradAt G2 z2 cotZ2 :=
    (hasGradAt_gatherRelu (poolSelIdx σ₁) z2 hx.z2
      ((hasGradAt_conv hkH hkW W₃ b₃ pl1 hZ3).congr_point hpt1)).of_eq rfl
  have hZ1 : HasGradAt (fun y => G2 (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₂ b₂
      (relu (c1 * (2 * (2 * h)) * (2 * (2 * w))) y))) z1 cotZ1 :=
    hasGradAt_relu z1 hx.z1 (hasGradAt_conv hkH hkW W₂ b₂ a1 hZ2)
  -- along a parameter, the real net agrees with the gather model near the point: the second
  -- pool alone (a stage-2 parameter), then the first under it (a stage-1 parameter)
  have germ2 : ∀ {P : Nat} (Z : Vec P → Vec (c2 * (2 * h) * (2 * w))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = z4 →
      (∀ θ (ci : Fin c2) (p q : Fin (2 * h) × Fin (2 * w)), CifarPoolTwin2 c1 c2 kH kW x p q →
        Z θ (t3Idx ci p.1 p.2) = Z θ (t3Idx ci q.1 q.2)) →
      (fun θ => Gp2 (maxPoolFlat c2 h w (relu _ (Z θ)))) =ᶠ[nhds θ₀] fun θ => G4 (Z θ) := by
    intro P Z θ₀ hZc h0 hT
    have hg := maxPool_relu_eventuallyEq_sel Z σ₂ (CifarPoolTwin2 c1 c2 kH kW x) hT θ₀ hZc
      (by rw [h0]; exact hx.z4) (by rw [h0]; exact hx.pool2) (by rw [h0]; exact hx.sel2)
    filter_upwards [hg] with θ hθ
    exact congrArg Gp2 hθ
  have germ1 : ∀ {P : Nat} (Z : Vec P → Vec (c1 * (2 * (2 * h)) * (2 * (2 * w)))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = z2 →
      (∀ θ (ci : Fin c1) (p q : Fin (2 * (2 * h)) × Fin (2 * (2 * w))), CnnPoolTwin c1 kH kW x p q →
        Z θ (t3Idx ci p.1 p.2) = Z θ (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c2) (p q : Fin (2 * h) × Fin (2 * w)), CifarPoolTwin2 c1 c2 kH kW x p q →
        cifarUp2 W₃ b₃ W₄ b₄ (Z θ) (t3Idx ci p.1 p.2)
          = cifarUp2 W₃ b₃ W₄ b₄ (Z θ) (t3Idx ci q.1 q.2)) →
      (fun θ => Gp2 (maxPoolFlat c2 h w (relu _ (cifarUp2 W₃ b₃ W₄ b₄ (Z θ)))))
        =ᶠ[nhds θ₀] fun θ => G2 (Z θ) := by
    intro P Z θ₀ hZc h0 hT1 hT2
    refine (germ2 (fun θ => cifarUp2 W₃ b₃ W₄ b₄ (Z θ)) θ₀
      ((cifarUp2_continuous W₃ b₃ W₄ b₄).continuousAt.comp hZc) (by rw [h0]; rfl) hT2).trans ?_
    have hg := maxPool_relu_eventuallyEq_sel Z σ₁ (CnnPoolTwin c1 kH kW x) hT1 θ₀ hZc
      (by rw [h0]; exact hx.z2) (by rw [h0]; exact hx.pool1) (by rw [h0]; exact hx.sel1)
    filter_upwards [hg] with θ hθ
    exact congrArg (fun v => G4 (flatConv (h := 2 * h) (w := 2 * w) W₄ b₄
      (relu (c2 * (2 * h) * (2 * w)) (flatConv (h := 2 * h) (w := 2 * w) W₃ b₃ v)))) hθ
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, denseW_hasGradAt xN cotN pl2 W₅ b₅ hH5,
    denseB_hasGradAt cotN W₅ pl2 b₅ hH5, denseW_hasGradAt xN cotN _ W₆ b₆ hH6,
    denseB_hasGradAt cotN W₆ _ b₆ hH6, denseW_hasGradAt xN cotN _ W₇ b₇ hL₇,
    denseB_hasGradAt cotN W₇ _ b₇ hL₇⟩
  -- stage 1: both pools move
  · refine (convW_hasGradAt xN cotN b₁ x W₁ hZ1).congr_of_eventuallyEq (germ1
      (fun θ => flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₂ b₂ (relu _
        (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) (Kernel4.unflatten θ) b₁ x))) _
      ((flatConv_differentiable W₂ b₂).continuous.comp ((relu_continuous _).comp
        (conv2d_weight_differentiable b₁ (Tensor3.unflatten x)).continuous)).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq (Kernel4.unflatten θ) b₁ W₂ b₂ ci)
      (fun θ ci p q hpq => hpq (Kernel4.unflatten θ) b₁ W₂ b₂ W₃ b₃ W₄ b₄ ci)).symm
  · refine (convB_hasGradAt cotN W₁ x b₁ hZ1).congr_of_eventuallyEq (germ1
      (fun θ => flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₂ b₂ (relu _
        (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₁ θ x))) _
      ((flatConv_differentiable W₂ b₂).continuous.comp ((relu_continuous _).comp
        (conv2d_bias_differentiable W₁ (Tensor3.unflatten x)).continuous)).continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ θ W₂ b₂ ci)
      (fun θ ci p q hpq => hpq W₁ θ W₂ b₂ W₃ b₃ W₄ b₄ ci)).symm
  · refine (convW_hasGradAt xN cotN b₂ a1 W₂ hZ2).congr_of_eventuallyEq (germ1
      (fun θ => flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) (Kernel4.unflatten θ) b₂ a1) _
      (conv2d_weight_differentiable b₂ (Tensor3.unflatten a1)).continuous.continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ (Kernel4.unflatten θ) b₂ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ (Kernel4.unflatten θ) b₂ W₃ b₃ W₄ b₄ ci)).symm
  · refine (convB_hasGradAt cotN W₂ a1 b₂ hZ2).congr_of_eventuallyEq (germ1
      (fun θ => flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₂ θ a1) _
      (conv2d_bias_differentiable W₂ (Tensor3.unflatten a1)).continuous.continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ θ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ θ W₃ b₃ W₄ b₄ ci)).symm
  -- stage 2: the second pool moves
  · refine (convW_hasGradAt xN cotN b₃ pl1 W₃ hZ3).congr_of_eventuallyEq (germ2
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) W₄ b₄ (relu _
        (flatConv (h := 2 * h) (w := 2 * w) (Kernel4.unflatten θ) b₃ pl1))) _
      ((flatConv_differentiable W₄ b₄).continuous.comp ((relu_continuous _).comp
        (conv2d_weight_differentiable b₃ (Tensor3.unflatten pl1)).continuous)).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ (Kernel4.unflatten θ) b₃ W₄ b₄ ci)).symm
  · refine (convB_hasGradAt cotN W₃ pl1 b₃ hZ3).congr_of_eventuallyEq (germ2
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) W₄ b₄ (relu _
        (flatConv (h := 2 * h) (w := 2 * w) W₃ θ pl1))) _
      ((flatConv_differentiable W₄ b₄).continuous.comp ((relu_continuous _).comp
        (conv2d_bias_differentiable W₃ (Tensor3.unflatten pl1)).continuous)).continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ θ W₄ b₄ ci)).symm
  · refine (convW_hasGradAt xN cotN b₄ a3 W₄ hZ4).congr_of_eventuallyEq (germ2
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) (Kernel4.unflatten θ) b₄ a3) _
      (conv2d_weight_differentiable b₄ (Tensor3.unflatten a3)).continuous.continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ (Kernel4.unflatten θ) b₄ ci)).symm
  · exact (convB_hasGradAt cotN W₄ a3 b₄ hZ4).congr_of_eventuallyEq (germ2
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) W₄ θ a3) _
      (conv2d_bias_differentiable W₄ (Tensor3.unflatten a3)).continuous.continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ θ ci)).symm

/-- **The artifact's loss**: every node is the gradient of the softmax cross-entropy at `label`,
    `g` the emitted loss cotangent. -/
theorem cifar_net_lossGrad_CE (xN cotN nlogN ohN : String) (hkH : 2 * ((kH - 1) / 2) + 1 = kH)
    (hkW : 2 * ((kW - 1) / 2) + 1 = kW) (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1) (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w))))
    (σ₁ : Fin c1 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin h → Fin w → Fin 2 × Fin 2)
    (hx : CifarLossSmoothAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ x σ₁ σ₂) (label : Fin nClasses) :
    CifarNetLossTied xN cotN W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x σ₁ σ₂
      (fun z _ => crossEntropy nClasses z label)
      (den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe
          (.operand nlogN (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x))))
        (.operand ohN (oneHot nClasses label)))) := by
  rw [softmaxCELossCot_den]
  exact cifar_net_lossGrad xN cotN hkH hkW W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x σ₁ σ₂ hx
    (hasGradAt_crossEntropy label _)

end Net

end Proofs.CifarFold
