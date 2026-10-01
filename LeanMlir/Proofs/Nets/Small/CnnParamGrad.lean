import LeanMlir.Proofs.Nets.Small.CnnFold
import LeanMlir.Proofs.Nets.Small.SmallParamGrad

/-! # The MNIST CNN — every parameter gradient node IS the loss's derivative, at real MNIST

`cnn_train_step_tied_certified` ties each of the ten SGD updates to the certified per-layer
Jacobian contracted with the cotangent the emitted chain threads to it, and leaves open whether
that cotangent is the loss gradient below the output. `cnn_net_lossGrad` closes it: the un-fused
`*Grad` node of each layer, at the chain cotangent, is the gradient of the loss in that parameter
(`HasGradAt`), for any loss `L` of the logits with gradient `g` there; `cnn_net_lossGrad_CE`
instantiates it at the softmax cross-entropy the render emits. The fused `*Sgd` ops are `θ − lr·`
these nodes (`SmallParamGrad.convWeightSgd_eq_grad` and its peers).

**The pool's clause is stated for the parameters, not the image.** On real MNIST almost every
image has a 2×2 window whose positive maximum sits at two cells, because the two cells read
identical input (a constant background patch): the net has no derivative in its input there, and
the pool none in its activation. But two such cells are the same function of the conv weights
(`CnnPoolTwin`, implied by identical two-layer receptive fields, `cnnPoolTwin_of_convPatchEq2`),
so along any parameter the pooled ReLU is the gather at a fixed selection
(`SmallParamGrad.maxPool_relu_eventuallyEq_sel`), and the loss IS differentiable in the
parameters. `CnnLossSmoothAt` allows exactly those ties. The probe
scripts/probes/mnist_pool_twin_probe.py checks this clause on the MNIST test set.

**Which cotangent.** At a tied window the pool's backward must pick one cell. The rendered
`select_and_scatter` (select = `GE`) does: it routes each window's cotangent to the window's first
maximal cell, the gather's adjoint. The capstone is stated at any selection `σ` naming a maximum of
every window (`cnnChainCotW2Sel`, `SmallParamGrad.PoolSelDom`). The step tie's `cnnChainCotW2`
reads the pool backward as `maxPoolBackDenote`, which routes to the first argmax
(`maxPool2Argmax`), so it IS `cnnChainCotW2Sel` at that selection, at every point
(`cnnChainCotW2_eq_sel`); `SmallParamGrad.poolSelDom_argmax` discharges the selection clause there.

**How.** The loss read at the logits is pulled back through the dense head
(`SmallParamGrad.hasGradAt_dense`, `SmallParamGrad.hasGradAt_relu`), through the pool as the
gather at `σ` (`SmallParamGrad.hasGradAt_gatherRelu`; at the point itself the pool IS that gather),
then through the convs (`SmallParamGrad.hasGradAt_conv`). Each conv node is the gather model's
parameter gradient, moved to the real net by the germ (`HasGradAt.congr_of_eventuallyEq`).

**Hypotheses.** Odd kernels (the rendered conv backward is the conv VJP there), and
`CnnLossSmoothAt`: every ReLU off its kink, and every pool window dead or tied only between twins.
**Scope.** One example (the emitted module batch-contracts; `den` is per-example).
-/

open Proofs Proofs.StableHLO Proofs.IR Proofs.SmallParamGrad

namespace Proofs.CnnFold

open scoped BigOperators

section Twins
variable {ic h w : Nat}

/-- The conv2 pre-activation, the input of the pool's ReLU. -/
noncomputable def cnnPoolPre {c kH kW : Nat} (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (x : Vec (ic * (2 * h) * (2 * w))) :
    Vec (c * (2 * h) * (2 * w)) :=
  flatConv (h := 2 * h) (w := 2 * w) W₂ b₂
    (relu (c * (2 * h) * (2 * w)) (flatConv (h := 2 * h) (w := 2 * w) W₁ b₁ x))

/-- **Two pool-input cells are twins**: they are equal at every conv weight, in every channel. At
    such a pair the pool can tie at a positive maximum, and the tie is the same at every weight,
    which is why the parameter gradient survives it. -/
def CnnPoolTwin (c kH kW : Nat) (x : Vec (ic * (2 * h) * (2 * w)))
    (p q : Fin (2 * h) × Fin (2 * w)) : Prop :=
  ∀ (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (ci : Fin c),
    cnnPoolPre W₁ b₁ W₂ b₂ x (t3Idx ci p.1 p.2) = cnnPoolPre W₁ b₁ W₂ b₂ x (t3Idx ci q.1 q.2)

/-- Cells with identical two-layer receptive fields in the image are twins. -/
theorem cnnPoolTwin_of_convPatchEq2 {c kH kW : Nat} {x : Vec (ic * (2 * h) * (2 * w))}
    {p q : Fin (2 * h) × Fin (2 * w)} (hpq : ConvPatchEq2 kH kW (Tensor3.unflatten x) p q) :
    CnnPoolTwin c kH kW x p q := by
  intro W₁ b₁ W₂ b₂ ci
  unfold cnnPoolPre flatConv
  rw [flatten_t3Idx, flatten_t3Idx]
  exact conv2d_eq_of_convPatchEq (convPatchEq_relu_conv hpq W₁ b₁) W₂ b₂ ci

end Twins

/-- **The conv2-output cotangent at a pool selection `σ`**: `cnnChainCotW2` with the pool backward
    routing each window's cotangent to the ONE cell `σ` names (`selScatter`), as the rendered
    `select_and_scatter` does, then the ReLU mask. -/
noncomputable def cnnChainCotW2Sel {c h w d1 nClasses : Nat}
    (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (W₃ : Mat (c * h * w) d1) (W₄ : Mat d1 d1) (W₅ : Mat d1 nClasses) (h3 h4 : Vec d1)
    (hc2 : Vec (c * (2 * h) * (2 * w))) (dy : Vec nClasses) : Vec (c * (2 * h) * (2 * w)) :=
  fun i => if hc2 i > 0
    then selScatter (poolSelIdx σ) ((cnnDenseHeadCot W₃ W₄ W₅ h3 h4).denote dy) i
    else 0

/-- **The step tie's conv₂ cotangent is the capstone's at the first argmax.** The rendered pool
    backward routes each window to its first maximum (`maxPool2Argmax`), so `cnnChainCotW2` is
    `cnnChainCotW2Sel` at that selection, at every point, ties included. -/
theorem cnnChainCotW2_eq_sel {c h w d1 nClasses : Nat}
    (W₃ : Mat (c * h * w) d1) (W₄ : Mat d1 d1) (W₅ : Mat d1 nClasses) (h3 h4 : Vec d1)
    (ac2 : Tensor3 c (2 * h) (2 * w)) (hc2 : Vec (c * (2 * h) * (2 * w))) (dy : Vec nClasses) :
    cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy
      = cnnChainCotW2Sel (maxPool2Argmax ac2) W₃ W₄ W₅ h3 h4 hc2 dy := by
  funext i
  simp only [cnnChainCotW2, cnnChainCotW2Sel, maxpool_flatDenote_eq_selScatter]

section Net
variable {ic c h w d1 nClasses kH kW : Nat}

/-- **The smooth-point bundle the loss gradient needs.** Every ReLU off its kink; every pool window
    dead or tied only between twins (`CnnPoolTwin`); and `σ` names a maximum of every window. -/
structure CnnLossSmoothAt (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW)
    (b₂ : Vec c) (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (x : Vec (ic * (2 * h) * (2 * w))) (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2) : Prop where
  z1 : ∀ k, flatConv (h := 2 * h) (w := 2 * w) W₁ b₁ x k ≠ 0
  z2 : ∀ k, cnnPoolPre W₁ b₁ W₂ b₂ x k ≠ 0
  pool : MaxPool2SmoothUpTo (CnnPoolTwin c kH kW x)
    (Tensor3.unflatten (cnnPoolPre W₁ b₁ W₂ b₂ x) : Tensor3 c (2 * h) (2 * w))
  sel : PoolSelDom σ (relu _ (cnnPoolPre W₁ b₁ W₂ b₂ x))
  z3 : ∀ k, dense W₃ b₃ (maxPoolFlat c h w (relu _ (cnnPoolPre W₁ b₁ W₂ b₂ x))) k ≠ 0
  z4 : ∀ k, dense W₄ b₄ (relu d1 (dense W₃ b₃
    (maxPoolFlat c h w (relu _ (cnnPoolPre W₁ b₁ W₂ b₂ x))))) k ≠ 0

/-- **Every MNIST-CNN parameter node is the gradient of `L`** in that parameter: the ten un-fused
    nodes, each at the cotangent the chain threads to its layer (the head's `mlpCotOut1` /
    `mlpCotOut0`, the pool routed at `σ`, the conv backward), stated against `L` of
    `mnistCnnNoBnForward` with that one parameter varied. -/
def CnnNetLossTied (xN cotN : String) (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1) (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
    (x : Vec (ic * (2 * h) * (2 * w))) (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (L : Vec nClasses → Vec 1) (g : Vec nClasses) : Prop :=
  let z1 := flatConv (h := 2 * h) (w := 2 * w) W₁ b₁ x
  let a1 := relu (c * (2 * h) * (2 * w)) z1
  let z2 := flatConv (h := 2 * h) (w := 2 * w) W₂ b₂ a1
  let pl := maxPoolFlat c h w (relu (c * (2 * h) * (2 * w)) z2)
  let h3 := dense W₃ b₃ pl
  let h4 := dense W₄ b₄ (relu d1 h3)
  let cotH4 := (mlpCotOut1 W₅ h4).denote g
  let cotH3 := (mlpCotOut0 W₄ W₅ h3 h4).denote g
  let cotZ2 := cnnChainCotW2Sel σ W₃ W₄ W₅ h3 h4 z2 g
  let cotZ1 := cnnChainCotW1 W₂ z1 cotZ2
  -- conv₁, conv₂
  HasGradAt (fun θ => L (mnistCnnNoBnForward (Kernel4.unflatten θ) b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x))
      (Kernel4.flatten W₁)
      (den (SHlo.convWeightGrad xN b₁ (Tensor3.unflatten x) W₁ (.operand cotN cotZ1)))
  ∧ HasGradAt (fun θ => L (mnistCnnNoBnForward W₁ θ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x)) b₁
      (den (SHlo.convBiasGrad W₁ (Tensor3.unflatten x) b₁ (.operand cotN cotZ1)))
  ∧ HasGradAt (fun θ => L (mnistCnnNoBnForward W₁ b₁ (Kernel4.unflatten θ) b₂ W₃ b₃ W₄ b₄ W₅ b₅ x))
      (Kernel4.flatten W₂)
      (den (SHlo.convWeightGrad xN b₂ (Tensor3.unflatten a1) W₂ (.operand cotN cotZ2)))
  ∧ HasGradAt (fun θ => L (mnistCnnNoBnForward W₁ b₁ W₂ θ W₃ b₃ W₄ b₄ W₅ b₅ x)) b₂
      (den (SHlo.convBiasGrad W₂ (Tensor3.unflatten a1) b₂ (.operand cotN cotZ2)))
  -- the dense head
  ∧ HasGradAt (fun θ => L (mnistCnnNoBnForward W₁ b₁ W₂ b₂ (Mat.unflatten θ) b₃ W₄ b₄ W₅ b₅ x))
      (Mat.flatten W₃) (den (SHlo.weightGrad xN pl (.operand cotN cotH3)))
  ∧ HasGradAt (fun θ => L (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ θ W₄ b₄ W₅ b₅ x)) b₃
      (den (SHlo.biasGrad (.operand cotN cotH3)))
  ∧ HasGradAt (fun θ => L (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ (Mat.unflatten θ) b₄ W₅ b₅ x))
      (Mat.flatten W₄) (den (SHlo.weightGrad xN (relu d1 h3) (.operand cotN cotH4)))
  ∧ HasGradAt (fun θ => L (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ θ W₅ b₅ x)) b₄
      (den (SHlo.biasGrad (.operand cotN cotH4)))
  ∧ HasGradAt (fun θ => L (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ (Mat.unflatten θ) b₅ x))
      (Mat.flatten W₅) (den (SHlo.weightGrad xN (relu d1 h4) (.operand cotN g)))
  ∧ HasGradAt (fun θ => L (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ θ x)) b₅
      (den (SHlo.biasGrad (.operand cotN g)))

/-- **Every MNIST-CNN parameter node is the gradient of `L` in that parameter**, whenever `g` is
    `L`'s gradient at the logits.

    Hypotheses: odd kernels, every ReLU off its kink, and every pool window dead or tied only
    between cells that are the same function of the conv weights (`CnnLossSmoothAt`), with `σ`
    naming a maximum of every window. -/
theorem cnn_net_lossGrad (xN cotN : String) (hkH : 2 * ((kH - 1) / 2) + 1 = kH)
    (hkW : 2 * ((kW - 1) / 2) + 1 = kW) (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1) (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
    (x : Vec (ic * (2 * h) * (2 * w))) (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (hx : CnnLossSmoothAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x σ)
    {L : Vec nClasses → Vec 1} {g : Vec nClasses}
    (hL : HasGradAt L (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x) g) :
    CnnNetLossTied xN cotN W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x σ L g := by
  unfold CnnNetLossTied
  intro z1 a1 z2 pl h3 h4 cotH4 cotH3 cotZ2 cotZ1
  -- the dense head, as the MLP's
  have hL₅ : HasGradAt L (dense W₅ b₅ (relu d1 h4)) g := hL
  have hH4 : HasGradAt (fun y => L (dense W₅ b₅ (relu d1 y))) h4 cotH4 :=
    (hasGradAt_relu h4 hx.z4 (hasGradAt_dense W₅ b₅ _ hL₅)).of_eq (denote_subst _ _ g).symm
  have hH3 : HasGradAt (fun y => L (dense W₅ b₅ (relu d1 (dense W₄ b₄ (relu d1 y))))) h3 cotH3 :=
    (hasGradAt_relu h3 hx.z3 (hasGradAt_dense W₄ b₄ _ hH4)).of_eq
      (by simp only [cotH3, mlpCotOut0, denote_subst]; rfl)
  -- the gather model: the pool frozen at `σ`, which at the point itself IS the pool
  let Gp : Vec (c * h * w) → Vec 1 := fun u =>
    L (dense W₅ b₅ (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃ u)))))
  let G2 : Vec (c * (2 * h) * (2 * w)) → Vec 1 := fun y =>
    Gp (fun k => relu (c * (2 * h) * (2 * w)) y (poolSelIdx σ k))
  have hpt : pl = fun k => relu (c * (2 * h) * (2 * w)) z2 (poolSelIdx σ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ _ hx.sel
  have hPl : HasGradAt Gp pl ((emitDenseBack W₃).denote cotH3) := hasGradAt_dense W₃ b₃ pl hH3
  have hZ2 : HasGradAt G2 z2 cotZ2 :=
    (hasGradAt_gatherRelu (poolSelIdx σ) z2 hx.z2 (hPl.congr_point hpt)).of_eq (by
      funext i
      simp only [cotZ2, cnnChainCotW2Sel, cnnDenseHeadCot, cotH3, mlpCotOut0, mlpCotOut1,
        denote_subst]
      rfl)
  have hZ1 : HasGradAt (fun y => G2 (flatConv (h := 2 * h) (w := 2 * w) W₂ b₂
      (relu (c * (2 * h) * (2 * w)) y))) z1 cotZ1 :=
    hasGradAt_relu z1 hx.z1 (hasGradAt_conv hkH hkW W₂ b₂ a1 hZ2)
  -- along a conv parameter, the real net agrees with the gather model near the point
  have germ : ∀ {P : Nat} (Z : Vec P → Vec (c * (2 * h) * (2 * w))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = z2 →
      (∀ θ (ci : Fin c) (p q : Fin (2 * h) × Fin (2 * w)), CnnPoolTwin c kH kW x p q →
        Z θ (t3Idx ci p.1 p.2) = Z θ (t3Idx ci q.1 q.2)) →
      (fun θ => Gp (maxPoolFlat c h w (relu _ (Z θ)))) =ᶠ[nhds θ₀] fun θ => G2 (Z θ) := by
    intro P Z θ₀ hZc h0 hT
    have hg := maxPool_relu_eventuallyEq_sel Z σ (CnnPoolTwin c kH kW x) hT θ₀ hZc
      (by rw [h0]; exact hx.z2) (by rw [h0]; exact hx.pool) (by rw [h0]; exact hx.sel)
    filter_upwards [hg] with θ hθ
    exact congrArg Gp hθ
  refine ⟨?_, ?_, ?_, ?_, denseW_hasGradAt xN cotN pl W₃ b₃ hH3,
    denseB_hasGradAt cotN W₃ pl b₃ hH3, denseW_hasGradAt xN cotN _ W₄ b₄ hH4,
    denseB_hasGradAt cotN W₄ _ b₄ hH4, denseW_hasGradAt xN cotN _ W₅ b₅ hL₅,
    denseB_hasGradAt cotN W₅ _ b₅ hL₅⟩
  · refine (convW_hasGradAt xN cotN b₁ x W₁ hZ1).congr_of_eventuallyEq (germ
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) W₂ b₂ (relu _
        (flatConv (h := 2 * h) (w := 2 * w) (Kernel4.unflatten θ) b₁ x))) _
      ((flatConv_differentiable W₂ b₂).continuous.comp ((relu_continuous _).comp
        (conv2d_weight_differentiable b₁ (Tensor3.unflatten x)).continuous)).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq (Kernel4.unflatten θ) b₁ W₂ b₂ ci)).symm
  · refine (convB_hasGradAt cotN W₁ x b₁ hZ1).congr_of_eventuallyEq (germ
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) W₂ b₂ (relu _
        (flatConv (h := 2 * h) (w := 2 * w) W₁ θ x))) _
      ((flatConv_differentiable W₂ b₂).continuous.comp ((relu_continuous _).comp
        (conv2d_bias_differentiable W₁ (Tensor3.unflatten x)).continuous)).continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ θ W₂ b₂ ci)).symm
  · refine (convW_hasGradAt xN cotN b₂ a1 W₂ hZ2).congr_of_eventuallyEq (germ
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) (Kernel4.unflatten θ) b₂ a1) _
      (conv2d_weight_differentiable b₂ (Tensor3.unflatten a1)).continuous.continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ (Kernel4.unflatten θ) b₂ ci)).symm
  · exact (convB_hasGradAt cotN W₂ a1 b₂ hZ2).congr_of_eventuallyEq (germ
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) W₂ θ a1) _
      (conv2d_bias_differentiable W₂ (Tensor3.unflatten a1)).continuous.continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ θ ci)).symm

/-- **The artifact's loss**: every node is the gradient of the softmax cross-entropy at `label`,
    `g` the emitted loss cotangent. -/
theorem cnn_net_lossGrad_CE (xN cotN nlogN ohN : String) (hkH : 2 * ((kH - 1) / 2) + 1 = kH)
    (hkW : 2 * ((kW - 1) / 2) + 1 = kW) (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1) (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
    (x : Vec (ic * (2 * h) * (2 * w))) (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (hx : CnnLossSmoothAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x σ) (label : Fin nClasses) :
    CnnNetLossTied xN cotN W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x σ
      (fun z _ => crossEntropy nClasses z label)
      (den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe
          (.operand nlogN (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x))))
        (.operand ohN (oneHot nClasses label)))) := by
  rw [softmaxCELossCot_den]
  exact cnn_net_lossGrad xN cotN hkH hkW W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x σ hx
    (hasGradAt_crossEntropy label _)

end Net

end Proofs.CnnFold
