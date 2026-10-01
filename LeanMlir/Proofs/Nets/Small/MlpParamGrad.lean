import LeanMlir.Proofs.Nets.Small.MlpFold
import LeanMlir.Proofs.Nets.Small.SmallParamGrad

/-! # The MNIST MLP — every parameter gradient node IS the loss's derivative

`mlp_train_step_tied_certified` ties each of the six SGD updates to the certified per-layer
Jacobian contracted with the cotangent the emitted chain threads to it (`mlpCotOut1`,
`mlpCotOut0`), and leaves open whether that cotangent is the loss gradient at the hidden layers.
`mlp_net_lossGrad` closes it: at the same cotangents, the un-fused `weightGrad` / `biasGrad` node
of each layer is the gradient of the loss in that parameter, for any loss `L` of the logits with
gradient `g` there; `mlp_net_lossGrad_CE` instantiates it at the softmax cross-entropy the render
emits. The fused `weightSgd` / `biasSgd` ops are `θ − lr·` these nodes
(`SmallParamGrad.weightSgd_eq_grad`, `SmallParamGrad.biasSgd_eq_grad`).

**How.** The loss read at the logits is pulled back one certified stage at a time
(`SmallParamGrad.hasGradAt_dense`, `SmallParamGrad.hasGradAt_relu`), each landing on the emitted
chain's cotangent; at each layer's output the node lemma (`SmallParamGrad.denseW_hasGradAt`)
turns it into the parameter gradient.

**Hypotheses.** Both hidden pre-activations off the ReLU kink (the pair `mlpHasVJPAt` takes).
**Scope.** One example (the emitted module batch-contracts; `den` is per-example).
-/

open Proofs Proofs.StableHLO Proofs.IR Proofs.SmallParamGrad

namespace Proofs.MlpFold

variable {d₀ d₁ d₂ d₃ : Nat}

/-- **Every MLP parameter node is the gradient of `L`** in that parameter: the six nodes, each at
    the cotangent the emitted chain threads to its layer (`g` at the logits, `mlpCotOut1`,
    `mlpCotOut0`), stated against `L` of `mlpForward` with that one parameter varied. -/
def MlpNetLossTied (aN cotN : String) (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (L : Vec d₃ → Vec 1)
    (g : Vec d₃) : Prop :=
  let p₀ := dense W₀ b₀ x
  let p₁ := dense W₁ b₁ (relu d₁ p₀)
  let c₁ := (mlpCotOut1 W₂ p₁).denote g
  let c₀ := (mlpCotOut0 W₁ W₂ p₀ p₁).denote g
  HasGradAt (fun θ => L (mlpForward (Mat.unflatten θ) b₀ W₁ b₁ W₂ b₂ x)) (Mat.flatten W₀)
      (den (SHlo.weightGrad aN x (.operand cotN c₀)))
  ∧ HasGradAt (fun θ => L (mlpForward W₀ θ W₁ b₁ W₂ b₂ x)) b₀
      (den (SHlo.biasGrad (.operand cotN c₀)))
  ∧ HasGradAt (fun θ => L (mlpForward W₀ b₀ (Mat.unflatten θ) b₁ W₂ b₂ x)) (Mat.flatten W₁)
      (den (SHlo.weightGrad aN (relu d₁ p₀) (.operand cotN c₁)))
  ∧ HasGradAt (fun θ => L (mlpForward W₀ b₀ W₁ θ W₂ b₂ x)) b₁
      (den (SHlo.biasGrad (.operand cotN c₁)))
  ∧ HasGradAt (fun θ => L (mlpForward W₀ b₀ W₁ b₁ (Mat.unflatten θ) b₂ x)) (Mat.flatten W₂)
      (den (SHlo.weightGrad aN (relu d₂ p₁) (.operand cotN g)))
  ∧ HasGradAt (fun θ => L (mlpForward W₀ b₀ W₁ b₁ W₂ θ x)) b₂
      (den (SHlo.biasGrad (.operand cotN g)))

/-- **Every MLP parameter node is the gradient of `L` in that parameter**, whenever `g` is `L`'s
    gradient at the logits and both hidden pre-activations are off the ReLU kink. -/
theorem mlp_net_lossGrad (aN cotN : String) (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀)
    (h₀ : ∀ k, dense W₀ b₀ x k ≠ 0) (h₁ : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0)
    {L : Vec d₃ → Vec 1} {g : Vec d₃} (hL : HasGradAt L (mlpForward W₀ b₀ W₁ b₁ W₂ b₂ x) g) :
    MlpNetLossTied aN cotN W₀ b₀ W₁ b₁ W₂ b₂ x L g := by
  unfold MlpNetLossTied
  intro p₀ p₁ c₁ c₀
  have hL₂ : HasGradAt L (dense W₂ b₂ (relu d₂ p₁)) g := hL
  have hP₁ : HasGradAt (fun y => L (dense W₂ b₂ (relu d₂ y))) p₁ c₁ :=
    (hasGradAt_relu p₁ h₁ (hasGradAt_dense W₂ b₂ _ hL₂)).of_eq (denote_subst _ _ g).symm
  have hP₀ : HasGradAt (fun y => L (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ y))))) p₀ c₀ :=
    (hasGradAt_relu p₀ h₀ (hasGradAt_dense W₁ b₁ _ hP₁)).of_eq
      (by simp only [c₀, mlpCotOut0, denote_subst]; rfl)
  exact ⟨denseW_hasGradAt aN cotN x W₀ b₀ hP₀, denseB_hasGradAt cotN W₀ x b₀ hP₀,
    denseW_hasGradAt aN cotN _ W₁ b₁ hP₁, denseB_hasGradAt cotN W₁ _ b₁ hP₁,
    denseW_hasGradAt aN cotN _ W₂ b₂ hL₂, denseB_hasGradAt cotN W₂ _ b₂ hL₂⟩

/-- **The artifact's loss**: every node is the gradient of the softmax cross-entropy at `label`,
    `g` the emitted loss cotangent. -/
theorem mlp_net_lossGrad_CE (aN cotN nlogN ohN : String) (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃)
    (h₀ : ∀ k, dense W₀ b₀ x k ≠ 0) (h₁ : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0) :
    MlpNetLossTied aN cotN W₀ b₀ W₁ b₁ W₂ b₂ x (fun z _ => crossEntropy d₃ z label)
      (den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe
          (.operand nlogN (mlpForward W₀ b₀ W₁ b₁ W₂ b₂ x))))
        (.operand ohN (oneHot d₃ label)))) := by
  rw [softmaxCELossCot_den]
  exact mlp_net_lossGrad aN cotN W₀ b₀ W₁ b₁ W₂ b₂ x h₀ h₁ (hasGradAt_crossEntropy label _)

end Proofs.MlpFold
