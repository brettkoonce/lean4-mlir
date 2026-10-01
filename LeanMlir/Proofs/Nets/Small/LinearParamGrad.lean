import LeanMlir.Proofs.Nets.Small.LinearFold
import LeanMlir.Proofs.Nets.Small.SmallParamGrad

/-! # The linear classifier — both parameter gradient nodes ARE the loss's derivative

`LinearFold` ties the Chapter-1 train step's two SGD updates to the certified per-layer Jacobian
contracted with the emitted loss cotangent. Here that contraction, as the un-fused `weightGrad` /
`biasGrad` node, is the gradient of the loss in the parameter: `linear_net_lossGrad` for any loss
`L` of the logits with gradient `g` there, `linear_net_lossGrad_CE` at the softmax cross-entropy
the render emits, with `g` the emitted loss cotangent. The fused `weightSgd` / `biasSgd` ops are
`θ − lr·` these nodes (`SmallParamGrad.weightSgd_eq_grad`, `SmallParamGrad.biasSgd_eq_grad`).

**Scope.** One example (the emitted module batch-contracts; `den` is per-example).
-/

open Proofs Proofs.StableHLO Proofs.SmallParamGrad

namespace Proofs.LinFold

variable {m n : Nat}

/-- **Both linear-classifier parameter nodes are the gradient of `L`** in that parameter, at the
    cotangent `g` the emitted chain starts from. -/
def LinNetLossTied (aN cotN : String) (W : Mat m n) (b : Vec n) (x : Vec m)
    (L : Vec n → Vec 1) (g : Vec n) : Prop :=
  HasGradAt (fun θ => L (mnistLinear (Mat.unflatten θ) b x)) (Mat.flatten W)
      (den (SHlo.weightGrad aN x (.operand cotN g)))
  ∧ HasGradAt (fun θ => L (mnistLinear W θ x)) b (den (SHlo.biasGrad (.operand cotN g)))

/-- **Every linear-classifier parameter node is the gradient of `L`** whenever `g` is `L`'s
    gradient at the logits. No smoothness hypothesis: the net is affine. -/
theorem linear_net_lossGrad (aN cotN : String) (W : Mat m n) (b : Vec n) (x : Vec m)
    {L : Vec n → Vec 1} {g : Vec n} (hL : HasGradAt L (mnistLinear W b x) g) :
    LinNetLossTied aN cotN W b x L g :=
  ⟨denseW_hasGradAt aN cotN x W b hL, denseB_hasGradAt cotN W x b hL⟩

/-- **The artifact's loss**: both nodes are the gradient of the softmax cross-entropy at `label`,
    `g` the emitted loss cotangent. -/
theorem linear_net_lossGrad_CE (aN cotN nlogN ohN : String) (W : Mat m n) (b : Vec n) (x : Vec m)
    (label : Fin n) :
    LinNetLossTied aN cotN W b x (fun z _ => crossEntropy n z label)
      (den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN (mnistLinear W b x))))
        (.operand ohN (oneHot n label)))) := by
  rw [softmaxCELossCot_den]
  exact linear_net_lossGrad aN cotN W b x (hasGradAt_crossEntropy label _)

end Proofs.LinFold
