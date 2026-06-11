import LeanMlir.Proofs.StableHLO
import LeanMlir.Proofs.AdamStep

/-! # AdamW render-close for the linear net — Phase 3b den-level faithfulness

The Adam analogue of `StableHLO.sgdW_descends_certified_grad`
(`LeanMlir/Proofs/StableHLO.lean`). The SGD close defines the emitted update as a
*math function of the certified gradient* — `sgdW = W − lr·wGrad x (den cotGraph)`
— and proves it equals `θ − lr·(certified ∂/∂θ Jacobian · denoted softmax-CE
cotangent)`. This file does the same with the optimizer map swapped for AdamW
(`Proofs.adamWScalar`, whose ℝ spec is `Proofs.adamWParam`, op-for-op the rendered
`ViTRender.emitAdamV` graph): the emitted weight/bias output is `adamWScalar`
*driven by the same certified gradient*. The cotangent stays denoted through the
proven `den (lossCotGraph …)`; only the per-entry update law changes.

**Faithfulness only, no descent** — Adam is not monotone (AMSGrad
counterexample), so unlike `sgdW_descends_certified_grad` there is *no* attached
loss-decrease claim; the theorem certifies that the emitted update is exactly
`adamWScalar` of the certified gradient. `wGrad`/`bGrad` are layer-agnostic, so
(as for SGD) the same close lifts verbatim to the MLP/CNN param grads. -/

namespace Proofs
namespace StableHLO

variable {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)

/-- The emitted **weight** AdamW update — entrywise `adamWScalar` driven by the
    *certified* ∂/∂W gradient `wGrad x (den cotGraph)`. The Adam peer of `sgdW`;
    `mW`/`vW` are the incoming first/second moment buffers. -/
noncomputable def adamW (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (mW vW : Mat m n)
    (label : Fin n) : Mat m n :=
  fun i j => adamWScalar β₁ β₂ ε lr wd bc₁ bc₂ (W i j) (mW i j) (vW i j)
              (wGrad x (den (lossCotGraph W b x (oneHot n label))) i j)

/-- The emitted **bias** AdamW update — entrywise over the certified ∂/∂b gradient. -/
noncomputable def adamB (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (mB vB : Vec n)
    (label : Fin n) : Vec n :=
  fun j => adamWScalar β₁ β₂ ε lr wd bc₁ bc₂ (b j) (mB j) (vB j)
            (bGrad (den (lossCotGraph W b x (oneHot n label))) j)

/-- **AdamW weight-step faithfulness.** The emitted update is `adamWScalar` of the
    *certified* ∂/∂W Jacobian contracted with the proven softmax-CE cotangent —
    AdamW promoted from trusted to proven, exactly as
    `sgdW_descends_certified_grad` does for plain SGD (the update map is now Adam,
    not `θ−lr·g`). -/
theorem adamW_certified_grad (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (mW vW : Mat m n)
    (label : Fin n) (i : Fin m) (j : Fin n) :
    adamW W b x β₁ β₂ ε lr wd bc₁ bc₂ mW vW label i j
      = adamWScalar β₁ β₂ ε lr wd bc₁ bc₂ (W i j) (mW i j) (vW i j)
          (∑ k : Fin n, pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
              (Mat.flatten W) (finProdFinEquiv (i, j)) k
            * den (lossCotGraph W b x (oneHot n label)) k) := by
  unfold adamW
  rw [wGrad_isWeightJacobian W b x (den (lossCotGraph W b x (oneHot n label))) i j]

/-- **AdamW bias-step faithfulness.** Likewise for `b`. -/
theorem adamB_certified_grad (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (mB vB : Vec n)
    (label : Fin n) (j : Fin n) :
    adamB W b x β₁ β₂ ε lr wd bc₁ bc₂ mB vB label j
      = adamWScalar β₁ β₂ ε lr wd bc₁ bc₂ (b j) (mB j) (vB j)
          (∑ i : Fin n, pdiv (fun b' : Vec n => dense W b' x) b j i
            * den (lossCotGraph W b x (oneHot n label)) i) := by
  unfold adamB
  rw [bGrad_isBiasJacobian W b x (den (lossCotGraph W b x (oneHot n label))) j]

end StableHLO
end Proofs
