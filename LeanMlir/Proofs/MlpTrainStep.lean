import LeanMlir.Proofs.IR
import LeanMlir.Proofs.LinearTrainStep

/-! # M2 — the MLP train step: per-layer parameter-gradient assembly

The MLP (`dense → relu → dense → relu → dense`) train step updates six parameters
`W₀,b₀,W₁,b₁,W₂,b₂`. Each layer's gradient is the **assembly** of backprop:
`∂L/∂Wᵢ = (layer-i input) ⊗ (cotangent the backward chain delivers at layer i)`,
`∂L/∂bᵢ = (that cotangent)`.

The generic per-layer bridges already certify this for *any* backward subgraph
`e : Back` (`weight_grad_bridge`/`bias_grad_bridge`, IR.lean) — `emit*Grad` equals
the certified Jacobian of that dense layer contracted with `e.denote dy`. So the
assembly is choosing the right cotangent subgraph per layer:

* **layer 2** (logits): the loss cotangent `g` itself — `Back.cotangent`
  (`weight_grad_bridge … Back.cotangent`, with `Back.cotangent.denote g = g`).
* **layer 1** (`p₁`): `relu'(p₁) ⊙ (W₂ · g)` — `mlpCotOut1`
  (already built; `mlp_layer1_weight_grad_bridge`).
* **layer 0** (`p₀`): `relu'(p₀) ⊙ (W₁ · mlpCotOut1)` — `mlpCotOut0`, below.

This file supplies the only missing piece, `mlpCotOut0`, and its weight/bias
bridges — completing the three-layer assembly. This is Crux A of
`planning/verified_train_step.md`: the multi-layer param-grad assembly, the step
`linear` couldn't show (one layer, no chain). The SGD wrapping `θ − lr·∇` on top is
identical to the linear case (`StableHLO.sgdW`).
-/

namespace Proofs.IR

open Proofs

/-- **Layer-0 cotangent subgraph** — the cotangent the backward chain delivers at
    the layer-0 dense output `p₀`: `relu'(p₀) ⊙ (W₁ · mlpCotOut1)`. Prepends one more
    `relu-back ∘ dense-back` to `mlpCotOut1`, exactly as `mlpCotOut1` extends
    `Back.cotangent`. -/
def mlpCotOut0 {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (p₀ : Vec d₁) (p₁ : Vec d₂) : Back d₃ d₁ :=
  (emitReluBack p₀).subst ((emitDenseBack W₁).subst (mlpCotOut1 W₂ p₁))

/-- **MLP layer-0 weight-gradient bridge.** The emitted layer-0 weight gradient
    (`x₀ ⊗ mlpCotOut0`) equals the certified Jacobian of the layer-0 dense wrt `W₀`,
    contracted with the cotangent the backward chain delivers there — the deepest
    chain (`relu'(p₀) ⊙ W₁ · relu'(p₁) ⊙ W₂ · dy`). The layer-0 peer of
    `mlp_layer1_weight_grad_bridge`. -/
theorem mlp_layer0_weight_grad_bridge {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (x₀ : Vec d₀) (p₀ : Vec d₁) (p₁ : Vec d₂) (dy : Vec d₃) (i : Fin d₀) (j : Fin d₁) :
    emitWeightGrad x₀ (mlpCotOut0 W₁ W₂ p₀ p₁) dy i j
      = ∑ k : Fin d₁,
          pdiv (fun v : Vec (d₀ * d₁) => dense (Mat.unflatten v) b₀ x₀)
               (Mat.flatten W₀) (finProdFinEquiv (i, j)) k
            * (mlpCotOut0 W₁ W₂ p₀ p₁).denote dy k :=
  weight_grad_bridge W₀ b₀ x₀ (mlpCotOut0 W₁ W₂ p₀ p₁) dy i j

/-- **MLP layer-0 bias-gradient bridge.** Likewise the layer-0 bias gradient is the
    certified ∂/∂b₀ Jacobian contracted with the same deepest cotangent. -/
theorem mlp_layer0_bias_grad_bridge {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (x₀ : Vec d₀) (p₀ : Vec d₁) (p₁ : Vec d₂) (dy : Vec d₃) (i : Fin d₁) :
    emitBiasGrad (mlpCotOut0 W₁ W₂ p₀ p₁) dy i
      = ∑ j : Fin d₁,
          pdiv (fun b' : Vec d₁ => dense W₀ b' x₀) b₀ i j
            * (mlpCotOut0 W₁ W₂ p₀ p₁).denote dy j :=
  bias_grad_bridge W₀ b₀ x₀ (mlpCotOut0 W₁ W₂ p₀ p₁) dy i

/-- **MLP layer-1 bias-gradient bridge** (the bias peer of the existing
    `mlp_layer1_weight_grad_bridge`). -/
theorem mlp_layer1_bias_grad_bridge {d₁ d₂ d₃ : Nat}
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (x₁ : Vec d₁) (p₁ : Vec d₂)
    (dy : Vec d₃) (i : Fin d₂) :
    emitBiasGrad (mlpCotOut1 W₂ p₁) dy i
      = ∑ j : Fin d₂,
          pdiv (fun b' : Vec d₂ => dense W₁ b' x₁) b₁ i j * (mlpCotOut1 W₂ p₁).denote dy j :=
  bias_grad_bridge W₁ b₁ x₁ (mlpCotOut1 W₂ p₁) dy i

/-- **MLP layer-2 (output) weight-gradient bridge** — the output layer's cotangent
    is the loss cotangent `dy` itself (`Back.cotangent`, `denote dy = dy`); the
    bridge is the generic one specialized there. -/
theorem mlp_layer2_weight_grad_bridge {d₂ d₃ : Nat}
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x₂ : Vec d₂) (dy : Vec d₃) (i : Fin d₂) (j : Fin d₃) :
    emitWeightGrad x₂ (Back.cotangent) dy i j
      = ∑ k : Fin d₃,
          pdiv (fun v : Vec (d₂ * d₃) => dense (Mat.unflatten v) b₂ x₂)
               (Mat.flatten W₂) (finProdFinEquiv (i, j)) k * dy k := by
  rw [weight_grad_bridge W₂ b₂ x₂ Back.cotangent dy i j]; rfl

-- ════════════════════════════════════════════════════════════════
-- § The chain: the composed cotangent subgraphs ARE the explicit backprop formulas
--
-- `mlpCotOut1`/`mlpCotOut0` are `Back.subst` compositions; `denote_subst` (the IR
-- chain rule) reduces their denotations to the closed `relu' ⊙ Wᵀ·…` forms. So "the
-- cotangent the backward chain delivers" is not an opaque graph but the genuine
-- multiplied-through gradient.
-- ════════════════════════════════════════════════════════════════

/-- **Layer-1 cotangent, explicit.** `mlpCotOut1.denote g = relu'(p₁) ⊙ (W₂ · g)`. -/
theorem mlpCotOut1_denote {d₂ d₃ : Nat} (W₂ : Mat d₂ d₃) (p₁ : Vec d₂) (g : Vec d₃) :
    (mlpCotOut1 W₂ p₁).denote g = fun i => if p₁ i > 0 then Mat.mulVec W₂ g i else 0 := by
  unfold mlpCotOut1 emitReluBack emitDenseBack
  rw [denote_subst]
  rfl

/-- **Layer-0 cotangent, explicit** — the deepest chain
    `relu'(p₀) ⊙ (W₁ · (relu'(p₁) ⊙ (W₂ · g)))`. -/
theorem mlpCotOut0_denote {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (p₀ : Vec d₁) (p₁ : Vec d₂) (g : Vec d₃) :
    (mlpCotOut0 W₁ W₂ p₀ p₁).denote g
      = fun i => if p₀ i > 0
          then Mat.mulVec W₁ (fun k => if p₁ k > 0 then Mat.mulVec W₂ g k else 0) i else 0 := by
  unfold mlpCotOut0 emitReluBack emitDenseBack
  rw [denote_subst, denote_subst]
  simp only [Back.denote, mlpCotOut1_denote W₂ p₁ g]

-- ════════════════════════════════════════════════════════════════
-- § The fold (output layer): the loss gradient wrt the TOP weights is `θ − lr·∂L/∂θ`
--
-- The output dense `W₂` classifies the layer-2 activation `a₁` and sits directly below
-- the softmax-CE loss — no ReLU between it and the loss — so its total-loss gradient
-- folds UNCONDITIONALLY: it is a direct instance of the linear fold at input `a₁`. The
-- hidden layers (`W₁`,`W₀`) fold only at smooth points (the chain runs back through the
-- ReLU kinks); that conditional fold is the remaining new proof.
-- ════════════════════════════════════════════════════════════════

/-- **Output-layer total-loss gradient.** For the top dense layer on activation `a₁`
    (in the MLP, `a₁ = relu(dense W₁ b₁ (relu(dense W₀ b₀ x)))`), the single gradient of
    the whole softmax-CE loss wrt `W₂` equals the certified `∂logits/∂W₂` contracted with
    the softmax-CE residual `softmax − onehot`. Unconditional — a direct instance of the
    linear fold `lossWeightGrad_eq_sum`. -/
theorem mlp_output_total_loss_grad {d₂ d₃ : Nat}
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (a₁ : Vec d₂) (label : Fin d₃) (i : Fin d₂) (j : Fin d₃) :
    pdiv (fun v : Vec (d₂ * d₃) => fun _ : Fin 1 =>
            crossEntropy d₃ (dense (Mat.unflatten v) b₂ a₁) label)
         (Mat.flatten W₂) (finProdFinEquiv (i, j)) 0
      = ∑ k : Fin d₃,
          pdiv (fun v : Vec (d₂ * d₃) => dense (Mat.unflatten v) b₂ a₁)
               (Mat.flatten W₂) (finProdFinEquiv (i, j)) k
            * (softmax d₃ (mnistLinear W₂ b₂ a₁) k - oneHot d₃ label k) :=
  StableHLO.lossWeightGrad_eq_sum W₂ b₂ a₁ label i j

end Proofs.IR
