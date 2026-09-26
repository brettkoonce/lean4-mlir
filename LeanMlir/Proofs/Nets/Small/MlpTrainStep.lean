import LeanMlir.Proofs.Foundation.IR
import LeanMlir.Proofs.Nets.Small.LinearTrainStep

/-! # The MLP train step: per-layer parameter-gradient assembly

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
  (IR.lean; `mlp_layer1_weight_grad_bridge`).
* **layer 0** (`p₀`): `relu'(p₀) ⊙ (W₁ · mlpCotOut1)` — `mlpCotOut0`, below.

This file supplies `mlpCotOut0` and its weight/bias bridges — completing the three-layer
assembly, the multi-layer step the one-layer `linear` net has no analogue of — and the
total-loss folds (`mlp_output_total_loss_grad`, and at smooth points
`mlp_hidden_total_loss_grad`, `mlp_input_total_loss_grad`, `mlp_whole_net_weight_grads`). The SGD
wrapping `θ − lr·∇` on top is identical to the linear case (`StableHLO.sgdW`).
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
-- ReLU kinks): `mlp_hidden_total_loss_grad` / `mlp_input_total_loss_grad` below.
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

/-- **Hidden-layer total-loss fold (conditional).** At a smooth point — the hidden
    pre-activation `p₁ = dense W₁ b₁ a₀` off the ReLU kinks — the single gradient of the
    whole softmax-CE loss wrt the hidden weights `W₁` folds, by the chain rule
    (`pdiv_comp`), into the certified `∂p₁/∂W₁` contracted with the loss gradient at the
    hidden pre-activation, `∂L/∂p₁`, left as a `pdiv`. (Off the kinks that factor is the
    cotangent the backward chain delivers at layer 1, `relu'(p₁) ⊙ (W₂ · (softmax−onehot))`,
    cf. `mlpCotOut1_denote`; that identification is not stated here.) Conditionality is intrinsic: the chain runs back through the
    ReLU kink, so — unlike the linear / output-layer fold — this needs the smoothness
    hypothesis. The hidden-layer analogue of `lossWeightGrad_eq_sum`. -/
theorem mlp_hidden_total_loss_grad {d₁ d₂ d₃ : Nat}
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (a₀ : Vec d₁) (label : Fin d₃) (h_smooth : ∀ k, dense W₁ b₁ a₀ k ≠ 0)
    (i : Fin d₁) (j : Fin d₂) :
    pdiv (fun v : Vec (d₁ * d₂) => fun _ : Fin 1 =>
            crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) label)
         (Mat.flatten W₁) (finProdFinEquiv (i, j)) 0
      = ∑ k : Fin d₂,
          pdiv (fun v : Vec (d₁ * d₂) => dense (Mat.unflatten v) b₁ a₀)
               (Mat.flatten W₁) (finProdFinEquiv (i, j)) k
            * pdiv (fun z : Vec d₂ => fun _ : Fin 1 =>
                     crossEntropy d₃ (dense W₂ b₂ (relu d₂ z)) label)
                   (dense W₁ b₁ a₀) k 0 := by
  -- `G = loss ∘ dense W₂ ∘ relu` is differentiable at `p₁` (ReLU smooth there).
  have hG_diff : DifferentiableAt ℝ
      (fun z : Vec d₂ => fun _ : Fin 1 => crossEntropy d₃ (dense W₂ b₂ (relu d₂ z)) label)
      (dense W₁ b₁ a₀) := by
    fun_prop (disch := assumption)
  -- The loss-of-W₁ map is `G ∘ (W₁-weight-map)`; apply the chain rule.
  rw [show (fun v : Vec (d₁ * d₂) => fun _ : Fin 1 =>
              crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) label)
        = (fun z : Vec d₂ => fun _ : Fin 1 => crossEntropy d₃ (dense W₂ b₂ (relu d₂ z)) label)
            ∘ (fun v : Vec (d₁ * d₂) => dense (Mat.unflatten v) b₁ a₀) from rfl,
      pdiv_comp _ _ _ ((StableHLO.denseWeightMap_differentiable b₁ a₀) _)
        (show DifferentiableAt ℝ
                (fun z : Vec d₂ => fun _ : Fin 1 => crossEntropy d₃ (dense W₂ b₂ (relu d₂ z)) label)
                (dense (Mat.unflatten (Mat.flatten W₁)) b₁ a₀)
           from by rw [Mat.unflatten_flatten]; exact hG_diff)]
  -- The inner point `F(flatten W₁) = dense (unflatten (flatten W₁)) b₁ a₀ = dense W₁ b₁ a₀`.
  simp only [Mat.unflatten_flatten]

/-- **Input-layer total-loss fold (conditional, deepest).** The same fold for the
    first layer `W₀`, whose chain runs back through *both* ReLUs — so it carries both
    smoothness hypotheses (the same pair as `mlpHasVJPAt`). The total loss gradient
    wrt `W₀` = certified `∂p₀/∂W₀` contracted with the loss gradient at `p₀`, left as a
    `pdiv` (its identification with the chain cotangent `mlpCotOut0_denote` computes is not
    stated here). -/
theorem mlp_input_total_loss_grad {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃)
    (h_smooth_0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h_smooth_1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0)
    (i : Fin d₀) (j : Fin d₁) :
    pdiv (fun v : Vec (d₀ * d₁) => fun _ : Fin 1 =>
            crossEntropy d₃
              (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x))))) label)
         (Mat.flatten W₀) (finProdFinEquiv (i, j)) 0
      = ∑ k : Fin d₁,
          pdiv (fun v : Vec (d₀ * d₁) => dense (Mat.unflatten v) b₀ x)
               (Mat.flatten W₀) (finProdFinEquiv (i, j)) k
            * pdiv (fun z : Vec d₁ => fun _ : Fin 1 =>
                     crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ z)))) label)
                   (dense W₀ b₀ x) k 0 := by
  have hG_diff : DifferentiableAt ℝ
      (fun z : Vec d₁ => fun _ : Fin 1 =>
        crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ z)))) label)
      (dense W₀ b₀ x) := by
    fun_prop (disch := assumption)
  rw [show (fun v : Vec (d₀ * d₁) => fun _ : Fin 1 =>
              crossEntropy d₃
                (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x))))) label)
        = (fun z : Vec d₁ => fun _ : Fin 1 =>
              crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ z)))) label)
            ∘ (fun v : Vec (d₀ * d₁) => dense (Mat.unflatten v) b₀ x) from rfl,
      pdiv_comp _ _ _ ((StableHLO.denseWeightMap_differentiable b₀ x) _)
        (show DifferentiableAt ℝ
                (fun z : Vec d₁ => fun _ : Fin 1 =>
                  crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ z)))) label)
                (dense (Mat.unflatten (Mat.flatten W₀)) b₀ x)
           from by rw [Mat.unflatten_flatten]; exact hG_diff)]
  simp only [Mat.unflatten_flatten]

-- ════════════════════════════════════════════════════════════════
-- § Whole-net capstone — every weight layer's total-loss gradient at once
--
-- One statement for the whole MLP's training: at a smooth point (both hidden
-- pre-activations off the ReLU kinks — the same pair `mlpHasVJPAt` uses), the
-- gradient of the WHOLE softmax-CE loss `crossEntropy ∘ mlpForward` with respect to
-- every weight layer is the certified assembled gradient. Output layer unconditionally,
-- the two hidden layers conditionally — folded from the per-layer results, with the
-- forward activations threaded through (`a₀ = relu(dense W₀ b₀ x)`, etc.).
-- ════════════════════════════════════════════════════════════════

/-- **Whole-network MLP weight-gradient capstone.** The three weight layers' total-loss
    gradients, jointly, under the two smoothness hypotheses. -/
theorem mlp_whole_net_weight_grads {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃)
    (h_smooth_0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h_smooth_1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0) :
    -- input layer W₀ (chain back through both ReLUs)
    (∀ (i : Fin d₀) (j : Fin d₁),
      pdiv (fun v : Vec (d₀ * d₁) => fun _ : Fin 1 =>
              crossEntropy d₃
                (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x))))) label)
           (Mat.flatten W₀) (finProdFinEquiv (i, j)) 0
        = ∑ k : Fin d₁,
            pdiv (fun v : Vec (d₀ * d₁) => dense (Mat.unflatten v) b₀ x)
                 (Mat.flatten W₀) (finProdFinEquiv (i, j)) k
              * pdiv (fun z : Vec d₁ => fun _ : Fin 1 =>
                       crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ z)))) label)
                     (dense W₀ b₀ x) k 0) ∧
    -- hidden layer W₁ (chain back through one ReLU)
    (∀ (i : Fin d₁) (j : Fin d₂),
      pdiv (fun v : Vec (d₁ * d₂) => fun _ : Fin 1 =>
              crossEntropy d₃
                (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten v) b₁ (relu d₁ (dense W₀ b₀ x))))) label)
           (Mat.flatten W₁) (finProdFinEquiv (i, j)) 0
        = ∑ k : Fin d₂,
            pdiv (fun v : Vec (d₁ * d₂) => dense (Mat.unflatten v) b₁ (relu d₁ (dense W₀ b₀ x)))
                 (Mat.flatten W₁) (finProdFinEquiv (i, j)) k
              * pdiv (fun z : Vec d₂ => fun _ : Fin 1 =>
                       crossEntropy d₃ (dense W₂ b₂ (relu d₂ z)) label)
                     (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))) k 0) ∧
    -- output layer W₂ (directly below the loss — unconditional)
    (∀ (i : Fin d₂) (j : Fin d₃),
      pdiv (fun v : Vec (d₂ * d₃) => fun _ : Fin 1 =>
              crossEntropy d₃
                (dense (Mat.unflatten v) b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
           (Mat.flatten W₂) (finProdFinEquiv (i, j)) 0
        = ∑ k : Fin d₃,
            pdiv (fun v : Vec (d₂ * d₃) =>
                    dense (Mat.unflatten v) b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))))
                 (Mat.flatten W₂) (finProdFinEquiv (i, j)) k
              * (softmax d₃ (mnistLinear W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k
                  - oneHot d₃ label k)) :=
  ⟨fun i j => mlp_input_total_loss_grad W₀ b₀ W₁ b₁ W₂ b₂ x label h_smooth_0 h_smooth_1 i j,
   fun i j => mlp_hidden_total_loss_grad W₁ b₁ W₂ b₂ (relu d₁ (dense W₀ b₀ x)) label h_smooth_1 i j,
   fun i j => mlp_output_total_loss_grad W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) label i j⟩

end Proofs.IR
