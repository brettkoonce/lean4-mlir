import LeanMlir.Proofs.Foundation.Batched.Basic

/-! # ParamGrad — the loss gradient in one layer's parameters, through a certified suffix VJP

A net's train-step tie says each parameter gradient node is that layer's own parameter Jacobian
contracted with the cotangent the backward chain threads to it, and its `*_eq_vjp` lemmas say the
chain's cotangents are certified VJP backwards. This file composes the two into the derivative
of the loss in the parameter: if the net with one parameter varied factors as
`L ∘ T ∘ layer` — `layer` the parameterised op at its fixed input, `T` everything after it — and
`T` has a certified VJP at the op's output, then

  `∂(L ∘ T ∘ layer)/∂θᵢ = Σⱼ ∂layerⱼ/∂θᵢ · (T's backward of ∇L)ⱼ`   (`pdiv_param_chain`).

`pdiv_param_chain_batchMap` is the batched form: the op is `batchMap N` of a per-example map at
parameter `θ`, and the Jacobian splits by example into the `Σ_n Σ_j` the gradient nodes denote.
`addConstHasVJPAt` is the one VJP combinator a residual suffix needs that the stage library lacks:
`u ↦ f u + c`, a skip branch held fixed while a parameter inside the body varies.
-/

namespace Proofs

open scoped BigOperators

/-- **Adding a constant keeps the VJP.** `u ↦ f u + c` has `f`'s backward: a residual block's skip
    is a constant once the parameter being varied sits inside the body. -/
noncomputable def addConstHasVJPAt {m n : Nat} (f : Vec m → Vec n) (c : Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hv : HasVJPAt f x) :
    HasVJPAt (fun u i => f u i + c i) x where
  backward dy := hv.backward dy
  correct dy i := by
    rw [hv.correct dy i]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [pdiv_add f (fun _ => c) x hf (differentiableAt_const c) i j, pdiv_const, add_zero]

theorem addConstHasVJPAt_backward {m n : Nat} (f : Vec m → Vec n) (c : Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hv : HasVJPAt f x) (dy : Vec n) :
    (addConstHasVJPAt f c x hf hv).backward dy = hv.backward dy := rfl

/-- **The loss gradient in a parameter, through a certified suffix VJP.** `L ∘ T ∘ layer` at `θ`:
    the parameter Jacobian of `layer` contracted with `T`'s certified backward of the loss gradient
    `g` (`hg`: `g` is `∇L` at the logits). -/
theorem pdiv_param_chain {P m k : Nat} (layer : Vec P → Vec m) (T : Vec m → Vec k)
    (L : Vec k → Vec 1) (θ : Vec P) (g : Vec k)
    (hl : DifferentiableAt ℝ layer θ) (hT : DifferentiableAt ℝ T (layer θ))
    (hL : DifferentiableAt ℝ L (T (layer θ))) (vT : HasVJPAt T (layer θ))
    (hg : ∀ j, pdiv L (T (layer θ)) j 0 = g j) (i : Fin P) :
    pdiv (fun θ' => L (T (layer θ'))) θ i 0 = ∑ j, pdiv layer θ i j * vT.backward g j := by
  change pdiv ((L ∘ T) ∘ layer) θ i 0 = _
  rw [pdiv_comp layer (L ∘ T) θ hl (hL.comp _ hT) i 0]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [pdiv_comp T L _ hT hL j 0, vT.correct g j]
  simp_rw [hg]

/-- The batched parameterised op `θ ↦ batchMap N (per θ) r` is differentiable when each
    example's map is differentiable in the parameter. -/
theorem batchMap_param_differentiableAt {P N a q : Nat} (per : Vec P → Vec a → Vec q)
    (r : Vec (N * a)) (θ : Vec P) (hper : ∀ y, DifferentiableAt ℝ (fun θ' => per θ' y) θ) :
    DifferentiableAt ℝ (fun θ' => StableHLO.batchMap N (per θ') r) θ := by
  refine differentiableAt_pi.2 fun J => ?_
  exact differentiableAt_pi.1 (hper _) _

/-- **`pdiv_param_chain` at a batched op.** The parameter Jacobian of `θ ↦ batchMap N (per θ) r`
    splits by example, so the loss gradient is the `Σ_n Σ_j` of each example's own parameter
    Jacobian against that example's slice of the certified backward — the shape every batched
    parameter gradient node denotes. -/
theorem pdiv_param_chain_batchMap {P N a q k : Nat} (per : Vec P → Vec a → Vec q)
    (r : Vec (N * a)) (T : Vec (N * q) → Vec k) (L : Vec k → Vec 1) (θ : Vec P) (g : Vec k)
    (hper : ∀ y, DifferentiableAt ℝ (fun θ' => per θ' y) θ)
    (hT : DifferentiableAt ℝ T (StableHLO.batchMap N (per θ) r))
    (hL : DifferentiableAt ℝ L (T (StableHLO.batchMap N (per θ) r)))
    (vT : HasVJPAt T (StableHLO.batchMap N (per θ) r))
    (hg : ∀ j, pdiv L (T (StableHLO.batchMap N (per θ) r)) j 0 = g j) (i : Fin P) :
    pdiv (fun θ' => L (T (StableHLO.batchMap N (per θ') r))) θ i 0
      = ∑ n : Fin N, ∑ j : Fin q,
          pdiv (fun θ' => per θ' (StableHLO.batchSlice N a r n)) θ i j
            * StableHLO.batchSlice N q (vT.backward g) n j := by
  rw [pdiv_param_chain (fun θ' => StableHLO.batchMap N (per θ') r) T L θ g
    (batchMap_param_differentiableAt per r θ hper) hT hL vT hg i]
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun n _ => Finset.sum_congr rfl fun j _ => ?_
  congr 1
  rw [pdiv_eq_fderiv_coord (batchMap_param_differentiableAt per r θ hper),
    pdiv_eq_fderiv_coord (hper _)]
  simp only [StableHLO.batchMap, Equiv.symm_apply_apply]
  rfl

end Proofs
