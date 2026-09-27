import LeanMlir.Proofs.Foundation.Batched.Basic

/-! # ParamGrad — the loss gradient in a parameter, from the gradient at its op's output

A net's train-step tie says each parameter gradient node is its layer's parameter Jacobian
contracted with the cotangent the backward chain threads there, and its `*_eq_vjp` lemmas say the
chain's cotangents are certified VJP backwards. This file is the calculus that composes the two
into a derivative of the loss:

* `HasGradAt G x dy` — a scalar `G` has gradient `dy` at `x`. The loss read at any activation of the
  net is such a `G`, and the chain's cotangent there is its `dy`.
* `HasGradAt.comp` — gradients pull back through a certified VJP: `G ∘ f` has gradient
  `f.backward dy`. Applied stage by stage, it walks the loss gradient down the net.
* `HasGradAt.pdiv_param` / `pdiv_param_batchMap` — with the gradient at a parameterised op's
  output known, the loss derivative in the parameter is the op's parameter Jacobian against it:
  the `Σ_n Σ_j` every batched gradient node denotes.

`addConstHasVJPAt` / `constAddHasVJPAt` are the VJP a residual needs when a parameter inside one
branch varies: the other branch is a constant.
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

/-- `addConstHasVJPAt` with the constant on the left: `u ↦ c + f u`. -/
noncomputable def constAddHasVJPAt {m n : Nat} (c : Vec n) (f : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hv : HasVJPAt f x) :
    HasVJPAt (fun u i => c i + f u i) x where
  backward dy := hv.backward dy
  correct dy i := by
    rw [hv.correct dy i]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [pdiv_add (fun _ => c) f x (differentiableAt_const c) hf i j, pdiv_const, zero_add]

/-- The batched parameterised op `θ ↦ batchMap N (per θ) r` is differentiable when each
    example's map is differentiable in the parameter. -/
theorem batchMap_param_differentiableAt {P N a q : Nat} (per : Vec P → Vec a → Vec q)
    (r : Vec (N * a)) (θ : Vec P) (hper : ∀ y, DifferentiableAt ℝ (fun θ' => per θ' y) θ) :
    DifferentiableAt ℝ (fun θ' => StableHLO.batchMap N (per θ') r) θ := by
  refine differentiableAt_pi.2 fun J => ?_
  exact differentiableAt_pi.1 (hper _) _

-- ════════════════════════════════════════════════════════════════
-- § `HasGradAt` — a scalar map's gradient at a point, propagated backwards through VJPs
-- ════════════════════════════════════════════════════════════════

/-- **`G : Vec m → Vec 1` has gradient `dy` at `x`**: differentiable there, and each partial is
    `dy`'s entry. The loss, read as a function of any activation of the net, is such a `G`; the
    backward chain's cotangent at that activation is its `dy`. -/
def HasGradAt {m : Nat} (G : Vec m → Vec 1) (x : Vec m) (dy : Vec m) : Prop :=
  DifferentiableAt ℝ G x ∧ ∀ j, pdiv G x j 0 = dy j

/-- **Gradients pull back through a certified VJP**: if `G` has gradient `dy` at `f x`, then
    `G ∘ f` has gradient `f`'s backward of `dy` at `x`. -/
theorem HasGradAt.comp {m n : Nat} {G : Vec n → Vec 1} {f : Vec m → Vec n} {x : Vec m}
    {dy : Vec n} (hG : HasGradAt G (f x) dy) (hf : DifferentiableAt ℝ f x) (vf : HasVJPAt f x) :
    HasGradAt (fun y => G (f y)) x (vf.backward dy) := by
  refine ⟨hG.1.comp x hf, fun i => ?_⟩
  change pdiv (G ∘ f) x i 0 = _
  rw [pdiv_comp f G x hf hG.1 i 0, vf.correct dy i]
  simp_rw [hG.2]

/-- Restate a gradient at an equal cotangent. -/
theorem HasGradAt.of_eq {m : Nat} {G : Vec m → Vec 1} {x dy dy' : Vec m}
    (hG : HasGradAt G x dy) (h : dy = dy') : HasGradAt G x dy' := h ▸ hG

/-- Restate a gradient at an equal point. -/
theorem HasGradAt.congr_point {m : Nat} {G : Vec m → Vec 1} {x x' dy : Vec m}
    (h : x = x') (hG : HasGradAt G x dy) : HasGradAt G x' dy := h ▸ hG

/-- **A parameter's loss derivative, from the gradient at its op's output.** -/
theorem HasGradAt.pdiv_param {P m : Nat} {G : Vec m → Vec 1} {layer : Vec P → Vec m}
    {θ : Vec P} {dy : Vec m} (hG : HasGradAt G (layer θ) dy) (hl : DifferentiableAt ℝ layer θ)
    (i : Fin P) :
    pdiv (fun θ' => G (layer θ')) θ i 0 = ∑ j, pdiv layer θ i j * dy j := by
  change pdiv (G ∘ layer) θ i 0 = _
  rw [pdiv_comp layer G θ hl hG.1 i 0]
  simp_rw [hG.2]

/-- **…at a batched op**: `θ ↦ batchMap N (per θ) r`, the Jacobian split by example — the
    `Σ_n Σ_j` every batched parameter gradient node denotes. -/
theorem HasGradAt.pdiv_param_batchMap {P N a q : Nat} {G : Vec (N * q) → Vec 1}
    (per : Vec P → Vec a → Vec q) (r : Vec (N * a)) {θ : Vec P} {dy : Vec (N * q)}
    (hG : HasGradAt G (StableHLO.batchMap N (per θ) r) dy)
    (hper : ∀ y, DifferentiableAt ℝ (fun θ' => per θ' y) θ) (i : Fin P) :
    pdiv (fun θ' => G (StableHLO.batchMap N (per θ') r)) θ i 0
      = ∑ n : Fin N, ∑ j : Fin q,
          pdiv (fun θ' => per θ' (StableHLO.batchSlice N a r n)) θ i j
            * StableHLO.batchSlice N q dy n j := by
  rw [hG.pdiv_param (layer := fun θ' => StableHLO.batchMap N (per θ') r)
    (batchMap_param_differentiableAt per r θ hper) i]
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun n _ => Finset.sum_congr rfl fun j _ => ?_
  congr 1
  rw [pdiv_eq_fderiv_coord (batchMap_param_differentiableAt per r θ hper),
    pdiv_eq_fderiv_coord (hper _)]
  simp only [StableHLO.batchMap, Equiv.symm_apply_apply]
  rfl

end Proofs
