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
* `HasGradAt.param` / `param_batchMap` — with the gradient at a parameterised op's output known,
  the loss, read as a function of the parameter, has a gradient there: the op's parameter
  Jacobian against the output gradient, the `Σ_n Σ_j` every batched gradient node denotes.

`addConstHasVJPAt` / `constAddHasVJPAt` are the VJP a residual needs when a parameter inside one
branch varies: the other branch is a constant. For a net whose every op is batch-separable,
`HasGradAt.param_batchMap_through` does the work per example against `linLoss dy`.
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
    backward chain's cotangent at that activation is its `dy`. Read as a function of a
    parameter, it is the statement "`dy` IS the loss gradient in that parameter": `pdiv` alone is
    `0` wherever `G` is not differentiable, so the differentiability is part of the claim. -/
structure HasGradAt {m : Nat} (G : Vec m → Vec 1) (x : Vec m) (dy : Vec m) : Prop where
  differentiableAt : DifferentiableAt ℝ G x
  pdiv_eq : ∀ j, pdiv G x j 0 = dy j

theorem hasGradAt_iff {m : Nat} {G : Vec m → Vec 1} {x dy : Vec m} :
    HasGradAt G x dy ↔ DifferentiableAt ℝ G x ∧ ∀ j, pdiv G x j 0 = dy j :=
  ⟨fun h => ⟨h.1, h.2⟩, fun h => ⟨h.1, h.2⟩⟩

/-- **Gradients pull back through a certified VJP**: if `G` has gradient `dy` at `f x`, then
    `G ∘ f` has gradient `f`'s backward of `dy` at `x`. -/
theorem HasGradAt.comp {m n : Nat} {G : Vec n → Vec 1} {f : Vec m → Vec n} {x : Vec m}
    {dy : Vec n} (hG : HasGradAt G (f x) dy) (hf : DifferentiableAt ℝ f x) (vf : HasVJPAt f x) :
    HasGradAt (fun y => G (f y)) x (vf.backward dy) := by
  refine ⟨hG.differentiableAt.comp x hf, fun i => ?_⟩
  -- `pdiv_comp` is stated on `G ∘ f`
  change pdiv (G ∘ f) x i 0 = _
  rw [pdiv_comp f G x hf hG.differentiableAt i 0, vf.correct dy i]
  simp_rw [hG.pdiv_eq]

/-- `HasGradAt.comp` through a global VJP, the cotangent spelled `vf.backward x dy`. At a large
    certified VJP the two spellings are definitionally equal but the unifier reaches the equality
    by unfolding the witness; stated once here, the equality is checked at a variable. -/
theorem HasGradAt.comp_global {m n : Nat} {G : Vec n → Vec 1} {f : Vec m → Vec n} {x : Vec m}
    {dy : Vec n} (hG : HasGradAt G (f x) dy) (hf : Differentiable ℝ f) (vf : HasVJP f) :
    HasGradAt (fun y => G (f y)) x (vf.backward x dy) :=
  hG.comp (hf x) (vf.toHasVJPAt x)

/-- Restate a gradient at an equal cotangent. -/
theorem HasGradAt.of_eq {m : Nat} {G : Vec m → Vec 1} {x dy dy' : Vec m}
    (hG : HasGradAt G x dy) (h : dy = dy') : HasGradAt G x dy' := h ▸ hG

/-- Restate a gradient at an equal point. -/
theorem HasGradAt.congr_point {m : Nat} {G : Vec m → Vec 1} {x x' dy : Vec m}
    (h : x = x') (hG : HasGradAt G x dy) : HasGradAt G x' dy := h ▸ hG

/-- Restate a gradient at an equal function. -/
theorem HasGradAt.congr_left {m : Nat} {G G' : Vec m → Vec 1} {x dy : Vec m}
    (hG : HasGradAt G x dy) (h : G = G') : HasGradAt G' x dy := h ▸ hG

/-- A gradient transports along a germ: `G' = G` near `x`. -/
theorem HasGradAt.congr_of_eventuallyEq {m : Nat} {G G' : Vec m → Vec 1} {x dy : Vec m}
    (hG : HasGradAt G x dy) (h : G =ᶠ[nhds x] G') : HasGradAt G' x dy := by
  refine ⟨hG.differentiableAt.congr_of_eventuallyEq h.symm, fun j => ?_⟩
  have hj := hG.pdiv_eq j
  unfold pdiv at hj ⊢
  rw [← h.fderiv_eq]
  exact hj

/-- **A parameter's loss gradient, from the gradient at its op's output**: the loss read as a
    function of the parameter is differentiable, with the op's parameter Jacobian contracted
    against `dy` as its gradient. -/
theorem HasGradAt.param {P m : Nat} {G : Vec m → Vec 1} {layer : Vec P → Vec m}
    {θ : Vec P} {dy : Vec m} (hG : HasGradAt G (layer θ) dy) (hl : DifferentiableAt ℝ layer θ) :
    HasGradAt (fun θ' => G (layer θ')) θ (fun i => ∑ j, pdiv layer θ i j * dy j) := by
  refine ⟨hG.differentiableAt.comp θ hl, fun i => ?_⟩
  -- `pdiv_comp` is stated on `G ∘ layer`
  change pdiv (G ∘ layer) θ i 0 = _
  rw [pdiv_comp layer G θ hl hG.differentiableAt i 0]
  simp_rw [hG.pdiv_eq]

/-- **…at a batched op**: `θ ↦ batchMap N (per θ) r`, the Jacobian split by example — the
    `Σ_n Σ_j` every batched parameter gradient node denotes. -/
theorem HasGradAt.param_batchMap {P N a q : Nat} {G : Vec (N * q) → Vec 1}
    (per : Vec P → Vec a → Vec q) (r : Vec (N * a)) {θ : Vec P} {dy : Vec (N * q)}
    (hG : HasGradAt G (StableHLO.batchMap N (per θ) r) dy)
    (hper : ∀ y, DifferentiableAt ℝ (fun θ' => per θ' y) θ) :
    HasGradAt (fun θ' => G (StableHLO.batchMap N (per θ') r)) θ
      (fun i => ∑ n : Fin N, ∑ j : Fin q,
          pdiv (fun θ' => per θ' (StableHLO.batchSlice N a r n)) θ i j
            * StableHLO.batchSlice N q dy n j) := by
  have hP := hG.param (layer := fun θ' => StableHLO.batchMap N (per θ') r)
    (batchMap_param_differentiableAt per r θ hper)
  refine ⟨hP.differentiableAt, fun i => ?_⟩
  rw [hP.pdiv_eq i]
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun n _ => Finset.sum_congr rfl fun j _ => ?_
  congr 1
  rw [pdiv_eq_fderiv_coord (batchMap_param_differentiableAt per r θ hper),
    pdiv_eq_fderiv_coord (hper _)]
  simp only [StableHLO.batchMap, Equiv.symm_apply_apply]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § A per-example stage, lifted: nets whose every op is batch-separable (ConvNeXt, ViT)
-- ════════════════════════════════════════════════════════════════

/-- The linear functional `u ↦ ⟨u, dy⟩`: its gradient is `dy` everywhere. Read per example, it
    turns "the chain's cotangent contracts a stage Jacobian" into a `HasGradAt` statement. -/
noncomputable def linLoss {m : Nat} (dy : Vec m) : Vec m → Vec 1 :=
  fun u _ => ∑ k, u k * dy k

theorem hasGradAt_linLoss {m : Nat} (dy x : Vec m) : HasGradAt (linLoss dy) x dy := by
  refine ⟨by unfold linLoss; fun_prop, fun j => ?_⟩
  rw [pdiv_of_linear (linLoss dy)
    (fun u v => by funext; simp [linLoss, add_mul, Finset.sum_add_distrib])
    (fun a v => by funext; simp [linLoss, Finset.mul_sum, mul_assoc])]
  simp [linLoss, basisVec]

/-- **A parameter inside a per-example stage, lifted over the batch.** Each example runs
    `y ↦ post y (per θ (pre y))`: the stage `per θ` at its input `pre y`, then the rest of the block
    `post y`. If, per example, the loss `⟨post y ·, dy⟩` has gradient `cot y dy` at the stage output,
    then the batched node `Σ_n Σ_j ∂per/∂θ · cotₙ` — at any saved activation `A` and cotangent `COT`
    whose slices are `pre yₙ` and `cot yₙ dyₙ` — is the gradient in `θ` of the whole batched
    block's loss. -/
theorem HasGradAt.param_batchMap_through {P N a b m q : Nat} {G : Vec (N * q) → Vec 1}
    (pre : Vec a → Vec b) (per : Vec P → Vec b → Vec m) (post : Vec a → Vec m → Vec q)
    (cot : Vec a → Vec q → Vec m) (X : Vec (N * a)) {θ : Vec P} {dY : Vec (N * q)}
    (hG : HasGradAt G (StableHLO.batchMap N (fun y => post y (per θ (pre y))) X) dY)
    (hper : ∀ y, DifferentiableAt ℝ (fun θ' => per θ' y) θ)
    (hpost : ∀ y, Differentiable ℝ (post y))
    (hcot : ∀ y dy, HasGradAt (fun u => linLoss dy (post y u)) (per θ (pre y)) (cot y dy))
    (A : Vec (N * b)) (COT : Vec (N * m))
    (hA : ∀ n, StableHLO.batchSlice N b A n = pre (StableHLO.batchSlice N a X n))
    (hC : ∀ n, StableHLO.batchSlice N m COT n
      = cot (StableHLO.batchSlice N a X n) (StableHLO.batchSlice N q dY n))
    :
    HasGradAt (fun θ' => G (StableHLO.batchMap N (fun y => post y (per θ' (pre y))) X)) θ
      (fun i => ∑ n : Fin N, ∑ j : Fin m,
        pdiv (fun θ' => per θ' (StableHLO.batchSlice N b A n)) θ i j
          * StableHLO.batchSlice N m COT n j) := by
  have hP := hG.param_batchMap (fun θ' y => post y (per θ' (pre y))) X
    (fun y => (hpost y _).comp θ (hper (pre y)))
  refine ⟨hP.differentiableAt, fun i => ?_⟩
  rw [hP.pdiv_eq i]
  refine Finset.sum_congr rfl fun n _ => ?_
  rw [hA, hC, ← ((hcot _ _).param
    (layer := fun θ' => per θ' (pre (StableHLO.batchSlice N a X n))) (hper _)).2 i]
  exact (((hasGradAt_linLoss _ _).param
    (layer := fun θ' => post (StableHLO.batchSlice N a X n)
      (per θ' (pre (StableHLO.batchSlice N a X n))))
    ((hpost _ _).comp θ (hper _))).2 i).symm

end Proofs
