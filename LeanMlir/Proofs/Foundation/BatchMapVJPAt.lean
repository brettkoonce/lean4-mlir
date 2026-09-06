import LeanMlir.Proofs.Architectures.EfficientNetChainClose

/-! # `batchMap` at a POINT — the pointwise peer of `batchMap_has_vjp`

`EfficientNetChainClose.lean` lifts a batch-separable op's VJP to the whole batch in the GLOBAL
form: `batchMap_has_vjp` takes `HasVJP f` and `Differentiable ℝ f`. EfficientNet never needed
anything weaker — swish is smooth everywhere and B0's stem has no pooling — so the pointwise peer
was never written.

ResNet-34 needs it. Its stem is `batchMap N (maxPool3s2Flat c h w) ∘ cbReluStridedB`, and a
max-pool has no derivative at a tie: `maxPool3s2Flat_has_vjp_at_vec` is `_at` by nature. Without
the lift below, r34's whole-net VJP at batch BN cannot be assembled — the one thing standing
between `ResNet34FullB.lean` and T1.

⭐ **The weakening is exactly as narrow as it looks.** `pdivMat_rowIndep` requires
`Differentiable ℝ g`, and its docstring explains why (a non-differentiable coordinate makes
`fderiv` junk and breaks the per-row decomposition) — but reading the proof, every use of that
hypothesis is at a ROW of the matrix it is stated about. So the hypothesis weakens to
`∀ r, DifferentiableAt ℝ g (A r)` with no change to the argument, only to where the row-projection
equation `(rowProj k) (Mat.flatten A) = A k` is applied: it moves to the top, so the coordinate
differentiability can be stated at the projected point.

⚠ The r34 stem's instance lives with r34's VJP, not here — `maxPool3s2Flat_has_vjp_at_vec` is
in the `Float` tier and this is a `Foundation` file. It is two lines there:
`batchMap_has_vjp_at _ v (fun r => maxPool3s2Flat_has_vjp_at_vec (Mat.unflatten v r) (hs r))
(fun r => maxPool3s2Flat_differentiableAt_vec (Mat.unflatten v r) (hs r) hc hh hw)`, with no
glue between the two, which is what says the lemma below has the right shape.
-/

namespace Proofs

open scoped BigOperators

/-- **Row-wise Jacobian decomposition at a point.** `pdivMat_rowIndep` (`Tensor.lean`) with global
    differentiability of `g` weakened to differentiability at each ROW of `A` — the only points at
    which the original proof uses it. -/
theorem pdivMat_rowIndep_at {m n p : Nat} (g : Vec n → Vec p)
    (A : Mat m n) (h_g_diff : ∀ r : Fin m, DifferentiableAt ℝ g (A r))
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0 := by
  unfold pdivMat pdiv
  set F : Vec (m * n) → Vec (m * p) :=
    fun v => Mat.flatten ((fun M : Mat m n => fun r => g (M r)) (Mat.unflatten v))
    with hF
  set rowProj : Fin m → (Vec (m * n) →L[ℝ] Vec n) := fun k' =>
    reindexCLM (fun j' : Fin n => finProdFinEquiv (k', j'))
  -- ⭐ Moved to the TOP (it is last in the global proof): the coordinate differentiability below
  -- must be stated at the projected point, and this is what identifies it with a row of `A`.
  have h_row_A : ∀ k' : Fin m, (rowProj k') (Mat.flatten A) = A k' := by
    intro k'
    funext j'
    show Mat.flatten A (finProdFinEquiv (k', j')) = A k' j'
    show A (finProdFinEquiv.symm (finProdFinEquiv (k', j'))).1
            (finProdFinEquiv.symm (finProdFinEquiv (k', j'))).2 = A k' j'
    simp
  have h_coord : ∀ (k' : Fin m) (l' : Fin p),
      (fun v : Vec (m * n) => F v (finProdFinEquiv (k', l'))) =
      (fun w : Vec n => g w l') ∘ (rowProj k') := by
    intro k' l'
    funext v
    show Mat.flatten ((fun M : Mat m n => fun r => g (M r)) (Mat.unflatten v))
        (finProdFinEquiv (k', l')) = g ((rowProj k') v) l'
    unfold Mat.flatten
    simp only [Equiv.symm_apply_apply]
    show g (Mat.unflatten v k') l' = g ((rowProj k') v) l'
    rfl
  have h_g_l : ∀ (l' : Fin p) (k' : Fin m),
      DifferentiableAt ℝ (fun w : Vec n => g w l') ((rowProj k') (Mat.flatten A)) := by
    intro l' k'
    rw [h_row_A k']
    exact differentiableAt_pi.mp (h_g_diff k') l'
  have h_coord_diff : ∀ (k' : Fin m) (l' : Fin p),
      DifferentiableAt ℝ (fun v' : Vec (m * n) => F v' (finProdFinEquiv (k', l'))) (Mat.flatten A) := by
    intro k' l'
    rw [h_coord k' l']
    exact (h_g_l l' k').comp (Mat.flatten A) (rowProj k').differentiableAt
  have h_F_diff : DifferentiableAt ℝ F (Mat.flatten A) := by
    rw [(differentiableAt_pi : DifferentiableAt ℝ F (Mat.flatten A) ↔ _)]
    intro idx
    have h_idx : finProdFinEquiv (finProdFinEquiv.symm idx) = idx :=
      Equiv.apply_symm_apply _ _
    have h_idx' : idx = finProdFinEquiv
        ((finProdFinEquiv.symm idx).1, (finProdFinEquiv.symm idx).2) := by
      conv_lhs => rw [← h_idx]
    rw [h_idx']
    exact h_coord_diff _ _
  have h_swap :
      fderiv ℝ F (Mat.flatten A) (basisVec (finProdFinEquiv (i, j))) (finProdFinEquiv (k, l)) =
      fderiv ℝ (fun v : Vec (m * n) => F v (finProdFinEquiv (k, l))) (Mat.flatten A)
        (basisVec (finProdFinEquiv (i, j))) := by
    rw [fderiv_apply h_F_diff (finProdFinEquiv (k, l))]
    rfl
  rw [h_swap]
  rw [h_coord k l]
  rw [fderiv_comp _ (h_g_l l k) (rowProj k).differentiableAt]
  rw [(rowProj k).fderiv]
  rw [h_row_A k]
  rw [fderiv_apply (h_g_diff k) l]
  simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.proj_apply]
  by_cases hik : i = k
  · subst hik
    rw [if_pos rfl]
    have h_basis : (rowProj i) (basisVec (finProdFinEquiv (i, j))) = basisVec j := by
      funext j'
      show basisVec (finProdFinEquiv (i, j)) (finProdFinEquiv (i, j')) = basisVec j j'
      simp only [basisVec_apply]
      by_cases hjj : j' = j
      · subst hjj; simp
      · rw [if_neg hjj, if_neg ?_]
        intro heq
        apply hjj
        exact (Prod.mk.inj (finProdFinEquiv.injective heq.symm)).2.symm
    rw [h_basis]
  · rw [if_neg hik]
    have h_basis : (rowProj k) (basisVec (finProdFinEquiv (i, j))) = (0 : Vec n) := by
      funext j'
      show basisVec (finProdFinEquiv (i, j)) (finProdFinEquiv (k, j')) = (0 : ℝ)
      simp only [basisVec_apply]
      rw [if_neg]
      intro heq
      apply hik
      exact (Prod.mk.inj (finProdFinEquiv.injective heq)).1.symm
    rw [h_basis]
    simp

-- ════════════════════════════════════════════════════════════════
-- § `batchMap` at a point
-- ════════════════════════════════════════════════════════════════

/-- **`batchMap N f` is differentiable at `v`** when `f` is differentiable at each of `v`'s rows.
    The pointwise peer of `batchMap_differentiable`. -/
theorem batchMap_differentiableAt {N a b : Nat} (f : Vec a → Vec b) (v : Vec (N * a))
    (hf : ∀ r : Fin N, DifferentiableAt ℝ f (Mat.unflatten v r)) :
    DifferentiableAt ℝ (StableHLO.batchMap N f) v := by
  rw [batchMap_eq_rowwiseFlat]
  apply differentiableAt_pi.mpr
  intro idx
  have hcoord :
      (fun w : Vec (N * a) =>
          Mat.flatten ((fun A : Mat N a => fun r => f (A r)) (Mat.unflatten w)) idx)
        = (fun z : Vec a => f z (finProdFinEquiv.symm idx).2) ∘
            (reindexCLM (fun i : Fin a => finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))) := by
    funext w; rfl
  rw [hcoord]
  refine DifferentiableAt.comp v ?_ (reindexCLM _).differentiableAt
  exact differentiableAt_pi.mp (hf (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2

/-- **`batchMap`'s Jacobian is block-diagonal across the batch, at a point.** `pdivMat_rowIndep_at`
    read through `batchMap_eq_rowwiseFlat`: entry `(idx, jdx)` vanishes unless the two indices name
    the same example, and is `f`'s own entry on that example's row otherwise. -/
theorem pdiv_batchMap_at {N a b : Nat} (f : Vec a → Vec b) (v : Vec (N * a))
    (hf_diff : ∀ r : Fin N, DifferentiableAt ℝ f (Mat.unflatten v r))
    (idx : Fin (N * a)) (jdx : Fin (N * b)) :
    pdiv (StableHLO.batchMap N f) v idx jdx =
      if (finProdFinEquiv.symm idx).1 = (finProdFinEquiv.symm jdx).1 then
        pdiv f (Mat.unflatten v (finProdFinEquiv.symm idx).1)
          (finProdFinEquiv.symm idx).2 (finProdFinEquiv.symm jdx).2
      else 0 := by
  have h := pdivMat_rowIndep_at f (Mat.unflatten v) hf_diff
      (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
      (finProdFinEquiv.symm jdx).1 (finProdFinEquiv.symm jdx).2
  unfold pdivMat at h
  simp only [Mat.flatten_unflatten, Prod.mk.eta, Equiv.apply_symm_apply] at h
  rw [batchMap_eq_rowwiseFlat]
  exact h

/-- ⭐ **`batchMap N f`'s VJP at a point** — the pointwise peer of `batchMap_has_vjp`, and the lift
    a batch-separable op with a KINK needs. The backward reshapes to `[N, ·]` and runs each
    example's own `_at` backward on its own row, exactly as the global one runs `f.backward`
    row-wise.

    ⚠ Unlike `batchMap_has_vjp` this is built field by field rather than transported along
    `batchMap_eq_rowwiseFlat` with `▸`: an `Eq.mpr` blocks `.backward` from reducing, which a
    whole-net certified-backward tie later needs. -/
noncomputable def batchMap_has_vjp_at {N a b : Nat} (f : Vec a → Vec b) (v : Vec (N * a))
    (hf : ∀ r : Fin N, HasVJPAt f (Mat.unflatten v r))
    (hf_diff : ∀ r : Fin N, DifferentiableAt ℝ f (Mat.unflatten v r)) :
    HasVJPAt (StableHLO.batchMap N f) v where
  backward := fun dy idx =>
    (hf (finProdFinEquiv.symm idx).1).backward
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2
  correct := by
    intro dy idx
    set i := finProdFinEquiv.symm idx with hi
    have hsum : (∑ jdx : Fin (N * b), pdiv (StableHLO.batchMap N f) v idx jdx * dy jdx)
        = ∑ q : Fin N × Fin b,
            pdiv (StableHLO.batchMap N f) v idx (finProdFinEquiv q) * dy (finProdFinEquiv q) :=
      (Fintype.sum_equiv finProdFinEquiv
        (fun q : Fin N × Fin b =>
          pdiv (StableHLO.batchMap N f) v idx (finProdFinEquiv q) * dy (finProdFinEquiv q))
        (fun jdx : Fin (N * b) => pdiv (StableHLO.batchMap N f) v idx jdx * dy jdx)
        (fun _ => rfl)).symm
    rw [hsum, Fintype.sum_prod_type]
    have hpd : ∀ (r : Fin N) (c : Fin b),
        pdiv (StableHLO.batchMap N f) v idx (finProdFinEquiv (r, c))
          = if i.1 = r then pdiv f (Mat.unflatten v i.1) i.2 c else 0 := by
      intro r c
      rw [pdiv_batchMap_at f v hf_diff]
      simp [hi]
    simp_rw [hpd]
    have hcollapse : ∀ r : Fin N,
        (∑ c : Fin b, (if i.1 = r then pdiv f (Mat.unflatten v i.1) i.2 c else 0)
            * dy (finProdFinEquiv (r, c)))
          = if i.1 = r then
              ∑ c : Fin b, pdiv f (Mat.unflatten v i.1) i.2 c * dy (finProdFinEquiv (r, c))
            else 0 := by
      intro r; by_cases h : i.1 = r <;> simp [h]
    simp_rw [hcollapse]
    rw [Finset.sum_ite_eq Finset.univ i.1
      (fun r => ∑ c : Fin b, pdiv f (Mat.unflatten v i.1) i.2 c * dy (finProdFinEquiv (r, c)))]
    simp only [Finset.mem_univ, ite_true]
    exact (hf i.1).correct (fun c => dy (finProdFinEquiv (i.1, c))) i.2

end Proofs
