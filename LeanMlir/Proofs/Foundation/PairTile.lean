import LeanMlir.Proofs.Foundation.MLP

/-!
# Pair-tile VJP

`Layer.pairTile` (the distogram stem): two residue blocks `Xi, Xj : Mat L D` — the host's
features, constants of the step — and two weights `W, Wj : Mat D C`; the pair map is
`y i j c = (Xi · W) i c + (Xj · Wj) j c`. It is linear in each weight, so each weight's VJP is
the adjoint of a linear map: `dW = Xiᵀ · (Σ_j dy)` and `dWj = Xjᵀ · (Σ_i dy)` — the cotangent
summed over the axis the weight was broadcast along, then the dense weight rule. These are the
two `stablehlo.reduce` + `dot_general` pairs the pairTile backward emits (`%ptb_du` / `%d_W`,
`%ptb_dv` / `%d_Wj`). The layer takes no input gradient: its input is the feature block.

Both witnesses are `pdiv_of_affine` with the other block's term as the constant, the shape
of `pdiv_dense_W`; the only work is the row-major `(i, (j, c))` index of the flat pair map.
-/

open Finset BigOperators

namespace Proofs
namespace PairTile

variable {L D C K : Nat}

/-- The `(i, j, c)` behind a flat index of a row-major `[L, L, C]` pair map. -/
@[reducible] def oidx (o : Fin (L * (L * C))) : Fin L × Fin L × Fin C :=
  ((finProdFinEquiv.symm o).1, (finProdFinEquiv.symm (finProdFinEquiv.symm o).2).1,
   (finProdFinEquiv.symm (finProdFinEquiv.symm o).2).2)

@[simp] theorem oidx_mk (a b : Fin L) (c : Fin C) :
    oidx (finProdFinEquiv (a, finProdFinEquiv (b, c)) : Fin (L * (L * C))) = (a, b, c) := by
  simp [oidx]

/-- A `Fin (a·(b·c))` sum is the row-major triple sum, the pair map's `(i, (j, c))` layout
    (`sum_finProdFinEquiv₃` is the left-nested `(a·b)·c`). -/
theorem sum_finProdFinEquiv_r {M : Type*} [AddCommMonoid M] {a b c : Nat}
    (f : Fin (a * (b * c)) → M) :
    ∑ k, f k = ∑ i : Fin a, ∑ j : Fin b, ∑ l : Fin c,
      f (finProdFinEquiv (i, finProdFinEquiv (j, l))) := by
  rw [sum_finProdFinEquiv]
  exact Finset.sum_congr rfl (fun _ _ => sum_finProdFinEquiv _)

/-- The pair map as a function of the flattened `i`-block weight (the `j` term is constant). -/
noncomputable def tileW (Xi Xj : Mat L D) (Wj : Mat D C) (v : Vec (D * C)) :
    Vec (L * (L * C)) :=
  fun o => Mat.mul Xi (Mat.unflatten v) (oidx o).1 (oidx o).2.2 +
    Mat.mul Xj Wj (oidx o).2.1 (oidx o).2.2

/-- The pair map as a function of the flattened `j`-block weight (the `i` term is constant). -/
noncomputable def tileWj (Xi Xj : Mat L D) (W : Mat D C) (v : Vec (D * C)) :
    Vec (L * (L * C)) :=
  fun o => Mat.mul Xi W (oidx o).1 (oidx o).2.2 +
    Mat.mul Xj (Mat.unflatten v) (oidx o).2.1 (oidx o).2.2

/-- `dW (d, c) = Σ_i Xi i d · Σ_j dy i j c` — the cotangent summed over `j`, contracted with
    `Xi` over `i` (the emitted `reduce … dimensions = [2]` then `dot_general … [0, 1] x [0, 1]`,
    per sample). -/
noncomputable def gradW (Xi : Mat L D) (dy : Vec (L * (L * C))) : Vec (D * C) :=
  fun k => ∑ a : Fin L, Xi a (finProdFinEquiv.symm k).1 *
    ∑ b : Fin L, dy (finProdFinEquiv (a, finProdFinEquiv (b, (finProdFinEquiv.symm k).2)))

/-- `dWj (d, c) = Σ_j Xj j d · Σ_i dy i j c` — summed over `i`, contracted with `Xj` over `j`. -/
noncomputable def gradWj (Xj : Mat L D) (dy : Vec (L * (L * C))) : Vec (D * C) :=
  fun k => ∑ b : Fin L, Xj b (finProdFinEquiv.symm k).1 *
    ∑ a : Fin L, dy (finProdFinEquiv (a, finProdFinEquiv (b, (finProdFinEquiv.symm k).2)))

/-- `∂ y_{i j c} / ∂ W_{d c'} = Xi i d · δ(c, c')`. -/
theorem pdiv_tileW (Xi Xj : Mat L D) (Wj : Mat D C) (v : Vec (D * C))
    (k : Fin (D * C)) (o : Fin (L * (L * C))) :
    pdiv (tileW Xi Xj Wj) v k o =
      if (oidx o).2.2 = (finProdFinEquiv.symm k).2
      then Xi (oidx o).1 (finProdFinEquiv.symm k).1 else 0 := by
  rw [show tileW Xi Xj Wj =
      fun v => (fun o => Mat.mul Xi (Mat.unflatten v) (oidx o).1 (oidx o).2.2) +
        (fun o => Mat.mul Xj Wj (oidx o).2.1 (oidx o).2.2) from rfl,
    pdiv_of_affine _ _
      (fun _ _ => by funext; simp [Mat.mul, Mat.unflatten, mul_add, Finset.sum_add_distrib])
      (fun _ _ => by funext; simp [Mat.mul, Mat.unflatten, Finset.mul_sum, mul_left_comm])]
  simp only [Mat.mul, Mat.unflatten, basisVec, ← Equiv.eq_symm_apply, Prod.ext_iff]
  simp [ite_and, Finset.sum_ite_eq']

/-- `∂ y_{i j c} / ∂ Wj_{d c'} = Xj j d · δ(c, c')`. -/
theorem pdiv_tileWj (Xi Xj : Mat L D) (W : Mat D C) (v : Vec (D * C))
    (k : Fin (D * C)) (o : Fin (L * (L * C))) :
    pdiv (tileWj Xi Xj W) v k o =
      if (oidx o).2.2 = (finProdFinEquiv.symm k).2
      then Xj (oidx o).2.1 (finProdFinEquiv.symm k).1 else 0 := by
  rw [show tileWj Xi Xj W =
      fun v => (fun o => Mat.mul Xj (Mat.unflatten v) (oidx o).2.1 (oidx o).2.2) +
        (fun o => Mat.mul Xi W (oidx o).1 (oidx o).2.2) from by
        funext v o; simp [tileWj, add_comm],
    pdiv_of_affine _ _
      (fun _ _ => by funext; simp [Mat.mul, Mat.unflatten, mul_add, Finset.sum_add_distrib])
      (fun _ _ => by funext; simp [Mat.mul, Mat.unflatten, Finset.mul_sum, mul_left_comm])]
  simp only [Mat.mul, Mat.unflatten, basisVec, ← Equiv.eq_symm_apply, Prod.ext_iff]
  simp [ite_and, Finset.sum_ite_eq']

/-- **The `i`-block weight's VJP** — proved. Backward `gradW`. -/
noncomputable def tileWHasVJP (Xi Xj : Mat L D) (Wj : Mat D C) : HasVJP (tileW Xi Xj Wj) where
  backward := fun _v dy => gradW Xi dy
  correct := by
    intro v dy k
    simp only [gradW]
    simp_rw [pdiv_tileW]
    rw [sum_finProdFinEquiv_r]
    simp [Finset.mul_sum, ite_mul, Finset.sum_ite_eq']

/-- **The `j`-block weight's VJP** — proved. Backward `gradWj`. -/
noncomputable def tileWjHasVJP (Xi Xj : Mat L D) (W : Mat D C) : HasVJP (tileWj Xi Xj W) where
  backward := fun _v dy => gradWj Xj dy
  correct := by
    intro v dy k
    simp only [gradWj]
    simp_rw [pdiv_tileWj]
    rw [sum_finProdFinEquiv_r]
    simp [Finset.mul_sum, ite_mul, Finset.sum_ite_eq']
    exact Finset.sum_comm

/-! ## Host pair planes (`Layer.pairTile`'s `pairIn`)

With `K` host pair planes the layer's output is the tile with the planes `P : Vec (L·(L·K))`
appended — constants of the step, like the feature blocks. In the flat layout the plane block
follows the tile block (`finSumFinEquiv`): the output is `tileW … v t` at `inl t` and `P q` at
`inr q`. Each weight's Jacobian is the tile's own on the first block and zero on the second,
so each VJP is the tile's backward on the cotangent's tile block (`tileBlock`) — the
`stablehlo.slice` of the first `C` channels the pairTile backward emits when `pairIn > 0`. -/

/-- The tile with `K` constant planes appended, as a function of the `i`-block weight. -/
noncomputable def tileWPair (Xi Xj : Mat L D) (Wj : Mat D C) (P : Vec (L * (L * K)))
    (v : Vec (D * C)) : Vec (L * (L * C) + L * (L * K)) :=
  fun o => Sum.elim (tileW Xi Xj Wj v) P (finSumFinEquiv.symm o)

/-- The same as a function of the `j`-block weight. -/
noncomputable def tileWjPair (Xi Xj : Mat L D) (W : Mat D C) (P : Vec (L * (L * K)))
    (v : Vec (D * C)) : Vec (L * (L * C) + L * (L * K)) :=
  fun o => Sum.elim (tileWj Xi Xj W v) P (finSumFinEquiv.symm o)

/-- The cotangent's tile block: its first `L·(L·C)` entries. -/
def tileBlock (dy : Vec (L * (L * C) + L * (L * K))) : Vec (L * (L * C)) :=
  fun t => dy (finSumFinEquiv (Sum.inl t))

/-- The `i`-block weight's Jacobian with planes appended: the tile's on the tile block, zero on
    the plane block. -/
theorem pdiv_tileWPair (Xi Xj : Mat L D) (Wj : Mat D C) (P : Vec (L * (L * K))) (v : Vec (D * C))
    (k : Fin (D * C)) (o : Fin (L * (L * C) + L * (L * K))) :
    pdiv (tileWPair Xi Xj Wj P) v k o =
      Sum.elim (fun t => pdiv (tileW Xi Xj Wj) v k t) (fun _ => (0 : ℝ)) (finSumFinEquiv.symm o) := by
  rw [show tileWPair Xi Xj Wj P =
      fun v => (fun o => Sum.elim (fun t => Mat.mul Xi (Mat.unflatten v) (oidx t).1 (oidx t).2.2)
          (fun _ => (0 : ℝ)) (finSumFinEquiv.symm o)) +
        (fun o => Sum.elim (fun t => Mat.mul Xj Wj (oidx t).2.1 (oidx t).2.2) P (finSumFinEquiv.symm o)) from by
        funext v o; rcases h : finSumFinEquiv.symm o with t | q <;> simp [tileWPair, tileW, h],
    pdiv_of_affine _ _
      (fun _ _ => by funext o; rcases h : finSumFinEquiv.symm o with t | q <;> simp [Mat.mul, Mat.unflatten, mul_add, Finset.sum_add_distrib, h])
      (fun _ _ => by funext o; rcases h : finSumFinEquiv.symm o with t | q <;> simp [Mat.mul, Mat.unflatten, Finset.mul_sum, mul_left_comm, h])]
  rcases h : finSumFinEquiv.symm o with t | q
  · simp only [Sum.elim_inl, pdiv_tileW]; simp only [Mat.mul, Mat.unflatten, basisVec, ← Equiv.eq_symm_apply, Prod.ext_iff]; simp [ite_and, Finset.sum_ite_eq']
  · simp

/-- The `j`-block weight's Jacobian with planes appended. -/
theorem pdiv_tileWjPair (Xi Xj : Mat L D) (W : Mat D C) (P : Vec (L * (L * K))) (v : Vec (D * C))
    (k : Fin (D * C)) (o : Fin (L * (L * C) + L * (L * K))) :
    pdiv (tileWjPair Xi Xj W P) v k o =
      Sum.elim (fun t => pdiv (tileWj Xi Xj W) v k t) (fun _ => (0 : ℝ)) (finSumFinEquiv.symm o) := by
  rw [show tileWjPair Xi Xj W P =
      fun v => (fun o => Sum.elim (fun t => Mat.mul Xj (Mat.unflatten v) (oidx t).2.1 (oidx t).2.2)
          (fun _ => (0 : ℝ)) (finSumFinEquiv.symm o)) +
        (fun o => Sum.elim (fun t => Mat.mul Xi W (oidx t).1 (oidx t).2.2) P (finSumFinEquiv.symm o)) from by
        funext v o; rcases h : finSumFinEquiv.symm o with t | q <;> simp [tileWjPair, tileWj, h, add_comm],
    pdiv_of_affine _ _
      (fun _ _ => by funext o; rcases h : finSumFinEquiv.symm o with t | q <;> simp [Mat.mul, Mat.unflatten, mul_add, Finset.sum_add_distrib, h])
      (fun _ _ => by funext o; rcases h : finSumFinEquiv.symm o with t | q <;> simp [Mat.mul, Mat.unflatten, Finset.mul_sum, mul_left_comm, h])]
  rcases h : finSumFinEquiv.symm o with t | q
  · simp only [Sum.elim_inl, pdiv_tileWj]; simp only [Mat.mul, Mat.unflatten, basisVec, ← Equiv.eq_symm_apply, Prod.ext_iff]; simp [ite_and, Finset.sum_ite_eq']
  · simp

/-- **The `i`-block weight's VJP with planes appended** — proved. Backward: `gradW` on the
    cotangent's tile block. -/
noncomputable def tileWPairHasVJP (Xi Xj : Mat L D) (Wj : Mat D C) (P : Vec (L * (L * K))) :
    HasVJP (tileWPair Xi Xj Wj P) where
  backward := fun _v dy => gradW Xi (tileBlock dy)
  correct := by
    intro v dy k
    rw [← finSumFinEquiv.sum_comp, Fintype.sum_sum_type]
    simp only [pdiv_tileWPair, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr, zero_mul,
      Finset.sum_const_zero, add_zero]
    exact (tileWHasVJP Xi Xj Wj).correct v (tileBlock dy) k

/-- **The `j`-block weight's VJP with planes appended** — proved. Backward: `gradWj` on the
    cotangent's tile block. -/
noncomputable def tileWjPairHasVJP (Xi Xj : Mat L D) (W : Mat D C) (P : Vec (L * (L * K))) :
    HasVJP (tileWjPair Xi Xj W P) where
  backward := fun _v dy => gradWj Xj (tileBlock dy)
  correct := by
    intro v dy k
    rw [← finSumFinEquiv.sum_comp, Fintype.sum_sum_type]
    simp only [pdiv_tileWjPair, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr, zero_mul,
      Finset.sum_const_zero, add_zero]
    exact (tileWjHasVJP Xi Xj W).correct v (tileBlock dy) k

end PairTile
end Proofs
