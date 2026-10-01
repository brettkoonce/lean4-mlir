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

variable {L D C : Nat}

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

end PairTile
end Proofs
