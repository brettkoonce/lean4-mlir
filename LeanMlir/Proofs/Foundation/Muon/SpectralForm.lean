module

public import Mathlib.Data.Matrix.Mul
public import Mathlib.Basic.Real.Basic

@[expose] public section

/-! # Spectral-form algebra for the Muon files

The two moves `MuonGeometry` (`shampoo_eq_muon`) and `MuonNewtonSchulz` (`nsStep_spectral`) both make
on a matrix written as `A (diagonal p) Bᵀ`: transposing it swaps the outer factors
(`svdForm_transpose`), and a product of two such forms whose inner factors contract (`BᵀC = 1`)
multiplies the diagonals pointwise (`svdForm_mul`).
-/

namespace Proofs.MuonSpectral

open scoped Matrix

variable {n : ℕ}

/-- **Transposing a spectral form swaps its outer factors**: `(A Σ Bᵀ)ᵀ = B Σ Aᵀ`. -/
theorem svdForm_transpose (A B : Matrix (Fin n) (Fin n) ℝ) (p : Fin n → ℝ) :
    (A * Matrix.diagonal p * Bᵀ)ᵀ = B * Matrix.diagonal p * Aᵀ := by
  simp only [Matrix.transpose_mul, Matrix.diagonal_transpose, Matrix.transpose_transpose,
    Matrix.mul_assoc]

/-- **Two spectral forms multiply pointwise** when the inner factors contract:
    `BᵀC = 1 ⟹ (A (diag p) Bᵀ)(C (diag q) Dᵀ) = A (diag p·q) Dᵀ`. -/
theorem svdForm_mul (A B C D : Matrix (Fin n) (Fin n) ℝ) (p q : Fin n → ℝ) (h : Bᵀ * C = 1) :
    (A * Matrix.diagonal p * Bᵀ) * (C * Matrix.diagonal q * Dᵀ)
      = A * Matrix.diagonal (fun i => p i * q i) * Dᵀ := by
  rw [show (A * Matrix.diagonal p * Bᵀ) * (C * Matrix.diagonal q * Dᵀ)
        = A * (Matrix.diagonal p * (Bᵀ * C) * Matrix.diagonal q) * Dᵀ from by
          simp only [Matrix.mul_assoc],
      h, Matrix.mul_one, Matrix.diagonal_mul_diagonal]

end Proofs.MuonSpectral
