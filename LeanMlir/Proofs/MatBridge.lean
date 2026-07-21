import LeanMlir.Proofs.Foundation.Tensor
import Mathlib.Data.Matrix.Mul

/-!
# `Proofs.Mat` ↔ Mathlib `Matrix` bridge

The proof suite works over `Mat m n := Fin m → Fin n → ℝ` (Tensor.lean),
which is *definitionally* Mathlib's `Matrix (Fin m) (Fin n) ℝ`. The suite
defines its own `mulVec` / `outer` / `mul` / `transpose` on that function
type rather than reusing Mathlib's. The four lemmas here reconcile the
operation names, so a downstream consumer can apply Mathlib's matrix-algebra
API (`mul_assoc`, `transpose_transpose`, `mulVec_mulVec`, …) to the suite's
outputs without re-deriving the correspondence by hand.

Opt-in by design: the core suite does **not** import this file, so
`Tensor.lean`'s import surface — and every build that doesn't need Mathlib's
`Matrix` — stays exactly as it was. Import `LeanMlir.Proofs.MatBridge` only
when you want the interop.
-/

open Matrix

namespace Proofs

variable {m n p : Nat}

/-! `Matrix.of` is the identity equiv `(Fin m → Fin n → ℝ) ≃ Matrix …`; it
carries no data (`Matrix.of A` is defeq to `A`) but lets instance resolution
see the value as a `Matrix`, so Mathlib's `*ᵥ` / `ᵀ` / `*` notation applies. -/

/-- `Mat.mulVec` is Mathlib's `Matrix.mulVec` (`*ᵥ`). -/
theorem Mat.mulVec_eq (A : Mat m n) (v : Vec n) :
    Mat.mulVec A v = Matrix.of A *ᵥ v := by
  funext i; simp [Mat.mulVec, Matrix.mulVec, Matrix.of_apply, dotProduct]

/-- `Mat.transpose` is Mathlib's `Matrix.transpose` (`ᵀ`). -/
theorem Mat.transpose_eq (A : Mat m n) :
    Mat.transpose A = (Matrix.of A)ᵀ := by
  funext j i; simp [Mat.transpose, Matrix.transpose_apply, Matrix.of_apply]

/-- `Mat.outer` is Mathlib's `Matrix.vecMulVec` (outer product). -/
theorem Mat.outer_eq (u : Vec m) (v : Vec n) :
    Mat.outer u v = Matrix.vecMulVec u v := by
  funext i j; simp [Mat.outer, Matrix.vecMulVec_apply]

/-- `Mat.mul` is Mathlib's matrix multiplication (`*`). -/
theorem Mat.mul_eq (A : Mat m n) (B : Mat n p) :
    Mat.mul A B = Matrix.of A * Matrix.of B := by
  funext i k; simp [Mat.mul, Matrix.mul_apply, Matrix.of_apply]

end Proofs
