import LeanMlir.Proofs.MuonGeometry
import Mathlib.Logic.Function.Iterate

/-! # Newton–Schulz convergence, P1: the iteration is a *scalar* map in disguise

The capstone of the Muon-geometry ladder (`planning/muon_ns_convergence.md`,
`planning/muon_geometry.md`, `LeanMlir/Proofs/MuonGeometry.lean`). L1–L6 proved that the polar factor
`UVᵀ` is the *right object* — operator-norm steepest descent (L3, von Neumann), the nuclear norm's
argmax, Shampoo's single step (L5), the nearest orthogonal matrix to `G` (L6). **What remains is that
the implementation actually computes it:** Muon's matmul iteration
`X ↦ aX + b(XXᵀ)X + c(XXᵀ)²X` (`OptimizerKind.muon`, `emitMuonUpdate`) converges to `UVᵀ`.

**This file is P1 — the spectral-step lemma, the bridge that turns the whole problem scalar.** The one
idea: a Newton–Schulz step never touches the singular *directions*, only the singular *values*. With
`X = U Σ Vᵀ` (`U,V` orthonormal, `Σ = diagonal σ`), since `XXᵀ = U Σ² Uᵀ`,
```
(XXᵀ)X = U Σ³ Vᵀ,   (XXᵀ)²X = U Σ⁵ Vᵀ   ⟹   nsStep a b c X = U (diagonal (φ ∘ σ)) Vᵀ,
```
where `φ(t) = a t + b t³ + c t⁵` (`nsScalar`) is applied per singular value, with `U,V` carried along
unchanged (`nsStep_spectral`). Iterating, `nsStep^[k] X = U (diagonal (φ^[k] ∘ σ)) Vᵀ`
(`nsStep_iterate_spectral`): **matrix convergence to `UVᵀ` reduces to scalar convergence `φ^[k](σᵢ) → 1`
per singular value.** This is the same `U Σ Vᵀ ↦ U f(Σ) Vᵀ` motif as L5's `conj_diag_pow`, now for the
polynomial `φ`. The downstream scalar analysis (P2) and the matrix-continuity assembly (P3) build on
these two lemmas. All `propext / Classical.choice / Quot.sound`-clean. -/

namespace Proofs.MuonNewtonSchulz

open scoped Matrix

variable {n : ℕ}

/-- **The Newton–Schulz step.** One iteration of Muon's gradient orthogonalizer:
    `nsStep a b c X = aX + b(XXᵀ)X + c(XXᵀ)²X`, the odd matrix polynomial `X·p(XᵀX)` written via the
    Gram matrix `XXᵀ`. The classic inverse-free polar iteration is `(a,b,c) = (3/2, −1/2, 0)`; Muon's
    tuned quintic is `(3.4445, −4.7750, 2.0315)`. The quintic monomial is associated as
    `(XXᵀ)·((XXᵀ)X)` so the spectral collapse threads cleanly. -/
def nsStep (a b c : ℝ) (X : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  a • X + b • (X * Xᵀ * X) + c • (X * Xᵀ * (X * Xᵀ * X))

/-- **The scalar map a Newton–Schulz step induces on each singular value:** `φ(t) = a t + b t³ + c t⁵`.
    `nsStep_spectral` shows `nsStep a b c` acts as `nsScalar a b c` on the singular values of `X`, so
    the matrix iteration's convergence is exactly this scalar iteration's (`φ^[k](σᵢ) → 1`). -/
def nsScalar (a b c t : ℝ) : ℝ := a * t + b * t ^ 3 + c * t ^ 5

/-- **P1 — the spectral-step lemma: a Newton–Schulz step is `nsScalar` applied per singular value.**
    For `X = U (diagonal σ) Vᵀ` with `U,V` orthonormal (`UᵀU = VᵀV = 1`),
    `nsStep a b c X = U (diagonal (fun i ↦ nsScalar a b c (σ i))) Vᵀ`. The singular vectors `U,V` are
    carried through untouched; only the singular values move, by the scalar polynomial `φ`. This is
    the **only matrix-level work** in the convergence proof — everything downstream is scalar.

    The proof is pure `UᵀU = 1` / `VᵀV = 1` collapse algebra (the `conj_diag_pow` motif of L5): the
    Gram matrix `XXᵀ = U (diagonal σ²) Uᵀ`, and each higher monomial `(XXᵀ)ᵏX` collapses to
    `U (diagonal σ^{2k+1}) Vᵀ` because the inner `UᵀU` contracts to the identity; the three scalar
    coefficients `a, b, c` ride through `•` onto the diagonal and sum pointwise to `φ`. -/
theorem nsStep_spectral (a b c : ℝ) (U V : Matrix (Fin n) (Fin n) ℝ) (σ : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) :
    nsStep a b c (U * Matrix.diagonal σ * Vᵀ)
      = U * Matrix.diagonal (fun i => nsScalar a b c (σ i)) * Vᵀ := by
  -- a scalar `r •` slides through the conjugation triple `U (diag d) Vᵀ` onto the diagonal
  have hdiag_smul : ∀ (r : ℝ) (d : Fin n → ℝ),
      r • Matrix.diagonal d = Matrix.diagonal (fun i => r * d i) := by
    intro r d; ext i j
    by_cases h : i = j <;> simp [Matrix.smul_apply, h, smul_eq_mul]
  have hsmul : ∀ (r : ℝ) (d : Fin n → ℝ),
      r • (U * Matrix.diagonal d * Vᵀ) = U * Matrix.diagonal (fun i => r * d i) * Vᵀ := by
    intro r d; rw [← Matrix.smul_mul, ← Matrix.mul_smul, hdiag_smul]
  -- `(U diag p Uᵀ)(U diag q Vᵀ) = U diag (p·q) Vᵀ` — the `UᵀU = 1` contraction
  have hcollapse : ∀ (p q : Fin n → ℝ),
      (U * Matrix.diagonal p * Uᵀ) * (U * Matrix.diagonal q * Vᵀ)
        = U * Matrix.diagonal (fun i => p i * q i) * Vᵀ := by
    intro p q
    rw [show (U * Matrix.diagonal p * Uᵀ) * (U * Matrix.diagonal q * Vᵀ)
          = U * (Matrix.diagonal p * (Uᵀ * U) * Matrix.diagonal q) * Vᵀ from by
            simp only [Matrix.mul_assoc],
        hU, Matrix.mul_one, Matrix.diagonal_mul_diagonal]
  -- the singular values only: `Xᵀ`, the Gram `XXᵀ`, and the cubic/quintic monomials stay `U diag(·) Vᵀ`
  have hXt : (U * Matrix.diagonal σ * Vᵀ)ᵀ = V * Matrix.diagonal σ * Uᵀ := by
    simp only [Matrix.transpose_mul, Matrix.diagonal_transpose, Matrix.transpose_transpose,
      Matrix.mul_assoc]
  have hXXt : (U * Matrix.diagonal σ * Vᵀ) * (U * Matrix.diagonal σ * Vᵀ)ᵀ
        = U * Matrix.diagonal (fun i => σ i * σ i) * Uᵀ := by
    rw [hXt, show (U * Matrix.diagonal σ * Vᵀ) * (V * Matrix.diagonal σ * Uᵀ)
          = U * (Matrix.diagonal σ * (Vᵀ * V) * Matrix.diagonal σ) * Uᵀ from by
            simp only [Matrix.mul_assoc], hV, Matrix.mul_one, Matrix.diagonal_mul_diagonal]
  have hcube : (U * Matrix.diagonal σ * Vᵀ) * (U * Matrix.diagonal σ * Vᵀ)ᵀ
                * (U * Matrix.diagonal σ * Vᵀ)
        = U * Matrix.diagonal (fun i => σ i * σ i * σ i) * Vᵀ := by
    rw [hXXt, hcollapse]
  have hquint : (U * Matrix.diagonal σ * Vᵀ) * (U * Matrix.diagonal σ * Vᵀ)ᵀ
                * ((U * Matrix.diagonal σ * Vᵀ) * (U * Matrix.diagonal σ * Vᵀ)ᵀ
                   * (U * Matrix.diagonal σ * Vᵀ))
        = U * Matrix.diagonal (fun i => σ i * σ i * (σ i * σ i * σ i)) * Vᵀ := by
    rw [hcube, hXXt, hcollapse]
  -- collect the three monomials onto a single diagonal, then identify it with `φ ∘ σ`
  have hsum3 : ∀ d1 d2 d3 : Fin n → ℝ,
      U * Matrix.diagonal d1 * Vᵀ + U * Matrix.diagonal d2 * Vᵀ + U * Matrix.diagonal d3 * Vᵀ
        = U * Matrix.diagonal (fun i => d1 i + d2 i + d3 i) * Vᵀ := by
    intro d1 d2 d3
    have hd : Matrix.diagonal (fun i => d1 i + d2 i + d3 i)
        = Matrix.diagonal d1 + Matrix.diagonal d2 + Matrix.diagonal d3 := by
      ext i j; by_cases h : i = j <;> simp [Matrix.add_apply, h]
    rw [hd, Matrix.mul_add, Matrix.mul_add, Matrix.add_mul, Matrix.add_mul]
  simp only [nsStep]
  rw [hquint, hcube, hsmul, hsmul, hsmul, hsum3,
      show (fun i => a * σ i + b * (σ i * σ i * σ i) + c * (σ i * σ i * (σ i * σ i * σ i)))
        = (fun i => nsScalar a b c (σ i)) from funext fun i => by simp only [nsScalar]; ring]

/-- **P1, iterated: `k` Newton–Schulz steps act as `nsScalar^[k]` per singular value.**
    `(nsStep a b c)^[k] (U (diagonal σ) Vᵀ) = U (diagonal (fun i ↦ (nsScalar a b c)^[k] (σ i))) Vᵀ`.
    A one-line induction reusing `nsStep_spectral` at each step: the singular vectors `U,V` are
    invariant under the whole orbit, so **convergence of the matrix iteration `nsStep^[k] X → UVᵀ`
    reduces to the scalar fixed-point convergence `(nsScalar a b c)^[k] (σ i) → 1`** for each singular
    value — the entry point for P2 (the cubic monotone argument) and P3 (the matrix-continuity glue). -/
theorem nsStep_iterate_spectral (a b c : ℝ) (U V : Matrix (Fin n) (Fin n) ℝ) (σ : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (k : ℕ) :
    (nsStep a b c)^[k] (U * Matrix.diagonal σ * Vᵀ)
      = U * Matrix.diagonal (fun i => (nsScalar a b c)^[k] (σ i)) * Vᵀ := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih, nsStep_spectral a b c U V _ hU hV]
    have : (fun i => nsScalar a b c ((nsScalar a b c)^[k] (σ i)))
        = (fun i => (nsScalar a b c)^[k + 1] (σ i)) :=
      funext fun i => (Function.iterate_succ_apply' (nsScalar a b c) k (σ i)).symm
    rw [this]

end Proofs.MuonNewtonSchulz
