import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Analysis.Matrix.Spectrum
import Mathlib.Analysis.Matrix.PosDef
import Mathlib.Analysis.Matrix.Order

/-! # Muon geometry: the optimizer as steepest descent under a norm

The geometric motivation for Muon (`planning/muon_geometry.md`), in the unifying frame **every
optimizer is steepest descent under a choice of norm**: the update direction is the dual-norm
maximizer `d⋆ = argmax_{‖d‖≤1} ⟨g,d⟩`, with optimal value the dual norm `‖g‖_*`.

| optimizer | step norm | `d⋆` | here |
|---|---|---|---|
| SGD | Euclidean `‖·‖₂` | `g/‖g‖`, value `‖g‖₂` | `steepest_l2_*` |
| sign / Adam-ish | `‖·‖∞` | `sign(g)`, value `‖g‖₁` | `steepest_linf_*` |
| **Muon** | operator norm | **polar factor `UVᵀ`**, value nuclear `Σσᵢ` | `muon_polar_achieves_nuclear` |

The SGD and sign rungs are the framework, proven outright. The Muon rung's *achievability* — the
polar factor `UVᵀ` of `G = UΣVᵀ` realizes the nuclear norm `Σσᵢ` — is proven here **given an SVD**
(pure trace algebra, `muon_polar_achieves_nuclear`).

**L4 (this layer): the SVD is now constructed, not hypothesized** — for an invertible (full-rank) `G`,
`svd_of_isUnit` builds `U, V` orthogonal and `s ≥ 0` with `G = U (diagonal s) Vᵀ` out of Mathlib's
spectral theorem of `GᵀG`: `V` = eigenvector basis, `sᵢ = √λᵢ` the singular values, `U = G V Σ⁻¹`.
No matrix square root is needed — only the spectral decomposition, scalar `√`, and diagonal inverses
(invertibility makes every `λᵢ > 0`, so `Σ⁻¹` exists). Composing with the achievability half gives
`muon_polar_achieves_nuclear_of_isUnit`: for any invertible `G`, the polar factor `UVᵀ` (Muon's
update direction) pairs with `G` to the nuclear norm `Σσᵢ` — the SVD hypothesis fully discharged.

The matching upper bound (it is the *max*: von Neumann's trace inequality, needs the matrix operator
norm) and the singular `G` case (the orthonormal completion of `U`) are the remaining layers. All
`propext / Classical.choice / Quot.sound`-clean. -/

namespace Proofs.MuonGeometry

open scoped BigOperators InnerProductSpace Matrix

variable {n : ℕ}

-- ════════════════════════════════════════════════════════════════
-- § L1 — SGD: steepest descent under the Euclidean norm (Cauchy–Schwarz)
-- ════════════════════════════════════════════════════════════════

/-- **Euclidean steepest ascent is bounded by `‖g‖`.** Over the unit `‖·‖₂` ball, no direction
    beats the gradient: `⟨g,d⟩ ≤ ‖g‖`. Pure Cauchy–Schwarz — the geometry behind plain SGD. -/
theorem steepest_l2_bound (g d : EuclideanSpace ℝ (Fin n)) (hd : ‖d‖ ≤ 1) :
    ⟪g, d⟫_ℝ ≤ ‖g‖ := by
  calc ⟪g, d⟫_ℝ ≤ ‖g‖ * ‖d‖ := real_inner_le_norm g d
    _ ≤ ‖g‖ * 1 := mul_le_mul_of_nonneg_left hd (norm_nonneg g)
    _ = ‖g‖ := mul_one _

/-- **…and the normalized gradient attains it.** `g/‖g‖` is a unit vector with `⟨g, g/‖g‖⟩ = ‖g‖`,
    so the SGD direction `g/‖g‖` is the steepest-ascent maximizer and the dual norm is `‖g‖₂`. -/
theorem steepest_l2_attained (g : EuclideanSpace ℝ (Fin n)) (hg : g ≠ 0) :
    ‖(‖g‖⁻¹ • g)‖ = 1 ∧ ⟪g, ‖g‖⁻¹ • g⟫_ℝ = ‖g‖ := by
  have hgn : ‖g‖ ≠ 0 := norm_ne_zero_iff.mpr hg
  refine ⟨?_, ?_⟩
  · rw [norm_smul, Real.norm_eq_abs, abs_inv, abs_of_nonneg (norm_nonneg g),
        inv_mul_cancel₀ hgn]
  · rw [real_inner_smul_right, real_inner_self_eq_norm_mul_norm]
    field_simp

-- ════════════════════════════════════════════════════════════════
-- § L2 — sign descent: steepest descent under the `‖·‖∞` norm (→ `‖·‖₁`)
-- ════════════════════════════════════════════════════════════════

/-- **`‖·‖∞`-steepest ascent is bounded by `Σ|gᵢ|`.** Over the box `|dᵢ| ≤ 1`, the pairing
    `Σ gᵢ dᵢ ≤ Σ|gᵢ| = ‖g‖₁`. The geometry behind sign and Adam-style coordinate updates. -/
theorem steepest_linf_bound (g d : Fin n → ℝ) (hd : ∀ i, |d i| ≤ 1) :
    ∑ i, g i * d i ≤ ∑ i, |g i| := by
  refine Finset.sum_le_sum (fun i _ => ?_)
  calc g i * d i ≤ |g i * d i| := le_abs_self _
    _ = |g i| * |d i| := abs_mul _ _
    _ ≤ |g i| * 1 := mul_le_mul_of_nonneg_left (hd i) (abs_nonneg _)
    _ = |g i| := mul_one _

/-- **…and `sign(g)` attains it.** The box-corner `dᵢ = ±1 = sign(gᵢ)` is feasible and gives
    `Σ gᵢ·sign(gᵢ) = Σ|gᵢ|`, so the sign update is the `‖·‖∞`-steepest direction, dual norm `‖g‖₁`. -/
theorem steepest_linf_attained (g : Fin n → ℝ) :
    (∀ i, |(if 0 ≤ g i then (1:ℝ) else -1)| ≤ 1) ∧
      ∑ i, g i * (if 0 ≤ g i then (1:ℝ) else -1) = ∑ i, |g i| := by
  refine ⟨fun i => ?_, Finset.sum_congr rfl (fun i _ => ?_)⟩
  · split <;> simp
  · split <;> rename_i h
    · rw [mul_one, abs_of_nonneg h]
    · rw [mul_neg_one]; exact (abs_of_neg (not_le.mp h)).symm

-- ════════════════════════════════════════════════════════════════
-- § L3 — Muon: the polar factor `UVᵀ` realizes the nuclear norm (given an SVD)
-- ════════════════════════════════════════════════════════════════

/-- Frobenius inner product `⟨A,B⟩_F = tr(Aᵀ B) = Σᵢⱼ AᵢⱼBᵢⱼ` — the inner product the update
    `⟨∇L, D⟩` is taken in. -/
def fInner (A B : Matrix (Fin n) (Fin n) ℝ) : ℝ := (Aᵀ * B).trace

/-- **Muon's update is the steepest ascent in operator-norm geometry — the achievability half.**
    Given an SVD `G = U Σ Vᵀ` (`U,V` orthogonal, `Σ = diagonal s`, `s ≥ 0`), the **polar factor**
    `U Vᵀ` — exactly Muon's update direction — pairs with `G` to give the **nuclear norm** `Σσᵢ`:
    `⟨G, UVᵀ⟩_F = Σ sᵢ`. (`UVᵀ` is on the operator-norm sphere, and by von Neumann's trace
    inequality `Σσᵢ` is the *max* of `⟨G,·⟩` over that ball — the upper half, next layer.) -/
theorem muon_polar_achieves_nuclear
    (U V : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) :
    fInner (U * Matrix.diagonal s * Vᵀ) (U * Vᵀ) = ∑ i, s i := by
  unfold fInner
  -- (U Σ Vᵀ)ᵀ (U Vᵀ) = V Σ (Uᵀ U) Vᵀ = V Σ Vᵀ, then trace (V Σ Vᵀ) = trace (Σ (Vᵀ V)) = Σ sᵢ
  simp only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.diagonal_transpose,
    Matrix.mul_assoc]
  rw [← Matrix.mul_assoc Uᵀ U Vᵀ, hU, Matrix.one_mul, Matrix.trace_mul_comm, Matrix.mul_assoc,
    hV, Matrix.mul_one, Matrix.trace_diagonal]

-- ════════════════════════════════════════════════════════════════
-- § L4 — build the SVD: `G = U (diagonal s) Vᵀ` from the spectral theorem of `GᵀG`
-- ════════════════════════════════════════════════════════════════

/-- **The SVD of an invertible matrix, constructed.** For invertible `G`, there are orthogonal
    `U, V` (`UᵀU = VᵀV = 1`) and nonnegative singular values `s` with `G = U (diagonal s) Vᵀ`.

    The build is spectral, not a black box: `A := GᵀG` is symmetric positive definite (positive
    definite ⇐ `G` invertible), so the spectral theorem gives an orthogonal eigenbasis `V` and
    eigenvalues `λ` with `A = V (diagonal λ) Vᵀ`, all `λᵢ > 0`. Set the singular values
    `sᵢ := √λᵢ` and `U := G V Σ⁻¹` (`Σ⁻¹ = diagonal (1/sᵢ)`, which exists because `λᵢ > 0`). Then
    `UᵀU = Σ⁻¹ (Vᵀ A V) Σ⁻¹ = Σ⁻¹ (diagonal λ) Σ⁻¹ = 1` and `U Σ Vᵀ = G V Vᵀ = G`. **No matrix
    square root is needed** — only the spectral decomposition, the scalar `√`, and diagonal
    inverses. This discharges the SVD hypothesis of `muon_polar_achieves_nuclear` for full-rank `G`
    (the singular case needs the orthonormal completion of `U`, the remaining layer). -/
theorem svd_of_isUnit (G : Matrix (Fin n) (Fin n) ℝ) (hG : IsUnit G) :
    ∃ (U V : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ),
      Uᵀ * U = 1 ∧ Vᵀ * V = 1 ∧ (∀ i, 0 ≤ s i) ∧ G = U * Matrix.diagonal s * Vᵀ := by
  -- `A := GᵀG` is symmetric and positive definite (PSD always, PD because `G` is a unit).
  have hAherm : (Gᵀ * G).IsHermitian := by
    have := Matrix.isHermitian_conjTranspose_mul_self G
    rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at this
  have hPSD : (Gᵀ * G).PosSemidef := by
    have := Matrix.posSemidef_conjTranspose_mul_self G
    rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at this
  have hAunit : IsUnit (Gᵀ * G) := ((Matrix.isUnit_transpose G).mpr hG).mul hG
  have hPD : (Gᵀ * G).PosDef := hPSD.posDef_iff_isUnit.mpr hAunit
  have hlampos : ∀ i, 0 < hAherm.eigenvalues i := hAherm.posDef_iff_eigenvalues_pos.mp hPD
  -- Spectral data: `V` the orthogonal eigenbasis, `lam` the (positive) eigenvalues of `GᵀG`.
  set V : Matrix (Fin n) (Fin n) ℝ := (hAherm.eigenvectorUnitary : Matrix (Fin n) (Fin n) ℝ)
    with hVdef
  set lam := hAherm.eigenvalues with hlamdef
  have hstar : star V = Vᵀ := by
    rw [Matrix.star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial]
  have hVtV : Vᵀ * V = 1 := by
    have h2 := hAherm.eigenvectorUnitary.2
    rw [Matrix.mem_unitaryGroup_iff'] at h2; rwa [hstar] at h2
  have hVVt : V * Vᵀ = 1 := by
    have h2 := hAherm.eigenvectorUnitary.2
    rw [Matrix.mem_unitaryGroup_iff] at h2; rwa [hstar] at h2
  have hAeq : Gᵀ * G = V * Matrix.diagonal lam * Vᵀ := by
    have hsp := hAherm.spectral_theorem
    rw [Unitary.conjStarAlgAut_apply, hstar] at hsp
    have hof : (RCLike.ofReal ∘ hAherm.eigenvalues : Fin n → ℝ) = lam := by funext i; simp [hlamdef]
    rw [hof] at hsp; exact hsp
  -- Singular values `s = √λ` (all positive, with `sᵢ² = λᵢ`); `Σ⁻¹ = diagonal (1/sᵢ)`; `U = G V Σ⁻¹`.
  set s : Fin n → ℝ := fun i => Real.sqrt (lam i) with hsdef
  have hspos : ∀ i, 0 < s i := fun i => Real.sqrt_pos.mpr (hlampos i)
  have hsq : ∀ i, s i * s i = lam i := fun i => Real.mul_self_sqrt (hlampos i).le
  set Dinv : Matrix (Fin n) (Fin n) ℝ := Matrix.diagonal (fun i => (s i)⁻¹) with hDinvdef
  set U : Matrix (Fin n) (Fin n) ℝ := G * V * Dinv with hUdef
  refine ⟨U, V, s, ?_, hVtV, fun i => (hspos i).le, ?_⟩
  · -- `UᵀU = Σ⁻¹ (Vᵀ (GᵀG) V) Σ⁻¹ = Σ⁻¹ (diagonal λ) Σ⁻¹ = 1`.
    have key : Vᵀ * (Gᵀ * G) * V = Matrix.diagonal lam := by
      rw [hAeq]
      calc Vᵀ * (V * Matrix.diagonal lam * Vᵀ) * V
          = (Vᵀ * V) * Matrix.diagonal lam * (Vᵀ * V) := by simp only [Matrix.mul_assoc]
        _ = Matrix.diagonal lam := by rw [hVtV, Matrix.one_mul, Matrix.mul_one]
    have hUtU : Uᵀ * U = Dinv * (Vᵀ * (Gᵀ * G) * V) * Dinv := by
      rw [hUdef, hDinvdef]
      simp only [Matrix.transpose_mul, Matrix.diagonal_transpose, Matrix.mul_assoc]
    rw [hUtU, key, hDinvdef, Matrix.diagonal_mul_diagonal, Matrix.diagonal_mul_diagonal]
    rw [show (fun i => (s i)⁻¹ * lam i * (s i)⁻¹) = (fun _ => (1 : ℝ)) from funext fun i => ?_,
        Matrix.diagonal_one]
    have hne : s i ≠ 0 := (hspos i).ne'
    field_simp
    linarith [hsq i]
  · -- `U Σ Vᵀ = G V (Σ⁻¹ Σ) Vᵀ = G V Vᵀ = G`.
    symm
    have hDs : Dinv * Matrix.diagonal s = 1 := by
      rw [hDinvdef, Matrix.diagonal_mul_diagonal,
          show (fun i => (s i)⁻¹ * s i) = (fun _ => (1 : ℝ)) from
            funext fun i => inv_mul_cancel₀ (hspos i).ne', Matrix.diagonal_one]
    calc U * Matrix.diagonal s * Vᵀ
        = G * V * (Dinv * Matrix.diagonal s) * Vᵀ := by rw [hUdef]; simp only [Matrix.mul_assoc]
      _ = G * V * Vᵀ := by rw [hDs, Matrix.mul_one]
      _ = G := by rw [Matrix.mul_assoc, hVVt, Matrix.mul_one]

/-- **Muon's update is the steepest ascent in operator-norm geometry — unconditionally, for any
    invertible `G`.** Combining the constructed SVD (`svd_of_isUnit`) with the achievability half
    (`muon_polar_achieves_nuclear`): for invertible `G` there exist orthogonal `U, V` and singular
    values `s ≥ 0` with `G = U (diagonal s) Vᵀ`, whose **polar factor `U Vᵀ`** — exactly Muon's
    update direction — pairs with `G` to the **nuclear norm** `⟨G, UVᵀ⟩_F = Σ sᵢ`. The SVD is no
    longer a hypothesis: it is built from the spectral theorem. (Von Neumann's trace inequality —
    that `Σσᵢ` is the *max* over the operator-norm ball, not merely achieved — is the next layer.) -/
theorem muon_polar_achieves_nuclear_of_isUnit (G : Matrix (Fin n) (Fin n) ℝ) (hG : IsUnit G) :
    ∃ (U V : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ),
      Uᵀ * U = 1 ∧ Vᵀ * V = 1 ∧ (∀ i, 0 ≤ s i) ∧
      G = U * Matrix.diagonal s * Vᵀ ∧ fInner G (U * Vᵀ) = ∑ i, s i := by
  obtain ⟨U, V, s, hU, hV, hs, hGeq⟩ := svd_of_isUnit G hG
  exact ⟨U, V, s, hU, hV, hs, hGeq, hGeq ▸ muon_polar_achieves_nuclear U V s hU hV⟩

end Proofs.MuonGeometry
