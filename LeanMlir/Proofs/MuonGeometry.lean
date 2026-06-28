import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.LinearAlgebra.Matrix.Trace

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
(pure trace algebra); the matching upper bound (it is the *max*: von Neumann's trace inequality) and
the construction of the SVD itself from Mathlib's spectral theorem are the next layers (L4/L3-upper
in the plan). All `propext / Classical.choice / Quot.sound`-clean. -/

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

end Proofs.MuonGeometry
