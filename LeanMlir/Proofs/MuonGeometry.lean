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
| **Muon** | operator norm | **polar factor `UVᵀ`**, value nuclear `Σσᵢ` | `muon_polar_steepest` |
| Shampoo (1-step) | Kronecker-factored | `(GGᵀ)^{-1/4}G(GᵀG)^{-1/4} = UVᵀ` = Muon | `shampoo_eq_muon` |

The SGD and sign rungs are the framework, proven outright. The Muon rung is now proved **both ways**:
the polar factor `UVᵀ` of `G = UΣVᵀ` *attains* the nuclear norm `Σσᵢ` (`muon_polar_achieves_nuclear`,
trace algebra) **and** is the *maximum* — von Neumann's trace inequality, `⟨G,D⟩_F ≤ Σσᵢ` for every
contraction `D` (`muon_polar_is_max`, per-singular-vector Cauchy–Schwarz). `muon_polar_steepest`
packages them: `UVᵀ` is feasible (an isometry), attains, and is unbeatable — Muon's update IS the
operator-norm steepest-ascent direction, the same `bound`+`attained` shape as the SGD/sign rungs.

**L4 (this layer): the SVD is now constructed, not hypothesized** — for an invertible (full-rank) `G`,
`svd_of_isUnit` builds `U, V` orthogonal and `s ≥ 0` with `G = U (diagonal s) Vᵀ` out of Mathlib's
spectral theorem of `GᵀG`: `V` = eigenvector basis, `sᵢ = √λᵢ` the singular values, `U = G V Σ⁻¹`.
No matrix square root is needed — only the spectral decomposition, scalar `√`, and diagonal inverses
(invertibility makes every `λᵢ > 0`, so `Σ⁻¹` exists). Composing with the achievability half gives
`muon_polar_achieves_nuclear_of_isUnit`: for any invertible `G`, the polar factor `UVᵀ` (Muon's
update direction) pairs with `G` to the nuclear norm `Σσᵢ` — the SVD hypothesis fully discharged.

**L5 (this layer): the Shampoo = Muon jewel.** Single-step Shampoo preconditions the gradient by the
inverse fourth-roots of its Gram matrices, `G ↦ (GGᵀ)^{-1/4} G (GᵀG)^{-1/4}`, and `shampoo_eq_muon`
proves this *equals* Muon's polar factor `UVᵀ` — two optimizers, one geometry. Reusing the L4 SVD
pieces `V, Σ`: the fourth-roots are spectral (`(GᵀG)^{-1/4} = V (diagonal s^{-1/2}) Vᵀ`), the helper
`conj_diag_pow` turns the matrix fourth-power into pointwise scalar powers, and the whole thing
collapses by `s^{-1/2}·s·s^{-1/2} = 1`. `shampoo_eq_muon_of_isUnit` makes it unconditional for any
invertible `G`.

**L6 (manifold view): the polar factor lands on `O(n)`, and is the *nearest* orthogonal matrix to
`G`.** `muon_polar_orthogonal` — `UVᵀ` is orthogonal (a point of the Stiefel manifold);
`muon_polar_nearest_orthogonal` — `‖G − UVᵀ‖_F ≤ ‖G − Q‖_F` for every orthogonal `Q`, the projection
of the gradient onto `O(n)`. The latter reuses the von Neumann bound: minimizing Frobenius distance
to `O(n)` *is* maximizing `⟨G,·⟩_F` over it, so "steepest" and "nearest orthogonal" are the same fact.

The only remaining layer is the singular `G` case (the orthonormal completion of `U`, which would
drop the invertibility hypothesis from the `_of_isUnit` capstones). All
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
-- § L3′ — the upper bound (von Neumann): `UVᵀ` is the *maximum*, so Muon IS steepest descent
-- ════════════════════════════════════════════════════════════════

/-- **No feasible direction beats the polar factor — von Neumann's trace inequality.** Over the
    operator-norm unit ball, the gradient pairing is bounded by the nuclear norm:
    `⟨G, D⟩_F ≤ Σσᵢ` for every contraction `D`. Here `‖D‖op ≤ 1` is spelled elementarily as the
    Euclidean contraction `(D x)·(D x) ≤ x·x` (`*ᵥ` = `mulVec`, `⬝ᵥ` = `dotProduct`), which avoids a
    matrix operator-norm instance while saying exactly that.

    Proof: with `G = U Σ Vᵀ`, cyclic trace gives `⟨G,D⟩_F = Σᵢ sᵢ Mᵢᵢ` for `M = Uᵀ D V`, and each
    diagonal entry `Mᵢᵢ = uᵢ · (D vᵢ)` is bounded by `1` — Cauchy–Schwarz (`‖uᵢ‖ = 1`) then the
    contraction (`‖D vᵢ‖ ≤ ‖vᵢ‖ = 1`), i.e. `Mᵢᵢ² ≤ (uᵢ·uᵢ)((Dvᵢ)·(Dvᵢ)) ≤ 1`. Since `sᵢ ≥ 0`,
    `Σ sᵢ Mᵢᵢ ≤ Σ sᵢ`. This is L1's per-singular-vector Cauchy–Schwarz, summed against `Σ`. -/
theorem muon_polar_is_max (U V D : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (hs : ∀ i, 0 ≤ s i)
    (hD : ∀ x : Fin n → ℝ, (D *ᵥ x) ⬝ᵥ (D *ᵥ x) ≤ x ⬝ᵥ x) :
    fInner (U * Matrix.diagonal s * Vᵀ) D ≤ ∑ i, s i := by
  -- dotProduct Cauchy–Schwarz `(a·b)² ≤ (a·a)(b·b)`
  have hCS : ∀ a b : Fin n → ℝ, (a ⬝ᵥ b) ^ 2 ≤ (a ⬝ᵥ a) * (b ⬝ᵥ b) := by
    intro a b
    have h := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ a b
    simp only [dotProduct]
    calc (∑ j, a j * b j) ^ 2 ≤ (∑ j, (a j) ^ 2) * (∑ j, (b j) ^ 2) := h
      _ = (∑ j, a j * a j) * (∑ j, b j * b j) := by simp only [sq]
  -- columns of an orthonormal matrix are unit vectors: `(W col i)·(W col i) = 1`
  have hcol : ∀ (W : Matrix (Fin n) (Fin n) ℝ), Wᵀ * W = 1 → ∀ i,
      (fun j => W j i) ⬝ᵥ (fun j => W j i) = 1 := by
    intro W hW i
    have h : (Wᵀ * W) i i = (fun j => W j i) ⬝ᵥ (fun j => W j i) := by
      simp [Matrix.mul_apply, Matrix.transpose_apply, dotProduct]
    rw [← h, hW, Matrix.one_apply_eq]
  -- cyclic-trace reduction: `⟨G,D⟩_F = Σᵢ sᵢ (Uᵀ D V)ᵢᵢ`
  have htrace : fInner (U * Matrix.diagonal s * Vᵀ) D = ∑ i, s i * (Uᵀ * D * V) i i := by
    unfold fInner
    have ht : (U * Matrix.diagonal s * Vᵀ)ᵀ = V * Matrix.diagonal s * Uᵀ := by
      simp only [Matrix.transpose_mul, Matrix.diagonal_transpose, Matrix.transpose_transpose,
        Matrix.mul_assoc]
    rw [ht, show V * Matrix.diagonal s * Uᵀ * D = V * (Matrix.diagonal s * Uᵀ * D) from by
          simp only [Matrix.mul_assoc], Matrix.trace_mul_comm,
        show Matrix.diagonal s * Uᵀ * D * V = Matrix.diagonal s * (Uᵀ * D * V) from by
          simp only [Matrix.mul_assoc], Matrix.trace]
    simp only [Matrix.diag_apply, Matrix.diagonal_mul]
  rw [htrace]
  refine Finset.sum_le_sum (fun i _ => ?_)
  -- `(Uᵀ D V)ᵢᵢ = uᵢ · (D vᵢ)`
  have hMrw : (Uᵀ * D * V) i i = (fun j => U j i) ⬝ᵥ (D *ᵥ (fun k => V k i)) := by
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.mulVec, dotProduct,
      Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl (fun a _ => Finset.sum_congr rfl (fun b _ => ?_))
    ring
  -- `Mᵢᵢ² ≤ 1` by Cauchy–Schwarz then the contraction
  have hMsq : ((Uᵀ * D * V) i i) ^ 2 ≤ 1 := by
    rw [hMrw]
    have haa : (fun j => U j i) ⬝ᵥ (fun j => U j i) = 1 := hcol U hU i
    have hbb : (D *ᵥ (fun k => V k i)) ⬝ᵥ (D *ᵥ (fun k => V k i)) ≤ 1 := by
      have := hD (fun k => V k i); rwa [hcol V hV i] at this
    calc ((fun j => U j i) ⬝ᵥ (D *ᵥ (fun k => V k i))) ^ 2
        ≤ ((fun j => U j i) ⬝ᵥ (fun j => U j i))
            * ((D *ᵥ (fun k => V k i)) ⬝ᵥ (D *ᵥ (fun k => V k i))) := hCS _ _
      _ = 1 * ((D *ᵥ (fun k => V k i)) ⬝ᵥ (D *ᵥ (fun k => V k i))) := by rw [haa]
      _ ≤ 1 * 1 := by apply mul_le_mul_of_nonneg_left hbb; norm_num
      _ = 1 := by norm_num
  have hMle : (Uᵀ * D * V) i i ≤ 1 := by nlinarith [hMsq]
  exact mul_le_of_le_one_right (hs i) hMle

/-- **Muon's update `UVᵀ` IS the steepest-ascent direction under the operator norm** — the L3 claim,
    both halves now proved. For an SVD `G = U Σ Vᵀ`, the polar factor `UVᵀ` is the `argmax` of
    `⟨G,·⟩_F` over the operator-norm unit ball, with optimal value the dual (nuclear) norm `Σσᵢ`:
    * **feasible** — `UVᵀ` is an isometry (`‖UVᵀ x‖ = ‖x‖`), hence a contraction, so it lies in the
      ball;
    * **attains** — `⟨G, UVᵀ⟩_F = Σσᵢ` (`muon_polar_achieves_nuclear`);
    * **unbeatable** — every contraction `D` has `⟨G,D⟩_F ≤ Σσᵢ` (`muon_polar_is_max`).
    Compare the SGD/sign rungs (`steepest_l2_*`, `steepest_linf_*`): same `bound`+`attained` shape,
    one norm up. This is *why* Muon's `den = UVᵀ` Newton–Schulz update is steepest descent. -/
theorem muon_polar_steepest (U V : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (hs : ∀ i, 0 ≤ s i) :
    (∀ x : Fin n → ℝ, ((U * Vᵀ) *ᵥ x) ⬝ᵥ ((U * Vᵀ) *ᵥ x) ≤ x ⬝ᵥ x) ∧
    fInner (U * Matrix.diagonal s * Vᵀ) (U * Vᵀ) = ∑ i, s i ∧
    (∀ D : Matrix (Fin n) (Fin n) ℝ, (∀ x, (D *ᵥ x) ⬝ᵥ (D *ᵥ x) ≤ x ⬝ᵥ x) →
        fInner (U * Matrix.diagonal s * Vᵀ) D ≤ ∑ i, s i) := by
  refine ⟨fun x => le_of_eq ?_, muon_polar_achieves_nuclear U V s hU hV,
          fun D hD => muon_polar_is_max U V D s hU hV hs hD⟩
  -- `UVᵀ` is an isometry: `(UVᵀ x)·(UVᵀ x) = x·((VUᵀ)(UVᵀ)) x = x·(V Vᵀ) x = x·x`
  rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, Matrix.mulVec_mulVec,
      Matrix.transpose_mul, Matrix.transpose_transpose,
      show (V * Uᵀ) * (U * Vᵀ) = V * (Uᵀ * U) * Vᵀ from by simp only [Matrix.mul_assoc],
      hU, Matrix.mul_one, mul_eq_one_comm.mp hV, Matrix.one_mulVec]

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

-- ════════════════════════════════════════════════════════════════
-- § L5 — the jewel: single-step Shampoo `(GGᵀ)^{-1/4} G (GᵀG)^{-1/4}` = Muon's `UVᵀ`
-- ════════════════════════════════════════════════════════════════

/-- **Powers of a diagonal conjugation become pointwise powers** — the spectral-calculus workhorse.
    For orthonormal `W` (`WᵀW = 1`), `(W (diagonal d) Wᵀ)^k = W (diagonal dᵏ) Wᵀ`: conjugating a
    diagonal commutes with raising to a power, sending a *matrix* power to a *scalar* power of each
    diagonal entry. This is what lets the Shampoo preconditioners' inverse fourth-roots
    `(GᵀG)^{-1/4}`, `(GGᵀ)^{-1/4}` collapse to diagonal arithmetic in `shampoo_eq_muon`. -/
theorem conj_diag_pow (W : Matrix (Fin n) (Fin n) ℝ) (d : Fin n → ℝ)
    (hWtW : Wᵀ * W = 1) (k : ℕ) :
    (W * Matrix.diagonal d * Wᵀ) ^ k = W * Matrix.diagonal (fun i => (d i) ^ k) * Wᵀ := by
  have hWWt : W * Wᵀ = 1 := mul_eq_one_comm.mp hWtW
  induction k with
  | zero => simp [hWWt]
  | succ m ih =>
    rw [pow_succ, ih,
       show W * Matrix.diagonal (fun i => d i ^ m) * Wᵀ * (W * Matrix.diagonal d * Wᵀ)
          = W * (Matrix.diagonal (fun i => d i ^ m) * (Wᵀ * W) * Matrix.diagonal d) * Wᵀ from by
            simp only [Matrix.mul_assoc],
       hWtW, Matrix.mul_one, Matrix.diagonal_mul_diagonal]
    simp only [pow_succ]

/-- **The Shampoo = Muon jewel.** Single-step Shampoo preconditions the gradient `G` by the inverse
    fourth-roots of its two Gram matrices: `G ↦ (GGᵀ)^{-1/4} G (GᵀG)^{-1/4}`. This **equals Muon's
    update** — the polar factor `UVᵀ` of `G = UΣVᵀ`. Two famous optimizers, one geometry.

    Given the SVD `G = U (diagonal s) Vᵀ` (`U,V` orthonormal, `s > 0`), the inverse fourth-roots are
    spectral: with `sᵢ^{-1/2} = (√sᵢ)⁻¹`, take `R := V (diagonal s^{-1/2}) Vᵀ` and
    `L := U (diagonal s^{-1/2}) Uᵀ`. The three conjuncts are:
    * `R⁴ · (GᵀG) = 1` — `R` really is `(GᵀG)^{-1/4}` (`GᵀG = V (diagonal s²) Vᵀ`, so `R⁴` inverts it);
    * `L⁴ · (GGᵀ) = 1` — `L` really is `(GGᵀ)^{-1/4}`;
    * `L · G · R = U Vᵀ` — **the jewel.** The collapse is the scalar identity
      `s^{-1/2} · s · s^{-1/2} = 1` applied to each singular value (`conj_diag_pow` turns the matrix
      fourth-roots into these pointwise powers). Cf. `muon_polar_achieves_nuclear` (the same `UVᵀ`,
      now reached from Shampoo's side instead of the nuclear-norm side). -/
theorem shampoo_eq_muon (U V : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (hs : ∀ i, 0 < s i) :
    ((V * Matrix.diagonal (fun i => (Real.sqrt (s i))⁻¹) * Vᵀ) ^ 4)
        * ((U * Matrix.diagonal s * Vᵀ)ᵀ * (U * Matrix.diagonal s * Vᵀ)) = 1 ∧
    ((U * Matrix.diagonal (fun i => (Real.sqrt (s i))⁻¹) * Uᵀ) ^ 4)
        * ((U * Matrix.diagonal s * Vᵀ) * (U * Matrix.diagonal s * Vᵀ)ᵀ) = 1 ∧
    (U * Matrix.diagonal (fun i => (Real.sqrt (s i))⁻¹) * Uᵀ)
        * (U * Matrix.diagonal s * Vᵀ)
        * (V * Matrix.diagonal (fun i => (Real.sqrt (s i))⁻¹) * Vᵀ) = U * Vᵀ := by
  set d : Fin n → ℝ := fun i => (Real.sqrt (s i))⁻¹ with hd
  -- reusable contraction `W (diag a) Wᵀ · W (diag b) Wᵀ = W (diag a·b) Wᵀ` (orthonormal `W`)
  have contract : ∀ (W : Matrix (Fin n) (Fin n) ℝ) (a b : Fin n → ℝ), Wᵀ * W = 1 →
      (W * Matrix.diagonal a * Wᵀ) * (W * Matrix.diagonal b * Wᵀ)
        = W * Matrix.diagonal (fun i => a i * b i) * Wᵀ := by
    intro W a b hW
    rw [show (W * Matrix.diagonal a * Wᵀ) * (W * Matrix.diagonal b * Wᵀ)
          = W * (Matrix.diagonal a * (Wᵀ * W) * Matrix.diagonal b) * Wᵀ from by
            simp only [Matrix.mul_assoc],
       hW, Matrix.mul_one, Matrix.diagonal_mul_diagonal]
  have hVVt : V * Vᵀ = 1 := mul_eq_one_comm.mp hV
  have hUUt : U * Uᵀ = 1 := mul_eq_one_comm.mp hU
  -- pointwise collapses: `(s^{-1/2})⁴·s² = 1` (fourth-root certs) and `s^{-1/2}·s·s^{-1/2} = 1` (jewel)
  have hpt4 : ∀ i, (d i) ^ 4 * (s i * s i) = 1 := by
    intro i
    have hsp := hs i
    have hsqrt : Real.sqrt (s i) * Real.sqrt (s i) = s i := Real.mul_self_sqrt hsp.le
    have hne : Real.sqrt (s i) ≠ 0 := (Real.sqrt_pos.mpr hsp).ne'
    simp only [hd]; field_simp; nlinarith [hsqrt]
  have hpt2 : ∀ i, d i * s i * d i = 1 := by
    intro i
    have hsp := hs i
    have hsqrt : Real.sqrt (s i) * Real.sqrt (s i) = s i := Real.mul_self_sqrt hsp.le
    have hne : Real.sqrt (s i) ≠ 0 := (Real.sqrt_pos.mpr hsp).ne'
    simp only [hd]; field_simp; nlinarith [hsqrt]
  -- the Gram matrices are spectral too: `GᵀG = V (diag s²) Vᵀ`, `GGᵀ = U (diag s²) Uᵀ`
  have hGt : (U * Matrix.diagonal s * Vᵀ)ᵀ = V * Matrix.diagonal s * Uᵀ := by
    simp only [Matrix.transpose_mul, Matrix.diagonal_transpose, Matrix.transpose_transpose,
      Matrix.mul_assoc]
  have hGtG : (U * Matrix.diagonal s * Vᵀ)ᵀ * (U * Matrix.diagonal s * Vᵀ)
      = V * Matrix.diagonal (fun i => s i * s i) * Vᵀ := by
    rw [hGt, show (V * Matrix.diagonal s * Uᵀ) * (U * Matrix.diagonal s * Vᵀ)
          = V * (Matrix.diagonal s * (Uᵀ * U) * Matrix.diagonal s) * Vᵀ from by
            simp only [Matrix.mul_assoc],
       hU, Matrix.mul_one, Matrix.diagonal_mul_diagonal]
  have hGGt : (U * Matrix.diagonal s * Vᵀ) * (U * Matrix.diagonal s * Vᵀ)ᵀ
      = U * Matrix.diagonal (fun i => s i * s i) * Uᵀ := by
    rw [hGt, show (U * Matrix.diagonal s * Vᵀ) * (V * Matrix.diagonal s * Uᵀ)
          = U * (Matrix.diagonal s * (Vᵀ * V) * Matrix.diagonal s) * Uᵀ from by
            simp only [Matrix.mul_assoc],
       hV, Matrix.mul_one, Matrix.diagonal_mul_diagonal]
  refine ⟨?_, ?_, ?_⟩
  · -- `R⁴ · GᵀG = V (diag (s^{-1/2})⁴·s²) Vᵀ = V Vᵀ = 1`
    rw [conj_diag_pow V d hV 4, hGtG, contract V _ _ hV,
        show (fun i => d i ^ 4 * (s i * s i)) = (fun _ => (1 : ℝ)) from funext hpt4,
        Matrix.diagonal_one, Matrix.mul_one, hVVt]
  · -- `L⁴ · GGᵀ = 1`, the same on the left
    rw [conj_diag_pow U d hU 4, hGGt, contract U _ _ hU,
        show (fun i => d i ^ 4 * (s i * s i)) = (fun _ => (1 : ℝ)) from funext hpt4,
        Matrix.diagonal_one, Matrix.mul_one, hUUt]
  · -- `L · G · R = U (diag (s^{-1/2}·s·s^{-1/2})) Vᵀ = U Vᵀ`
    rw [show (U * Matrix.diagonal d * Uᵀ) * (U * Matrix.diagonal s * Vᵀ)
              * (V * Matrix.diagonal d * Vᵀ)
            = U * (Matrix.diagonal d * (Uᵀ * U) * Matrix.diagonal s * (Vᵀ * V) * Matrix.diagonal d)
                * Vᵀ from by simp only [Matrix.mul_assoc],
         hU, hV]
    simp only [Matrix.mul_one]
    rw [Matrix.diagonal_mul_diagonal, Matrix.diagonal_mul_diagonal,
        show (fun i => d i * s i * d i) = (fun _ => (1 : ℝ)) from funext hpt2,
        Matrix.diagonal_one, Matrix.mul_one]

/-- **The Shampoo = Muon jewel, unconditional for any invertible `G`.** Composing the constructed
    SVD (`svd_of_isUnit`) with `shampoo_eq_muon`: for invertible `G` there are orthonormal `U, V` and
    singular values `s > 0` (strict, since `diagonal s = Uᵀ G V` is a unit) with `G = U (diagonal s)
    Vᵀ`, such that the two factors `R = V (diagonal s^{-1/2}) Vᵀ`, `L = U (diagonal s^{-1/2}) Uᵀ` are
    the inverse fourth-roots of `GᵀG`, `GGᵀ` (`R⁴·GᵀG = L⁴·GGᵀ = 1`) and Shampoo's preconditioned
    gradient is Muon's polar factor: `L · G · R = U Vᵀ`. -/
theorem shampoo_eq_muon_of_isUnit (G : Matrix (Fin n) (Fin n) ℝ) (hG : IsUnit G) :
    ∃ (U V : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ),
      Uᵀ * U = 1 ∧ Vᵀ * V = 1 ∧ (∀ i, 0 < s i) ∧ G = U * Matrix.diagonal s * Vᵀ ∧
      ((V * Matrix.diagonal (fun i => (Real.sqrt (s i))⁻¹) * Vᵀ) ^ 4) * (Gᵀ * G) = 1 ∧
      ((U * Matrix.diagonal (fun i => (Real.sqrt (s i))⁻¹) * Uᵀ) ^ 4) * (G * Gᵀ) = 1 ∧
      (U * Matrix.diagonal (fun i => (Real.sqrt (s i))⁻¹) * Uᵀ) * G
        * (V * Matrix.diagonal (fun i => (Real.sqrt (s i))⁻¹) * Vᵀ) = U * Vᵀ := by
  obtain ⟨U, V, s, hU, hV, hs0, hGeq⟩ := svd_of_isUnit G hG
  -- `diagonal s = Uᵀ G V` is a unit (product of units), so every `sᵢ ≠ 0`, hence `> 0`.
  have hdiag : Matrix.diagonal s = Uᵀ * G * V := by
    rw [hGeq, show Uᵀ * (U * Matrix.diagonal s * Vᵀ) * V
          = (Uᵀ * U) * Matrix.diagonal s * (Vᵀ * V) from by simp only [Matrix.mul_assoc],
       hU, hV, Matrix.one_mul, Matrix.mul_one]
  have huUt : IsUnit Uᵀ := ⟨⟨Uᵀ, U, hU, mul_eq_one_comm.mp hU⟩, rfl⟩
  have huV : IsUnit V := ⟨⟨V, Vᵀ, mul_eq_one_comm.mp hV, hV⟩, rfl⟩
  have hunit : IsUnit (Matrix.diagonal s) := hdiag ▸ (huUt.mul hG).mul huV
  have hspos : ∀ i, 0 < s i := fun i =>
    lt_of_le_of_ne (hs0 i) (Ne.symm (Pi.isUnit_iff.mp (Matrix.isUnit_diagonal.mp hunit) i).ne_zero)
  obtain ⟨hR, hL, hJ⟩ := shampoo_eq_muon U V s hU hV hspos
  exact ⟨U, V, s, hU, hV, hspos, hGeq, hGeq ▸ hR, hGeq ▸ hL, hGeq ▸ hJ⟩

-- ════════════════════════════════════════════════════════════════
-- § L6 — manifold view: `UVᵀ` lands on `O(n)`, and is the *nearest* orthogonal matrix to `G`
-- ════════════════════════════════════════════════════════════════

/-- **Muon's update lands on the orthogonal group.** The polar factor `UVᵀ` is orthogonal —
    `(UVᵀ)ᵀ(UVᵀ) = (UVᵀ)(UVᵀ)ᵀ = 1` — i.e. a point of `O(n)` (the Stiefel manifold of orthonormal
    frames). This is the geometric content of "Muon orthogonalizes the gradient": the update is not
    a vector in flat weight space but a point on the manifold of orthogonal maps, and the
    implementation's Newton–Schulz iteration is the retraction that computes this projection. -/
theorem muon_polar_orthogonal (U V : Matrix (Fin n) (Fin n) ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) :
    (U * Vᵀ)ᵀ * (U * Vᵀ) = 1 ∧ (U * Vᵀ) * (U * Vᵀ)ᵀ = 1 := by
  constructor
  · rw [Matrix.transpose_mul, Matrix.transpose_transpose,
        show (V * Uᵀ) * (U * Vᵀ) = V * (Uᵀ * U) * Vᵀ from by simp only [Matrix.mul_assoc],
        hU, Matrix.mul_one, mul_eq_one_comm.mp hV]
  · rw [Matrix.transpose_mul, Matrix.transpose_transpose,
        show (U * Vᵀ) * (V * Uᵀ) = U * (Vᵀ * V) * Uᵀ from by simp only [Matrix.mul_assoc],
        hV, Matrix.mul_one, mul_eq_one_comm.mp hU]

/-- **Muon's update is the nearest orthogonal matrix to `G`** — the projection of the raw gradient
    onto `O(n)` in Frobenius distance: `‖G − UVᵀ‖_F ≤ ‖G − Q‖_F` for every orthogonal `Q` (stated in
    squared `fInner` form to avoid `√`). This is *why* the polar factor is "the orthogonalized
    gradient", and it is the ladder's punchline reusing its own prize: expanding
    `‖G − Q‖_F² = ‖G‖_F² − 2⟨G,Q⟩_F + n` (orthogonal `Q` has `‖Q‖_F² = tr(QᵀQ) = n`), minimizing the
    distance is *maximizing* `⟨G,Q⟩_F` over `O(n) ⊆ {contractions}` — exactly the von Neumann bound
    `muon_polar_is_max`, attained at `UVᵀ`. The same inequality that makes `UVᵀ` the steepest
    direction makes it the nearest orthogonal matrix. -/
theorem muon_polar_nearest_orthogonal (U V Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (hs : ∀ i, 0 ≤ s i) (hQ : Qᵀ * Q = 1) :
    fInner (U * Matrix.diagonal s * Vᵀ - U * Vᵀ) (U * Matrix.diagonal s * Vᵀ - U * Vᵀ)
      ≤ fInner (U * Matrix.diagonal s * Vᵀ - Q) (U * Matrix.diagonal s * Vᵀ - Q) := by
  -- `fInner` is a symmetric bilinear form; expand both squared distances
  have hsym : ∀ A B : Matrix (Fin n) (Fin n) ℝ, fInner A B = fInner B A := by
    intro A B; unfold fInner
    rw [← Matrix.trace_transpose (Bᵀ * A), Matrix.transpose_mul, Matrix.transpose_transpose]
  have hexp : ∀ A B : Matrix (Fin n) (Fin n) ℝ,
      fInner (A - B) (A - B) = fInner A A - fInner A B - fInner B A + fInner B B := by
    intro A B; unfold fInner
    simp only [Matrix.transpose_sub, Matrix.sub_mul, Matrix.mul_sub, Matrix.trace_sub]; ring
  -- an orthogonal matrix has constant Frobenius norm² `= n`, and is a contraction (isometry)
  have hnn : ∀ W : Matrix (Fin n) (Fin n) ℝ, Wᵀ * W = 1 → fInner W W = (n : ℝ) := by
    intro W hW; unfold fInner; rw [hW, Matrix.trace_one, Fintype.card_fin]
  have hiso : ∀ x : Fin n → ℝ, (Q *ᵥ x) ⬝ᵥ (Q *ᵥ x) ≤ x ⬝ᵥ x := by
    intro x; refine le_of_eq ?_
    rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, Matrix.mulVec_mulVec, hQ,
        Matrix.one_mulVec]
  have hGUV : fInner (U * Matrix.diagonal s * Vᵀ) (U * Vᵀ) = ∑ i, s i :=
    muon_polar_achieves_nuclear U V s hU hV
  have hUVUV : fInner (U * Vᵀ) (U * Vᵀ) = (n : ℝ) := hnn _ (muon_polar_orthogonal U V hU hV).1
  have hQQ : fInner Q Q = (n : ℝ) := hnn Q hQ
  -- the cross term `⟨G,Q⟩_F ≤ Σσᵢ = ⟨G,UVᵀ⟩_F` is the von Neumann bound — the whole inequality
  have hmax : fInner (U * Matrix.diagonal s * Vᵀ) Q ≤ ∑ i, s i :=
    muon_polar_is_max U V Q s hU hV hs hiso
  rw [hexp, hexp, hsym (U * Vᵀ) (U * Matrix.diagonal s * Vᵀ),
      hsym Q (U * Matrix.diagonal s * Vᵀ), hGUV, hUVUV, hQQ]
  linarith [hmax]

end Proofs.MuonGeometry
