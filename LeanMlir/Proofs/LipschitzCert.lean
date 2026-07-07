import LeanMlir.Proofs.Tensor
import Mathlib.Analysis.InnerProductSpace.PiL2

/-! # Lipschitz-margin certified robustness radius (Tsuzuku–Sato–Sugiyama 2018)

The verification payoff behind the `mnist-{linear,mlp,cnn}-pgd` demos
(`planning/robustness.md`, `planning/robustness_ladder.md`): the *certificate* — the lower
bound of the `cert ≤ TRUE ≤ PGD` sandwich — turned from a number into a **theorem**.

The classifier's logit map `f : ℝ^d → ℝ^k` is `L`-Lipschitz in L2. At an input `x`, the
margin is `m = f(x)_{top} − f(x)_{runner-up}`. The theorem: **every perturbation `δ` with
`‖δ‖₂ < m / (√2·L)` leaves the argmax class unchanged** — a provable safe radius against
*all* attacks (vs PGD, which only finds one). The `√2` is the L2 distance `‖eᵢ − eⱼ‖₂`
between two one-hot class directions: a pairwise logit gap is `(√2·L)`-Lipschitz.

The `L` is supplied numerically by `specNormW` / `specNormConvTapSum`
(`LeanMlir/VerifiedTrain.lean`); `lipschitzL2_comp` + `clm_lipschitzL2` show *why* the naive
per-layer **product** `L = ∏ᵢ ‖Wᵢ‖₂` is a sound (if loose) global constant — the looseness
the demos make visual (linear tight → MLP/CNN vacuous).

The second half of the file formalizes the *other* certificate — **randomized smoothing**
(Cohen–Rosenfeld–Kolter 2019, the `*-smooth` demos): `smoothing_certified_radius` gives the
`σ·Φ⁻¹(p_A)` radius the driver reports, as the same Lipschitz-margin argument on the per-class
probit score fields `Φ⁻¹∘P[f(x+η)=·]` — depth-independent, non-vacuous where the product collapses.
`smoothing_certified_radius_probit` is the `Ioo (0,1)` variant that the REAL Gaussian quantile can
instantiate (`SmoothingGaussian.lean` discharges its `hmono`/`hanti` at the true `Φ⁻¹`).

All results are `propext / Classical.choice / Quot.sound`-clean (`tests/AuditAxioms.lean`). -/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § A pointwise-bound Lipschitz predicate (L2, explicit ε–δ form)
-- ════════════════════════════════════════════════════════════════

/-- `f` is `L`-Lipschitz in the (L2) norm: `‖f u − f w‖ ≤ L · ‖u − w‖`. The explicit
    ε–δ form (vs Mathlib's `LipschitzWith`, which is `ℝ≥0∞`-valued) — it reads like the
    math and composes by plain arithmetic. -/
def LipschitzL2 {α β : Type*} [NormedAddCommGroup α] [NormedAddCommGroup β]
    (L : ℝ) (f : α → β) : Prop :=
  ∀ u w, ‖f u - f w‖ ≤ L * ‖u - w‖

/-- Composition multiplies Lipschitz constants: `g ∘ h` is `(Lg·Lh)`-Lipschitz. This is
    exactly the per-layer **product** bound `L = ∏ᵢ ‖Wᵢ‖₂` — sound, and (past one layer)
    loose by construction, the depth-cliff the PGD demos expose. -/
theorem LipschitzL2.comp {α β γ : Type*} [NormedAddCommGroup α] [NormedAddCommGroup β]
    [NormedAddCommGroup γ] {Lg Lh : ℝ} {g : β → γ} {h : α → β}
    (hg : LipschitzL2 Lg g) (hh : LipschitzL2 Lh h) (hLg : 0 ≤ Lg) :
    LipschitzL2 (Lg * Lh) (g ∘ h) := by
  intro u w
  calc ‖g (h u) - g (h w)‖ ≤ Lg * ‖h u - h w‖ := hg _ _
    _ ≤ Lg * (Lh * ‖u - w‖) := by
        exact mul_le_mul_of_nonneg_left (hh _ _) hLg
    _ = (Lg * Lh) * ‖u - w‖ := by ring

/-- A continuous linear map `A` (a net's affine layer, bias dropped) is `‖A‖`-Lipschitz —
    and for a weight matrix the operator norm `‖A‖` **is** the spectral norm `‖W‖₂` that
    `specNormW` estimates by power iteration. So each linear layer contributes its spectral
    norm to the product. -/
theorem clm_lipschitzL2 {α β : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]
    [NormedAddCommGroup β] [NormedSpace ℝ β] (A : α →L[ℝ] β) :
    LipschitzL2 ‖A‖ A := by
  intro u w
  rw [← map_sub]
  exact A.le_opNorm _

-- ════════════════════════════════════════════════════════════════
-- § Euclidean coordinate geometry: the √2 of a pairwise logit gap
-- ════════════════════════════════════════════════════════════════

variable {k : ℕ}

/-- `‖v‖² = Σᵢ (vᵢ)²` on `EuclideanSpace ℝ (Fin k)` (unfold the L2 norm). -/
theorem euclid_norm_sq (v : EuclideanSpace ℝ (Fin k)) :
    ‖v‖ ^ 2 = ∑ i, (v i) ^ 2 := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
  simp [Real.norm_eq_abs, sq_abs]

/-- Two distinct coordinates of `v` carry at most `2‖v‖²` of squared mass:
    `(vᵢ − vⱼ)² ≤ 2‖v‖²`. Equivalently `|vᵢ − vⱼ| ≤ √2·‖v‖` — the `√2` in the radius. -/
theorem coord_pair_bound (v : EuclideanSpace ℝ (Fin k)) {i j : Fin k} (hij : i ≠ j) :
    (v i - v j) ^ 2 ≤ 2 * ‖v‖ ^ 2 := by
  have hpair : (v i) ^ 2 + (v j) ^ 2 ≤ ‖v‖ ^ 2 := by
    rw [euclid_norm_sq]
    have hsub : ({i, j} : Finset (Fin k)) ⊆ Finset.univ := Finset.subset_univ _
    have hle := Finset.sum_le_sum_of_subset_of_nonneg hsub
      (fun l _ _ => by positivity : ∀ l ∈ Finset.univ, l ∉ ({i, j} : Finset (Fin k)) →
        0 ≤ (v l) ^ 2)
    rwa [Finset.sum_pair hij] at hle
  nlinarith [sq_nonneg (v i + v j)]

-- ════════════════════════════════════════════════════════════════
-- § The certified radius
-- ════════════════════════════════════════════════════════════════

variable {E : Type*} [NormedAddCommGroup E]
  {f : E → EuclideanSpace ℝ (Fin k)} {L : ℝ}

/-- **Pairwise logit gaps are `(√2·L)`-stable.** Perturbing the input by `δ` moves the gap
    `f(·)ᵢ − f(·)ⱼ` (any two classes) down by at most `√2·L·‖δ‖`:
    `f(x)ᵢ − f(x)ⱼ − √2·L·‖δ‖ ≤ f(x+δ)ᵢ − f(x+δ)ⱼ`. The engine of the certificate. -/
theorem logit_gap_stable (hf : LipschitzL2 L f)
    (x δ : E) {i j : Fin k} (hij : i ≠ j) :
    (f x i - f x j) - Real.sqrt 2 * L * ‖δ‖ ≤ f (x + δ) i - f (x + δ) j := by
  set v : EuclideanSpace ℝ (Fin k) := f (x + δ) - f x with hv
  -- ‖v‖ ≤ L‖δ‖
  have hvnorm : ‖v‖ ≤ L * ‖δ‖ := by
    have := hf (x + δ) x
    rwa [add_sub_cancel_left] at this
  -- |vᵢ − vⱼ| ≤ √2·‖v‖
  have hcoord : |v i - v j| ≤ Real.sqrt 2 * ‖v‖ := by
    rw [← Real.sqrt_sq_eq_abs]
    calc Real.sqrt ((v i - v j) ^ 2)
        ≤ Real.sqrt (2 * ‖v‖ ^ 2) := Real.sqrt_le_sqrt (coord_pair_bound v hij)
      _ = Real.sqrt 2 * ‖v‖ := by
          rw [Real.sqrt_mul (by norm_num), Real.sqrt_sq (norm_nonneg _)]
  -- chain: √2‖v‖ ≤ √2·L·‖δ‖
  have hchain : |v i - v j| ≤ Real.sqrt 2 * L * ‖δ‖ := by
    refine hcoord.trans ?_
    have h2 : (0:ℝ) ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
    calc Real.sqrt 2 * ‖v‖ ≤ Real.sqrt 2 * (L * ‖δ‖) :=
          mul_le_mul_of_nonneg_left hvnorm h2
      _ = Real.sqrt 2 * L * ‖δ‖ := by ring
  -- decompose the gap difference and bound below by −|vᵢ − vⱼ|
  have hvi : v i = f (x + δ) i - f x i := rfl
  have hvj : v j = f (x + δ) j - f x j := rfl
  have hlow : -(Real.sqrt 2 * L * ‖δ‖) ≤ v i - v j := neg_le_of_abs_le hchain
  -- f(x+δ)ᵢ − f(x+δ)ⱼ = (f x i − f x j) + (vᵢ − vⱼ)
  have hdecomp : f (x + δ) i - f (x + δ) j = (f x i - f x j) + (v i - v j) := by
    rw [hvi, hvj]; ring
  rw [hdecomp]; linarith [hlow]

/-- **Lipschitz-margin certified radius (Tsuzuku et al. 2018).** If the logit map `f` is
    `L`-Lipschitz in L2 (`L > 0`), class `i` leads every other class at `x` by at least the
    margin `m` (`hmargin`), and `‖δ‖₂ < m / (√2·L)`, then **`i` still strictly leads every
    other class at `x + δ`** — the prediction provably cannot flip inside the L2 ball of
    radius `m / (√2·L)`, against *any* attack. -/
theorem lipschitz_margin_certified_radius
    (hf : LipschitzL2 L f) (hL : 0 < L)
    {x δ : E} {i : Fin k} {m : ℝ}
    (hmargin : ∀ j, j ≠ i → m ≤ f x i - f x j)
    (hδ : ‖δ‖ < m / (Real.sqrt 2 * L)) :
    ∀ j, j ≠ i → f (x + δ) j < f (x + δ) i := by
  have hsqrt2 : (0:ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hden : 0 < Real.sqrt 2 * L := mul_pos hsqrt2 hL
  -- ‖δ‖ < m/(√2 L)  ⇒  √2·L·‖δ‖ < m
  have hswing : Real.sqrt 2 * L * ‖δ‖ < m := by
    rw [lt_div_iff₀ hden] at hδ
    linarith [hδ]
  intro j hj
  have hgap := logit_gap_stable hf x δ hj.symm
  have hmj := hmargin j hj
  linarith [hgap, hmj, hswing]

-- ════════════════════════════════════════════════════════════════
-- § Randomized-smoothing certified radius (Cohen–Rosenfeld–Kolter 2019)
-- ════════════════════════════════════════════════════════════════

/-! The *other* certificate of the `cifar-smooth` / `mnist-{mlp,cnn}-smooth` demos — the one that
stays non-vacuous where the Lipschitz product collapses. The smoothed classifier
`ĝ(x) = argmax_c P[f(x+η)=c]`, `η ~ N(0,σ²I)`, is certifiably robust in L2 with radius
`σ·Φ⁻¹(p_A)` (the radius the driver reports). The proof factors exactly like the Tsuzuku theorem:
a **Lipschitz-margin** argument over per-class **probit score fields**

  `gᶜ(x) = Φ⁻¹(P[f(x+η)=c])`,

each of which is `(1/σ)`-Lipschitz. That `(1/σ)`-Lipschitzness is the analytic heart of Cohen 2019
(Neyman–Pearson over the Gaussian likelihood ratio; equivalently Gaussian isoperimetry, Salman et
al. 2019 Lemma 2) — it is taken here as the **hypothesis** `hg`, in exactly the way
`lipschitz_margin_certified_radius` takes the logit-map Lipschitz constant `L` as a hypothesis
rather than re-deriving the spectral norms. Given it, the certified radius is pure margin algebra:
no Gaussian measure theory leaks past the hypothesis, and the result is `Classical`-clean. -/

/-- **Core smoothing margin step.** If the smoothed classifier's per-class probit score fields
    `g c` are each `(1/σ)`-Lipschitz in L2 (`hg` — the Cohen/Salman Gaussian content), and class
    `i` leads every other class in probit score at `x` by margin `m`, then for every `‖δ‖₂ < σ·m/2`
    it **still leads** at `x+δ`. The radius is `σ·m/2`; with `m = Φ⁻¹(p_A)−Φ⁻¹(p_B)` this is Cohen's
    `R = (σ/2)(Φ⁻¹(p_A)−Φ⁻¹(p_B))`. Since `Φ⁻¹` is increasing the `g`-argmax IS the smoothed
    prediction `ĝ`, so `ĝ` cannot flip inside the L2 ball of radius `σ·m/2`. -/
theorem smoothed_margin_certified_radius {σ : ℝ} (hσ : 0 < σ)
    {g : Fin k → E → ℝ} (hg : ∀ c, LipschitzL2 (1 / σ) (g c))
    {x δ : E} {i : Fin k} {m : ℝ}
    (hmargin : ∀ j, j ≠ i → m ≤ g i x - g j x)
    (hδ : ‖δ‖ < σ * m / 2) :
    ∀ j, j ≠ i → g j (x + δ) < g i (x + δ) := by
  intro j hj
  -- each score field moves by at most (1/σ)·‖δ‖ under the perturbation δ
  have hi := hg i (x + δ) x
  have hjj := hg j (x + δ) x
  rw [add_sub_cancel_left, Real.norm_eq_abs] at hi hjj
  have hi' := (abs_le.mp hi).1   -- −(1/σ·‖δ‖) ≤ g i (x+δ) − g i x
  have hj' := (abs_le.mp hjj).2  -- g j (x+δ) − g j x ≤ 1/σ·‖δ‖
  have hmar := hmargin j hj
  -- the perturbation can erode the margin by at most 2·(1/σ)·‖δ‖, which is < m
  have hfrac : 2 * (1 / σ) * ‖δ‖ < m := by
    have e : 2 * (1 / σ) * ‖δ‖ = 2 * ‖δ‖ / σ := by ring
    rw [e, div_lt_iff₀ hσ]
    nlinarith [hδ]
  nlinarith [hi', hj', hmar, hfrac]

/-- **Randomized-smoothing certified radius (Cohen–Rosenfeld–Kolter 2019).** The form the
    `*-smooth` drivers report: with `p c x = P[f(x+η)=c]`, an (abstract) probit `Φ⁻¹ = Phiinv`
    that is increasing (`hmono`) and odd about ½ (`hanti : Φ⁻¹(1−p) = −Φ⁻¹(p)` — the real inverse
    Gaussian CDF is both), per-class scores `Φ⁻¹∘(p c)` each `(1/σ)`-Lipschitz (`hg`), and the
    runner-up bound `p_j(x) ≤ 1 − p_A(x)` (the non-top mass; `hrunner`), **every `‖δ‖₂ < σ·Φ⁻¹(p_A(x))`
    keeps class `i` the strict argmax of the noise-probabilities** — i.e. `ĝ(x+δ) = i`. This is the
    `σ·Φ⁻¹(p_A)` radius, derived from the core step via `p_B ≤ 1−p_A ⇒ Φ⁻¹(p_B) ≤ −Φ⁻¹(p_A)`,
    so the margin `Φ⁻¹(p_A)−Φ⁻¹(p_B) ≥ 2·Φ⁻¹(p_A)`. Depth-independent: no per-layer norm, no product. -/
theorem smoothing_certified_radius {σ : ℝ} (hσ : 0 < σ)
    {Phiinv : ℝ → ℝ} (hmono : Monotone Phiinv)
    (hanti : ∀ p : ℝ, Phiinv (1 - p) = -Phiinv p)
    {p : Fin k → E → ℝ}
    (hg : ∀ c, LipschitzL2 (1 / σ) (fun x => Phiinv (p c x)))
    {x δ : E} {i : Fin k}
    (hrunner : ∀ j, j ≠ i → p j x ≤ 1 - p i x)
    (hδ : ‖δ‖ < σ * Phiinv (p i x)) :
    ∀ j, j ≠ i → p j (x + δ) < p i (x + δ) := by
  -- probit scores: margin ≥ 2·Φ⁻¹(p_A) at x, then the core step in g-space
  have hscore : ∀ j, j ≠ i →
      Phiinv (p j (x + δ)) < Phiinv (p i (x + δ)) := by
    refine smoothed_margin_certified_radius (g := fun c x => Phiinv (p c x)) hσ hg
      (m := 2 * Phiinv (p i x)) ?_ ?_
    · intro j hj
      have h1 : Phiinv (p j x) ≤ Phiinv (1 - p i x) := hmono (hrunner j hj)
      have h2 : Phiinv (1 - p i x) = -Phiinv (p i x) := hanti _
      show 2 * Phiinv (p i x) ≤ Phiinv (p i x) - Phiinv (p j x)
      linarith [h1, h2]
    · have e : σ * (2 * Phiinv (p i x)) / 2 = σ * Phiinv (p i x) := by ring
      rw [e]; exact hδ
  -- Φ⁻¹ increasing ⇒ the strict score order forces the strict probability order
  intro j hj
  by_contra h
  exact absurd (hmono (not_lt.mp h)) (not_le.mpr (hscore j hj))

/-- **The radius theorem at an honest probit (Ioo variant).** The TRUE quantile `Φ⁻¹` is
    unbounded on `(0,1)`, so no total real-valued `Phiinv` can satisfy the global `hmono`
    of `smoothing_certified_radius` while agreeing with it — the abstract theorem is fine,
    but it can never be *instantiated* at the real inverse Gaussian CDF. This variant fixes
    that: all class probabilities live in `(0,1)` (`hp` — Monte-Carlo/Clopper–Pearson
    estimates are never exactly 0 or 1), and monotonicity/oddness are only required ON
    `Ioo 0 1`, which the real `Φ⁻¹` satisfies (`SmoothingGaussian.lean` discharges both,
    making the Cohen radius a theorem about the genuine Gaussian quantile with only the
    Neyman–Pearson Lipschitz core `hg` left as a hypothesis). Same proof, with the Ioo
    memberships threaded through. -/
theorem smoothing_certified_radius_probit {σ : ℝ} (hσ : 0 < σ)
    {Phiinv : ℝ → ℝ} (hmono : MonotoneOn Phiinv (Set.Ioo 0 1))
    (hanti : ∀ q ∈ Set.Ioo (0:ℝ) 1, Phiinv (1 - q) = -Phiinv q)
    {p : Fin k → E → ℝ}
    (hp : ∀ c y, p c y ∈ Set.Ioo (0:ℝ) 1)
    (hg : ∀ c, LipschitzL2 (1 / σ) (fun x => Phiinv (p c x)))
    {x δ : E} {i : Fin k}
    (hrunner : ∀ j, j ≠ i → p j x ≤ 1 - p i x)
    (hδ : ‖δ‖ < σ * Phiinv (p i x)) :
    ∀ j, j ≠ i → p j (x + δ) < p i (x + δ) := by
  have h1mem : (1 - p i x) ∈ Set.Ioo (0:ℝ) 1 := by
    have h := hp i x
    exact ⟨by linarith [h.2], by linarith [h.1]⟩
  have hscore : ∀ j, j ≠ i →
      Phiinv (p j (x + δ)) < Phiinv (p i (x + δ)) := by
    refine smoothed_margin_certified_radius (g := fun c x => Phiinv (p c x)) hσ hg
      (m := 2 * Phiinv (p i x)) ?_ ?_
    · intro j hj
      have h1 : Phiinv (p j x) ≤ Phiinv (1 - p i x) :=
        hmono (hp j x) h1mem (hrunner j hj)
      have h2 : Phiinv (1 - p i x) = -Phiinv (p i x) := hanti _ (hp i x)
      show 2 * Phiinv (p i x) ≤ Phiinv (p i x) - Phiinv (p j x)
      linarith [h1, h2]
    · have e : σ * (2 * Phiinv (p i x)) / 2 = σ * Phiinv (p i x) := by ring
      rw [e]; exact hδ
  intro j hj
  by_contra h
  exact absurd (hmono (hp i (x + δ)) (hp j (x + δ)) (not_lt.mp h))
    (not_le.mpr (hscore j hj))

end Proofs
