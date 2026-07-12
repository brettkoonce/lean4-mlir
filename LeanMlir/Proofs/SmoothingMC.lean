import LeanMlir.Proofs.SmoothingGaussian
import Mathlib.Probability.Moments.SubGaussian

/-! # The Monte-Carlo tie for randomized smoothing

`smoothing_certified_radius_classifier` (SmoothingGaussian.lean) certifies the
radius `σ·Φ⁻¹(p)` for the TRUE class probability `p = ∫ 1[C(x+σz)=y] dγ(z)`.
The `*-smooth` drivers can only ESTIMATE `p` from `N` Gaussian samples — the
honest gap flagged since the smoothing theorems landed.

This file closes it: a one-sided Hoeffding bound for `[0,1]`-valued
Monte-Carlo means over the product measure (`mc_mean_lower_bound`, built on
Mathlib's `HasSubgaussianMGF` machinery), so that with probability
`≥ 1 − exp(−2Nt²)` over the samples, the true `p` is at least the empirical
`p̂ − t` — and on that event the reported radius `σ·Φ⁻¹(p̂ − t)` is genuinely
certified (`smoothing_mc_certified`, composed with the classifier theorem
through the quantile's monotonicity). The guarantee has exactly the shape of
Cohen et al.'s CERTIFY procedure: a confidence-qualified radius, now end to
end a theorem. -/

namespace Proofs

open MeasureTheory ProbabilityTheory Real
open scoped BigOperators ENNReal NNReal

section MCBound

variable {E : Type*} [MeasurableSpace E]

/-- Coordinates of a finite product of probability measures are iid: the
    `iIndepFun` fact the Hoeffding sum bound needs. -/
lemma iIndepFun_eval_pi (ν : Measure E) [IsProbabilityMeasure ν] (N : ℕ) :
    iIndepFun (fun (i : Fin N) (ω : Fin N → E) => ω i)
      (Measure.pi fun _ : Fin N => ν) := by
  rw [iIndepFun_iff_map_fun_eq_pi_map (fun i => (measurable_pi_apply i).aemeasurable)]
  have h1 : (fun (ω : Fin N → E) (i : Fin N) => ω i) = id := rfl
  have h2 : (fun i : Fin N =>
      Measure.map (fun ω : Fin N → E => ω i) (Measure.pi fun _ : Fin N => ν))
      = fun _ : Fin N => ν := by
    funext i
    exact (measurePreserving_eval (fun _ : Fin N => ν) i).map_eq
  rw [h1, Measure.map_id, h2]

/-- **One-sided Hoeffding for a `[0,1]`-valued Monte-Carlo mean.** With
    probability `≥ 1 − exp(−2Nt²)` over `N` iid samples from `ν`, the
    empirical mean minus `t` lower-bounds the true mean. -/
theorem mc_mean_lower_bound (ν : Measure E) [IsProbabilityMeasure ν]
    {f : E → ℝ} (hfm : Measurable f) (hf01 : ∀ z, f z ∈ Set.Icc (0 : ℝ) 1)
    (N : ℕ) (hN : 0 < N) {t : ℝ} (ht : 0 ≤ t) :
    1 - Real.exp (-2 * N * t ^ 2)
      ≤ (Measure.pi fun _ : Fin N => ν).real
          {ω | (∑ i, f (ω i)) / N - t ≤ ∫ z, f z ∂ν} := by
  set μN : Measure (Fin N → E) := Measure.pi fun _ : Fin N => ν with hμN
  haveI : IsProbabilityMeasure μN := by
    rw [hμN]; infer_instance
  set p : ℝ := ∫ z, f z ∂ν with hp
  -- the centered coordinate variables
  set X : Fin N → (Fin N → E) → ℝ := fun i ω => f (ω i) - p with hX
  -- each is (1/2)²-subgaussian by Hoeffding's lemma for bounded variables
  have hmean : ∀ i : Fin N, (∫ ω, f (ω i) ∂μN) = p := by
    intro i
    calc (∫ ω, f (ω i) ∂μN)
        = ∫ y, f y ∂(Measure.map (fun ω : Fin N → E => ω i) μN) :=
          (integral_map (measurable_pi_apply i).aemeasurable
            hfm.aestronglyMeasurable).symm
      _ = p := by
          rw [hp]
          exact congrArg (fun m => ∫ y, f y ∂m)
            (measurePreserving_eval (fun _ : Fin N => ν) i).map_eq
  have hsubG : ∀ i : Fin N,
      HasSubgaussianMGF (X i) (((1 : ℝ≥0) / 2) ^ 2) μN := by
    intro i
    have hb : ∀ᵐ ω ∂μN, f (ω i) ∈ Set.Icc (0 : ℝ) 1 :=
      ae_of_all _ fun ω => hf01 (ω i)
    have h := hasSubgaussianMGF_of_mem_Icc
      (X := fun ω : Fin N → E => f (ω i))
      ((hfm.comp (measurable_pi_apply i)).aemeasurable) hb
    have hc : ((‖(1 : ℝ) - 0‖₊ / 2) ^ 2 : ℝ≥0) = ((1 : ℝ≥0) / 2) ^ 2 := by
      norm_num
    rw [hc] at h
    have hXi : X i = fun ω => f (ω i) - ∫ ω', f (ω' i) ∂μN := by
      funext ω
      rw [hX, hmean i]
    rw [hXi]
    exact h
  -- iid: compose coordinate independence with the (measurable) shift
  have hindep : iIndepFun X μN := by
    have h0 : iIndepFun (fun (i : Fin N) (ω : Fin N → E) => ω i) μN :=
      iIndepFun_eval_pi ν N
    exact h0.comp (fun i => fun v => f v - p) (fun i => (hfm.sub measurable_const))
  -- Hoeffding on the sum
  have hHoeff := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hindep
    (c := fun _ : Fin N => ((1 : ℝ≥0) / 2) ^ 2)
    (s := Finset.univ) (fun i _ => hsubG i)
    (ε := N * t) (by positivity)
  -- the exponent simplifies to −2Nt²
  have hexp : -(N * t) ^ 2 / (2 * ∑ _i : Fin N, (((1 : ℝ≥0) / 2) ^ 2 : ℝ≥0))
      = -2 * N * t ^ 2 := by
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
    push_cast
    have hN' : (0 : ℝ) < N := by exact_mod_cast hN
    field_simp
    ring
  rw [hexp] at hHoeff
  -- complement: the good event contains the complement of the bad event
  have hsub : {ω : Fin N → E | N * t ≤ ∑ i, X i ω}ᶜ
      ⊆ {ω : Fin N → E | (∑ i, f (ω i)) / N - t ≤ p} := by
    intro ω hω
    simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_le] at hω
    simp only [Set.mem_setOf_eq]
    have hsum : (∑ i, X i ω) = (∑ i, f (ω i)) - N * p := by
      rw [hX]
      rw [Finset.sum_sub_distrib]
      simp [Finset.card_univ]
    have hN' : (0 : ℝ) < N := by exact_mod_cast hN
    rw [hsum] at hω
    rw [sub_le_iff_le_add, div_le_iff₀ hN']
    nlinarith
  have hmeas_bad : MeasurableSet {ω : Fin N → E | N * t ≤ ∑ i, X i ω} := by
    apply measurableSet_le measurable_const
    exact Finset.measurable_sum _ fun i _ =>
      (hfm.comp (measurable_pi_apply i)).sub measurable_const
  calc 1 - Real.exp (-2 * N * t ^ 2)
      ≤ 1 - μN.real {ω | N * t ≤ ∑ i, X i ω} := by linarith [hHoeff]
    _ = μN.real {ω | N * t ≤ ∑ i, X i ω}ᶜ := by
        rw [measureReal_compl hmeas_bad, probReal_univ]
    _ ≤ μN.real {ω | (∑ i, f (ω i)) / N - t ≤ p} :=
        measureReal_mono hsub

end MCBound

-- ════════════════════════════════════════════════════════════════
-- § The quantile below zero: junk value, vacuous certificate
-- ════════════════════════════════════════════════════════════════

/-- Below `0` the quantile's defining set is empty (`Φ > 0` everywhere), so
    `Φ⁻¹` takes the junk value `sSup ∅ = 0` — and the radius `σ·Φ⁻¹(p̂−t)`
    certifies vacuously. -/
lemma stdNormalQuantile_of_nonpos {q : ℝ} (hq : q ≤ 0) : stdNormalQuantile q = 0 := by
  have hset : {s : ℝ | stdNormalCDF s < q} = ∅ := by
    ext s
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_lt]
    exact hq.trans (stdNormalCDF_pos s).le
  rw [stdNormalQuantile, hset, Real.sSup_empty]

-- ════════════════════════════════════════════════════════════════
-- § The composed guarantee: Cohen's CERTIFY, end to end
-- ════════════════════════════════════════════════════════════════

/-- **The Monte-Carlo smoothing certificate.** Sample `N` iid standard
    Gaussians; report the radius `σ·Φ⁻¹(p̂ − t)` from the empirical class
    frequency `p̂`. With probability `≥ 1 − exp(−2Nt²)` over the samples, the
    reported radius is GENUINELY certified: every `‖δ‖ < σ·Φ⁻¹(p̂ − t)` keeps
    `y` the strict argmax of the smoothed classifier. This is the guarantee
    shape of Cohen–Rosenfeld–Kolter's CERTIFY procedure, with the Neyman–
    Pearson side (`smoothing_probit_lipschitz`), the radius algebra
    (`smoothing_certified_radius_classifier`), and now the sampling
    confidence (`mc_mean_lower_bound`, Hoeffding) all theorems. -/
theorem smoothing_mc_certified {n k : ℕ} {σ : ℝ} (hσ : 0 < σ)
    {C : EuclideanSpace ℝ (Fin (n + 1)) → Fin k} (hC : Measurable C)
    (hp : ∀ c x, (∫ z, (if C (x + σ • z) = c then (1:ℝ) else 0)
      ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))) ∈ Set.Ioo (0:ℝ) 1)
    (x : EuclideanSpace ℝ (Fin (n + 1))) (y : Fin k)
    (N : ℕ) (hN : 0 < N) {t : ℝ} (ht : 0 ≤ t) :
    1 - Real.exp (-2 * N * t ^ 2)
      ≤ (Measure.pi fun _ : Fin N => stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))).real
          {ω | ∀ δ : EuclideanSpace ℝ (Fin (n + 1)),
            ‖δ‖ < σ * stdNormalQuantile
              ((∑ i, (if C (x + σ • ω i) = y then (1:ℝ) else 0)) / N - t) →
            ∀ j, j ≠ y →
              (∫ z, (if C (x + δ + σ • z) = j then (1:ℝ) else 0)
                  ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
                < ∫ z, (if C (x + δ + σ • z) = y then (1:ℝ) else 0)
                    ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))} := by
  set γ := stdGaussian (EuclideanSpace ℝ (Fin (n + 1))) with hγ
  set f : EuclideanSpace ℝ (Fin (n + 1)) → ℝ :=
    fun v => if C (x + σ • v) = y then (1:ℝ) else 0 with hf
  have hfm : Measurable f := by
    exact measurable_const.ite
      ((hC.comp (measurable_const.add (measurable_id.const_smul σ)))
        (measurableSet_singleton y)) measurable_const
  have hf01 : ∀ v, f v ∈ Set.Icc (0:ℝ) 1 := by
    intro v
    by_cases h : C (x + σ • v) = y <;> simp [hf, h]
  -- the Hoeffding event implies the certificate
  have hsub : {ω : Fin N → EuclideanSpace ℝ (Fin (n + 1)) |
        (∑ i, f (ω i)) / N - t ≤ ∫ z, f z ∂γ}
      ⊆ {ω | ∀ δ : EuclideanSpace ℝ (Fin (n + 1)),
            ‖δ‖ < σ * stdNormalQuantile ((∑ i, (if C (x + σ • ω i) = y then (1:ℝ) else 0)) / N - t) →
            ∀ j, j ≠ y →
              (∫ z, (if C (x + δ + σ • z) = j then (1:ℝ) else 0) ∂γ)
                < ∫ z, (if C (x + δ + σ • z) = y then (1:ℝ) else 0) ∂γ} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    intro δ hδ j hj
    set q : ℝ := (∑ i, f (ω i)) / N - t with hq
    have hδq : ‖δ‖ < σ * stdNormalQuantile q := hδ
    rcases le_or_gt q 0 with hq0 | hq0
    · rw [stdNormalQuantile_of_nonpos hq0, mul_zero] at hδq
      exact absurd hδq (not_lt.mpr (norm_nonneg δ))
    · -- 0 < q ≤ p < 1: quantile monotone on (0,1) lifts the radius
      have hpy := hp y x
      have hqIoo : q ∈ Set.Ioo (0:ℝ) 1 := ⟨hq0, lt_of_le_of_lt hω hpy.2⟩
      have hmono := stdNormalQuantile_monotoneOn hqIoo hpy hω
      have hδ' : ‖δ‖ < σ * stdNormalQuantile (∫ z, f z ∂γ) :=
        lt_of_lt_of_le hδq (mul_le_mul_of_nonneg_left hmono hσ.le)
      exact smoothing_certified_radius_classifier hσ hC hp hδ' j hj
  calc 1 - Real.exp (-2 * N * t ^ 2)
      ≤ (Measure.pi fun _ : Fin N => γ).real
          {ω | (∑ i, f (ω i)) / N - t ≤ ∫ z, f z ∂γ} :=
        mc_mean_lower_bound γ hfm hf01 N hN ht
    _ ≤ _ := measureReal_mono hsub

end Proofs
