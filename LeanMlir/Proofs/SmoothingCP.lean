import LeanMlir.Proofs.SmoothingMC

/-! # The exact Clopper–Pearson tie for randomized smoothing

`smoothing_mc_certified` (SmoothingMC.lean) qualifies the reported radius with
Hoeffding's `1 − exp(−2Nt²)` — a crude bound. Cohen–Rosenfeld–Kolter's CERTIFY
actually deploys the EXACT binomial lower confidence limit (Clopper–Pearson,
`proportion_confint` one-sided). This file closes that last arithmetic gap:

* `pi_hitCount_eq_binomial` — **the count of successes over `Measure.pi` is
  binomial** (the piece Mathlib doesn't have: `Bin(n,p)` exists as
  `setBer(Iio n, p).map ncard`, but no law-of-the-iid-indicator-count):
  induction on `N` through `measurePreserving_piFinSuccAbove`, splitting the
  first coordinate by Fubini and closing with Pascal's rule;
* `cpLower α N k = sInf {q ∈ [0,1] | α < binomTail N k q}` — the CP lower
  bound, with `binomTail` the literal tail polynomial the driver evaluates;
* `cp_coverage` — **with probability ≥ 1 − α the CP bound is a genuine lower
  bound for the true probability.** The `sInf` definition makes coverage need
  NO tail-monotonicity in `q`: the minimal counterexample count `k₀` has
  `p < cpLower α N k₀`, so `binomTail N k₀ p ≤ α` directly (contrapositive of
  `csInf_le`), and every count below `k₀` certifies by minimality of `k₀`;
* `smoothing_cp_certified` — composed with
  `smoothing_certified_radius_classifier`: with probability `≥ 1 − α` over the
  `N` Gaussian samples, the radius `σ·Φ⁻¹(cpLower α N k)` reported from the
  observed class count `k` is genuinely certified. Guarantee AND arithmetic
  now match the deployed CERTIFY;
* the SOLVED form (the per-image scorecard shape): `binomTail_monotoneOn` —
  monotone in the success probability by COUPLING (uniform measure on `[0,1]`,
  nested `[0,q] ⊆ [0,p]`, tail events nest — the tail law again, no calculus)
  — gives `le_cpLower_of_tail_le`: ONE kernel tail check
  `binomTail N k₀ q₀ ≤ α` certifies the driver's reported `q₀`, and
  `smoothing_cp_certified_solved` turns it into "if the count comes out `k₀`,
  the radius `σ·Φ⁻¹(q₀)` is certified" (w.p. `≥ 1 − α`). Demo checks at
  99/100 and 999/1000 close the file (`exponentiation.threshold` gotcha).

All results are `propext / Classical.choice / Quot.sound`-clean. -/

namespace Proofs

open MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal NNReal

-- ════════════════════════════════════════════════════════════════
-- § The binomial tail and the Clopper–Pearson lower bound
-- ════════════════════════════════════════════════════════════════

/-- Upper tail of the Binomial(N, q) distribution: `P(X ≥ k)` as a real
    polynomial in `q` — exactly what the driver evaluates. -/
noncomputable def binomTail (N k : ℕ) (q : ℝ) : ℝ :=
  ∑ j ∈ Finset.Icc k N, (N.choose j : ℝ) * q ^ j * (1 - q) ^ (N - j)

/-- The Clopper–Pearson lower confidence bound at level `α` for `k` successes
    in `N` trials: the smallest success probability whose upper tail at `k`
    still exceeds `α`. -/
noncomputable def cpLower (α : ℝ) (N k : ℕ) : ℝ :=
  sInf {q | q ∈ Set.Icc (0:ℝ) 1 ∧ α < binomTail N k q}

/-- The CP key inequality: if the true `p` lies strictly below the CP lower
    bound, its tail is at most `α` (contrapositive of `sInf ≤`). -/
lemma binomTail_le_of_lt_cpLower {α p : ℝ} (hp : p ∈ Set.Icc (0:ℝ) 1) {N k : ℕ}
    (h : p < cpLower α N k) : binomTail N k p ≤ α := by
  by_contra hgt
  have hmem : p ∈ {q | q ∈ Set.Icc (0:ℝ) 1 ∧ α < binomTail N k q} :=
    ⟨hp, not_le.mp hgt⟩
  exact absurd (csInf_le ⟨0, fun q hq => hq.1.1⟩ hmem) (not_le.mpr h)

-- ════════════════════════════════════════════════════════════════
-- § The count of successes over the product measure is binomial
-- ════════════════════════════════════════════════════════════════

section BinomialCount

variable {E : Type*} [MeasurableSpace E]

/-- The number of samples landing in `A`, as a `Fin N`-indexed indicator sum
    (`Set.indicator`: no decidability needed). -/
noncomputable def hitCount (A : Set E) (N : ℕ) (ω : Fin N → E) : ℕ :=
  ∑ i, A.indicator 1 (ω i)

omit [MeasurableSpace E] in
/-- The count is monotone in the target set. -/
lemma hitCount_mono {A B : Set E} (hAB : A ⊆ B) (N : ℕ) (ω : Fin N → E) :
    hitCount A N ω ≤ hitCount B N ω := by
  simp only [hitCount]
  refine Finset.sum_le_sum fun i _ => ?_
  by_cases h : ω i ∈ A
  · rw [Set.indicator_of_mem h, Set.indicator_of_mem (hAB h)]
  · rw [Set.indicator_of_notMem h]
    omega

omit [MeasurableSpace E] in
lemma hitCount_le (A : Set E) (N : ℕ) (ω : Fin N → E) : hitCount A N ω ≤ N := by
  simp only [hitCount]
  calc (∑ i, A.indicator 1 (ω i)) ≤ ∑ _i : Fin N, 1 :=
        Finset.sum_le_sum fun i _ => by
          by_cases h : ω i ∈ A
          · rw [Set.indicator_of_mem h]; exact le_rfl
          · rw [Set.indicator_of_notMem h]; omega
    _ = N := by simp

lemma measurable_hitCount {A : Set E} (hA : MeasurableSet A) (N : ℕ) :
    Measurable (hitCount A N) :=
  Finset.measurable_sum _ fun i _ =>
    (measurable_const.indicator hA).comp (measurable_pi_apply i)

omit [MeasurableSpace E] in
/-- Prepending a sample to the tuple adds its indicator to the count. -/
lemma hitCount_insertNth_zero (A : Set E) (N : ℕ) (x : E) (τ : Fin N → E) :
    hitCount A (N + 1) ((0 : Fin (N + 1)).insertNth x τ)
      = A.indicator 1 x + hitCount A N τ := by
  simp only [hitCount, Fin.sum_univ_succ]
  congr 1

/-- **The count of successes over the product measure is binomial** (point law,
    ℝ≥0∞ form): `P(hitCount = j) = C(N,j)·ν(A)^j·ν(Aᶜ)^(N−j)`. Induction on `N`
    through the coordinate-0 product split, Fubini on the first sample, and
    Pascal's rule. -/
lemma pi_hitCount_eq_binomial (ν : Measure E) [IsProbabilityMeasure ν] {A : Set E}
    (hA : MeasurableSet A) (N : ℕ) : ∀ j : ℕ,
    Measure.pi (fun _ : Fin N => ν) {ω | hitCount A N ω = j}
      = (N.choose j : ℝ≥0∞) * ν A ^ j * ν Aᶜ ^ (N - j) := by
  induction N with
  | zero =>
    intro j
    cases j with
    | zero => simp [hitCount]
    | succ m => simp [hitCount, Nat.choose]
  | succ N ih =>
    intro j
    set μN : Measure (Fin N → E) := Measure.pi fun _ : Fin N => ν with hμN
    -- transfer to the product measure via the measurable equiv at coordinate 0
    set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (N + 1) => E) 0 with he
    have hpres := measurePreserving_piFinSuccAbove (fun _ : Fin (N + 1) => ν) 0
    have hSm : MeasurableSet {ω : Fin (N + 1) → E | hitCount A (N + 1) ω = j} :=
      (measurable_hitCount hA (N + 1)) (measurableSet_singleton j)
    have hsymm : ∀ y : E × (Fin N → E),
        e.symm y = (0 : Fin (N + 1)).insertNth y.1 y.2 := fun y => rfl
    have hpre : e.symm ⁻¹' {ω | hitCount A (N + 1) ω = j}
        = {y : E × (Fin N → E) | A.indicator 1 y.1 + hitCount A N y.2 = j} := by
      ext y
      simp only [Set.mem_preimage, Set.mem_setOf_eq, hsymm, hitCount_insertNth_zero]
    have hstep : (Measure.pi fun _ : Fin (N + 1) => ν)
          {ω | hitCount A (N + 1) ω = j}
        = (ν.prod μN)
            {y : E × (Fin N → E) | A.indicator 1 y.1 + hitCount A N y.2 = j} := by
      rw [← hpre]
      exact ((hpres.symm e).measure_preimage hSm.nullMeasurableSet).symm
    have hTm : MeasurableSet
        {y : E × (Fin N → E) | A.indicator 1 y.1 + hitCount A N y.2 = j} :=
      (((measurable_const.indicator hA).comp measurable_fst).add
        ((measurable_hitCount hA N).comp measurable_snd)) (measurableSet_singleton j)
    rw [hstep, Measure.prod_apply hTm,
      ← lintegral_add_compl (fun x => μN (Prod.mk x ⁻¹'
        {y : E × (Fin N → E) | A.indicator 1 y.1 + hitCount A N y.2 = j})) hA]
    -- the two conditional slices
    have hOnA : ∫⁻ x in A, μN (Prod.mk x ⁻¹'
          {y : E × (Fin N → E) | A.indicator 1 y.1 + hitCount A N y.2 = j}) ∂ν
        = ν A * μN {τ | 1 + hitCount A N τ = j} := by
      rw [setLIntegral_congr_fun hA (fun x hx => ?_), setLIntegral_const, mul_comm]
      have hset : Prod.mk x ⁻¹'
            {y : E × (Fin N → E) | A.indicator 1 y.1 + hitCount A N y.2 = j}
          = {τ : Fin N → E | 1 + hitCount A N τ = j} := by
        ext τ
        simp [Set.indicator_of_mem hx]
      rw [hset]
    have hOnAc : ∫⁻ x in Aᶜ, μN (Prod.mk x ⁻¹'
          {y : E × (Fin N → E) | A.indicator 1 y.1 + hitCount A N y.2 = j}) ∂ν
        = ν Aᶜ * μN {τ | hitCount A N τ = j} := by
      rw [setLIntegral_congr_fun hA.compl (fun x hx => ?_), setLIntegral_const, mul_comm]
      have hset : Prod.mk x ⁻¹'
            {y : E × (Fin N → E) | A.indicator 1 y.1 + hitCount A N y.2 = j}
          = {τ : Fin N → E | hitCount A N τ = j} := by
        ext τ
        simp [Set.indicator_of_notMem (Set.mem_compl_iff A x |>.mp hx)]
      rw [hset]
    rw [hOnA, hOnAc, ih j]
    -- evaluate the shifted slice and close the binomial recurrence
    cases j with
    | zero =>
      have hempty : {τ : Fin N → E | 1 + hitCount A N τ = 0} = ∅ := by
        ext τ; simp
      rw [hempty]
      simp only [measure_empty, mul_zero, zero_add, Nat.choose_zero_right,
        Nat.cast_one, pow_zero, Nat.sub_zero, one_mul]
      rw [pow_succ, mul_comm]
    | succ m =>
      have hshift : {τ : Fin N → E | 1 + hitCount A N τ = m + 1}
          = {τ | hitCount A N τ = m} := by
        ext τ
        simp only [Set.mem_setOf_eq]
        omega
      rw [hshift, ih m]
      rcases lt_trichotomy m N with hmN | hmN | hmN
      · -- m < N: Pascal + exponent bookkeeping
        have h1 : N + 1 - (m + 1) = N - m := by omega
        have h2 : N - m = (N - (m + 1)) + 1 := by omega
        rw [Nat.choose_succ_succ, h1, h2, Nat.cast_add, pow_succ, pow_succ]
        ring
      · -- m = N: the top term
        subst hmN
        simp only [Nat.choose_self, Nat.cast_one, Nat.sub_self, pow_zero,
          Nat.choose_succ_self, Nat.cast_zero, mul_one, zero_mul, mul_zero,
          add_zero, one_mul]
        rw [pow_succ, mul_comm]
      · -- N < m: everything vanishes
        rw [Nat.choose_eq_zero_of_lt hmN, Nat.choose_eq_zero_of_lt (by omega),
          Nat.choose_eq_zero_of_lt (by omega)]
        simp

/-- The point law in real form. -/
lemma pi_hitCount_real_eq_binomial (ν : Measure E) [IsProbabilityMeasure ν] {A : Set E}
    (hA : MeasurableSet A) (N j : ℕ) :
    (Measure.pi fun _ : Fin N => ν).real {ω | hitCount A N ω = j}
      = (N.choose j : ℝ) * (ν.real A) ^ j * (1 - ν.real A) ^ (N - j) := by
  have hc : (ν Aᶜ).toReal = 1 - ν.real A := by
    rw [← measureReal_def, measureReal_compl hA, probReal_univ]
  rw [measureReal_def, pi_hitCount_eq_binomial ν hA N j, ENNReal.toReal_mul,
    ENNReal.toReal_mul, ENNReal.toReal_pow, ENNReal.toReal_pow,
    ENNReal.toReal_natCast, hc, measureReal_def]

/-- **The upper-tail law**: `P(hitCount ≥ k)` over the product measure is
    exactly the `binomTail` polynomial the CP bound is defined from. -/
lemma pi_hitCount_tail_real (ν : Measure E) [IsProbabilityMeasure ν] {A : Set E}
    (hA : MeasurableSet A) (N k : ℕ) :
    (Measure.pi fun _ : Fin N => ν).real {ω | k ≤ hitCount A N ω}
      = binomTail N k (ν.real A) := by
  have hunion : {ω : Fin N → E | k ≤ hitCount A N ω}
      = ⋃ j ∈ Finset.Icc k N, {ω | hitCount A N ω = j} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_Icc]
    constructor
    · exact fun h => ⟨hitCount A N ω, ⟨h, hitCount_le A N ω⟩, rfl⟩
    · rintro ⟨j, ⟨hkj, _⟩, hj⟩
      omega
  have hmeas : ∀ j ∈ Finset.Icc k N,
      MeasurableSet {ω : Fin N → E | hitCount A N ω = j} :=
    fun j _ => (measurable_hitCount hA N) (measurableSet_singleton j)
  have hdisj : (Finset.Icc k N : Set ℕ).PairwiseDisjoint
      (fun j => {ω : Fin N → E | hitCount A N ω = j}) := by
    intro a _ b _ hab
    refine Set.disjoint_left.mpr fun ω ha hb => hab ?_
    simp only [Set.mem_setOf_eq] at ha hb
    omega
  rw [hunion, measureReal_biUnion_finset hdisj hmeas]
  exact Finset.sum_congr rfl fun j _ => pi_hitCount_real_eq_binomial ν hA N j

-- ════════════════════════════════════════════════════════════════
-- § Clopper–Pearson coverage
-- ════════════════════════════════════════════════════════════════

/-- **Clopper–Pearson coverage.** With probability at least `1 − α` over the
    `N` iid samples, the CP lower bound computed from the observed count is a
    genuine lower bound for the true probability `ν(A)`. No tail-monotonicity
    in the parameter is needed: the minimal-counterexample count `k₀` has
    `p < cpLower α N k₀`, so its tail at `p` is `≤ α` by the `sInf` defining
    property, and every count below `k₀` certifies by minimality. -/
theorem cp_coverage (ν : Measure E) [IsProbabilityMeasure ν] {A : Set E}
    (hA : MeasurableSet A) (N : ℕ) {α : ℝ} (hα : 0 ≤ α) :
    1 - α ≤ (Measure.pi fun _ : Fin N => ν).real
        {ω | cpLower α N (hitCount A N ω) ≤ ν.real A} := by
  classical
  set μN : Measure (Fin N → E) := Measure.pi fun _ : Fin N => ν with hμN
  haveI : IsProbabilityMeasure μN := by rw [hμN]; infer_instance
  set p : ℝ := ν.real A with hp
  by_cases hK : ∃ m, p < cpLower α N m ∧ m ≤ N
  · set k₀ := Nat.find hK with hk₀
    obtain ⟨hk₀p, hk₀N⟩ := Nat.find_spec hK
    have hgood : {ω : Fin N → E | hitCount A N ω < k₀}
        ⊆ {ω | cpLower α N (hitCount A N ω) ≤ p} := by
      intro ω hω
      simp only [Set.mem_setOf_eq] at hω ⊢
      by_contra hgt
      exact absurd ⟨not_le.mp hgt, hitCount_le A N ω⟩ (Nat.find_min hK hω)
    have hp01 : p ∈ Set.Icc (0:ℝ) 1 := ⟨measureReal_nonneg, measureReal_le_one⟩
    have htail : μN.real {ω | k₀ ≤ hitCount A N ω} ≤ α := by
      rw [hμN, pi_hitCount_tail_real ν hA N k₀]
      exact binomTail_le_of_lt_cpLower hp01 hk₀p
    have hmeasTail : MeasurableSet {ω : Fin N → E | k₀ ≤ hitCount A N ω} :=
      measurableSet_le measurable_const (measurable_hitCount hA N)
    have hcompl : μN.real {ω | hitCount A N ω < k₀}
        = 1 - μN.real {ω | k₀ ≤ hitCount A N ω} := by
      have hcset : {ω : Fin N → E | hitCount A N ω < k₀}
          = {ω | k₀ ≤ hitCount A N ω}ᶜ := by
        ext ω
        simp [not_le]
      rw [hcset, measureReal_compl hmeasTail, probReal_univ]
    calc 1 - α ≤ 1 - μN.real {ω | k₀ ≤ hitCount A N ω} := by linarith
      _ = μN.real {ω | hitCount A N ω < k₀} := hcompl.symm
      _ ≤ μN.real {ω | cpLower α N (hitCount A N ω) ≤ p} :=
          measureReal_mono hgood
  · have huniv : {ω : Fin N → E | cpLower α N (hitCount A N ω) ≤ p}
        = Set.univ := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_univ, iff_true]
      by_contra hgt
      exact hK ⟨hitCount A N ω, not_le.mp hgt, hitCount_le A N ω⟩
    rw [huniv, probReal_univ]
    linarith

end BinomialCount

-- ════════════════════════════════════════════════════════════════
-- § The solved form: one tail check certifies the reported bound
-- ════════════════════════════════════════════════════════════════

/-- **The binomial tail is monotone in the success probability** — by
    COUPLING, not calculus: instantiate the tail law at the uniform measure
    on `[0,1]` with the nested sets `[0,q] ⊆ [0,p]`, and the tail events
    nest pointwise. -/
lemma binomTail_monotoneOn (N k : ℕ) :
    MonotoneOn (binomTail N k) (Set.Icc (0:ℝ) 1) := by
  intro q hq p hp hqp
  show binomTail N k q ≤ binomTail N k p
  set ν : Measure ℝ := volume.restrict (Set.Icc (0:ℝ) 1) with hν
  haveI : IsProbabilityMeasure ν := by
    constructor
    rw [hν, Measure.restrict_apply_univ, Real.volume_Icc]
    norm_num
  have hreal : ∀ r : ℝ, r ∈ Set.Icc (0:ℝ) 1 → ν.real (Set.Icc 0 r) = r := by
    intro r hr
    rw [hν, measureReal_def, Measure.restrict_apply measurableSet_Icc,
      Set.inter_eq_left.mpr (Set.Icc_subset_Icc le_rfl hr.2), Real.volume_Icc,
      ENNReal.toReal_ofReal (by linarith [hr.1])]
    ring
  have htailq := pi_hitCount_tail_real ν
    (measurableSet_Icc : MeasurableSet (Set.Icc (0:ℝ) q)) N k
  have htailp := pi_hitCount_tail_real ν
    (measurableSet_Icc : MeasurableSet (Set.Icc (0:ℝ) p)) N k
  rw [hreal q hq] at htailq
  rw [hreal p hp] at htailp
  rw [← htailq, ← htailp]
  refine measureReal_mono (fun ω hω => ?_)
  simp only [Set.mem_setOf_eq] at hω ⊢
  exact hω.trans (hitCount_mono (Set.Icc_subset_Icc le_rfl hqp) N ω)

/-- At `q = 1` the tail is exactly 1 (only the `j = N` term survives). -/
lemma binomTail_one {N k : ℕ} (hk : k ≤ N) : binomTail N k 1 = 1 := by
  rw [binomTail, Finset.sum_eq_single N]
  · simp
  · intro j hj hjN
    have hNj : N - j ≠ 0 := by
      rw [Finset.mem_Icc] at hj
      omega
    simp [zero_pow hNj]
  · intro h
    exact absurd (Finset.mem_Icc.mpr ⟨hk, le_rfl⟩) h

/-- **The solved-form CP bound**: one in-kernel tail check
    `binomTail N k q₀ ≤ α` certifies `q₀` as a lower bound for `cpLower` —
    the form the driver's reported `q₀` can be verified in. -/
lemma le_cpLower_of_tail_le {α q₀ : ℝ} (hα : α < 1) {N k : ℕ} (hk : k ≤ N)
    (hq₀ : q₀ ∈ Set.Icc (0:ℝ) 1) (h : binomTail N k q₀ ≤ α) :
    q₀ ≤ cpLower α N k := by
  have hone : (1:ℝ) ∈ {q | q ∈ Set.Icc (0:ℝ) 1 ∧ α < binomTail N k q} :=
    ⟨Set.right_mem_Icc.mpr zero_le_one, by rwa [binomTail_one hk]⟩
  refine le_csInf ⟨1, hone⟩ fun q hq => ?_
  by_contra hlt
  have hmono := binomTail_monotoneOn N k hq.1 hq₀ (not_le.mp hlt).le
  linarith [hq.2]

-- ════════════════════════════════════════════════════════════════
-- § The composed guarantee: CERTIFY's exact arithmetic, end to end
-- ════════════════════════════════════════════════════════════════

/-- **The Clopper–Pearson smoothing certificate.** Sample `N` iid standard
    Gaussians; report the radius `σ·Φ⁻¹(cpLower α N k)` from the exact
    binomial lower confidence limit at the observed class count `k`. With
    probability `≥ 1 − α` over the samples, the reported radius is GENUINELY
    certified: every `‖δ‖ < σ·Φ⁻¹(cpLower α N k)` keeps `y` the strict argmax
    of the smoothed classifier. Same guarantee shape as
    `smoothing_mc_certified`, but the confidence bound is now the arithmetic
    Cohen's CERTIFY actually deploys. -/
theorem smoothing_cp_certified {n k : ℕ} {σ : ℝ} (hσ : 0 < σ)
    {C : EuclideanSpace ℝ (Fin (n + 1)) → Fin k} (hC : Measurable C)
    (hp : ∀ c x, (∫ z, (if C (x + σ • z) = c then (1:ℝ) else 0)
      ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))) ∈ Set.Ioo (0:ℝ) 1)
    (x : EuclideanSpace ℝ (Fin (n + 1))) (y : Fin k)
    (N : ℕ) {α : ℝ} (hα : 0 ≤ α) :
    1 - α
      ≤ (Measure.pi fun _ : Fin N => stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))).real
          {ω | ∀ δ : EuclideanSpace ℝ (Fin (n + 1)),
            ‖δ‖ < σ * stdNormalQuantile
              (cpLower α N (∑ i, if C (x + σ • ω i) = y then 1 else 0)) →
            ∀ j, j ≠ y →
              (∫ z, (if C (x + δ + σ • z) = j then (1:ℝ) else 0)
                  ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
                < ∫ z, (if C (x + δ + σ • z) = y then (1:ℝ) else 0)
                    ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))} := by
  classical
  set γ := stdGaussian (EuclideanSpace ℝ (Fin (n + 1))) with hγ
  set A : Set (EuclideanSpace ℝ (Fin (n + 1))) := {v | C (x + σ • v) = y} with hA_def
  have hA : MeasurableSet A :=
    (hC.comp (measurable_const.add (measurable_id.const_smul σ)))
      (measurableSet_singleton y)
  -- the true class probability IS the Gaussian measure of A
  have hpA : γ.real A = ∫ z, (if C (x + σ • z) = y then (1:ℝ) else 0) ∂γ := by
    rw [← integral_indicator_one hA]
    refine integral_congr_ae (ae_of_all _ fun z => ?_)
    by_cases h : C (x + σ • z) = y
    · rw [Set.indicator_of_mem (show z ∈ A from h), Pi.one_apply]
      simp [h]
    · rw [Set.indicator_of_notMem (show z ∉ A from h)]
      simp [h]
  -- the driver's count IS hitCount
  have hcount : ∀ ω : Fin N → EuclideanSpace ℝ (Fin (n + 1)),
      (∑ i, if C (x + σ • ω i) = y then 1 else 0) = hitCount A N ω := by
    intro ω
    refine Finset.sum_congr rfl fun i _ => ?_
    by_cases h : C (x + σ • ω i) = y
    · rw [if_pos h, Set.indicator_of_mem (show ω i ∈ A from h), Pi.one_apply]
    · rw [if_neg h, Set.indicator_of_notMem (show ω i ∉ A from h)]
  -- the coverage event implies the certificate
  have hsub : {ω : Fin N → EuclideanSpace ℝ (Fin (n + 1)) |
        cpLower α N (hitCount A N ω) ≤ γ.real A}
      ⊆ {ω | ∀ δ : EuclideanSpace ℝ (Fin (n + 1)),
            ‖δ‖ < σ * stdNormalQuantile
              (cpLower α N (∑ i, if C (x + σ • ω i) = y then 1 else 0)) →
            ∀ j, j ≠ y →
              (∫ z, (if C (x + δ + σ • z) = j then (1:ℝ) else 0) ∂γ)
                < ∫ z, (if C (x + δ + σ • z) = y then (1:ℝ) else 0) ∂γ} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    intro δ hδ j hj
    rw [hcount ω] at hδ
    set q : ℝ := cpLower α N (hitCount A N ω) with hq
    rcases le_or_gt q 0 with hq0 | hq0
    · rw [stdNormalQuantile_of_nonpos hq0, mul_zero] at hδ
      exact absurd hδ (not_lt.mpr (norm_nonneg δ))
    · have hpy := hp y x
      have hple : q ≤ ∫ z, (if C (x + σ • z) = y then (1:ℝ) else 0) ∂γ :=
        hpA ▸ hω
      have hqIoo : q ∈ Set.Ioo (0:ℝ) 1 := ⟨hq0, lt_of_le_of_lt hple hpy.2⟩
      have hmono := stdNormalQuantile_monotoneOn hqIoo hpy hple
      have hδ' : ‖δ‖ < σ * stdNormalQuantile
          (∫ z, (if C (x + σ • z) = y then (1:ℝ) else 0) ∂γ) :=
        lt_of_lt_of_le hδ (mul_le_mul_of_nonneg_left hmono hσ.le)
      exact smoothing_certified_radius_classifier hσ hC hp hδ' j hj
  calc 1 - α
      ≤ (Measure.pi fun _ : Fin N => γ).real
          {ω | cpLower α N (hitCount A N ω) ≤ γ.real A} :=
        cp_coverage γ hA N hα
    _ ≤ _ := measureReal_mono hsub

/-- **The driver tie, solved form.** For the OBSERVED count `k₀` and the
    driver's reported CP lower bound `q₀` (rationalized down), ONE in-kernel
    tail check `binomTail N k₀ q₀ ≤ α` certifies: with probability `≥ 1 − α`
    over the samples, IF the count comes out `k₀` THEN the radius
    `σ·Φ⁻¹(q₀)` is genuinely certified. This is the per-image scorecard
    theorem shape: instantiate at the driver's `(N, k₀, α, q₀)` and the
    hypothesis is pure kernel rational arithmetic. -/
theorem smoothing_cp_certified_solved {n k : ℕ} {σ : ℝ} (hσ : 0 < σ)
    {C : EuclideanSpace ℝ (Fin (n + 1)) → Fin k} (hC : Measurable C)
    (hp : ∀ c x, (∫ z, (if C (x + σ • z) = c then (1:ℝ) else 0)
      ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))) ∈ Set.Ioo (0:ℝ) 1)
    (x : EuclideanSpace ℝ (Fin (n + 1))) (y : Fin k)
    {N k₀ : ℕ} (hk₀ : k₀ ≤ N) {α q₀ : ℝ} (hα : 0 ≤ α) (hα1 : α < 1)
    (hq₀ : q₀ ∈ Set.Ioo (0:ℝ) 1) (htail : binomTail N k₀ q₀ ≤ α) :
    1 - α
      ≤ (Measure.pi fun _ : Fin N => stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))).real
          {ω | (∑ i, if C (x + σ • ω i) = y then 1 else 0) = k₀ →
            ∀ δ : EuclideanSpace ℝ (Fin (n + 1)),
            ‖δ‖ < σ * stdNormalQuantile q₀ →
            ∀ j, j ≠ y →
              (∫ z, (if C (x + δ + σ • z) = j then (1:ℝ) else 0)
                  ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
                < ∫ z, (if C (x + δ + σ • z) = y then (1:ℝ) else 0)
                    ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))} := by
  classical
  set γ := stdGaussian (EuclideanSpace ℝ (Fin (n + 1))) with hγ
  set A : Set (EuclideanSpace ℝ (Fin (n + 1))) := {v | C (x + σ • v) = y} with hA_def
  have hA : MeasurableSet A :=
    (hC.comp (measurable_const.add (measurable_id.const_smul σ)))
      (measurableSet_singleton y)
  have hpA : γ.real A = ∫ z, (if C (x + σ • z) = y then (1:ℝ) else 0) ∂γ := by
    rw [← integral_indicator_one hA]
    refine integral_congr_ae (ae_of_all _ fun z => ?_)
    by_cases h : C (x + σ • z) = y
    · rw [Set.indicator_of_mem (show z ∈ A from h), Pi.one_apply]
      simp [h]
    · rw [Set.indicator_of_notMem (show z ∉ A from h)]
      simp [h]
  have hcount : ∀ ω : Fin N → EuclideanSpace ℝ (Fin (n + 1)),
      (∑ i, if C (x + σ • ω i) = y then 1 else 0) = hitCount A N ω := by
    intro ω
    refine Finset.sum_congr rfl fun i _ => ?_
    by_cases h : C (x + σ • ω i) = y
    · rw [if_pos h, Set.indicator_of_mem (show ω i ∈ A from h), Pi.one_apply]
    · rw [if_neg h, Set.indicator_of_notMem (show ω i ∉ A from h)]
  have hsub : {ω : Fin N → EuclideanSpace ℝ (Fin (n + 1)) |
        cpLower α N (hitCount A N ω) ≤ γ.real A}
      ⊆ {ω | (∑ i, if C (x + σ • ω i) = y then 1 else 0) = k₀ →
            ∀ δ : EuclideanSpace ℝ (Fin (n + 1)),
            ‖δ‖ < σ * stdNormalQuantile q₀ →
            ∀ j, j ≠ y →
              (∫ z, (if C (x + δ + σ • z) = j then (1:ℝ) else 0) ∂γ)
                < ∫ z, (if C (x + δ + σ • z) = y then (1:ℝ) else 0) ∂γ} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    intro hcnt δ hδ j hj
    rw [hcount ω] at hcnt
    have hqcp : q₀ ≤ cpLower α N (hitCount A N ω) := by
      rw [hcnt]
      exact le_cpLower_of_tail_le hα1 hk₀ (Set.Ioo_subset_Icc_self hq₀) htail
    have hpy := hp y x
    have hple : q₀ ≤ ∫ z, (if C (x + σ • z) = y then (1:ℝ) else 0) ∂γ := by
      rw [← hpA]
      exact hqcp.trans hω
    have hmono := stdNormalQuantile_monotoneOn hq₀ hpy hple
    have hδ' : ‖δ‖ < σ * stdNormalQuantile
        (∫ z, (if C (x + σ • z) = y then (1:ℝ) else 0) ∂γ) :=
      lt_of_lt_of_le hδ (mul_le_mul_of_nonneg_left hmono hσ.le)
    exact smoothing_certified_radius_classifier hσ hC hp hδ' j hj
  calc 1 - α
      ≤ (Measure.pi fun _ : Fin N => γ).real
          {ω | cpLower α N (hitCount A N ω) ≤ γ.real A} :=
        cp_coverage γ hA N hα
    _ ≤ _ := measureReal_mono hsub

-- ════════════════════════════════════════════════════════════════
-- § Demo tail checks: the hypothesis is kernel rational arithmetic
-- ════════════════════════════════════════════════════════════════

/-- Demo tail check, kernel-arithmetic only: 99 hits in 100 draws certify
    `q₀ = 0.9` at `α = 1/1000` (radius `σ·Φ⁻¹(0.9) ≈ 1.28σ`). -/
lemma binomTail_check_99of100 : binomTail 100 99 (9/10) ≤ 1/1000 := by
  have hset : Finset.Icc 99 100 = {99, 100} := by decide
  have h99 : Nat.choose 100 99 = 100 := by decide
  rw [binomTail, hset, Finset.sum_insert (by decide), Finset.sum_singleton,
    h99, Nat.choose_self]
  norm_num

set_option maxRecDepth 8000 in
set_option exponentiation.threshold 4000 in
/-- Mid-scale demo: 999 hits in 1000 draws certify `q₀ = 0.985` at
    `α = 1/1000` (radius `σ·Φ⁻¹(0.985) ≈ 2.17σ`). Driver-scale `N = 10112`
    instances are generator territory (CertsHeavy): raise
    `exponentiation.threshold`, ladder the `Nat.choose` literals via
    `Nat.choose_symm`/`choose_succ_right_eq` instead of `decide`. -/
lemma binomTail_check_999of1000 : binomTail 1000 999 (985/1000) ≤ 1/1000 := by
  have hset : Finset.Icc 999 1000 = {999, 1000} := by decide
  have h999 : Nat.choose 1000 999 = 1000 := by
    rw [← Nat.choose_symm (by norm_num : 999 ≤ 1000)]
    norm_num
  rw [binomTail, hset, Finset.sum_insert (by decide), Finset.sum_singleton,
    h999, Nat.choose_self]
  norm_num

end Proofs
