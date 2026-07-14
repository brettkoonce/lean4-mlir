import Mathlib.Probability.CDF
import Mathlib.Probability.Distributions.Gaussian.Real

/-! # Mathlib upstreaming drafts — CI-guarded copies

Compiles the contents of `planning/mathlib_upstream_drafts/PR1_CDF.lean` (generic cdf
lemmas, target `Mathlib/Probability/CDF.lean`) and
`planning/mathlib_upstream_drafts/PR2_GaussianReal.lean` (Gaussian cdf facts, target
`Mathlib/Probability/Distributions/Gaussian/Real.lean`) against this repo's pinned Mathlib,
inside the namespace `MathlibUpstream` so nothing clashes with Mathlib or with
`LeanMlir.Proofs`. Keep in sync with those two files.

A `Certs` root (audited in `tests/AuditAxioms.lean`) so the drafts can't rot between
Mathlib pin bumps while the PRs are in flight. Once a PR merges and the pin catches up,
delete the corresponding section here and cite Mathlib directly.

Fast check: `lake env lean LeanMlir/Proofs/UpstreamDraft.lean` -/

namespace MathlibUpstream

open MeasureTheory Measure Set Filter ProbabilityTheory
open scoped Topology ENNReal NNReal

-- ════════════════════════════════════════════════════════════════
-- PR 1 — generic cdf lemmas (append to Mathlib/Probability/CDF.lean)
-- ════════════════════════════════════════════════════════════════

section StrictMono

variable (μ : Measure ℝ) [IsProbabilityMeasure μ]

/-- If a probability measure on `ℝ` gives positive mass to every nonempty open set, then its
cdf is strictly monotone. -/
theorem strictMono_cdf [μ.IsOpenPosMeasure] : StrictMono (cdf μ) := by
  intro s t hst
  have hIoc : 0 < μ (Ioc s t) :=
    (isOpen_Ioo.measure_pos μ (nonempty_Ioo.mpr hst)).trans_le (measure_mono Ioo_subset_Ioc_self)
  have hreal : 0 < μ.real (Ioc s t) := ENNReal.toReal_pos hIoc.ne' (measure_ne_top _ _)
  have hsplit : μ.real (Iic t) = μ.real (Iic s) + μ.real (Ioc s t) := by
    rw [← measureReal_union (Iic_disjoint_Ioc le_rfl) measurableSet_Ioc,
      Iic_union_Ioc_eq_Iic hst.le]
  simp only [cdf_eq_real]
  linarith

/-- If a probability measure on `ℝ` gives positive mass to every nonempty open set, then its
cdf is everywhere positive: there is mass below every point. -/
theorem cdf_pos [μ.IsOpenPosMeasure] (x : ℝ) : 0 < cdf μ x := by
  have h : 0 < μ (Iic x) :=
    (isOpen_Iio.measure_pos μ ⟨x - 1, sub_one_lt x⟩).trans_le (measure_mono Iio_subset_Iic_self)
  rw [cdf_eq_real]
  exact ENNReal.toReal_pos h.ne' (measure_ne_top _ _)

/-- If a probability measure on `ℝ` gives positive mass to every nonempty open set, then its
cdf is everywhere less than 1: there is mass above every point. -/
theorem cdf_lt_one [μ.IsOpenPosMeasure] (x : ℝ) : cdf μ x < 1 := by
  have h : 0 < μ.real (Ioi x) :=
    ENNReal.toReal_pos (isOpen_Ioi.measure_pos μ ⟨x + 1, lt_add_one x⟩).ne' (measure_ne_top _ _)
  have hc : μ.real (Ioi x) = 1 - μ.real (Iic x) := by
    rw [← compl_Iic, measureReal_compl measurableSet_Iic, probReal_univ]
  rw [cdf_eq_real]
  linarith

/-- If a probability measure on `ℝ` gives positive mass to every nonempty open set, then its
cdf maps into the open unit interval. -/
theorem cdf_mem_Ioo [μ.IsOpenPosMeasure] (x : ℝ) : cdf μ x ∈ Ioo (0 : ℝ) 1 :=
  ⟨cdf_pos μ x, cdf_lt_one μ x⟩

/-- If the cdf of a probability measure on `ℝ` is strictly monotone, the measure gives positive
mass to every nonempty open set. Converse of `strictMono_cdf`. -/
theorem isOpenPosMeasure_of_strictMono_cdf (h : StrictMono (cdf μ)) : μ.IsOpenPosMeasure := by
  refine ⟨fun U hU hUne hU0 => ?_⟩
  obtain ⟨a, b, hab, habU⟩ := hU.exists_Ioo_subset hUne
  obtain ⟨c, hac, hcb⟩ := exists_between hab
  have h0 : μ (Ioc a c) = 0 :=
    measure_mono_null (fun x hx => habU ⟨hx.1, hx.2.trans_lt hcb⟩) hU0
  have hsplit : μ.real (Iic c) = μ.real (Iic a) + μ.real (Ioc a c) := by
    rw [← measureReal_union (Iic_disjoint_Ioc le_rfl) measurableSet_Ioc,
      Iic_union_Ioc_eq_Iic hac.le]
  have h0' : μ.real (Ioc a c) = 0 := by simp [measureReal_def, h0]
  have hlt := h hac
  rw [cdf_eq_real, cdf_eq_real, hsplit, h0', add_zero] at hlt
  exact lt_irrefl _ hlt

/-- The cdf of a probability measure on `ℝ` is strictly monotone iff the measure gives positive
mass to every nonempty open set. -/
theorem strictMono_cdf_iff : StrictMono (cdf μ) ↔ μ.IsOpenPosMeasure :=
  ⟨isOpenPosMeasure_of_strictMono_cdf μ, fun h => haveI := h; strictMono_cdf μ⟩

end StrictMono

section Continuous

variable (μ : Measure ℝ) [IsProbabilityMeasure μ]

/-- The cdf of a probability measure on `ℝ` without atoms has no jumps: its left limit at every
point equals its value there. -/
theorem leftLim_cdf [NullSingletonClass μ] (x : ℝ) : Function.leftLim (cdf μ) x = cdf μ x := by
  have hsing : μ {x} = 0 := measure_singleton x
  rw [← measure_cdf μ, StieltjesFunction.measure_singleton] at hsing
  have h1 : cdf μ x - Function.leftLim (cdf μ) x ≤ 0 := ENNReal.ofReal_eq_zero.mp hsing
  have h2 : Function.leftLim (cdf μ) x ≤ cdf μ x := (cdf μ).mono.leftLim_le le_rfl
  linarith

/-- The cdf of a probability measure on `ℝ` without atoms is continuous: it is monotone and
right-continuous, and by `leftLim_cdf` it has no jumps. -/
theorem continuous_cdf [NullSingletonClass μ] : Continuous (cdf μ) := by
  rw [continuous_iff_continuousAt]
  intro x
  rw [(cdf μ).mono.continuousAt_iff_leftLim_eq_rightLim, leftLim_cdf μ x]
  exact ((cdf μ).mono.continuousWithinAt_Ioi_iff_rightLim_eq.mp
    (((cdf μ).right_continuous x).mono Ioi_subset_Ici_self)).symm

/-- If the cdf of a probability measure on `ℝ` is continuous, the measure has no atoms.
Converse of `continuous_cdf`. -/
theorem nullSingletonClass_of_continuous_cdf (h : Continuous (cdf μ)) : NullSingletonClass μ := by
  refine ⟨fun x => ?_⟩
  have hll : Function.leftLim (cdf μ) x = cdf μ x :=
    (cdf μ).mono.continuousWithinAt_Iio_iff_leftLim_eq.mp h.continuousAt.continuousWithinAt
  rw [← measure_cdf μ, StieltjesFunction.measure_singleton, hll, sub_self, ENNReal.ofReal_zero]

/-- The cdf of a probability measure on `ℝ` is continuous iff the measure has no atoms. -/
theorem continuous_cdf_iff : Continuous (cdf μ) ↔ NullSingletonClass μ :=
  ⟨nullSingletonClass_of_continuous_cdf μ, fun h => haveI := h; continuous_cdf μ⟩

end Continuous

-- ════════════════════════════════════════════════════════════════
-- PR 2 — Gaussian cdf facts
--   (append to Mathlib/Probability/Distributions/Gaussian/Real.lean)
-- ════════════════════════════════════════════════════════════════

section GaussianCDF

/-- A real Gaussian measure with nonzero variance gives positive mass to every nonempty open
set: its density is everywhere positive, so it dominates the Lebesgue measure. -/
lemma isOpenPosMeasure_gaussianReal (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    (gaussianReal μ v).IsOpenPosMeasure :=
  (gaussianReal_absolutelyContinuous' μ hv).isOpenPosMeasure

instance instIsOpenPosMeasureGaussianReal (μ : ℝ) (v : ℝ≥0) [NeZero v] :
    (gaussianReal μ v).IsOpenPosMeasure :=
  isOpenPosMeasure_gaussianReal μ (NeZero.ne v)

/-- The cdf of a real Gaussian measure with nonzero variance is strictly monotone. -/
lemma strictMono_cdf_gaussianReal (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    StrictMono (cdf (gaussianReal μ v)) :=
  haveI := isOpenPosMeasure_gaussianReal μ hv
  strictMono_cdf _

/-- The cdf of a real Gaussian measure with nonzero variance is continuous. -/
lemma continuous_cdf_gaussianReal (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    Continuous (cdf (gaussianReal μ v)) :=
  haveI := nullSingletonClass_gaussianReal (μ := μ) hv
  continuous_cdf _

/-- The cdf of a real Gaussian measure with nonzero variance is everywhere positive. -/
lemma cdf_gaussianReal_pos (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) (x : ℝ) :
    0 < cdf (gaussianReal μ v) x :=
  haveI := isOpenPosMeasure_gaussianReal μ hv
  cdf_pos _ x

/-- The cdf of a real Gaussian measure with nonzero variance is everywhere less than 1. -/
lemma cdf_gaussianReal_lt_one (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) (x : ℝ) :
    cdf (gaussianReal μ v) x < 1 :=
  haveI := isOpenPosMeasure_gaussianReal μ hv
  cdf_lt_one _ x

/-- The cdf of a real Gaussian measure with nonzero variance maps into the open unit
interval. -/
lemma cdf_gaussianReal_mem_Ioo (μ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) (x : ℝ) :
    cdf (gaussianReal μ v) x ∈ Ioo (0 : ℝ) 1 :=
  ⟨cdf_gaussianReal_pos μ hv x, cdf_gaussianReal_lt_one μ hv x⟩

/-- Symmetry of the centered Gaussian cdf: `Φ_v (-x) = 1 - Φ_v x`. The centered Gaussian is
invariant under negation, so the mass of `Iic (-x)` is the mass of `Ici x`, which (no atoms)
is the mass of the complement of `Iic x`. -/
lemma cdf_gaussianReal_neg {v : ℝ≥0} (hv : v ≠ 0) (x : ℝ) :
    cdf (gaussianReal 0 v) (-x) = 1 - cdf (gaussianReal 0 v) x := by
  haveI : NullSingletonClass (gaussianReal 0 v) := nullSingletonClass_gaussianReal hv
  have hmap : (gaussianReal 0 v).map (fun y => -y) = gaussianReal 0 v := by
    simpa using gaussianReal_map_neg (μ := 0) (v := v)
  have hpre : (fun y : ℝ => -y) ⁻¹' Iic (-x) = Ici x := by ext y; simp
  have hIic : gaussianReal 0 v (Iic (-x)) = gaussianReal 0 v (Ici x) := by
    conv_lhs => rw [← hmap]
    rw [Measure.map_apply measurable_neg measurableSet_Iic, hpre]
  have hIci : gaussianReal 0 v (Ici x) = gaussianReal 0 v (Ioi x) :=
    measure_congr Ioi_ae_eq_Ici.symm
  have hcompl : (gaussianReal 0 v).real (Ioi x) = 1 - (gaussianReal 0 v).real (Iic x) := by
    rw [← compl_Iic, measureReal_compl measurableSet_Iic, probReal_univ]
  rw [cdf_eq_real, cdf_eq_real, measureReal_def, hIic, hIci, ← measureReal_def, hcompl]

/-- Shifting the mean of a Gaussian shifts its cdf: `cdf (gaussianReal (μ + δ) v) x
= cdf (gaussianReal μ v) (x - δ)`. Holds for `v = 0` as well (both sides are Dirac cdfs). -/
lemma cdf_gaussianReal_sub_const (μ δ : ℝ) (v : ℝ≥0) (x : ℝ) :
    cdf (gaussianReal (μ + δ) v) x = cdf (gaussianReal μ v) (x - δ) := by
  have hpre : (· + δ) ⁻¹' Iic x = Iic (x - δ) := by
    ext y; simp [le_sub_iff_add_le]
  rw [cdf_eq_real, cdf_eq_real, ← gaussianReal_map_add_const δ, measureReal_def,
    Measure.map_apply (measurable_add_const δ) measurableSet_Iic, hpre, ← measureReal_def]

end GaussianCDF

end MathlibUpstream
