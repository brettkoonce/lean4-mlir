/-
PR 2 — feat(Probability/Distributions/Gaussian/Real): cdf of the real Gaussian distribution
(depends on PR 1: `strictMono_cdf`, `continuous_cdf`, `cdf_pos`, `cdf_lt_one`)

Content below is to be APPENDED to `Mathlib/Probability/Distributions/Gaussian/Real.lean`,
inside the existing `namespace ProbabilityTheory` (insert before its final `end`). It needs
ONE new import in that file's header:

  public import Mathlib.Probability.CDF

(the file currently does not import the cdf; the sibling distribution files Exponential/
Gamma/Pareto already import it, so this matches existing practice — DECIDED: take this
route; a new `Gaussian/CDF.lean` file remains the fallback if reviewers prefer it). Add
`Brett Koonce` to Real.lean's `Authors:` line. The section opens `Set` locally (Real.lean
does not open it; PR 1's host file does).

Verified to compile against the pinned Mathlib by `LeanMlir/Proofs/UpstreamDraft.lean`
(namespace `MathlibUpstream`); keep the two in sync.
-/

section GaussianCDF

open Set

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
