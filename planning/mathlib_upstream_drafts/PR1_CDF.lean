/-
PR 1 — feat(Probability/CDF): strict monotonicity and continuity of the cdf

Content below is to be APPENDED to `Mathlib/Probability/CDF.lean`, inside the existing
`namespace ProbabilityTheory` (i.e. insert before its `end ProbabilityTheory`). The two new
sections reuse that file's existing opens (`open MeasureTheory Measure Set Filter`,
`open scoped Topology`); no new imports are needed (`Mathlib.MeasureTheory.Measure.OpenPos`
and `Mathlib.Topology.Order.LeftRightLim` are already transitive). Add `Brett Koonce` to the
file's `Authors:` line.

Verified to compile against the pinned Mathlib by `LeanMlir/Proofs/UpstreamDraft.lean`
(namespace `MathlibUpstream`); keep the two in sync.

Also update the module docstring's "Main statements" with:
* `ProbabilityTheory.strictMono_cdf_iff`: the cdf of a probability measure is strictly
  monotone iff the measure is positive on nonempty open sets.
* `ProbabilityTheory.continuous_cdf_iff`: the cdf of a probability measure is continuous iff
  the measure has no atoms.
-/

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
