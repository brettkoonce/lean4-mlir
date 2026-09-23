import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.CDF

/-! # The standard normal: `Φ`, its quantile `Φ⁻¹`, and full support

`stdNormalCDF` (Mathlib's `cdf` of `gaussianReal 0 1`) and `stdNormalQuantile` (`sSup {t | Φ t < p}`,
the honest inverse on `(0,1)`), with the facts every smoothing certificate uses: `Φ` is strictly
monotone and symmetric; on `(0,1)` the quantile is monotone, odd about `½`, continuous and inverts
`Φ` both ways; below `0` it takes the junk value `0`. The `IsOpenPosMeasure` instances say the
1-D and the multivariate standard Gaussian charge every nonempty open set. Mathlib has none of
this for the Gaussian quantile.
-/

namespace Proofs

open MeasureTheory ProbabilityTheory Filter
open scoped Topology

/-- `N(0,1)` charges every nonempty open set (the pdf is everywhere positive) —
    packaged as the Mathlib `IsOpenPosMeasure` class. -/
instance : (gaussianReal 0 1).IsOpenPosMeasure :=
  (gaussianReal_absolutelyContinuous' 0 one_ne_zero).isOpenPosMeasure

/-- The standard Gaussian on a finite-dimensional inner-product space charges
    every nonempty open set: it is the pushforward of the pi-Gaussian (open-pos
    by `pi.isOpenPosMeasure`) under the surjective continuous basis sum. -/
instance stdGaussian.instIsOpenPosMeasure {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E]
    [BorelSpace E] : (stdGaussian E).IsOpenPosMeasure := by
  refine Continuous.isOpenPosMeasure_map (by fun_prop) fun e => ?_
  exact ⟨fun i => (stdOrthonormalBasis ℝ E).repr e i,
    by simpa using (stdOrthonormalBasis ℝ E).sum_repr e⟩

-- ════════════════════════════════════════════════════════════════
-- § Φ: the standard-normal CDF, strictly monotone and symmetric
-- ════════════════════════════════════════════════════════════════

/-- The standard-normal CDF `Φ` — Mathlib's `cdf` of the genuine `gaussianReal 0 1`. -/
noncomputable def stdNormalCDF : ℝ → ℝ := fun t => cdf (gaussianReal 0 1) t

/-- The standard-normal quantile `Φ⁻¹`, as `sSup {t | Φ t < p}`. Total on ℝ (junk value
    outside `(0,1)`, where the defining set is empty or unbounded); the honest inverse on
    `(0,1)`, which is where every guarded use below lives. -/
noncomputable def stdNormalQuantile (p : ℝ) : ℝ := sSup {t | stdNormalCDF t < p}

/-- Every open interval carries positive standard-Gaussian mass (the pdf is everywhere
    positive). The engine of strict monotonicity. -/
lemma stdGaussian_Ioo_pos {s t : ℝ} (hst : s < t) :
    0 < gaussianReal 0 1 (Set.Ioo s t) :=
  isOpen_Ioo.measure_pos _ (Set.nonempty_Ioo.2 hst)

/-- `Φ` is strictly monotone: `Φ t − Φ s = P(Ioc s t) > 0` for `s < t`. -/
lemma stdNormalCDF_strictMono : StrictMono stdNormalCDF := by
  intro s t hst
  have hIoc : 0 < gaussianReal 0 1 (Set.Ioc s t) :=
    lt_of_lt_of_le (stdGaussian_Ioo_pos hst) (measure_mono Set.Ioo_subset_Ioc_self)
  have hreal : 0 < (gaussianReal 0 1).real (Set.Ioc s t) :=
    ENNReal.toReal_pos hIoc.ne' (measure_ne_top _ _)
  have hsplit : (gaussianReal 0 1).real (Set.Iic t)
      = (gaussianReal 0 1).real (Set.Iic s) + (gaussianReal 0 1).real (Set.Ioc s t) := by
    rw [← measureReal_union (by exact Set.Iic_disjoint_Ioc le_rfl) measurableSet_Ioc,
      Set.Iic_union_Ioc_eq_Iic hst.le]
  simp only [stdNormalCDF, cdf_eq_real]
  linarith

/-- Gaussian symmetry `Φ(−t) = 1 − Φ(t)`: the standard Gaussian is invariant under
    negation, so `P(Iic (−t)) = P(Ici t)`, and (no atoms) the complement gives the rest. -/
lemma stdNormalCDF_neg (t : ℝ) : stdNormalCDF (-t) = 1 - stdNormalCDF t := by
  have : NullSingletonClass (gaussianReal 0 1) := nullSingletonClass_gaussianReal one_ne_zero
  have hmap : (gaussianReal 0 1).map (fun x => -x) = gaussianReal 0 1 := by
    simpa using gaussianReal_map_neg (μ := 0) (v := 1)
  have hpre : (fun x : ℝ => -x) ⁻¹' Set.Iic (-t) = Set.Ici t := by
    ext x; simp
  have hIic : gaussianReal 0 1 (Set.Iic (-t)) = gaussianReal 0 1 (Set.Ici t) := by
    conv_lhs => rw [← hmap]
    rw [Measure.map_apply measurable_neg measurableSet_Iic, hpre]
  have hIci : gaussianReal 0 1 (Set.Ici t) = gaussianReal 0 1 (Set.Ioi t) :=
    measure_congr Ioi_ae_eq_Ici.symm
  have hcompl : (gaussianReal 0 1).real (Set.Ioi t)
      = 1 - (gaussianReal 0 1).real (Set.Iic t) := by
    rw [← Set.compl_Iic, measureReal_compl measurableSet_Iic, probReal_univ]
  simp only [stdNormalCDF, cdf_eq_real]
  rw [Measure.real, hIic, hIci, ← Measure.real, hcompl]

-- ════════════════════════════════════════════════════════════════
-- § Φ⁻¹ on (0,1): the defining sets behave, mono + odd-about-½
-- ════════════════════════════════════════════════════════════════

/-- `Φ → 0` at `−∞`, so for `p > 0` some `t` has `Φ t < p` — the quantile's set is
    nonempty. -/
lemma stdNormalCDF_exists_lt {p : ℝ} (hp : 0 < p) : ∃ t, stdNormalCDF t < p :=
  ((tendsto_cdf_atBot (μ := gaussianReal 0 1)).eventually_lt_const hp).exists

/-- `Φ → 1` at `+∞`, so for `p < 1` some `t` has `Φ t > p`. -/
lemma stdNormalCDF_exists_gt {p : ℝ} (hp : p < 1) : ∃ t, p < stdNormalCDF t :=
  ((tendsto_cdf_atTop (μ := gaussianReal 0 1)).eventually_const_lt hp).exists

/-- For `p < 1` the sub-level set `{Φ < p}` is bounded above (anything past a point with
    `Φ > p` is excluded). -/
lemma stdNormalCDF_sublevel_bddAbove {p : ℝ} (hp : p < 1) :
    BddAbove {t | stdNormalCDF t < p} := by
  obtain ⟨T, hT⟩ := stdNormalCDF_exists_gt hp
  exact ⟨T, fun t ht =>
    (stdNormalCDF_strictMono.monotone.reflect_lt (lt_trans ht hT)).le⟩

/-- **`hmono` discharged:** the real quantile is monotone on `(0,1)` — larger `p`, larger
    sub-level set, larger `sSup`. -/
lemma stdNormalQuantile_monotoneOn :
    MonotoneOn stdNormalQuantile (Set.Ioo 0 1) := by
  intro a ha b hb hab
  exact csSup_le_csSup (stdNormalCDF_sublevel_bddAbove hb.2)
    (stdNormalCDF_exists_lt ha.1) (fun t ht => lt_of_lt_of_le ht hab)

/-- **No flat step at level `q`:** `sSup {Φ < q} = sInf {Φ > q}`. Any gap between them
    would contain two points where `Φ = q` exactly — impossible for a strictly monotone
    `Φ`. The bridge between the quantile's `sSup` form and its mirrored `sInf` form. -/
lemma stdNormalCDF_sSup_lt_eq_sInf_gt {q : ℝ} (hq : q ∈ Set.Ioo (0:ℝ) 1) :
    sSup {t | stdNormalCDF t < q} = sInf {t | q < stdNormalCDF t} := by
  have hAne : Set.Nonempty {t | stdNormalCDF t < q} := stdNormalCDF_exists_lt hq.1
  have hBne : Set.Nonempty {t | q < stdNormalCDF t} := stdNormalCDF_exists_gt hq.2
  have hAbdd : BddAbove {t | stdNormalCDF t < q} := stdNormalCDF_sublevel_bddAbove hq.2
  have hBbdd : BddBelow {t | q < stdNormalCDF t} := by
    obtain ⟨s, hs⟩ := hAne
    exact ⟨s, fun t ht =>
      (stdNormalCDF_strictMono.monotone.reflect_lt (lt_trans hs ht)).le⟩
  have hle : sSup {t | stdNormalCDF t < q} ≤ sInf {t | q < stdNormalCDF t} :=
    csSup_le hAne (fun a ha => le_csInf hBne (fun b hb =>
      (stdNormalCDF_strictMono.monotone.reflect_lt (lt_trans ha hb)).le))
  refine le_antisymm hle (le_of_not_gt fun hgap => ?_)
  set sA := sSup {t | stdNormalCDF t < q}
  set iB := sInf {t | q < stdNormalCDF t}
  -- inside the (putative) gap the cdf is pinned to exactly q…
  have hmid : ∀ m, sA < m → m < iB → stdNormalCDF m = q := by
    intro m hm₁ hm₂
    have hnotA : ¬ stdNormalCDF m < q := fun h => absurd (le_csSup hAbdd h) (not_le.mpr hm₁)
    have hnotB : ¬ q < stdNormalCDF m := fun h => absurd (csInf_le hBbdd h) (not_le.mpr hm₂)
    exact le_antisymm (not_lt.mp hnotB) (not_lt.mp hnotA)
  -- …and a gap has room for two such points, killing strict monotonicity
  have h₁ : stdNormalCDF (sA + (iB - sA) / 3) = q :=
    hmid _ (by linarith) (by linarith)
  have h₂ : stdNormalCDF (sA + 2 * (iB - sA) / 3) = q :=
    hmid _ (by linarith) (by linarith)
  have := stdNormalCDF_strictMono
    (show sA + (iB - sA) / 3 < sA + 2 * (iB - sA) / 3 by linarith)
  rw [h₁, h₂] at this
  exact lt_irrefl q this

/-- **`hanti` discharged:** the real quantile is odd about ½, `Φ⁻¹(1−q) = −Φ⁻¹(q)` on
    `(0,1)`. Symmetry turns `{Φ < 1−q}` into the negation of `{Φ > q}`, `sSup ∘ neg`
    into `−sInf`, and the no-flat-step lemma closes the `sInf`/`sSup` mismatch. -/
lemma stdNormalQuantile_anti {q : ℝ} (hq : q ∈ Set.Ioo (0:ℝ) 1) :
    stdNormalQuantile (1 - q) = -stdNormalQuantile q := by
  have hset : {t | stdNormalCDF t < 1 - q} = -{t | q < stdNormalCDF t} := by
    ext t
    simp only [Set.mem_ofPred_eq, Set.mem_neg, stdNormalCDF_neg]
    constructor <;> intro h <;> linarith
  rw [stdNormalQuantile, hset, Real.sSup_neg, stdNormalQuantile,
    stdNormalCDF_sSup_lt_eq_sInf_gt hq]

-- ════════════════════════════════════════════════════════════════
-- § Quantile inversion: Φ(Φ⁻¹ p) = p on (0,1)
-- ════════════════════════════════════════════════════════════════

/-- **The quantile genuinely inverts Φ** on `(0,1)`: `Φ(Φ⁻¹ p) = p`. Right continuity of
    the Stieltjes cdf gives `≥` (a value below `p` at the sup would push the sup further
    right); no-atoms gives `≤` (the cdf equals its left limit, and everything left of the
    sup is `< p`). The lemma that makes `stdNormalQuantile` an inverse, not just a
    monotone-odd stand-in — G2's Neyman–Pearson bound enters through it. -/
lemma stdNormalCDF_quantile {p : ℝ} (hp : p ∈ Set.Ioo (0:ℝ) 1) :
    stdNormalCDF (stdNormalQuantile p) = p := by
  have : NullSingletonClass (gaussianReal 0 1) := nullSingletonClass_gaussianReal one_ne_zero
  have hAne : Set.Nonempty {t | stdNormalCDF t < p} := stdNormalCDF_exists_lt hp.1
  have hAbdd : BddAbove {t | stdNormalCDF t < p} := stdNormalCDF_sublevel_bddAbove hp.2
  set q := stdNormalQuantile p with hq
  -- (≥): right continuity — if Φ q < p, some u > q also has Φ u < p, beating the sSup
  have hge : p ≤ stdNormalCDF q := by
    by_contra hlt
    rw [not_le] at hlt
    have hrc : ContinuousWithinAt stdNormalCDF (Set.Ici q) q :=
      (cdf (gaussianReal 0 1)).right_continuous q
    have hev : ∀ᶠ u in 𝓝[>] q, stdNormalCDF u < p :=
      nhdsWithin_mono q Set.Ioi_subset_Ici_self
        (Filter.Tendsto.eventually_lt_const hlt hrc)
    obtain ⟨u, huq, hu⟩ := (hev.and eventually_mem_nhdsWithin).exists
    exact absurd (le_csSup hAbdd huq) (not_le.mpr hu)
  -- (≤): no atoms — Φ q equals its left limit, and everything left of q is < p
  have hle : stdNormalCDF q ≤ p := by
    have hsing : (gaussianReal 0 1) {q} = 0 := measure_singleton q
    rw [← measure_cdf (μ := gaussianReal 0 1), StieltjesFunction.measure_singleton] at hsing
    have h1 : stdNormalCDF q - Function.leftLim (cdf (gaussianReal 0 1)) q ≤ 0 :=
      ENNReal.ofReal_eq_zero.mp hsing
    have h2 : Function.leftLim (cdf (gaussianReal 0 1)) q ≤ stdNormalCDF q :=
      (cdf (gaussianReal 0 1)).mono.leftLim_le le_rfl
    have hll : Function.leftLim (cdf (gaussianReal 0 1)) q = stdNormalCDF q := by linarith
    rw [show stdNormalCDF q = Function.leftLim (cdf (gaussianReal 0 1)) q from hll.symm]
    refine le_of_tendsto ((cdf (gaussianReal 0 1)).mono.tendsto_leftLim q) ?_
    filter_upwards [self_mem_nhdsWithin] with u hu
    obtain ⟨a, ha, hua⟩ := exists_lt_of_lt_csSup hAne hu
    exact ((cdf (gaussianReal 0 1)).mono hua.le).trans (le_of_lt ha)
  linarith


/-- `Φ⁻¹(Φ s) = s` — the quantile inverts the cdf everywhere (strict monotonicity makes
    the strict sub-level set of `Φ s` exactly `Iio s`). -/
lemma stdNormalQuantile_cdf (s : ℝ) : stdNormalQuantile (stdNormalCDF s) = s := by
  have hset : {r | stdNormalCDF r < stdNormalCDF s} = Set.Iio s := by
    ext r
    simp only [Set.mem_ofPred_eq, Set.mem_Iio]
    exact ⟨fun h => stdNormalCDF_strictMono.lt_iff_lt.mp h,
      fun h => stdNormalCDF_strictMono h⟩
  rw [stdNormalQuantile, hset, csSup_Iio]

/-- `Φ` never reaches 0: there is Gaussian mass below every point. -/
lemma stdNormalCDF_pos (s : ℝ) : 0 < stdNormalCDF s := by
  have h := stdGaussian_Ioo_pos (show s - 1 < s by linarith)
  have hle : gaussianReal 0 1 (Set.Ioo (s - 1) s) ≤ gaussianReal 0 1 (Set.Iic s) :=
    measure_mono (fun x hx => le_of_lt hx.2)
  have : 0 < (gaussianReal 0 1).real (Set.Iic s) :=
    ENNReal.toReal_pos (lt_of_lt_of_le h hle).ne' (measure_ne_top _ _)
  rw [stdNormalCDF, cdf_eq_real]
  exact this

/-- `Φ` never reaches 1 (symmetry + `stdNormalCDF_pos`). -/
lemma stdNormalCDF_lt_one (s : ℝ) : stdNormalCDF s < 1 := by
  have h := stdNormalCDF_pos (-s)
  have hneg := stdNormalCDF_neg s
  linarith

/-- `Φ` maps into the open unit interval. -/
lemma stdNormalCDF_mem_Ioo (s : ℝ) : stdNormalCDF s ∈ Set.Ioo (0:ℝ) 1 :=
  ⟨stdNormalCDF_pos s, stdNormalCDF_lt_one s⟩

/-- `Φ⁻¹` is STRICTLY monotone on `(0,1)` (upgrade of `stdNormalQuantile_monotoneOn`):
    reflect strictness through `Φ` via the two-sided inverse `stdNormalCDF_quantile`. -/
lemma stdNormalQuantile_strictMonoOn :
    StrictMonoOn stdNormalQuantile (Set.Ioo (0:ℝ) 1) := by
  intro p hp q hq hpq
  have h := stdNormalCDF_strictMono.lt_iff_lt
    (a := stdNormalQuantile p) (b := stdNormalQuantile q)
  rw [stdNormalCDF_quantile hp, stdNormalCDF_quantile hq] at h
  exact h.mp hpq

/-- `Φ⁻¹` maps `(0,1)` ONTO ℝ: every real `s` is `Φ⁻¹(Φ s)`. -/
lemma stdNormalQuantile_surjOn :
    Set.SurjOn stdNormalQuantile (Set.Ioo (0:ℝ) 1) Set.univ := fun s _ =>
  ⟨stdNormalCDF s, stdNormalCDF_mem_Ioo s, stdNormalQuantile_cdf s⟩

/-- `Φ⁻¹` is continuous at every `p ∈ (0,1)`: strictly monotone on the open interval
    with image all of ℝ (a neighborhood of anything). -/
lemma stdNormalQuantile_continuousAt {p : ℝ} (hp : p ∈ Set.Ioo (0:ℝ) 1) :
    ContinuousAt stdNormalQuantile p := by
  apply stdNormalQuantile_strictMonoOn.continuousAt_of_image_mem_nhds
    (isOpen_Ioo.mem_nhds hp)
  have himg : stdNormalQuantile '' Set.Ioo (0:ℝ) 1 = Set.univ :=
    Set.eq_univ_of_univ_subset stdNormalQuantile_surjOn
  rw [himg]
  exact Filter.univ_mem

/-- `Φ⁻¹` is continuous on `(0,1)`. -/
lemma stdNormalQuantile_continuousOn :
    ContinuousOn stdNormalQuantile (Set.Ioo (0:ℝ) 1) := fun _ hp =>
  (stdNormalQuantile_continuousAt hp).continuousWithinAt

/-- Below `0` the quantile's defining set is empty (`Φ > 0` everywhere), so
    `Φ⁻¹` takes the junk value `sSup ∅ = 0` — and the radius `σ·Φ⁻¹(p̂−t)`
    certifies vacuously. -/
lemma stdNormalQuantile_of_nonpos {q : ℝ} (hq : q ≤ 0) : stdNormalQuantile q = 0 := by
  have hset : {s : ℝ | stdNormalCDF s < q} = ∅ := by
    ext s
    simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_lt]
    exact hq.trans (stdNormalCDF_pos s).le
  rw [stdNormalQuantile, hset, Real.sSup_empty]

end Proofs
