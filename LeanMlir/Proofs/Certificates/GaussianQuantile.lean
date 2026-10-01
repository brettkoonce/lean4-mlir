import Mathlib.Probability.Distributions.Gaussian.Multivariate
import LeanMlir.Proofs.Foundation.UpstreamDraft

/-! # The standard normal: `Φ`, its quantile `Φ⁻¹`, and full support

`stdNormalCDF` (Mathlib's `cdf` of `gaussianReal 0 1`) and `stdNormalQuantile` (`sSup {t | Φ t < p}`,
the honest inverse on `(0,1)`), with the facts every smoothing certificate uses: `Φ` is strictly
monotone and symmetric; on `(0,1)` the quantile is monotone, odd about `½`, continuous and inverts
`Φ` both ways; below `0` it takes the junk value `0`. The `IsOpenPosMeasure` instance says the
multivariate standard Gaussian charges every nonempty open set; the 1-D instance is
`MathlibUpstream.instIsOpenPosMeasureGaussianReal`. Mathlib has none of
this for the Gaussian quantile.
-/

namespace Proofs

open MeasureTheory ProbabilityTheory Filter
open scoped Topology

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

private theorem stdNormalCDF_eq_real (t : ℝ) : stdNormalCDF t = (gaussianReal 0 1).real (Set.Iic t) :=
  cdf_eq_real _ t

/-- The interval split `Φ b − Φ a = P(Ioc a b)`. -/
theorem stdNormalCDF_sub {a b : ℝ} (hab : a ≤ b) :
    stdNormalCDF b - stdNormalCDF a = (gaussianReal 0 1).real (Set.Ioc a b) := by
  rw [stdNormalCDF_eq_real, stdNormalCDF_eq_real, ← Set.Iic_union_Ioc_eq_Iic hab,
    measureReal_union (Set.Iic_disjoint_Ioc le_rfl) measurableSet_Ioc]
  ring

/-- `Φ` is strictly monotone (`MathlibUpstream.strictMono_cdf_gaussianReal`). -/
lemma stdNormalCDF_strictMono : StrictMono stdNormalCDF :=
  MathlibUpstream.strictMono_cdf_gaussianReal 0 one_ne_zero

/-- `Φ` is continuous (`MathlibUpstream.continuous_cdf_gaussianReal`). -/
theorem continuous_stdNormalCDF : Continuous stdNormalCDF :=
  MathlibUpstream.continuous_cdf_gaussianReal 0 one_ne_zero

/-- Gaussian symmetry `Φ(−t) = 1 − Φ(t)` (`MathlibUpstream.cdf_gaussianReal_neg`). -/
lemma stdNormalCDF_neg (t : ℝ) : stdNormalCDF (-t) = 1 - stdNormalCDF t :=
  MathlibUpstream.cdf_gaussianReal_neg one_ne_zero t

-- ════════════════════════════════════════════════════════════════
-- § Φ⁻¹ on (0,1): the defining sets behave, mono
-- ════════════════════════════════════════════════════════════════

/-- `Φ → 0` at `−∞`, so for `p > 0` some `t` has `Φ t < p` — the quantile's set is
    nonempty. -/
private lemma stdNormalCDF_exists_lt {p : ℝ} (hp : 0 < p) : ∃ t, stdNormalCDF t < p :=
  ((tendsto_cdf_atBot (μ := gaussianReal 0 1)).eventually_lt_const hp).exists

/-- `Φ → 1` at `+∞`, so for `p < 1` some `t` has `Φ t > p`. -/
private lemma stdNormalCDF_exists_gt {p : ℝ} (hp : p < 1) : ∃ t, p < stdNormalCDF t :=
  ((tendsto_cdf_atTop (μ := gaussianReal 0 1)).eventually_const_lt hp).exists

/-- For `p < 1` the sub-level set `{Φ < p}` is bounded above (anything past a point with
    `Φ > p` is excluded). -/
private lemma stdNormalCDF_sublevel_bddAbove {p : ℝ} (hp : p < 1) :
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

-- ════════════════════════════════════════════════════════════════
-- § Quantile inversion: Φ(Φ⁻¹ p) = p on (0,1), so Φ⁻¹ is odd about ½
-- ════════════════════════════════════════════════════════════════

/-- **The quantile genuinely inverts Φ** on `(0,1)`: `Φ(Φ⁻¹ p) = p`. Right continuity of
    the Stieltjes cdf gives `≥` (a value below `p` at the sup would push the sup further
    right); no-atoms gives `≤` (the cdf equals its left limit, and everything left of the
    sup is `< p`). The lemma that makes `stdNormalQuantile` an inverse, not just a
    monotone-odd stand-in; `smoothing_probit_lipschitz` applies it to feed the Neyman–Pearson
    bound its threshold. -/
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
    have hll : Function.leftLim (cdf (gaussianReal 0 1)) q = stdNormalCDF q :=
      MathlibUpstream.leftLim_cdf _ q
    rw [show stdNormalCDF q = Function.leftLim (cdf (gaussianReal 0 1)) q from hll.symm]
    refine le_of_tendsto ((cdf (gaussianReal 0 1)).mono.tendsto_leftLim q) ?_
    filter_upwards [self_mem_nhdsWithin] with u hu
    obtain ⟨a, ha, hua⟩ := exists_lt_of_lt_csSup hAne hu
    exact ((cdf (gaussianReal 0 1)).mono hua.le).trans (le_of_lt ha)
  linarith

/-- **`hanti` discharged:** the real quantile is odd about ½, `Φ⁻¹(1−q) = −Φ⁻¹(q)` on
    `(0,1)`. `Φ` is injective, and symmetry plus `stdNormalCDF_quantile` send both sides to
    `1 − q`. -/
lemma stdNormalQuantile_anti {q : ℝ} (hq : q ∈ Set.Ioo (0:ℝ) 1) :
    stdNormalQuantile (1 - q) = -stdNormalQuantile q := by
  apply stdNormalCDF_strictMono.injective
  rw [stdNormalCDF_neg, stdNormalCDF_quantile hq,
    stdNormalCDF_quantile ⟨by linarith [hq.2], by linarith [hq.1]⟩]


/-- `Φ⁻¹(Φ s) = s` — the quantile inverts the cdf everywhere (strict monotonicity makes
    the strict sub-level set of `Φ s` exactly `Iio s`). -/
lemma stdNormalQuantile_cdf (s : ℝ) : stdNormalQuantile (stdNormalCDF s) = s := by
  have hset : {r | stdNormalCDF r < stdNormalCDF s} = Set.Iio s := by
    ext r
    simp only [Set.mem_ofPred_eq, Set.mem_Iio]
    exact ⟨fun h => stdNormalCDF_strictMono.lt_iff_lt.mp h,
      fun h => stdNormalCDF_strictMono h⟩
  rw [stdNormalQuantile, hset, csSup_Iio]

/-- `Φ` never reaches 0 (`MathlibUpstream.cdf_gaussianReal_pos`). -/
private lemma stdNormalCDF_pos (s : ℝ) : 0 < stdNormalCDF s :=
  MathlibUpstream.cdf_gaussianReal_pos 0 one_ne_zero s

/-- `Φ` never reaches 1 (`MathlibUpstream.cdf_gaussianReal_lt_one`). -/
private lemma stdNormalCDF_lt_one (s : ℝ) : stdNormalCDF s < 1 :=
  MathlibUpstream.cdf_gaussianReal_lt_one 0 one_ne_zero s

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
