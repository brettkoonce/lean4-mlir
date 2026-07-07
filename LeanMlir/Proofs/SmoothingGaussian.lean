import LeanMlir.Proofs.LipschitzCert
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.CDF
import Mathlib.Analysis.InnerProductSpace.Projection.Reflection

/-! # The real Gaussian probit: Φ, Φ⁻¹, and the Cohen radius at the true quantile

G1 of `planning/smoothing_gaussian_lemma.md`: instantiate the randomized-smoothing certified
radius at the REAL standard-normal quantile, so the only smoothing-side hypothesis left is
the Neyman–Pearson Lipschitz core (`hg` — G2–G4 of the plan).

`stdNormalCDF` is Mathlib's `cdf (gaussianReal 0 1)` — the genuine `Φ`, no bespoke integral.
This file proves the three facts Mathlib doesn't have:

* **strict monotonicity** (`stdNormalCDF_strictMono`) — the Gaussian pdf is everywhere
  positive, so every interval carries positive mass;
* **symmetry** `Φ(−t) = 1 − Φ(t)` (`stdNormalCDF_neg`) — the standard Gaussian is invariant
  under negation (`gaussianReal_map_neg`);
* the quantile `stdNormalQuantile p = sSup {t | Φ t < p}` is **monotone on `(0,1)`**
  (`stdNormalQuantile_monotoneOn`) and **odd about ½** (`stdNormalQuantile_anti`,
  `Φ⁻¹(1−q) = −Φ⁻¹(q)`), via the no-flat-step lemma `stdNormalCDF_sSup_lt_eq_sInf_gt`
  (a flat step at level `q` would give two points with `Φ = q`, against strict mono).

G2 (the 1-D Neyman–Pearson core) also lives here: `stdNormalCDF_quantile` upgrades the
quantile to a genuine inverse (`Φ(Φ⁻¹ p) = p` on `(0,1)`, right-continuity + no-atoms), and
`gaussian_np_shift` is the Cohen bound — a `[0,1]` function with `N(0,1)`-mean ≥ `Φ(t)` keeps
mean ≥ `Φ(t−δ)` under a `δ ≥ 0` shift, by the monotone-likelihood-ratio pointwise inequality
`(f − 1_{z≤t})·(LR − LR(t)) ≥ 0` (no layer-cake, no rearrangement machinery).

Capstone: `smoothing_certified_radius_gaussian` — `smoothing_certified_radius_probit` with
`Phiinv := stdNormalQuantile`, its `hmono`/`hanti` DISCHARGED. The quantile is total on ℝ
(junk `sSup` outside `(0,1)`) but every use here is guarded by `hp : p c y ∈ Ioo 0 1` — the
realistic regime, since Monte-Carlo/Clopper–Pearson class-probability estimates are never
exactly 0 or 1. See `planning/smoothing_gaussian_lemma.md` for why the ORIGINAL abstract
theorem's global `Monotone Phiinv` can never be met by the true (unbounded) quantile.

All results are `propext / Classical.choice / Quot.sound`-clean (`tests/AuditAxioms.lean`). -/

namespace Proofs

open MeasureTheory ProbabilityTheory Filter
open scoped Topology

variable {k : ℕ} {E : Type*} [NormedAddCommGroup E]

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
    0 < gaussianReal 0 1 (Set.Ioo s t) := by
  rw [gaussianReal_apply 0 one_ne_zero, setLIntegral_pos_iff (measurable_gaussianPDF 0 1),
    support_gaussianPDF one_ne_zero, Set.univ_inter]
  simpa [Real.volume_Ioo] using sub_pos.mpr hst

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
  haveI : NoAtoms (gaussianReal 0 1) := noAtoms_gaussianReal one_ne_zero
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
    simp only [Set.mem_setOf_eq, Set.mem_neg, stdNormalCDF_neg]
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
  haveI : NoAtoms (gaussianReal 0 1) := noAtoms_gaussianReal one_ne_zero
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

-- ════════════════════════════════════════════════════════════════
-- § G2: the 1-D Neyman–Pearson core (σ = 1, shift δ ≥ 0)
-- ════════════════════════════════════════════════════════════════

/-! The analytic heart of Cohen 2019, in its 1-D normalized form. The classic proof needs
no layer-cake and no rearrangement machinery: with `h` the halfspace indicator at the
threshold `t` and `LR` the (monotone) Gaussian likelihood ratio, the pointwise inequality
`(f − h)·(LR − LR(t)) ≥ 0` — sign-checked on each side of `t` — integrates against the
base Gaussian into exactly the Neyman–Pearson optimality of the halfspace. -/

/-- The Gaussian likelihood ratio: `pdf_{N(δ,1)}(z) = exp(δz − δ²/2) · pdf_{N(0,1)}(z)` —
    monotone in `z` (for `δ ≥ 0`), which is all Neyman–Pearson needs. -/
lemma gaussianPDFReal_shift (δ z : ℝ) :
    gaussianPDFReal δ 1 z = Real.exp (δ * z - δ ^ 2 / 2) * gaussianPDFReal 0 1 z := by
  simp only [gaussianPDFReal, NNReal.coe_one]
  have harg : -(z - δ) ^ 2 / (2 * 1)
      = δ * z - δ ^ 2 / 2 + -(z - 0) ^ 2 / (2 * 1) := by ring
  rw [mul_left_comm, ← Real.exp_add, harg]

/-- The shifted Gaussian's cdf is a shifted `Φ`: `cdf_{N(δ,1)}(t) = Φ(t − δ)`. -/
lemma cdf_gaussianReal_shift (δ t : ℝ) :
    cdf (gaussianReal δ 1) t = stdNormalCDF (t - δ) := by
  have hmap : gaussianReal δ 1 = (gaussianReal 0 1).map (· + δ) := by
    rw [gaussianReal_map_add_const]; norm_num
  have hpre : (· + δ) ⁻¹' Set.Iic t = Set.Iic (t - δ) := by
    ext z; simp [le_sub_iff_add_le]
  rw [stdNormalCDF, cdf_eq_real, cdf_eq_real, hmap, Measure.real,
    Measure.map_apply (measurable_add_const δ) measurableSet_Iic, hpre, ← Measure.real]

/-- The halfspace indicator's Gaussian mass is the cdf at the threshold. -/
lemma integral_indicator_Iic_gaussianReal (μ t : ℝ) :
    ∫ z, (Set.Iic t).indicator (1 : ℝ → ℝ) z ∂(gaussianReal μ 1)
      = cdf (gaussianReal μ 1) t := by
  rw [integral_indicator_one measurableSet_Iic, cdf_eq_real]

/-- **G2 — the 1-D Neyman–Pearson core (Cohen 2019, Lemma 3 specialization).** For
    measurable `f : ℝ → [0,1]` with standard-Gaussian mean at least `Φ(t)`, the mean under
    the `δ`-shifted Gaussian (`δ ≥ 0`) is at least `Φ(t − δ)`: among all `[0,1]` functions
    of given Gaussian mass, the halfspace indicator `1_{z ≤ t}` loses the most mass under a
    shift — the likelihood ratio `exp(δz − δ²/2)` is monotone, so
    `(f − 1_{z ≤ t})·(LR − LR(t)) ≥ 0` pointwise, and integrating it against `N(0,1)`
    forces `∫f dN(δ,1) ≥ ∫1_{z ≤ t} dN(δ,1) = Φ(t − δ)`. -/
theorem gaussian_np_shift {f : ℝ → ℝ} (hfm : Measurable f)
    (hf0 : ∀ z, 0 ≤ f z) (hf1 : ∀ z, f z ≤ 1)
    {δ t : ℝ} (hδ : 0 ≤ δ)
    (hp : stdNormalCDF t ≤ ∫ z, f z ∂(gaussianReal 0 1)) :
    stdNormalCDF (t - δ) ≤ ∫ z, f z ∂(gaussianReal δ 1) := by
  classical
  set h : ℝ → ℝ := (Set.Iic t).indicator (1 : ℝ → ℝ) with hh
  have hhm : Measurable h := measurable_const.indicator measurableSet_Iic
  have hh0 : ∀ z, 0 ≤ h z := fun z =>
    Set.indicator_nonneg (fun _ _ => zero_le_one) z
  have hh1 : ∀ z, h z ≤ 1 := by
    intro z
    by_cases hz : z ∈ Set.Iic t <;> simp [hh, hz]
  -- bounded-by-1 measurable functions integrate against any Gaussian pdf
  have hbdd : ∀ (μ' : ℝ) (g : ℝ → ℝ), Measurable g → (∀ z, |g z| ≤ 1) →
      Integrable (fun z => gaussianPDFReal μ' 1 z * g z) volume := by
    intro μ' g hgm hg1
    refine (integrable_gaussianPDFReal μ' 1).mono
      ((measurable_gaussianPDFReal μ' 1).mul hgm).aestronglyMeasurable
      (ae_of_all _ fun z => ?_)
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_mul,
      abs_of_nonneg (gaussianPDFReal_nonneg μ' 1 z)]
    calc gaussianPDFReal μ' 1 z * |g z| ≤ gaussianPDFReal μ' 1 z * 1 :=
          mul_le_mul_of_nonneg_left (hg1 z) (gaussianPDFReal_nonneg μ' 1 z)
      _ = gaussianPDFReal μ' 1 z := mul_one _
  have habs_f : ∀ z, |f z| ≤ 1 := fun z => abs_le.mpr ⟨by linarith [hf0 z], hf1 z⟩
  have habs_h : ∀ z, |h z| ≤ 1 := fun z => abs_le.mpr ⟨by linarith [hh0 z], hh1 z⟩
  have habs_fh : ∀ z, |f z - h z| ≤ 1 := fun z =>
    abs_le.mpr ⟨by linarith [hf0 z, hh1 z], by linarith [hf1 z, hh0 z]⟩
  -- the likelihood ratio at the threshold
  set c : ℝ := Real.exp (δ * t - δ ^ 2 / 2) with hc
  have hcpos : 0 < c := Real.exp_pos _
  -- pointwise: (f − h)·pdf_δ ≥ c·(f − h)·pdf₀ — the NP rearrangement, case split at t
  have hpoint : ∀ z, c * (gaussianPDFReal 0 1 z * (f z - h z))
      ≤ gaussianPDFReal δ 1 z * (f z - h z) := by
    intro z
    rw [gaussianPDFReal_shift δ z]
    rcases le_or_gt z t with hz | hz
    · -- z ≤ t: f − h ≤ 0 and LR ≤ c
      have hhz : h z = 1 := by simp [hh, hz]
      have hfh : f z - h z ≤ 0 := by rw [hhz]; linarith [hf1 z]
      have hlr : Real.exp (δ * z - δ ^ 2 / 2) ≤ c := by
        apply Real.exp_le_exp.mpr
        have : δ * z ≤ δ * t := mul_le_mul_of_nonneg_left hz hδ
        linarith
      nlinarith [mul_nonneg (mul_nonneg (sub_nonneg.mpr hlr) (neg_nonneg.mpr hfh))
        (gaussianPDFReal_nonneg 0 1 z)]
    · -- z > t: f − h ≥ 0 and LR ≥ c
      have hhz : h z = 0 := by
        simp [hh, not_le.mpr hz]
      have hfh : 0 ≤ f z - h z := by rw [hhz]; linarith [hf0 z]
      have hlr : c ≤ Real.exp (δ * z - δ ^ 2 / 2) := by
        apply Real.exp_le_exp.mpr
        have : δ * t ≤ δ * z := mul_le_mul_of_nonneg_left hz.le hδ
        linarith
      nlinarith [mul_nonneg (mul_nonneg (sub_nonneg.mpr hlr) hfh)
        (gaussianPDFReal_nonneg 0 1 z)]
  -- integrate the pointwise bound
  have hint0 : Integrable (fun z => gaussianPDFReal 0 1 z * (f z - h z)) volume :=
    hbdd 0 _ (hfm.sub hhm) habs_fh
  have hintδ : Integrable (fun z => gaussianPDFReal δ 1 z * (f z - h z)) volume :=
    hbdd δ _ (hfm.sub hhm) habs_fh
  have hkey : c * ∫ z, gaussianPDFReal 0 1 z * (f z - h z)
      ≤ ∫ z, gaussianPDFReal δ 1 z * (f z - h z) := by
    rw [← integral_const_mul]
    exact integral_mono (hint0.const_mul c) hintδ hpoint
  -- split the products back into μ-integral differences
  have hsplit : ∀ μ' : ℝ, (∫ z, gaussianPDFReal μ' 1 z * (f z - h z))
      = (∫ z, f z ∂(gaussianReal μ' 1)) - ∫ z, h z ∂(gaussianReal μ' 1) := by
    intro μ'
    rw [integral_gaussianReal_eq_integral_smul one_ne_zero,
      integral_gaussianReal_eq_integral_smul one_ne_zero, ← integral_sub
        (by simpa [smul_eq_mul] using hbdd μ' f hfm habs_f)
        (by simpa [smul_eq_mul] using hbdd μ' h hhm habs_h)]
    congr 1
    funext z
    simp [smul_eq_mul, mul_sub]
  rw [hsplit 0, hsplit δ] at hkey
  -- endpoints: ∫ h = Φ at the (shifted) threshold
  have hend0 : ∫ z, h z ∂(gaussianReal 0 1) = stdNormalCDF t :=
    integral_indicator_Iic_gaussianReal 0 t
  have hendδ : ∫ z, h z ∂(gaussianReal δ 1) = stdNormalCDF (t - δ) := by
    rw [integral_indicator_Iic_gaussianReal δ t, cdf_gaussianReal_shift]
  rw [hend0] at hkey
  rw [← hendδ]
  nlinarith [hkey, mul_nonneg hcpos.le (sub_nonneg.mpr hp)]

-- ════════════════════════════════════════════════════════════════
-- § Capstone: the Cohen radius at the REAL Gaussian quantile
-- ════════════════════════════════════════════════════════════════

/-- **Randomized-smoothing certified radius at the true Gaussian probit.** With class
    probabilities honestly inside `(0,1)` (`hp`), per-class probit scores
    `Φ⁻¹ ∘ p c` each `(1/σ)`-Lipschitz (`hg` — the Neyman–Pearson core, the ONE remaining
    smoothing-side hypothesis, G2–G4 of `planning/smoothing_gaussian_lemma.md`), and the
    runner-up bound, every `‖δ‖₂ < σ·Φ⁻¹(p_A(x))` keeps class `i` the strict argmax —
    where `Φ⁻¹` is now the genuine standard-normal quantile, not an abstract stand-in.
    Exactly the `σ·Φ⁻¹(p_A)` radius the `*-smooth` drivers report. -/
theorem smoothing_certified_radius_gaussian {σ : ℝ} (hσ : 0 < σ)
    {p : Fin k → E → ℝ}
    (hp : ∀ c y, p c y ∈ Set.Ioo (0:ℝ) 1)
    (hg : ∀ c, LipschitzL2 (1 / σ) (fun x => stdNormalQuantile (p c x)))
    {x δ : E} {i : Fin k}
    (hrunner : ∀ j, j ≠ i → p j x ≤ 1 - p i x)
    (hδ : ‖δ‖ < σ * stdNormalQuantile (p i x)) :
    ∀ j, j ≠ i → p j (x + δ) < p i (x + δ) :=
  smoothing_certified_radius_probit hσ stdNormalQuantile_monotoneOn
    (fun _ hq => stdNormalQuantile_anti hq) hp hg hrunner hδ

-- ════════════════════════════════════════════════════════════════
-- § G3: dimension reduction — the n-D Neyman–Pearson bound
-- ════════════════════════════════════════════════════════════════

/-! The n-D Cohen bound `∫f(·+δ) dγ ≥ Φ(Φ⁻¹(∫f dγ) − ‖δ‖)` for the standard Gaussian `γ`
on `EuclideanSpace ℝ (Fin (n+1))`. Structure: (i) a 1-D Cameron–Martin change of variables
turns the shifted integral into a monotone-likelihood-ratio-weighted one; (ii) Fubini over
`Measure.pi` (split at coordinate 0 via `piFinSuccAbove`) lifts it to the iid pi measure —
only coordinate 0 carries the shift; (iii) the same pointwise MLR trick as G2, now with the
weight `exp(d·z₀ − d²/2)` and the halfspace `{z₀ ≤ t}`, gives the pi-space NP theorem;
(iv) an adapted orthonormal basis (a reflection carries `e₀` to `δ/‖δ‖`, and Mathlib's
`stdGaussian_eq_map_pi_orthonormalBasis` says the standard Gaussian doesn't care) rotates
the general shift onto coordinate 0. The plan's riskiest item — pi-Gaussian rotational
invariance — turned out to ship with Mathlib. -/

/-- The iid standard-Gaussian product measure on `Fin (n+1) → ℝ`. -/
noncomputable abbrev stdGaussianPi (n : ℕ) : Measure (Fin (n + 1) → ℝ) :=
  Measure.pi fun _ => gaussianReal 0 1

-- ── the 1-D Cameron–Martin shift identity ──

lemma integral_gaussianReal_shift_eq {g : ℝ → ℝ} (hgm : Measurable g) (d : ℝ) :
    ∫ s, g (s + d) ∂(gaussianReal 0 1)
      = ∫ s, Real.exp (d * s - d ^ 2 / 2) * g s ∂(gaussianReal 0 1) := by
  have hmap : gaussianReal d 1 = (gaussianReal 0 1).map (· + d) := by
    rw [gaussianReal_map_add_const]; norm_num
  have h1 : ∫ s, g (s + d) ∂(gaussianReal 0 1) = ∫ s, g s ∂(gaussianReal d 1) := by
    rw [hmap, integral_map (measurable_add_const d).aemeasurable hgm.aestronglyMeasurable]
  rw [h1, integral_gaussianReal_eq_integral_smul one_ne_zero,
    integral_gaussianReal_eq_integral_smul one_ne_zero]
  congr 1
  funext z
  rw [smul_eq_mul, smul_eq_mul, gaussianPDFReal_shift]
  ring

/-- The exponential weight is Gaussian-integrable. -/
lemma integrable_expWeight (d : ℝ) :
    Integrable (fun s => Real.exp (d * s - d ^ 2 / 2)) (gaussianReal 0 1) := by
  have heq : (fun s => Real.exp (d * s - d ^ 2 / 2))
      = fun s => Real.exp (-(d ^ 2) / 2) * Real.exp (d * s) := by
    funext s
    rw [← Real.exp_add]
    congr 1
    ring
  rw [heq]
  exact (integrable_exp_mul_gaussianReal d).const_mul _

-- ── coordinate-0 pushforward of the pi measure ──

lemma pi_gaussian_integral_eval {n : ℕ} {g : ℝ → ℝ} (hg : Measurable g) :
    ∫ z, g (z 0) ∂(stdGaussianPi n) = ∫ s, g s ∂(gaussianReal 0 1) := by
  have h := (measurePreserving_eval (fun _ : Fin (n + 1) => gaussianReal 0 1) 0).map_eq
  conv_rhs => rw [← h]
  rw [integral_map (measurable_pi_apply 0).aemeasurable hg.aestronglyMeasurable]

-- ── the pi-space shift: only coordinate 0 moves ──

lemma insertNth_zero_add_single {n : ℕ} (s d : ℝ) (w : Fin n → ℝ) :
    (0 : Fin (n + 1)).insertNth s w + d • Pi.single 0 1
      = (0 : Fin (n + 1)).insertNth (s + d) w := by
  rw [Fin.insertNth_zero', Fin.insertNth_zero']
  funext j
  refine Fin.cases ?_ (fun i => ?_) j
  · simp
  · simp

lemma pi_gaussian_shift_eq {n : ℕ} {F : (Fin (n + 1) → ℝ) → ℝ} (hFm : Measurable F)
    (hFb : ∀ z, |F z| ≤ 1) (d : ℝ) :
    ∫ z, F (z + d • Pi.single 0 1) ∂(stdGaussianPi n)
      = ∫ z, Real.exp (d * z 0 - d ^ 2 / 2) * F z ∂(stdGaussianPi n) := by
  classical
  set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) 0 with he
  have hpres := measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => gaussianReal 0 1) 0
  -- transfer any measurable integrand to the product side
  have htrans : ∀ G : (Fin (n + 1) → ℝ) → ℝ, Measurable G →
      ∫ z, G z ∂(stdGaussianPi n) = ∫ y, G (e.symm y) ∂((stdGaussianPi n).map e) := by
    intro G hG
    have hGm : Measurable fun y : ℝ × (Fin n → ℝ) => G (e.symm y) :=
      hG.comp e.symm.measurable
    rw [integral_map e.measurable.aemeasurable hGm.aestronglyMeasurable]
    simp only [MeasurableEquiv.symm_apply_apply]
  have hmF : Measurable fun z : Fin (n + 1) → ℝ => F (z + d • Pi.single 0 1) :=
    hFm.comp (measurable_id.add_const _)
  have hz0 : Measurable fun z : Fin (n + 1) → ℝ => z 0 := measurable_pi_apply 0
  have hmW : Measurable fun z : Fin (n + 1) → ℝ => Real.exp (d * z 0 - d ^ 2 / 2) * F z :=
    (Real.measurable_exp.comp ((hz0.const_mul d).sub measurable_const)).mul hFm
  -- e.symm is insertNth at 0
  have hsymm : ∀ y : ℝ × (Fin n → ℝ), e.symm y = (0 : Fin (n + 1)).insertNth y.1 y.2 := by
    intro y; rfl
  -- integrability on the product side
  have hprodF : Integrable (fun y : ℝ × (Fin n → ℝ) =>
      F (e.symm (y.1 + d, y.2))) (((gaussianReal 0 1)).prod (Measure.pi fun _ : Fin n => gaussianReal 0 1)) := by
    refine (integrable_const 1).mono'
      ((hFm.comp (e.symm.measurable.comp ((measurable_fst.add_const d).prodMk measurable_snd))).aestronglyMeasurable)
      (ae_of_all _ fun y => ?_)
    simpa using hFb _
  have hprodW : Integrable (fun y : ℝ × (Fin n → ℝ) =>
      Real.exp (d * y.1 - d ^ 2 / 2) * F (e.symm y))
      (((gaussianReal 0 1)).prod (Measure.pi fun _ : Fin n => gaussianReal 0 1)) := by
    have hWfst : Integrable (fun y : ℝ × (Fin n → ℝ) => Real.exp (d * y.1 - d ^ 2 / 2))
        (((gaussianReal 0 1)).prod (Measure.pi fun _ : Fin n => gaussianReal 0 1)) := by
      exact (integrable_expWeight d).comp_fst _
    refine hWfst.mono'
      (((Real.measurable_exp.comp (measurable_fst.const_mul d |>.sub measurable_const)).mul
        (hFm.comp e.symm.measurable)).aestronglyMeasurable)
      (ae_of_all _ fun y => ?_)
    rw [Real.norm_eq_abs, abs_mul, Real.abs_exp]
    calc Real.exp (d * y.1 - d ^ 2 / 2) * |F (e.symm y)|
        ≤ Real.exp (d * y.1 - d ^ 2 / 2) * 1 :=
          mul_le_mul_of_nonneg_left (hFb _) (Real.exp_pos _).le
      _ = Real.exp (d * y.1 - d ^ 2 / 2) := mul_one _
  -- the chain
  rw [htrans _ hmF, htrans _ hmW, hpres.map_eq]
  have hstep1 : ∀ y : ℝ × (Fin n → ℝ),
      F (e.symm y + d • Pi.single 0 1) = F (e.symm (y.1 + d, y.2)) := by
    intro y
    rw [hsymm, hsymm, insertNth_zero_add_single]
  have hcoord : ∀ y : ℝ × (Fin n → ℝ), e.symm y 0 = y.1 := by
    intro y
    rw [hsymm]
    simp
  simp only [hstep1, hcoord]
  rw [integral_prod_symm _ hprodF, integral_prod_symm _ hprodW]
  congr 1
  funext w
  have hslice : Measurable fun s => F (e.symm (s, w)) :=
    hFm.comp (e.symm.measurable.comp (measurable_id.prodMk measurable_const))
  have := integral_gaussianReal_shift_eq (g := fun s => F (e.symm (s, w))) hslice d
  simpa using this

-- ── the pi-space Neyman–Pearson theorem ──

theorem pi_gaussian_np_shift {n : ℕ} {F : (Fin (n + 1) → ℝ) → ℝ} (hFm : Measurable F)
    (hF0 : ∀ z, 0 ≤ F z) (hF1 : ∀ z, F z ≤ 1) {d t : ℝ} (hd : 0 ≤ d)
    (hp : stdNormalCDF t ≤ ∫ z, F z ∂(stdGaussianPi n)) :
    stdNormalCDF (t - d) ≤ ∫ z, F (z + d • Pi.single 0 1) ∂(stdGaussianPi n) := by
  classical
  have hFb : ∀ z, |F z| ≤ 1 := fun z => abs_le.mpr ⟨by linarith [hF0 z], hF1 z⟩
  rw [pi_gaussian_shift_eq hFm hFb d]
  set h : (Fin (n + 1) → ℝ) → ℝ := fun z => (Set.Iic t).indicator (1 : ℝ → ℝ) (z 0) with hh
  have hh1d : Measurable ((Set.Iic t).indicator (1 : ℝ → ℝ)) :=
    measurable_const.indicator measurableSet_Iic
  have hz0 : Measurable fun z : Fin (n + 1) → ℝ => z 0 := measurable_pi_apply 0
  have hhm : Measurable h := hh1d.comp hz0
  have hh0 : ∀ z, 0 ≤ h z := fun z => Set.indicator_nonneg (fun _ _ => zero_le_one) _
  have hh1 : ∀ z, h z ≤ 1 := by
    intro z
    by_cases hz : z 0 ∈ Set.Iic t <;> simp [hh, hz]
  have habs_fh : ∀ z, |F z - h z| ≤ 1 := fun z =>
    abs_le.mpr ⟨by linarith [hF0 z, hh1 z], by linarith [hF1 z, hh0 z]⟩
  set c : ℝ := Real.exp (d * t - d ^ 2 / 2) with hc
  have hcpos : 0 < c := Real.exp_pos _
  -- pointwise MLR inequality (no pdf factor: the measure carries it)
  have hpoint : ∀ z, c * (F z - h z) ≤ Real.exp (d * z 0 - d ^ 2 / 2) * (F z - h z) := by
    intro z
    rcases le_or_gt (z 0) t with hz | hz
    · have hhz : h z = 1 := by simp [hh, hz]
      have hfh : F z - h z ≤ 0 := by rw [hhz]; linarith [hF1 z]
      have hlr : Real.exp (d * z 0 - d ^ 2 / 2) ≤ c := by
        apply Real.exp_le_exp.mpr
        have := mul_le_mul_of_nonneg_left hz hd
        linarith
      nlinarith [mul_nonneg (sub_nonneg.mpr hlr) (neg_nonneg.mpr hfh)]
    · have hhz : h z = 0 := by simp [hh, not_le.mpr hz]
      have hfh : 0 ≤ F z - h z := by rw [hhz]; linarith [hF0 z]
      have hlr : c ≤ Real.exp (d * z 0 - d ^ 2 / 2) := by
        apply Real.exp_le_exp.mpr
        have := mul_le_mul_of_nonneg_left hz.le hd
        linarith
      nlinarith [mul_nonneg (sub_nonneg.mpr hlr) hfh]
  -- integrability
  have hWm : Measurable fun s : ℝ => Real.exp (d * s - d ^ 2 / 2) :=
    Real.measurable_exp.comp ((measurable_id.const_mul d).sub measurable_const)
  have hWΓ : Integrable (fun z : Fin (n + 1) → ℝ => Real.exp (d * z 0 - d ^ 2 / 2))
      (stdGaussianPi n) := by
    have h₁ := integrable_expWeight d
    rw [← (measurePreserving_eval (fun _ : Fin (n + 1) => gaussianReal 0 1) 0).map_eq] at h₁
    exact (integrable_map_measure hWm.aestronglyMeasurable
      (measurable_pi_apply 0).aemeasurable).mp h₁
  have hFhΓ : Integrable (fun z => F z - h z) (stdGaussianPi n) :=
    (integrable_const 1).mono' (hFm.sub hhm).aestronglyMeasurable
      (ae_of_all _ fun z => by rw [Real.norm_eq_abs]; exact habs_fh z)
  have hintR : Integrable (fun z => Real.exp (d * z 0 - d ^ 2 / 2) * (F z - h z))
      (stdGaussianPi n) := by
    refine hWΓ.mono' (((hWm.comp hz0).mul (hFm.sub hhm)).aestronglyMeasurable)
      (ae_of_all _ fun z => ?_)
    rw [Real.norm_eq_abs, abs_mul, Real.abs_exp]
    calc Real.exp (d * z 0 - d ^ 2 / 2) * |F z - h z|
        ≤ Real.exp (d * z 0 - d ^ 2 / 2) * 1 :=
          mul_le_mul_of_nonneg_left (habs_fh z) (Real.exp_pos _).le
      _ = Real.exp (d * z 0 - d ^ 2 / 2) := mul_one _
  have hkey : c * ∫ z, (F z - h z) ∂(stdGaussianPi n)
      ≤ ∫ z, Real.exp (d * z 0 - d ^ 2 / 2) * (F z - h z) ∂(stdGaussianPi n) := by
    rw [← integral_const_mul]
    exact integral_mono (hFhΓ.const_mul c) hintR hpoint
  -- endpoint integrability + splitting
  have hFint : Integrable F (stdGaussianPi n) :=
    (integrable_const 1).mono' hFm.aestronglyMeasurable
      (ae_of_all _ fun z => by rw [Real.norm_eq_abs]; exact hFb z)
  have hhint : Integrable h (stdGaussianPi n) :=
    (integrable_const 1).mono' hhm.aestronglyMeasurable
      (ae_of_all _ fun z => by
        rw [Real.norm_eq_abs]; exact abs_le.mpr ⟨by linarith [hh0 z], hh1 z⟩)
  have hWFint : Integrable (fun z => Real.exp (d * z 0 - d ^ 2 / 2) * F z)
      (stdGaussianPi n) := by
    refine hWΓ.mono' (((hWm.comp hz0).mul hFm).aestronglyMeasurable)
      (ae_of_all _ fun z => ?_)
    rw [Real.norm_eq_abs, abs_mul, Real.abs_exp]
    calc Real.exp (d * z 0 - d ^ 2 / 2) * |F z|
        ≤ Real.exp (d * z 0 - d ^ 2 / 2) * 1 :=
          mul_le_mul_of_nonneg_left (hFb z) (Real.exp_pos _).le
      _ = Real.exp (d * z 0 - d ^ 2 / 2) := mul_one _
  have hWhint : Integrable (fun z => Real.exp (d * z 0 - d ^ 2 / 2) * h z)
      (stdGaussianPi n) := by
    refine hWΓ.mono' (((hWm.comp hz0).mul hhm).aestronglyMeasurable)
      (ae_of_all _ fun z => ?_)
    rw [Real.norm_eq_abs, abs_mul, Real.abs_exp]
    calc Real.exp (d * z 0 - d ^ 2 / 2) * |h z|
        ≤ Real.exp (d * z 0 - d ^ 2 / 2) * 1 :=
          mul_le_mul_of_nonneg_left
            (abs_le.mpr ⟨by linarith [hh0 z], hh1 z⟩) (Real.exp_pos _).le
      _ = Real.exp (d * z 0 - d ^ 2 / 2) := mul_one _
  rw [integral_sub hFint hhint] at hkey
  have hsplitR : ∫ z, Real.exp (d * z 0 - d ^ 2 / 2) * (F z - h z) ∂(stdGaussianPi n)
      = (∫ z, Real.exp (d * z 0 - d ^ 2 / 2) * F z ∂(stdGaussianPi n))
        - ∫ z, Real.exp (d * z 0 - d ^ 2 / 2) * h z ∂(stdGaussianPi n) := by
    rw [← integral_sub hWFint hWhint]
    congr 1
    funext z
    ring
  rw [hsplitR] at hkey
  -- endpoints: ∫ h = Φ t and ∫ w·h = Φ (t − d)
  have hhval : ∫ z, h z ∂(stdGaussianPi n) = stdNormalCDF t := by
    simp only [hh]
    rw [pi_gaussian_integral_eval hh1d, integral_indicator_Iic_gaussianReal]
    rfl
  have hWhval : ∫ z, Real.exp (d * z 0 - d ^ 2 / 2) * h z ∂(stdGaussianPi n)
      = stdNormalCDF (t - d) := by
    simp only [hh]
    rw [pi_gaussian_integral_eval
      (g := fun s => Real.exp (d * s - d ^ 2 / 2) * (Set.Iic t).indicator (1 : ℝ → ℝ) s)
      (hWm.mul hh1d), ← integral_gaussianReal_shift_eq hh1d d]
    have hind : ∀ s : ℝ, (Set.Iic t).indicator (1 : ℝ → ℝ) (s + d)
        = (Set.Iic (t - d)).indicator (1 : ℝ → ℝ) s := by
      intro s
      have hiff : s + d ≤ t ↔ s ≤ t - d := by constructor <;> intro <;> linarith
      by_cases hs : s ≤ t - d
      · rw [Set.indicator_of_mem (Set.mem_Iic.mpr (hiff.mpr hs)),
          Set.indicator_of_mem (Set.mem_Iic.mpr hs)]
        rfl
      · rw [Set.indicator_of_notMem (fun hmem => hs (hiff.mp (Set.mem_Iic.mp hmem))),
          Set.indicator_of_notMem (fun hmem => hs (Set.mem_Iic.mp hmem))]
    simp only [hind]
    rw [integral_indicator_Iic_gaussianReal]
    rfl
  rw [hhval] at hkey
  rw [hWhval] at hkey
  linarith [hkey, mul_nonneg hcpos.le (sub_nonneg.mpr hp)]

-- ── G3: transfer to the standard Gaussian on Euclidean space ──

theorem stdGaussian_np_shift {n : ℕ} {f : EuclideanSpace ℝ (Fin (n + 1)) → ℝ}
    (hfm : Measurable f) (hf0 : ∀ z, 0 ≤ f z) (hf1 : ∀ z, f z ≤ 1)
    {δ : EuclideanSpace ℝ (Fin (n + 1))} {t : ℝ}
    (hp : stdNormalCDF t ≤ ∫ z, f z ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))) :
    stdNormalCDF (t - ‖δ‖)
      ≤ ∫ z, f (z + δ) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) := by
  rcases eq_or_ne δ 0 with rfl | hδ0
  · simpa using hp
  -- adapted orthonormal basis: b 0 = the unit direction of δ
  set u : EuclideanSpace ℝ (Fin (n + 1)) := ‖δ‖⁻¹ • δ with hu
  have hδnorm : ‖δ‖ ≠ 0 := norm_ne_zero_iff.mpr hδ0
  have hunorm : ‖u‖ = 1 := by
    rw [hu, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hδnorm]
  set b0 := EuclideanSpace.basisFun (Fin (n + 1)) ℝ with hb0def
  have hb0norm : ‖b0 0‖ = ‖u‖ := by rw [hunorm, b0.orthonormal.1 0]
  have hρ : Submodule.reflection (ℝ ∙ (b0 0 - u))ᗮ (b0 0) = u :=
    Submodule.reflection_sub hb0norm
  set b := b0.map (Submodule.reflection (ℝ ∙ (b0 0 - u))ᗮ) with hb
  have hb0' : b 0 = u := by rw [hb, OrthonormalBasis.map_apply, hρ]
  -- transfer along the basis expansion
  have hmapeq := stdGaussian_eq_map_pi_orthonormalBasis b
  have hsum_meas : Measurable fun x : Fin (n + 1) → ℝ => ∑ i, x i • b i := by
    refine Finset.measurable_sum _ fun i _ => ?_
    exact (measurable_pi_apply i).smul_const (b i)
  have htransfer : ∀ g : EuclideanSpace ℝ (Fin (n + 1)) → ℝ, Measurable g →
      ∫ z, g z ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))
        = ∫ x, g (∑ i, x i • b i) ∂(stdGaussianPi n) := by
    intro g hg
    rw [hmapeq, integral_map hsum_meas.aemeasurable hg.aestronglyMeasurable]
  -- the δ-shift is the coordinate-0 shift in the adapted basis
  have hshift : ∀ x : Fin (n + 1) → ℝ,
      (∑ i, (x + ‖δ‖ • (Pi.single 0 1 : Fin (n + 1) → ℝ)) i • b i)
        = (∑ i, x i • b i) + δ := by
    intro x
    have hterm : ∀ i, (x + ‖δ‖ • (Pi.single 0 1 : Fin (n + 1) → ℝ)) i • b i
        = x i • b i + (‖δ‖ * (Pi.single 0 1 : Fin (n + 1) → ℝ) i) • b i := by
      intro i
      rw [Pi.add_apply, Pi.smul_apply, smul_eq_mul, add_smul]
    simp only [hterm, Finset.sum_add_distrib]
    congr 1
    rw [Finset.sum_eq_single 0]
    · rw [Pi.single_eq_same, mul_one, hb0', hu, smul_smul,
        mul_inv_cancel₀ hδnorm, one_smul]
    · intro i _ hi
      rw [Pi.single_eq_of_ne hi, mul_zero, zero_smul]
    · intro habs
      exact absurd (Finset.mem_univ 0) habs
  -- apply the pi-space NP theorem to the pulled-back function
  have hnp := pi_gaussian_np_shift (F := fun x => f (∑ i, x i • b i))
    (hfm.comp hsum_meas) (fun x => hf0 _) (fun x => hf1 _) (norm_nonneg δ)
    (by rw [← htransfer f hfm]; exact hp)
  calc stdNormalCDF (t - ‖δ‖)
      ≤ ∫ x, f (∑ i, (x + ‖δ‖ • (Pi.single 0 1 : Fin (n + 1) → ℝ)) i • b i)
          ∂(stdGaussianPi n) := hnp
    _ = ∫ z, f (z + δ) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) := by
        rw [htransfer (fun z => f (z + δ)) (hfm.comp (measurable_id.add_const δ))]
        congr 1
        funext x
        rw [hshift]

end Proofs
