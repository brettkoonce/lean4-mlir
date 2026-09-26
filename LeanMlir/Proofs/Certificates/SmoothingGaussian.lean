import LeanMlir.Proofs.Certificates.GaussianQuantile
import LeanMlir.Proofs.Certificates.LipschitzCert
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.CDF
import Mathlib.Analysis.InnerProductSpace.Projection.Reflection

/-! # The real Gaussian probit: Φ, Φ⁻¹, and the Cohen radius as a THEOREM

Endpoint: `smoothing_certified_radius_classifier` — for a measurable classifier under
`N(0,σ²I)` smoothing, every `‖δ‖ < σ·Φ⁻¹(p_A(x))` provably cannot flip the smoothed argmax, with
`Φ⁻¹` the genuine standard-normal quantile. The Cohen–Rosenfeld–Kolter `(1/σ)`-Lipschitz probit
(`hg` of `smoothing_certified_radius_probit`, the Neyman–Pearson half) is a theorem here,
`smoothing_probit_lipschitz`. The one remaining hypothesis on the smoothed classifier is `hp`:
every class's smoothed probability lies in `(0,1)` at every point. `SmoothingNetSemantics.lean`
discharges it for argmax nets.

`Φ` (`stdNormalCDF`, Mathlib's `cdf (gaussianReal 0 1)`), `Φ⁻¹` (`stdNormalQuantile`) and their
monotonicity, symmetry, two-sided inversion and continuity live in `GaussianQuantile.lean`. This
file proves:

* the Neyman–Pearson bounds: `pi_gaussian_np_shift` — a `[0,1]` function with mean ≥ `Φ(t)`
  under the iid product `stdGaussianPi n` keeps mean ≥ `Φ(t−d)` under a `d ≥ 0` shift along
  coordinate 0, by the monotone-likelihood-ratio pointwise inequality
  `(F − 1_{z₀≤t})·(LR − LR(t)) ≥ 0` (no layer-cake, no rearrangement machinery) — and its
  rotation to an arbitrary shift on Euclidean space, `stdGaussian_np_shift`;
* the `(1/σ)`-Lipschitz probit, `smoothing_probit_lipschitz`;
* the Cohen radius at the real quantile: `smoothing_certified_radius_gaussian` (abstract scores,
  `hg` still a hypothesis), `smoothing_certified_radius_cohen` (Gaussian-smoothed `[0,1]`
  scores), `smoothing_certified_radius_classifier` (hard classifier), and
  `smoothing_certified_of_le` (any lower bound on the top-class probability certifies).

The quantile is total on ℝ (junk `sSup` outside `(0,1)`), and every use here is guarded by
`hp`. `smoothing_certified_radius_probit` asks monotonicity of the probit only on `(0,1)`
because no globally monotone function agrees with the true (unbounded) quantile there.

All results are `propext / Classical.choice / Quot.sound`-clean ([`tests/AuditAxioms.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/AuditAxioms.lean)). -/

namespace Proofs

open MeasureTheory ProbabilityTheory Filter
open scoped Topology

variable {k : ℕ} {E : Type*} [NormedAddCommGroup E]

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

/-- The halfspace indicator's Gaussian mass is the cdf at the threshold. -/
lemma integral_indicator_Iic_gaussianReal (μ t : ℝ) :
    ∫ z, (Set.Iic t).indicator (1 : ℝ → ℝ) z ∂(gaussianReal μ 1)
      = cdf (gaussianReal μ 1) t := by
  rw [integral_indicator_one measurableSet_Iic, cdf_eq_real]

-- ════════════════════════════════════════════════════════════════
-- § Capstone: the Cohen radius at the REAL Gaussian quantile
-- ════════════════════════════════════════════════════════════════

/-- **Randomized-smoothing certified radius at the true Gaussian probit.** With every class
    probability `p c y` inside `(0,1)` at every point `y` (`hp`), per-class probit scores
    `Φ⁻¹ ∘ p c` each `(1/σ)`-Lipschitz (`hg` — for Gaussian-smoothed `[0,1]` scores this is
    `smoothing_probit_lipschitz`, applied in `smoothing_certified_radius_cohen`), and the
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
only coordinate 0 carries the shift; (iii) the same pointwise MLR trick as the 1-D case, now
with the weight `exp(d·z₀ − d²/2)` and the halfspace `{z₀ ≤ t}`, gives the pi-space NP theorem;
(iv) an adapted orthonormal basis (a reflection carries `e₀` to `δ/‖δ‖`, and Mathlib's
`stdGaussian_eq_map_pi_orthonormalBasis` says the standard Gaussian doesn't care) rotates
the general shift onto coordinate 0. -/

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
      (((gaussianReal 0 1)).prod (Measure.pi fun _ : Fin n => gaussianReal 0 1)) :=
    ((integrable_expWeight d).comp_fst _).mul_bdd (hFm.comp e.symm.measurable).aestronglyMeasurable
      (ae_of_all _ fun y => (Real.norm_eq_abs _).trans_le (hFb _))
  -- the chain, on the product side
  rw [← hpres.symm.integral_comp', ← hpres.symm.integral_comp']
  have hstep1 : ∀ y : ℝ × (Fin n → ℝ),
      F (e.symm y + d • Pi.single 0 1) = F (e.symm (y.1 + d, y.2)) := by
    intro y
    rw [hsymm, hsymm, insertNth_zero_add_single]
  have hcoord : ∀ y : ℝ × (Fin n → ℝ), e.symm y 0 = y.1 := by
    intro y
    rw [hsymm]
    simp
  simp only [← he, hstep1, hcoord]
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
      (stdGaussianPi n) :=
    (measurePreserving_eval (fun _ : Fin (n + 1) => gaussianReal 0 1) 0).integrable_comp_of_integrable
      (integrable_expWeight d)
  -- a `|·| ≤ 1` factor keeps the weight integrable
  have hWbdd : ∀ {G : (Fin (n + 1) → ℝ) → ℝ}, Measurable G → (∀ z, |G z| ≤ 1) →
      Integrable (fun z => Real.exp (d * z 0 - d ^ 2 / 2) * G z) (stdGaussianPi n) :=
    fun hG hG1 => hWΓ.mul_bdd hG.aestronglyMeasurable
      (ae_of_all _ fun z => (Real.norm_eq_abs _).trans_le (hG1 z))
  have hFhΓ : Integrable (fun z => F z - h z) (stdGaussianPi n) :=
    Integrable.of_mem_Icc (-1) 1 (hFm.sub hhm).aemeasurable
      (ae_of_all _ fun z => abs_le.mp (habs_fh z))
  have hintR := hWbdd (hFm.sub hhm) habs_fh
  have hkey : c * ∫ z, (F z - h z) ∂(stdGaussianPi n)
      ≤ ∫ z, Real.exp (d * z 0 - d ^ 2 / 2) * (F z - h z) ∂(stdGaussianPi n) := by
    rw [← integral_const_mul]
    exact integral_mono (hFhΓ.const_mul c) hintR hpoint
  -- endpoint integrability + splitting
  have hFint : Integrable F (stdGaussianPi n) :=
    Integrable.of_mem_Icc 0 1 hFm.aemeasurable (ae_of_all _ fun z => ⟨hF0 z, hF1 z⟩)
  have hhint : Integrable h (stdGaussianPi n) :=
    Integrable.of_mem_Icc 0 1 hhm.aemeasurable (ae_of_all _ fun z => ⟨hh0 z, hh1 z⟩)
  have hWFint := hWbdd hFm hFb
  have hWhint := hWbdd hhm fun z => abs_le.mpr ⟨by linarith [hh0 z], hh1 z⟩
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
    rw [integral_comp_eval hh1d.aestronglyMeasurable, integral_indicator_Iic_gaussianReal]
    rfl
  have hWhval : ∫ z, Real.exp (d * z 0 - d ^ 2 / 2) * h z ∂(stdGaussianPi n)
      = stdNormalCDF (t - d) := by
    simp only [hh]
    rw [integral_comp_eval (μ := fun _ : Fin (n + 1) => gaussianReal 0 1) (i := 0)
      (f := fun s => Real.exp (d * s - d ^ 2 / 2) * (Set.Iic t).indicator (1 : ℝ → ℝ) s)
      (hWm.mul hh1d).aestronglyMeasurable, ← integral_gaussianReal_shift_eq hh1d d]
    have hind : ∀ s : ℝ, (Set.Iic t).indicator (1 : ℝ → ℝ) (s + d)
        = (Set.Iic (t - d)).indicator (1 : ℝ → ℝ) s := fun s => by
      simp only [Set.indicator_apply, Set.mem_Iic, Pi.one_apply, le_sub_iff_add_le]
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
    rw [Fintype.sum_eq_single 0 fun i hi => by rw [Pi.single_eq_of_ne hi, mul_zero, zero_smul],
      Pi.single_eq_same, mul_one, hb0', hu, smul_smul, mul_inv_cancel₀ hδnorm, one_smul]
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

-- ════════════════════════════════════════════════════════════════
-- § G4: the smoothed probit is (1/σ)-Lipschitz — hg becomes a theorem
-- ════════════════════════════════════════════════════════════════

/-! Assembly. `stdNormalQuantile_cdf` (the other inversion direction, `Φ⁻¹(Φ s) = s`, from
strict monotonicity) plus `stdNormalCDF_mem_Ioo` let the n-D bound `stdGaussian_np_shift` be
pushed through `Φ⁻¹`: applying it in both directions gives `|Φ⁻¹(p(x)) − Φ⁻¹(p(y))| ≤ ‖x−y‖/σ`
— the Cohen/Salman `(1/σ)`-Lipschitz probit, `smoothing_probit_lipschitz`. Instantiating
`smoothing_certified_radius_gaussian` with it yields `smoothing_certified_radius_cohen` (soft scores) and
`smoothing_certified_radius_classifier` (hard classifier — the `[0,1]` bounds AND the
runner-up bound come free from decision-region disjointness). The `σ`-smoothed mean is
written `∫ f(x + σ•z) dγ(z)` with `γ` the STANDARD Gaussian — i.e. noise `N(0, σ²I)`,
exactly what the `*-smooth` drivers sample. -/

/-- **The smoothed probit is (1/σ)-Lipschitz** (Cohen 2019 / Salman 2019 Lemma 2). For
    measurable `f : EuclideanSpace ℝ (Fin (n+1)) → [0,1]` whose σ-smoothed mean
    `p(x) = ∫ f(x + σz) dγ(z)` lies inside `(0,1)` at every `x`, the probit score
    `x ↦ Φ⁻¹(p x)` is `(1/σ)`-Lipschitz in L2. -/
theorem smoothing_probit_lipschitz {n : ℕ} {σ : ℝ} (hσ : 0 < σ)
    {f : EuclideanSpace ℝ (Fin (n + 1)) → ℝ}
    (hfm : Measurable f) (hf0 : ∀ z, 0 ≤ f z) (hf1 : ∀ z, f z ≤ 1)
    (hp : ∀ x : EuclideanSpace ℝ (Fin (n + 1)),
      (∫ z, f (x + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
        ∈ Set.Ioo (0:ℝ) 1) :
    LipschitzL2 (1 / σ)
      (fun x => stdNormalQuantile
        (∫ z, f (x + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))) := by
  -- one-sided bound, both directions
  have hside : ∀ x y : EuclideanSpace ℝ (Fin (n + 1)),
      stdNormalQuantile (∫ z, f (x + σ • z) ∂(stdGaussian _)) - (1 / σ) * ‖y - x‖
        ≤ stdNormalQuantile (∫ z, f (y + σ • z) ∂(stdGaussian _)) := by
    intro x y
    set px := ∫ z, f (x + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) with hpx
    set py := ∫ z, f (y + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) with hpy
    -- the smoothed mean at y is a shifted smoothed mean at x
    have hshift : py = ∫ z, (fun w => f (x + σ • w))
        (z + σ⁻¹ • (y - x)) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) := by
      rw [hpy]
      congr 1
      funext z
      congr 1
      rw [smul_add, smul_smul, mul_inv_cancel₀ (ne_of_gt hσ), one_smul]
      abel
    -- NP bound with the true threshold t = Φ⁻¹(px)
    have hnp := stdGaussian_np_shift (f := fun w => f (x + σ • w))
      (hfm.comp (measurable_const.add (measurable_id.const_smul σ)))
      (fun w => hf0 _) (fun w => hf1 _)
      (δ := σ⁻¹ • (y - x)) (t := stdNormalQuantile px)
      (by rw [stdNormalCDF_quantile (hp x)])
    rw [← hshift] at hnp
    -- Φ⁻¹ is monotone on (0,1): push it through the NP bound
    have hmem : stdNormalCDF (stdNormalQuantile px - ‖σ⁻¹ • (y - x)‖) ∈ Set.Ioo (0:ℝ) 1 :=
      stdNormalCDF_mem_Ioo _
    have hmono := stdNormalQuantile_monotoneOn hmem (hp y) hnp
    rw [stdNormalQuantile_cdf] at hmono
    -- ‖σ⁻¹ • (y − x)‖ = (1/σ)‖y − x‖
    have hnorm : ‖σ⁻¹ • (y - x)‖ = (1 / σ) * ‖y - x‖ := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hσ), one_div]
    rw [hnorm] at hmono
    linarith [hmono]
  -- combine the two one-sided bounds
  intro u w
  dsimp only
  have h1 := hside w u
  have h2 := hside u w
  rw [norm_sub_rev w u] at h2
  rw [Real.norm_eq_abs, abs_le]
  constructor <;> linarith [h1, h2]

/-- **The Cohen radius, Neyman–Pearson side DISCHARGED.** For a family of measurable
    `[0,1]` class scores whose σ-smoothed means stay in `(0,1)`, the smoothed prediction
    cannot flip within `‖δ‖ < σ·Φ⁻¹(p_i(x))`. No Lipschitz hypothesis: `hg` is now the
    theorem `smoothing_probit_lipschitz`. -/
theorem smoothing_certified_radius_cohen {n k : ℕ} {σ : ℝ} (hσ : 0 < σ)
    {f : Fin k → EuclideanSpace ℝ (Fin (n + 1)) → ℝ}
    (hfm : ∀ c, Measurable (f c)) (hf0 : ∀ c z, 0 ≤ f c z) (hf1 : ∀ c z, f c z ≤ 1)
    (hp : ∀ c x, (∫ z, f c (x + σ • z)
      ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))) ∈ Set.Ioo (0:ℝ) 1)
    {x δ : EuclideanSpace ℝ (Fin (n + 1))} {i : Fin k}
    (hrunner : ∀ j, j ≠ i →
      (∫ z, f j (x + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
        ≤ 1 - ∫ z, f i (x + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
    (hδ : ‖δ‖ < σ * stdNormalQuantile
      (∫ z, f i (x + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))) :
    ∀ j, j ≠ i →
      (∫ z, f j (x + δ + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
        < ∫ z, f i (x + δ + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) :=
  smoothing_certified_radius_gaussian hσ
    (p := fun c y => ∫ z, f c (y + σ • z)
      ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
    (fun c y => hp c y)
    (fun c => smoothing_probit_lipschitz hσ (hfm c) (hf0 c) (hf1 c) (hp c))
    hrunner hδ

/-- **The classifier form.** For a measurable hard classifier `C`, class scores are the
    decision-region indicators, so `[0,1]`-boundedness AND the runner-up bound are both
    automatic (regions are disjoint: `p_j + p_i ≤ 1`). Hypotheses: measurability of `C`;
    `hp` — every class's smoothed probability lies in `(0,1)` at every point, i.e. no decision
    region is Gaussian-null or conull (`Φ⁻¹` is only meaningful on `(0,1)`); for an argmax net
    it follows from one strict-argmax witness per class (`argmaxNet_smoothProb_mem_Ioo`); and
    the margin `‖δ‖ < σ·Φ⁻¹(p_i(x))`. The radius has the form the `*-smooth` drivers report,
    stated at the true class probability. -/
theorem smoothing_certified_radius_classifier {n k : ℕ} {σ : ℝ} (hσ : 0 < σ)
    {C : EuclideanSpace ℝ (Fin (n + 1)) → Fin k} (hC : Measurable C)
    (hp : ∀ c x, (∫ z, (if C (x + σ • z) = c then (1:ℝ) else 0)
      ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))) ∈ Set.Ioo (0:ℝ) 1)
    {x δ : EuclideanSpace ℝ (Fin (n + 1))} {i : Fin k}
    (hδ : ‖δ‖ < σ * stdNormalQuantile
      (∫ z, (if C (x + σ • z) = i then (1:ℝ) else 0)
        ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))) :
    ∀ j, j ≠ i →
      (∫ z, (if C (x + δ + σ • z) = j then (1:ℝ) else 0)
          ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
        < ∫ z, (if C (x + δ + σ • z) = i then (1:ℝ) else 0)
            ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) := by
  set f : Fin k → EuclideanSpace ℝ (Fin (n + 1)) → ℝ :=
    fun c y => if C y = c then (1:ℝ) else 0 with hf
  have hfm : ∀ c, Measurable (f c) := fun c =>
    measurable_const.ite (hC (measurableSet_singleton c)) measurable_const
  have hf0 : ∀ c z, 0 ≤ f c z := fun c z => by
    by_cases h : C z = c <;> simp [hf, h]
  have hf1 : ∀ c z, f c z ≤ 1 := fun c z => by
    by_cases h : C z = c <;> simp [hf, h]
  -- disjoint decision regions: p_j(x) + p_i(x) ≤ 1
  have hrunner : ∀ j, j ≠ i →
      (∫ z, f j (x + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
        ≤ 1 - ∫ z, f i (x + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) := by
    intro j hj
    have hint : ∀ c : Fin k, Integrable (fun z => f c (x + σ • z))
        (stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) := fun c =>
      (integrable_const 1).mono'
        ((hfm c).comp (measurable_const.add (measurable_id.const_smul σ))).aestronglyMeasurable
        (ae_of_all _ fun z => by
          rw [Real.norm_eq_abs]
          exact abs_le.mpr ⟨by linarith [hf0 c (x + σ • z)], hf1 c (x + σ • z)⟩)
    have hsum : (∫ z, f j (x + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
        + ∫ z, f i (x + σ • z) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))
        ≤ 1 := by
      rw [← integral_add (hint j) (hint i)]
      calc ∫ z, (f j (x + σ • z) + f i (x + σ • z))
            ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))
          ≤ ∫ _z, (1:ℝ) ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) := by
            refine integral_mono ((hint j).add (hint i)) (integrable_const 1) fun z => ?_
            by_cases hcj : C (x + σ • z) = j
            · have hci : C (x + σ • z) ≠ i := by rw [hcj]; exact hj
              simp [hf, hcj, hj]
            · by_cases hci : C (x + σ • z) = i <;>
                simp [hf, hcj, hci, hj.symm]
        _ = 1 := by simp
    linarith
  exact smoothing_certified_radius_cohen hσ hfm hf0 hf1 hp hrunner hδ


/-- The decision region `{v | C (x + σ•v) = y}` of a measurable classifier is measurable. -/
lemma measurableSet_smoothRegion {n k : ℕ} {C : EuclideanSpace ℝ (Fin (n + 1)) → Fin k}
    (hC : Measurable C) (x : EuclideanSpace ℝ (Fin (n + 1))) (σ : ℝ) (y : Fin k) :
    MeasurableSet {v | C (x + σ • v) = y} :=
  (hC.comp (measurable_const.add (measurable_id.const_smul σ))) (measurableSet_singleton y)

/-- **The indicator bridge**: a class probability (the integral of the vote indicator) is the
    measure of the class's decision region. -/
lemma smoothProb_eq_real {n k : ℕ} {C : EuclideanSpace ℝ (Fin (n + 1)) → Fin k}
    (hC : Measurable C) (μ : Measure (EuclideanSpace ℝ (Fin (n + 1))))
    (x : EuclideanSpace ℝ (Fin (n + 1))) (σ : ℝ) (y : Fin k) :
    μ.real {v | C (x + σ • v) = y} = ∫ z, (if C (x + σ • z) = y then (1:ℝ) else 0) ∂μ := by
  rw [← integral_indicator_one (measurableSet_smoothRegion hC x σ y)]
  refine integral_congr_ae (ae_of_all _ fun z => ?_)
  by_cases h : C (x + σ • z) = y <;> simp [h]

/-- **Any lower bound certifies.** `smoothing_certified_radius_classifier` at a radius
    `σ·Φ⁻¹(q)` for any `q` below the true class probability — the step every confidence bound
    (Hoeffding, Clopper–Pearson) takes: `Φ⁻¹` is monotone on `(0,1)`, and `q ≤ 0` gives radius
    0, which no `δ` beats. -/
theorem smoothing_certified_of_le {n k : ℕ} {σ : ℝ} (hσ : 0 < σ)
    {C : EuclideanSpace ℝ (Fin (n + 1)) → Fin k} (hC : Measurable C)
    (hp : ∀ c x, (∫ z, (if C (x + σ • z) = c then (1:ℝ) else 0)
      ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))) ∈ Set.Ioo (0:ℝ) 1)
    {x δ : EuclideanSpace ℝ (Fin (n + 1))} {i : Fin k} {q : ℝ}
    (hq : q ≤ ∫ z, (if C (x + σ • z) = i then (1:ℝ) else 0)
      ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
    (hδ : ‖δ‖ < σ * stdNormalQuantile q) :
    ∀ j, j ≠ i →
      (∫ z, (if C (x + δ + σ • z) = j then (1:ℝ) else 0)
          ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
        < ∫ z, (if C (x + δ + σ • z) = i then (1:ℝ) else 0)
            ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))) := by
  rcases le_or_gt q 0 with hq0 | hq0
  · rw [stdNormalQuantile_of_nonpos hq0, mul_zero] at hδ
    exact absurd hδ (not_lt.mpr (norm_nonneg δ))
  · have hpi := hp i x
    refine smoothing_certified_radius_classifier hσ hC hp (lt_of_lt_of_le hδ ?_)
    exact mul_le_mul_of_nonneg_left
      (stdNormalQuantile_monotoneOn ⟨hq0, lt_of_le_of_lt hq hpi.2⟩ hpi hq) hσ.le

end Proofs
