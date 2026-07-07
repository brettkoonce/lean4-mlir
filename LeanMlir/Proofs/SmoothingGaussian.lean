import LeanMlir.Proofs.LipschitzCert
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.CDF

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

end Proofs
