import LeanMlir.Proofs.Certificates.SmoothingCP
import LeanMlir.Proofs.Certificates.LipschitzCertInstance

/-! # Net semantics for the smoothing chain — the classifier becomes a NET

The smoothing chain up through `smoothing_cp_certified_solved` quantifies over
an ABSTRACT measurable classifier `C` with an interiority hypothesis
`hp : ∀ c x, p_c(x) ∈ (0,1)` — the last informality flagged in the scorecard
headers. This file closes it:

* `argmaxNet` — the argmax classifier of a logit map (lowest index wins ties;
  the tie-break is irrelevant at strict-argmax points), with
  `measurable_argmaxNet` from logit measurability alone (fibers are finite
  boolean combinations of `{f·j ≤ f·c}` sets);
* `isOpen_strictRegion` — strict decision regions of continuous logits are
  open;
* `stdGaussian` full support — `IsOpenPosMeasure` instances for
  `gaussianReal 0 1` (from `stdGaussian_Ioo_pos`) and for the multivariate
  `stdGaussian E` (pushforward of the pi-Gaussian under the surjective
  continuous basis sum; `pi.isOpenPosMeasure` does the product);
* `argmaxNet_smoothProb_mem_Ioo` — **the `hp` discharge**: ONE strict-argmax
  witness per class ⇒ every smoothed class probability at every point is in
  `(0,1)` (the witness's open region has positive Gaussian mass everywhere;
  any OTHER class's region caps it below 1);
* `smoothing_cp_certified_net` — the capstone: CERTIFY's guarantee with the
  classifier INSTANTIATED as `argmaxNet f`, measurability and `hp` both
  discharged from continuity + witnesses;
* `mlpT_logit_continuous` — the trained /128-rationalized pooled-MNIST MLP's
  logits are continuous (the concrete instantiation lives in the generated
  `SmoothingNetWitness.lean`).

All results are `propext / Classical.choice / Quot.sound`-clean. -/

namespace Proofs

open MeasureTheory ProbabilityTheory Real
open scoped BigOperators ENNReal NNReal RealInnerProductSpace

-- ════════ § the argmax classifier ════════

open scoped Classical in
/-- The argmax classifier of a logit map, LOWEST index winning ties: the
    least index among the maximizers. Total and deterministic — the
    tie-break never matters at strict-argmax points. -/
noncomputable def argmaxNet {E : Type*} {k : ℕ} (f : E → Fin (k + 1) → ℝ)
    (x : E) : Fin (k + 1) :=
  (Finset.univ.filter fun c => ∀ j, f x j ≤ f x c).min' (by
    obtain ⟨c, -, hc⟩ :=
      Finset.exists_max_image Finset.univ (f x) ⟨0, Finset.mem_univ 0⟩
    exact ⟨c, Finset.mem_filter.mpr
      ⟨Finset.mem_univ c, fun j => hc j (Finset.mem_univ j)⟩⟩)

lemma argmaxNet_isMax {E : Type*} {k : ℕ} (f : E → Fin (k + 1) → ℝ) (x : E) :
    ∀ j, f x j ≤ f x (argmaxNet f x) := by
  classical
  have h := Finset.min'_mem
    (Finset.univ.filter fun c => ∀ j, f x j ≤ f x c)
    (by
      obtain ⟨c, -, hc⟩ :=
        Finset.exists_max_image Finset.univ (f x) ⟨0, Finset.mem_univ 0⟩
      exact ⟨c, Finset.mem_filter.mpr
        ⟨Finset.mem_univ c, fun j => hc j (Finset.mem_univ j)⟩⟩)
  exact (Finset.mem_filter.mp h).2

/-- At a STRICT argmax the tie-break is irrelevant: `argmaxNet` returns it. -/
lemma argmaxNet_eq_of_strict {E : Type*} {k : ℕ} {f : E → Fin (k + 1) → ℝ}
    {x : E} {c : Fin (k + 1)} (h : ∀ j, j ≠ c → f x j < f x c) :
    argmaxNet f x = c := by
  classical
  rw [argmaxNet, Finset.min'_eq_iff]
  constructor
  · exact Finset.mem_filter.mpr
      ⟨Finset.mem_univ c, fun j => by
        by_cases hj : j = c
        · rw [hj]
        · exact (h j hj).le⟩
  · intro b hb
    have hbmax := (Finset.mem_filter.mp hb).2
    by_contra hlt
    have hbc : b ≠ c := fun e => hlt (e ▸ le_refl c)
    exact absurd (hbmax c) (not_le.mpr (h b hbc))

/-- Fibers of `argmaxNet` from logit measurability: `argmaxNet f x = c` iff
    `c` maximizes at `x` and no smaller index does. -/
lemma measurable_argmaxNet {E : Type*} [MeasurableSpace E] {k : ℕ}
    {f : E → Fin (k + 1) → ℝ} (hf : ∀ j, Measurable fun x => f x j) :
    Measurable (argmaxNet f) := by
  classical
  refine measurable_to_countable' fun c => ?_
  have hfiber : argmaxNet f ⁻¹' {c}
      = ({x | ∀ j, f x j ≤ f x c}
          ∩ ⋂ b : Fin (k + 1), {x | (∀ j, f x j ≤ f x b) → c ≤ b}) := by
    ext x
    simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_inter_iff,
      Set.mem_setOf_eq, Set.mem_iInter]
    constructor
    · intro hx
      subst hx
      refine ⟨argmaxNet_isMax f x, fun b hb => ?_⟩
      rw [argmaxNet]
      exact Finset.min'_le _ b (Finset.mem_filter.mpr ⟨Finset.mem_univ b, hb⟩)
    · rintro ⟨hmax, hleast⟩
      rw [argmaxNet, Finset.min'_eq_iff]
      exact ⟨Finset.mem_filter.mpr ⟨Finset.mem_univ c, hmax⟩,
        fun b hb => hleast b (Finset.mem_filter.mp hb).2⟩
  have hmaxSet : ∀ b : Fin (k + 1), MeasurableSet {x : E | ∀ j, f x j ≤ f x b} := by
    intro b
    have : {x : E | ∀ j, f x j ≤ f x b} = ⋂ j, {x | f x j ≤ f x b} :=
      Set.setOf_forall _
    rw [this]
    exact MeasurableSet.iInter fun j => measurableSet_le (hf j) (hf b)
  rw [hfiber]
  refine (hmaxSet c).inter (MeasurableSet.iInter fun b => ?_)
  by_cases hcb : c ≤ b
  · have : {x : E | (∀ j, f x j ≤ f x b) → c ≤ b} = Set.univ :=
      Set.eq_univ_of_forall fun x _ => hcb
    rw [this]; exact MeasurableSet.univ
  · have : {x : E | (∀ j, f x j ≤ f x b) → c ≤ b}
        = {x | ∀ j, f x j ≤ f x b}ᶜ := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_compl_iff]
      exact ⟨fun h hx => hcb (h hx), fun h hx => absurd hx h⟩
    rw [this]
    exact (hmaxSet b).compl

-- ════════ § strict decision regions are open ════════

lemma isOpen_strictRegion {E : Type*} [TopologicalSpace E] {k : ℕ}
    {f : E → Fin (k + 1) → ℝ} (hf : ∀ j, Continuous fun x => f x j)
    (c : Fin (k + 1)) :
    IsOpen {x | ∀ j, j ≠ c → f x j < f x c} := by
  have hrw : {x | ∀ j, j ≠ c → f x j < f x c}
      = ⋂ j : Fin (k + 1), (if j = c then Set.univ else {x | f x j < f x c}) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iInter]
    constructor
    · intro h j
      by_cases hj : j = c
      · simp [hj]
      · simpa [hj] using h j hj
    · intro h j hj
      have := h j
      simpa [hj] using this
  rw [hrw]
  refine isOpen_iInter_of_finite fun j => ?_
  by_cases hj : j = c
  · simp [hj]
  · simpa [hj] using isOpen_lt (hf j) (hf c)

-- ════════ § stdGaussian has full support ════════

/-- `N(0,1)` charges every nonempty open set (the pdf is everywhere positive) —
    packaged as the Mathlib `IsOpenPosMeasure` class. -/
instance : (gaussianReal 0 1).IsOpenPosMeasure := by
  refine ⟨fun U hU ⟨x, hx⟩ => ?_⟩
  obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.mp hU x hx
  have hsub : Set.Ioo (x - ε) (x + ε) ⊆ U := by
    rw [← Real.ball_eq_Ioo]; exact hball
  intro h0
  exact absurd (measure_mono_null hsub h0)
    (stdGaussian_Ioo_pos (by linarith)).ne'

/-- The standard Gaussian on a finite-dimensional inner-product space charges
    every nonempty open set: it is the pushforward of the pi-Gaussian (open-pos
    by `pi.isOpenPosMeasure`) under the surjective continuous basis sum. -/
instance stdGaussian.instIsOpenPosMeasure {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E]
    [BorelSpace E] : (stdGaussian E).IsOpenPosMeasure := by
  constructor
  intro U hU hne
  set T : (Fin (Module.finrank ℝ E) → ℝ) → E :=
    fun x => ∑ i, x i • stdOrthonormalBasis ℝ E i with hT
  have hTcont : Continuous T := by
    refine continuous_finsetSum _ fun i _ => ?_
    exact (continuous_apply i).smul continuous_const
  rw [stdGaussian, Measure.map_apply hTcont.measurable hU.measurableSet]
  have hopen : IsOpen (T ⁻¹' U) := hU.preimage hTcont
  have hnonempty : (T ⁻¹' U).Nonempty := by
    obtain ⟨e, he⟩ := hne
    refine ⟨fun i => (stdOrthonormalBasis ℝ E).repr e i, ?_⟩
    show T _ ∈ U
    have := (stdOrthonormalBasis ℝ E).sum_repr e
    rw [hT]
    simpa [this] using he
  exact (hopen.measure_pos _ hnonempty).ne'

-- ════════ § the hp discharge: witnesses ⇒ interior class probabilities ════════

/-- **`hp` interiority from per-class witnesses.** If every class has a point
    where it is the STRICT argmax of the continuous logits, then under
    Gaussian smoothing EVERY class probability at EVERY point is in `(0,1)`:
    exactly the `hp` hypothesis of the smoothing chain. Needs ≥ 2 classes
    (`Fin (k+2)`) — with one class `p ≡ 1`. -/
theorem argmaxNet_smoothProb_mem_Ioo {n k : ℕ} {σ : ℝ} (hσ : 0 < σ)
    {f : EuclideanSpace ℝ (Fin (n + 1)) → Fin (k + 2) → ℝ}
    (hf : ∀ j, Continuous fun x => f x j)
    (w : Fin (k + 2) → EuclideanSpace ℝ (Fin (n + 1)))
    (hw : ∀ c, ∀ j, j ≠ c → f (w c) j < f (w c) c) :
    ∀ c x, (∫ z, (if argmaxNet f (x + σ • z) = c then (1:ℝ) else 0)
      ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))) ∈ Set.Ioo (0:ℝ) 1 := by
  classical
  intro c x
  set γ := stdGaussian (EuclideanSpace ℝ (Fin (n + 1))) with hγ
  set C : EuclideanSpace ℝ (Fin (n + 1)) → Fin (k + 2) := argmaxNet f with hCdef
  have hC : Measurable C := measurable_argmaxNet fun j => (hf j).measurable
  set A : Set (EuclideanSpace ℝ (Fin (n + 1))) := {v | C (x + σ • v) = c} with hA_def
  have hA : MeasurableSet A :=
    (hC.comp (measurable_const.add (measurable_id.const_smul σ)))
      (measurableSet_singleton c)
  -- the indicator-integral bridge (same shape as SmoothingCP's hpA)
  have hpA : γ.real A = ∫ z, (if C (x + σ • z) = c then (1:ℝ) else 0) ∂γ := by
    rw [← integral_indicator_one hA]
    refine integral_congr_ae (ae_of_all _ fun z => ?_)
    by_cases h : C (x + σ • z) = c
    · rw [Set.indicator_of_mem (show z ∈ A from h), Pi.one_apply]
      simp [h]
    · rw [Set.indicator_of_notMem (show z ∉ A from h)]
      simp [h]
  rw [← hpA]
  -- the affine noise map and the per-class strict open regions
  have haff : Continuous fun z : EuclideanSpace ℝ (Fin (n + 1)) => x + σ • z :=
    continuous_const.add (continuous_id.const_smul σ)
  set S : Fin (k + 2) → Set (EuclideanSpace ℝ (Fin (n + 1))) :=
    fun b => {v | ∀ j, j ≠ b → f v j < f v b} with hS
  have hSopen : ∀ b, IsOpen ((fun z => x + σ • z) ⁻¹' S b) :=
    fun b => (isOpen_strictRegion hf b).preimage haff
  have hSne : ∀ b, ((fun z => x + σ • z) ⁻¹' S b).Nonempty := by
    intro b
    refine ⟨σ⁻¹ • (w b - x), ?_⟩
    show x + σ • σ⁻¹ • (w b - x) ∈ S b
    rw [smul_smul, mul_inv_cancel₀ hσ.ne', one_smul, add_sub_cancel]
    exact hw b
  have hSsub : ∀ b, (fun z => x + σ • z) ⁻¹' S b ⊆ {v | C (x + σ • v) = b} := by
    intro b z hz
    exact argmaxNet_eq_of_strict hz
  constructor
  · -- 0 < p_c: the c-witness region sits inside A
    have hpos : 0 < γ ((fun z => x + σ • z) ⁻¹' S c) :=
      (hSopen c).measure_pos γ (hSne c)
    have hle : γ.real ((fun z => x + σ • z) ⁻¹' S c) ≤ γ.real A :=
      measureReal_mono (hSsub c) (measure_ne_top _ _)
    have : 0 < γ.real ((fun z => x + σ • z) ⁻¹' S c) :=
      ENNReal.toReal_pos hpos.ne' (measure_ne_top _ _)
    linarith
  · -- p_c < 1: any OTHER class's witness region is disjoint from A
    set c' : Fin (k + 2) := if c = 0 then 1 else 0 with hc'
    have hcc' : c' ≠ c := by
      by_cases h : c = 0
      · subst h
        simp only [hc']
        exact one_ne_zero
      · simp only [hc', if_neg h]
        exact Ne.symm h
    have hdisj : Disjoint A ((fun z => x + σ • z) ⁻¹' S c') := by
      rw [Set.disjoint_right]
      intro z hz
      have : C (x + σ • z) = c' := hSsub c' hz
      simp only [hA_def, Set.mem_setOf_eq, this]
      exact fun h => hcc' h
    have hpos' : 0 < γ.real ((fun z => x + σ • z) ⁻¹' S c') :=
      ENNReal.toReal_pos ((hSopen c').measure_pos γ (hSne c')).ne'
        (measure_ne_top _ _)
    have hsum : γ.real A + γ.real ((fun z => x + σ • z) ⁻¹' S c') ≤ 1 := by
      have hunion := measureReal_union (μ := γ) hdisj
        ((hSopen c').measurableSet)
      have hle : γ.real (A ∪ (fun z => x + σ • z) ⁻¹' S c') ≤ 1 := by
        rw [show (1:ℝ) = γ.real Set.univ from (probReal_univ (μ := γ)).symm]
        exact measureReal_mono (Set.subset_univ _) (measure_ne_top _ _)
      rw [hunion] at hle
      linarith
    linarith

-- ════════ § the capstone: CERTIFY for the NET's argmax ════════

/-- **The net-semantics capstone.** `smoothing_cp_certified_solved` with the
    classifier INSTANTIATED as the argmax of a concrete continuous logit map:
    measurability and `hp` interiority are DISCHARGED (from continuity and
    per-class strict-argmax witnesses). With probability `≥ 1 − α` over the
    `N` Gaussian samples: if the vote count for class `y` comes out `k₀`
    (where `binomTail N k₀ q₀ ≤ α` is one kernel check), every perturbation
    `‖δ‖ < σ·Φ⁻¹(q₀)` leaves `y` the strict argmax of the SMOOTHED net. -/
theorem smoothing_cp_certified_net {n k : ℕ} {σ : ℝ} (hσ : 0 < σ)
    {f : EuclideanSpace ℝ (Fin (n + 1)) → Fin (k + 2) → ℝ}
    (hf : ∀ j, Continuous fun x => f x j)
    (w : Fin (k + 2) → EuclideanSpace ℝ (Fin (n + 1)))
    (hw : ∀ c, ∀ j, j ≠ c → f (w c) j < f (w c) c)
    (x : EuclideanSpace ℝ (Fin (n + 1))) (y : Fin (k + 2))
    {N k₀ : ℕ} (hk₀ : k₀ ≤ N) {α q₀ : ℝ} (hα : 0 ≤ α) (hα1 : α < 1)
    (hq₀ : q₀ ∈ Set.Ioo (0:ℝ) 1) (htail : binomTail N k₀ q₀ ≤ α) :
    1 - α
      ≤ (Measure.pi fun _ : Fin N => stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))).real
          {ω | (∑ i, if argmaxNet f (x + σ • ω i) = y then 1 else 0) = k₀ →
            ∀ δ : EuclideanSpace ℝ (Fin (n + 1)),
            ‖δ‖ < σ * stdNormalQuantile q₀ →
            ∀ j, j ≠ y →
              (∫ z, (if argmaxNet f (x + δ + σ • z) = j then (1:ℝ) else 0)
                  ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1)))))
                < ∫ z, (if argmaxNet f (x + δ + σ • z) = y then (1:ℝ) else 0)
                    ∂(stdGaussian (EuclideanSpace ℝ (Fin (n + 1))))} :=
  smoothing_cp_certified_solved hσ
    (measurable_argmaxNet fun j => (hf j).measurable)
    (argmaxNet_smoothProb_mem_Ioo hσ hf w hw)
    x y hk₀ hα hα1 hq₀ htail

end Proofs

namespace Proofs.LipschitzCertDemo

/-- Each logit of the trained pooled-MNIST MLP is continuous: the coordinate
    formula `∑ k, W2ⱼₖ·max(∑ l, W1ₖₗ·xₗ, 0)` is definitional. -/
theorem mlpT_logit_continuous : ∀ j : Fin 10, Continuous fun x => mlpT x j := by
  intro j
  show Continuous fun x : EuclideanSpace ℝ (Fin 49) =>
    ∑ k : Fin 8, W2t j k * max (∑ l, W1t k l * x l) 0
  refine continuous_finsetSum _ fun k _ => continuous_const.mul ?_
  refine Continuous.max ?_ continuous_const
  exact continuous_finsetSum _ fun l _ =>
    continuous_const.mul (EuclideanSpace.proj l).continuous

end Proofs.LipschitzCertDemo
