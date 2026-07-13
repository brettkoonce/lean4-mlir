import LeanMlir.Proofs.SmoothingGaussian
import Mathlib.Analysis.Real.Pi.Bounds

/-! # Certified decimal quantile bounds — `Φ⁻¹` leaves symbolic-land

The scorecard's radii are `σ·Φ⁻¹(q₀)` with `Φ⁻¹` SYMBOLIC; the driver prints
decimals via float Acklam. This file closes that gap with certified rational
LOWER bounds on `Φ⁻¹`:

* `stdNormalCDF_panel` — one upper-Riemann panel `Φ(b) ≤ Φ(a) + (b−a)·φ(a)`
  on `[0,∞)` (the density is antitone there), by bounding the Gaussian
  measure of `Ioc a b` through `gaussianReal_apply` + `setLIntegral` — no
  FTC, no improper integrals;
* `ratExpLB`/`ratPdfUB`/`ratCeil9` — a KERNEL-COMPUTABLE rational upper
  bound for the density: 32-term Taylor lower bound for `exp`
  (`Real.sum_le_exp_of_nonneg`), `√(2π) ≥ 2.5066282` (`Real.pi_gt_d20`),
  each panel value ceiling-rounded to `10⁻⁹` so the grid fold's
  denominators stay bounded (the exact values' lcm explodes);
* `phiGridUB` + `stdNormalCDF_le_phiGridUB` — the cumulative grid fold:
  `Φ(m·h) ≤ phiGridUB h m`, a computable ℚ;
* `le_stdNormalQuantile_of_grid` — the workhorse: ONE `decide +kernel`
  rational check `phiGridUB h m ≤ q₀` certifies `m·h ≤ Φ⁻¹(q₀)`.

Demo instances: `Φ⁻¹(0.9) ≥ 1.27` (true 1.2816) and `Φ⁻¹(0.9952) ≥ 2.54`
(true 2.590) at `h = 1/500` (~0.001 slack in Φ), plus the radius form for
the scorecard's MNIST-MLP image 1.

For the whole corpus, the §prefix-scan section makes per-image checks cheap:
`phiScanRev` computes ALL grid values in one kernel pass (the head's two uses
stay shared through the kernel's whnf cache), `phiScanRev_getD` indexes it,
and `le_stdNormalQuantile_of_scan`/`smooth_radius_dec` turn one O(index)
lookup against the one-shot literal into a certified decimal radius — see
the generated `SmoothingDecScorecard.lean` (279 images). The literal is
verified in CHUNKS (`phiScanRevFrom` + `phiScanRevFrom_append`): one
whole-grid evaluation peaks at 15 GB of retained kernel-cache bignums (an
OOM on 16 GB CI runners); per-declaration chunks are freed in between.

All results are `propext / Classical.choice / Quot.sound`-clean. -/

namespace Proofs

open MeasureTheory ProbabilityTheory Real
open scoped BigOperators ENNReal NNReal

-- ════════ § Φ(0) = 1/2 and the one-panel bound ════════

lemma stdNormalCDF_zero : stdNormalCDF 0 = 1/2 := by
  have h := stdNormalCDF_neg 0
  rw [neg_zero] at h
  linarith

/-- One upper Riemann panel: on `[a,b] ⊆ [0,∞)` the density is at most its
    left-endpoint value, so `Φ(b) ≤ Φ(a) + (b−a)·φ(a)`. -/
lemma stdNormalCDF_panel {a b : ℝ} (h0 : 0 ≤ a) (hab : a ≤ b) :
    stdNormalCDF b ≤ stdNormalCDF a + (b - a) * gaussianPDFReal 0 1 a := by
  have hsplit : stdNormalCDF b
      = stdNormalCDF a + (gaussianReal 0 1).real (Set.Ioc a b) := by
    rw [stdNormalCDF, stdNormalCDF, cdf_eq_real, cdf_eq_real,
      ← measureReal_union (Set.Iic_disjoint_Ioc le_rfl) measurableSet_Ioc,
      Set.Iic_union_Ioc_eq_Iic hab]
  rw [hsplit]
  have hpdf_mono : ∀ x ∈ Set.Ioc a b,
      gaussianPDF 0 1 x ≤ gaussianPDF 0 1 a := by
    intro x hx
    rw [gaussianPDF, gaussianPDF]
    refine ENNReal.ofReal_le_ofReal ?_
    rw [gaussianPDFReal, gaussianPDFReal]
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    refine Real.exp_le_exp.mpr ?_
    simp only [sub_zero, NNReal.coe_one, mul_one]
    have hsq : a^2 ≤ x^2 := by nlinarith [hx.1.le]
    linarith
  have hmeas : (gaussianReal 0 1) (Set.Ioc a b)
      ≤ ENNReal.ofReal ((b - a) * gaussianPDFReal 0 1 a) := by
    rw [gaussianReal_apply 0 one_ne_zero]
    calc ∫⁻ x in Set.Ioc a b, gaussianPDF 0 1 x
        ≤ ∫⁻ _x in Set.Ioc a b, gaussianPDF 0 1 a :=
          setLIntegral_mono' measurableSet_Ioc hpdf_mono
      _ = gaussianPDF 0 1 a * volume (Set.Ioc a b) := setLIntegral_const _ _
      _ = ENNReal.ofReal ((b - a) * gaussianPDFReal 0 1 a) := by
          rw [Real.volume_Ioc, gaussianPDF,
            ← ENNReal.ofReal_mul (gaussianPDFReal_nonneg 0 1 a), mul_comm]
  have hnn : (0:ℝ) ≤ (b - a) * gaussianPDFReal 0 1 a :=
    mul_nonneg (sub_nonneg.mpr hab) (gaussianPDFReal_nonneg 0 1 a)
  have hreal : (gaussianReal 0 1).real (Set.Ioc a b)
      ≤ (b - a) * gaussianPDFReal 0 1 a := by
    rw [measureReal_def]
    refine le_trans (ENNReal.toReal_mono (by finiteness) hmeas) ?_
    rw [ENNReal.toReal_ofReal hnn]
  linarith

-- ════════ § rational exp lower bound + √(2π) lower bound ════════

/-- Truncated Taylor sum — a computable rational lower bound for `exp` on
    `[0,∞)` (32 terms: relative error < e⁻²⁴ at `x = 5.12`, our largest use). -/
def ratExpLB (x : ℚ) : ℚ := ∑ i ∈ Finset.range 32, x ^ i / i.factorial

lemma ratExpLB_le {x : ℚ} (hx : 0 ≤ x) : (ratExpLB x : ℝ) ≤ Real.exp x := by
  have h := Real.sum_le_exp_of_nonneg (x := (x:ℝ)) (by exact_mod_cast hx) 32
  refine le_trans (le_of_eq ?_) h
  rw [ratExpLB]
  push_cast
  rfl

lemma one_le_ratExpLB {x : ℚ} (hx : 0 ≤ x) : 1 ≤ ratExpLB x := by
  rw [ratExpLB]
  calc (1:ℚ) = ∑ i ∈ Finset.range 1, x ^ i / i.factorial := by
        simp [Nat.factorial]
    _ ≤ ∑ i ∈ Finset.range 32, x ^ i / i.factorial :=
        Finset.sum_le_sum_of_subset_of_nonneg
          (Finset.range_subset.mpr (by norm_num))
          (fun i _ _ => by positivity)

lemma sqrt_two_pi_ge : (2.5066282 : ℝ) ≤ Real.sqrt (2 * π) := by
  refine (Real.le_sqrt' (by norm_num)).mpr ?_
  nlinarith [Real.pi_gt_d20]

-- ════════ § the computable pdf upper bound (ceiling-rounded) ════════

/-- Computable rational upper bound for the standard-normal density at a
    rational point: `φ(a) = 1/(√(2π)·exp(a²/2)) ≤ 1/(2.5066282·ratExpLB(a²/2))`. -/
def ratPdfUB (a : ℚ) : ℚ := ((25066282 : ℚ)/10000000 * ratExpLB (a^2/2))⁻¹

lemma gaussianPDFReal_le_ratPdfUB (a : ℚ) :
    gaussianPDFReal 0 1 (a:ℝ) ≤ (ratPdfUB a : ℝ) := by
  have hx : (0:ℚ) ≤ a^2/2 := by positivity
  have hLB := ratExpLB_le hx
  have hLB1 : (1:ℝ) ≤ (ratExpLB (a^2/2) : ℝ) := by
    exact_mod_cast one_le_ratExpLB hx
  have hden_pos : (0:ℝ) < 2.5066282 * (ratExpLB (a^2/2) : ℝ) := by nlinarith
  have hcast : ((a^2/2 : ℚ) : ℝ) = (a:ℝ)^2/2 := by push_cast; ring
  rw [gaussianPDFReal]
  simp only [sub_zero, NNReal.coe_one, mul_one]
  have hexp : Real.exp (-(a:ℝ)^2/2) = (Real.exp ((a:ℝ)^2/2))⁻¹ := by
    rw [← Real.exp_neg]
    ring_nf
  rw [hexp, ← mul_inv]
  have hchain : (2.5066282 : ℝ) * (ratExpLB (a^2/2) : ℝ)
      ≤ Real.sqrt (2*π) * Real.exp ((a:ℝ)^2/2) := by
    refine mul_le_mul sqrt_two_pi_ge ?_ (by linarith) (Real.sqrt_nonneg _)
    calc (ratExpLB (a^2/2) : ℝ) ≤ Real.exp ((a^2/2 : ℚ) : ℝ) := hLB
      _ = Real.exp ((a:ℝ)^2/2) := by rw [hcast]
  have hub : (Real.sqrt (2*π) * Real.exp ((a:ℝ)^2/2))⁻¹
      ≤ ((2.5066282 : ℝ) * (ratExpLB (a^2/2) : ℝ))⁻¹ := by
    rw [← one_div, ← one_div]
    exact one_div_le_one_div_of_le hden_pos hchain
  refine hub.trans (le_of_eq ?_)
  rw [ratPdfUB]
  push_cast
  norm_num

/-- Round a rational UP to denominator `10⁹` — keeps the grid fold's
    denominators from exploding (the exact `ratPdfUB` values have ~190-digit
    numerators whose lcm across 640 grid points is astronomical). -/
def ratCeil9 (q : ℚ) : ℚ := (⌈q * 1000000000⌉ : ℤ) / 1000000000

lemma le_ratCeil9 (q : ℚ) : q ≤ ratCeil9 q := by
  rw [ratCeil9]
  rw [le_div_iff₀ (by norm_num : (0:ℚ) < 1000000000)]
  exact_mod_cast Int.le_ceil (q * 1000000000)

-- ════════ § the grid fold + the master bound ════════

/-- Computable upper bound for `Φ(m·h)`: cumulative left-endpoint upper
    Riemann panels from 0, each pdf bound ceiling-rounded to `10⁻⁹`. -/
def phiGridUB (h : ℚ) : ℕ → ℚ
  | 0 => 1/2
  | m+1 => phiGridUB h m + h * ratCeil9 (ratPdfUB ((m:ℚ) * h))

lemma stdNormalCDF_le_phiGridUB {h : ℚ} (hh : 0 ≤ h) (m : ℕ) :
    stdNormalCDF ((m:ℚ) * h : ℚ) ≤ (phiGridUB h m : ℝ) := by
  induction m with
  | zero => simp [phiGridUB, stdNormalCDF_zero]
  | succ m ih =>
    have hh' : (0:ℝ) ≤ (h:ℚ) := by exact_mod_cast hh
    have h0 : (0:ℝ) ≤ ((m:ℚ) * h : ℚ) := by
      have : (0:ℚ) ≤ (m:ℚ) * h := by positivity
      exact_mod_cast this
    have hab : (((m:ℚ) * h : ℚ) : ℝ) ≤ (((m+1:ℕ):ℚ) * h : ℚ) := by
      have : ((m:ℚ) * h : ℚ) ≤ ((m+1:ℕ):ℚ) * h := by
        push_cast
        nlinarith
      exact_mod_cast this
    calc stdNormalCDF (((m+1:ℕ):ℚ) * h : ℚ)
        ≤ stdNormalCDF ((m:ℚ) * h : ℚ)
            + ((((m+1:ℕ):ℚ) * h : ℚ) - (((m:ℚ) * h : ℚ) : ℝ))
              * gaussianPDFReal 0 1 ((m:ℚ) * h : ℚ) :=
          stdNormalCDF_panel h0 hab
      _ ≤ (phiGridUB h m : ℝ)
            + (h : ℝ) * (ratCeil9 (ratPdfUB ((m:ℚ) * h)) : ℝ) := by
          have hw : ((((m+1:ℕ):ℚ) * h : ℚ) : ℝ) - (((m:ℚ) * h : ℚ) : ℝ) = (h:ℝ) := by
            push_cast
            ring
          have hpdf : gaussianPDFReal 0 1 (((m:ℚ) * h : ℚ) : ℝ)
              ≤ (ratCeil9 (ratPdfUB ((m:ℚ) * h)) : ℝ) := by
            refine (gaussianPDFReal_le_ratPdfUB ((m:ℚ) * h)).trans ?_
            exact_mod_cast le_ratCeil9 (ratPdfUB ((m:ℚ) * h))
          have hpdf_nn : (0:ℝ) ≤ gaussianPDFReal 0 1 (((m:ℚ) * h : ℚ) : ℝ) :=
            gaussianPDFReal_nonneg 0 1 _
          rw [hw]
          nlinarith [ih]
      _ = ((phiGridUB h (m+1) : ℚ) : ℝ) := by
          rw [phiGridUB]
          push_cast
          ring

-- ════════ § transfer through the quantile ════════

lemma le_stdNormalQuantile_of_cdf_le {t q : ℝ} (hq : q ∈ Set.Ioo (0:ℝ) 1)
    (h : stdNormalCDF t ≤ q) : t ≤ stdNormalQuantile q := by
  have hm := stdNormalQuantile_monotoneOn (stdNormalCDF_mem_Ioo t) hq h
  rwa [stdNormalQuantile_cdf] at hm

/-- **The certified-decimal-radius workhorse**: one rational grid check
    `phiGridUB h m ≤ q₀` certifies `m·h ≤ Φ⁻¹(q₀)`. -/
lemma le_stdNormalQuantile_of_grid {h : ℚ} (hh : 0 ≤ h) (m : ℕ) {q : ℝ}
    (hq : q ∈ Set.Ioo (0:ℝ) 1) (hcheck : ((phiGridUB h m : ℚ) : ℝ) ≤ q) :
    (((m:ℚ) * h : ℚ) : ℝ) ≤ stdNormalQuantile q :=
  le_stdNormalQuantile_of_cdf_le hq ((stdNormalCDF_le_phiGridUB hh m).trans hcheck)

-- ════════ § demo decimal bounds (kernel panels; ~15-30 s each) ════════

set_option maxHeartbeats 1000000 in
/-- `Φ⁻¹(0.9) ≥ 1.27` (true value 1.2816...) — 635 kernel panels at `h=1/500`. -/
lemma stdNormalQuantile_ge_of_09 : (1.27 : ℝ) ≤ stdNormalQuantile (9/10) := by
  have hcheck : ((phiGridUB (1/500) 635 : ℚ) : ℝ) ≤ (9:ℝ)/10 := by
    have h : phiGridUB (1/500) 635 ≤ (9:ℚ)/10 := by decide +kernel
    have h' := (Rat.cast_le (K := ℝ)).mpr h
    simpa using h'
  have h := le_stdNormalQuantile_of_grid (h := 1/500) (by norm_num) 635
    (q := (9:ℝ)/10) (by norm_num) hcheck
  refine le_trans (le_of_eq ?_) h
  norm_num

set_option maxHeartbeats 2000000 in
/-- `Φ⁻¹(0.9952) ≥ 2.54` (true value 2.5899...) — 1270 kernel panels. `0.9952`
    is the scorecard's MNIST-MLP image-1 CP lower bound (count 10084/10112). -/
lemma stdNormalQuantile_ge_of_9952 :
    (2.54 : ℝ) ≤ stdNormalQuantile ((9952:ℝ)/10000) := by
  have hcheck : ((phiGridUB (1/500) 1270 : ℚ) : ℝ) ≤ (9952:ℝ)/10000 := by
    have h : phiGridUB (1/500) 1270 ≤ (9952:ℚ)/10000 := by decide +kernel
    have h' := (Rat.cast_le (K := ℝ)).mpr h
    simpa using h'
  have h := le_stdNormalQuantile_of_grid (h := 1/500) (by norm_num) 1270
    (q := (9952:ℝ)/10000) (by norm_num) hcheck
  refine le_trans (le_of_eq ?_) h
  norm_num

/-- The scorecard radius in DECIMALS: MNIST-MLP image 1's certified radius
    `σ·Φ⁻¹(q₀)` is at least `1.27` (driver float printout: 1.295; `σ = 1/2`). -/
lemma smooth_cp_mlp_i1_radius_dec :
    (1.27 : ℝ) ≤ (1/2 : ℝ) * stdNormalQuantile ((9952:ℝ)/10000) := by
  linarith [stdNormalQuantile_ge_of_9952]

-- ════════ § the prefix scan — one kernel pass prices the whole grid ════════

/-- Descending prefix scan of the grid fold: `phiScanRev h n =
    [phiGridUB h n, …, phiGridUB h 0]`. Each step reuses the previous head, so
    ONE kernel evaluation prices the whole grid at O(n) pdf bounds (a per-image
    `phiGridUB` decide re-folds its whole prefix instead; the kernel's whnf
    cache keeps the head's two uses shared — measured 81 s for the full
    `h = 1/1000`, 3300-panel scan). -/
def phiScanRev (h : ℚ) : ℕ → List ℚ
  | 0 => [1/2]
  | m+1 =>
    match phiScanRev h m with
    | [] => []
    | x :: xs => (x + h * ratCeil9 (ratPdfUB ((m:ℚ) * h))) :: x :: xs

lemma phiScanRev_ne_nil (h : ℚ) (n : ℕ) : phiScanRev h n ≠ [] := by
  induction n with
  | zero => simp [phiScanRev]
  | succ m ih =>
    cases hm : phiScanRev h m with
    | nil => exact absurd hm ih
    | cons x xs => simp [phiScanRev, hm]

lemma phiScanRev_headI (h : ℚ) (n : ℕ) :
    (phiScanRev h n).headI = phiGridUB h n := by
  induction n with
  | zero => simp [phiScanRev, phiGridUB]
  | succ m ih =>
    cases hm : phiScanRev h m with
    | nil => exact absurd hm (phiScanRev_ne_nil h m)
    | cons x xs =>
      rw [hm] at ih
      simp only [List.headI] at ih
      simp [phiScanRev, hm, phiGridUB, ih]

lemma phiScanRev_succ (h : ℚ) (m : ℕ) :
    phiScanRev h (m+1) = phiGridUB h (m+1) :: phiScanRev h m := by
  cases hm : phiScanRev h m with
  | nil => exact absurd hm (phiScanRev_ne_nil h m)
  | cons x xs =>
    have hx : x = phiGridUB h m := by
      have := phiScanRev_headI h m
      rw [hm] at this
      simpa using this
    simp [phiScanRev, hm, phiGridUB, hx]

/-- Index the scan: entry `n − m` (descending order) is `phiGridUB h m`. -/
lemma phiScanRev_getD (h : ℚ) : ∀ {n m : ℕ}, m ≤ n →
    (phiScanRev h n).getD (n - m) 1 = phiGridUB h m := by
  intro n
  induction n with
  | zero =>
    intro m hm
    rw [Nat.le_zero.mp hm]
    simp [phiScanRev, phiGridUB]
  | succ k ih =>
    intro m hm
    rw [phiScanRev_succ]
    rcases Nat.eq_or_lt_of_le hm with h1 | h2
    · rw [h1]
      simp
    · have hm' : m ≤ k := Nat.lt_succ_iff.mp h2
      have hidx : k + 1 - m = (k - m) + 1 := by omega
      rw [hidx, List.getD_cons_succ]
      exact ih hm'

/-- **The scan workhorse**: against a ONE-shot kernel-evaluated literal
    `phiScanRev h n = L`, a single O(index) lookup `L.getD (n−m) 1 ≤ q₀`
    certifies `m·h ≤ Φ⁻¹(q₀)` — per-image checks stop re-folding the grid. -/
lemma le_stdNormalQuantile_of_scan {h : ℚ} (hh : 0 ≤ h) {n : ℕ} {L : List ℚ}
    (hL : phiScanRev h n = L) {m : ℕ} (hm : m ≤ n) {qr : ℚ}
    (hq : (qr:ℝ) ∈ Set.Ioo (0:ℝ) 1) (hcheck : L.getD (n - m) 1 ≤ qr) :
    (((m:ℚ) * h : ℚ) : ℝ) ≤ stdNormalQuantile (qr:ℝ) := by
  have hg : phiGridUB h m ≤ qr := by
    rw [← phiScanRev_getD h hm, hL]
    exact hcheck
  exact le_stdNormalQuantile_of_grid hh m hq ((Rat.cast_le (K := ℝ)).mpr hg)

/-- The scan CONTINUED from a checkpoint: given `v = phiGridUB h k`,
    `phiScanRevFrom h k v j = [phiGridUB h (k+j), …, phiGridUB h k]`
    (established by `phiScanRevFrom_append`). Lets the whole-grid kernel
    evaluation — 15 GB peak at 3300 panels, an OOM on 16 GB CI runners — be
    split into per-declaration chunks: the kernel's whnf cache (which retains
    every intermediate bignum) is freed between declarations. -/
def phiScanRevFrom (h : ℚ) (k : ℕ) (v : ℚ) : ℕ → List ℚ
  | 0 => [v]
  | j+1 =>
    match phiScanRevFrom h k v j with
    | [] => []
    | x :: xs => (x + h * ratCeil9 (ratPdfUB (((k + j : ℕ):ℚ) * h))) :: x :: xs

lemma phiScanRevFrom_ne_nil (h : ℚ) (k : ℕ) (v : ℚ) (j : ℕ) :
    phiScanRevFrom h k v j ≠ [] := by
  induction j with
  | zero => simp [phiScanRevFrom]
  | succ m ih =>
    cases hm : phiScanRevFrom h k v m with
    | nil => exact absurd hm ih
    | cons x xs => simp [phiScanRevFrom, hm]

/-- **The chunk glue**: a scan continued from the checkpoint `phiGridUB h k`
    extends the full scan from `k` to `k + j`. -/
lemma phiScanRevFrom_append (h : ℚ) (k : ℕ) : ∀ j : ℕ,
    phiScanRevFrom h k (phiGridUB h k) j ++ (phiScanRev h k).tail
      = phiScanRev h (k + j) := by
  intro j
  induction j with
  | zero =>
    cases hk : phiScanRev h k with
    | nil => exact absurd hk (phiScanRev_ne_nil h k)
    | cons x xs =>
      have hx : x = phiGridUB h k := by
        have := phiScanRev_headI h k
        rw [hk] at this
        simpa using this
      simp [phiScanRevFrom, hx]
  | succ m ih =>
    cases hm : phiScanRevFrom h k (phiGridUB h k) m with
    | nil => exact absurd hm (phiScanRevFrom_ne_nil h k _ m)
    | cons x xs =>
      have hx : x = phiGridUB h (k + m) := by
        have h1 := ih
        rw [hm] at h1
        have h2 := phiScanRev_headI h (k + m)
        rw [← h1] at h2
        simpa using h2
      have hidx : k + (m + 1) = (k + m) + 1 := by omega
      rw [hidx, phiScanRev_succ, ← ih, hm]
      simp only [phiScanRevFrom, hm, List.cons_append]
      rw [phiGridUB, hx]

/-- The scorecard-protocol decimal-radius form (σ = 1/2, grid `h = 1/1000`,
    4-decimal `q₀ = a/10000`): one scan lookup certifies
    `m/2000 ≤ σ·Φ⁻¹(q₀)` — the certified decimal counterpart of the driver's
    float radius printout. -/
lemma smooth_radius_dec {n : ℕ} {L : List ℚ}
    (hL : phiScanRev (1/1000) n = L) (m a : ℕ) (hm : m ≤ n)
    (ha0 : 5000 ≤ a) (ha1 : a < 10000)
    (hcheck : L.getD (n - m) 1 ≤ (a:ℚ)/10000) :
    (m:ℝ)/2000 ≤ (1/2:ℝ) * stdNormalQuantile ((a:ℝ)/10000) := by
  have hq : (((a:ℚ)/10000 : ℚ) : ℝ) ∈ Set.Ioo (0:ℝ) 1 := by
    have h0 : (0:ℝ) < (a:ℝ) := by
      have : (0:ℕ) < a := lt_of_lt_of_le (by norm_num) ha0
      exact_mod_cast this
    have h1 : (a:ℝ) < 10000 := by exact_mod_cast ha1
    constructor
    · push_cast
      exact div_pos h0 (by norm_num)
    · push_cast
      rw [div_lt_one (by norm_num : (0:ℝ) < 10000)]
      exact h1
  have h := le_stdNormalQuantile_of_scan (by norm_num) hL hm hq hcheck
  have hcast : (((a:ℚ)/10000 : ℚ) : ℝ) = (a:ℝ)/10000 := by push_cast; ring
  have hlhs : (((m:ℚ) * (1/1000) : ℚ) : ℝ) = (m:ℝ)/1000 := by push_cast; ring
  rw [hcast, hlhs] at h
  linarith

end Proofs
