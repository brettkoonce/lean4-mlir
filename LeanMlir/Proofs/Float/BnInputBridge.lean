import LeanMlir.Proofs.Float.BnFloatBridge

/-!
# Real-BN input-sensitivity (the per-block composition enabler)

Composing the float bridge through a ResNet block chains each op as
"float-op-at-float-input vs real-op-at-real-input". `flatConvF_close`, `mul_close`,
`add_close`, `relu_close` all take an input-error hypothesis — but
`bnForward_close_of` compares the float and real BN *at the same input* (it models
only rounding). So block composition needs the missing piece: how the **real**
`bnForward` moves when its input moves — its input-Lipschitz behavior.

This file proves that sensitivity chain over `ℝ`: mean (`bnMean_input_close`),
variance (`bnVar_input_close`), and inverse-stddev (`bnIstd_input_close`, which
reuses `rsqrt_lipschitz`). With these + the rounding wrappers, a per-block
closeness `|bnForwardF(float input) − bnForward(real input)|` splits as
rounding-at-fixed-input (`bnForward_close_of`) + input-shift (these), so the block
chain is mechanical from here.
-/

namespace Proofs

open scoped Real

/-- The real mean stays within the input range: `|μ| ≤ A` under `|xᵢ| ≤ A`. -/
theorem bnMean_abs_le {n : ℕ} (x : Vec n) {A : ℝ} (hn : 0 < n) (hA : ∀ i, |x i| ≤ A) :
    |bnMean n x| ≤ A := by
  have hnR : (0:ℝ) < (n:ℝ) := by exact_mod_cast hn
  unfold bnMean
  rw [abs_div, abs_of_pos hnR, div_le_iff₀ hnR]
  calc |∑ i, x i| ≤ ∑ i, |x i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin n, A := Finset.sum_le_sum fun i _ => hA i
    _ = A * (n:ℝ) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring

/-- **Mean input-sensitivity.** `|μ(x) − μ(y)| ≤ (Σ|xᵢ−yᵢ|)/n`. -/
theorem bnMean_input_close {n : ℕ} (x y : Vec n) (hn : 0 < n) :
    |bnMean n x - bnMean n y| ≤ (∑ i, |x i - y i|) / (n:ℝ) := by
  have hnR : (0:ℝ) < (n:ℝ) := by exact_mod_cast hn
  unfold bnMean
  rw [div_sub_div_same, abs_div, abs_of_pos hnR]
  gcongr
  calc |∑ i, x i - ∑ i, y i| = |∑ i, (x i - y i)| := by rw [Finset.sum_sub_distrib]
    _ ≤ ∑ i, |x i - y i| := Finset.abs_sum_le_sum_abs _ _

/-- `|a − μ| ≤ |a| + |μ|` (triangle through 0). -/
private theorem abs_sub_le_add (a b : ℝ) : |a - b| ≤ |a| + |b| := by
  have h := abs_sub_le a 0 b; simp only [sub_zero, zero_sub, abs_neg] at h; exact h

/-- **Variance input-sensitivity.** `|σ²(x) − σ²(y)| ≤ 8A·(Σ|xᵢ−yᵢ|)/n` under
    `|xᵢ|,|yᵢ| ≤ A` — each centered-square difference factors as
    `((cx)−(cy))·((cx)+(cy))`, bounded by `(|xᵢ−yᵢ|+δ)·4A` with `δ = (Σ|x−y|)/n`
    the mean shift; the `nδ = Σ|x−y|` collapse gives the `8A` constant. -/
theorem bnVar_input_close {n : ℕ} (x y : Vec n) {A : ℝ} (hn : 0 < n)
    (hAx : ∀ i, |x i| ≤ A) (hAy : ∀ i, |y i| ≤ A) :
    |bnVar n x - bnVar n y| ≤ 8 * A * ((∑ i, |x i - y i|) / (n:ℝ)) := by
  have hnR : (0:ℝ) < (n:ℝ) := by exact_mod_cast hn
  have hA0 : 0 ≤ A := (abs_nonneg _).trans (hAx ⟨0, hn⟩)
  set μx := bnMean n x with hμx
  set μy := bnMean n y with hμy
  set δ := (∑ i, |x i - y i|) / (n:ℝ) with hδ
  have hδ0 : 0 ≤ δ := by
    rw [hδ]; exact div_nonneg (Finset.sum_nonneg fun i _ => abs_nonneg _) (le_of_lt hnR)
  have hμxA : |μx| ≤ A := bnMean_abs_le x hn hAx
  have hμyA : |μy| ≤ A := bnMean_abs_le y hn hAy
  have hμd : |μx - μy| ≤ δ := bnMean_input_close x y hn
  -- per-term centered-square difference bound
  have hterm : ∀ i, |(x i - μx) * (x i - μx) - (y i - μy) * (y i - μy)| ≤
      (|x i - y i| + δ) * (4 * A) := by
    intro i
    have hfac : (x i - μx) * (x i - μx) - (y i - μy) * (y i - μy)
        = ((x i - μx) - (y i - μy)) * ((x i - μx) + (y i - μy)) := by ring
    rw [hfac, abs_mul]
    have hd1 : |(x i - μx) - (y i - μy)| ≤ |x i - y i| + δ := by
      have he : (x i - μx) - (y i - μy) = (x i - y i) - (μx - μy) := by ring
      rw [he]; exact (abs_sub_le_add _ _).trans (add_le_add le_rfl hμd)
    have hd2 : |(x i - μx) + (y i - μy)| ≤ 4 * A := by
      have hx1 : |x i - μx| ≤ A + A := (abs_sub_le_add _ _).trans (add_le_add (hAx i) hμxA)
      have hy1 : |y i - μy| ≤ A + A := (abs_sub_le_add _ _).trans (add_le_add (hAy i) hμyA)
      calc |(x i - μx) + (y i - μy)| ≤ |x i - μx| + |y i - μy| := abs_add_le _ _
        _ ≤ (A + A) + (A + A) := add_le_add hx1 hy1
        _ = 4 * A := by ring
    exact mul_le_mul hd1 hd2 (abs_nonneg _) (by linarith [hd1, abs_nonneg (x i - y i)])
  -- sum + divide
  have hsumterm : ∑ i, |(x i - μx) * (x i - μx) - (y i - μy) * (y i - μy)| ≤
      8 * A * (∑ i, |x i - y i|) := by
    calc ∑ i, |(x i - μx) * (x i - μx) - (y i - μy) * (y i - μy)|
        ≤ ∑ i, (|x i - y i| + δ) * (4 * A) := Finset.sum_le_sum fun i _ => hterm i
      _ = (∑ i, |x i - y i|) * (4 * A) + ((n:ℝ) * δ) * (4 * A) := by
          rw [← Finset.sum_mul, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
            Fintype.card_fin, nsmul_eq_mul]; ring
      _ = 8 * A * (∑ i, |x i - y i|) := by
          rw [hδ]; field_simp; ring
  calc |bnVar n x - bnVar n y|
      = |(∑ i, (x i - μx) * (x i - μx) - ∑ i, (y i - μy) * (y i - μy)) / (n:ℝ)| := by
        simp only [bnVar, ← hμx, ← hμy]; rw [div_sub_div_same]
    _ = |∑ i, ((x i - μx) * (x i - μx) - (y i - μy) * (y i - μy))| / (n:ℝ) := by
        rw [abs_div, abs_of_pos hnR, Finset.sum_sub_distrib]
    _ ≤ (8 * A * (∑ i, |x i - y i|)) / (n:ℝ) := by
        gcongr; exact (Finset.abs_sum_le_sum_abs _ _).trans hsumterm
    _ = 8 * A * ((∑ i, |x i - y i|) / (n:ℝ)) := by ring

/-- **Inverse-stddev input-sensitivity.** `|istd(x) − istd(y)| ≤
    8A·(Σ|xᵢ−yᵢ|)/n / (2ε√ε)` — the variance shift pushed through
    `rsqrt_lipschitz` (both `σ²+ε ≥ ε`). The BN-input piece that, with the
    rounding `bnIstd_close`, the per-block composition needs. -/
theorem bnIstd_input_close {n : ℕ} (x y : Vec n) {A ε : ℝ} (hn : 0 < n) (hε : 0 < ε)
    (hAx : ∀ i, |x i| ≤ A) (hAy : ∀ i, |y i| ≤ A) :
    |bnIstd n x ε - bnIstd n y ε| ≤
      (8 * A * ((∑ i, |x i - y i|) / (n:ℝ))) / (2 * ε * Real.sqrt ε) := by
  have hbεx : ε ≤ bnVar n x + ε := by have := bnVar_nonneg n x; linarith
  have hbεy : ε ≤ bnVar n y + ε := by have := bnVar_nonneg n y; linarith
  unfold bnIstd
  calc |1 / Real.sqrt (bnVar n x + ε) - 1 / Real.sqrt (bnVar n y + ε)|
      ≤ |(bnVar n x + ε) - (bnVar n y + ε)| / (2 * ε * Real.sqrt ε) :=
        rsqrt_lipschitz hε hbεx hbεy
    _ = |bnVar n x - bnVar n y| / (2 * ε * Real.sqrt ε) := by ring_nf
    _ ≤ (8 * A * ((∑ i, |x i - y i|) / (n:ℝ))) / (2 * ε * Real.sqrt ε) := by
        gcongr
        exact bnVar_input_close x y hn hAx hAy

/-- `|istd| ≤ 1/√ε` (the `ε`-floor caps the inverse-stddev). -/
theorem bnIstd_abs_le {n : ℕ} (x : Vec n) {ε : ℝ} (hε : 0 < ε) :
    |bnIstd n x ε| ≤ 1 / Real.sqrt ε := by
  have hbε : ε ≤ bnVar n x + ε := by have := bnVar_nonneg n x; linarith
  have hsε : 0 < Real.sqrt ε := Real.sqrt_pos.mpr hε
  have hpos : 0 < bnIstd n x ε := by
    unfold bnIstd
    exact div_pos one_pos (Real.sqrt_pos.mpr (by linarith))
  rw [abs_of_pos hpos]
  unfold bnIstd
  exact one_div_le_one_div_of_le hsε (Real.sqrt_le_sqrt hbε)

/-- **BN forward input-sensitivity (the assembled real-BN input-Lipschitz).**
    `|bnForward x − bnForward y|` per coordinate, under `|xᵢ|,|yᵢ| ≤ A`: the `β`
    cancels and `γ·x̂` splits as `(centered shift)·istd + centered·(istd shift)`,
    bounded via the mean shift `δ = (Σ|x−y|)/n`, `|istd| ≤ 1/√ε`, and
    `bnIstd_input_close`. The normalize stage of the real-BN input-Lipschitz;
    with the rounding `bnForward_close_of`, the per-block float composition is
    `|bnForwardF(float) − bnForward(real)| ≤ rounding + this`. -/
theorem bnForward_input_close {n : ℕ} (x y : Vec n) {A ε γ β : ℝ} (hn : 0 < n) (hε : 0 < ε)
    (hAx : ∀ i, |x i| ≤ A) (hAy : ∀ i, |y i| ≤ A) (i : Fin n) :
    |bnForward n ε γ β x i - bnForward n ε γ β y i| ≤
      |γ| * ((|x i - y i| + (∑ j, |x j - y j|) / (n:ℝ)) * (1 / Real.sqrt ε)
             + 2 * A * ((8 * A * ((∑ j, |x j - y j|) / (n:ℝ))) / (2 * ε * Real.sqrt ε))) := by
  set μx := bnMean n x with hμx
  set μy := bnMean n y with hμy
  set sx := bnIstd n x ε with hsx
  set sy := bnIstd n y ε with hsy
  set δ := (∑ j, |x j - y j|) / (n:ℝ) with hδ
  have hsxA : |sx| ≤ 1 / Real.sqrt ε := bnIstd_abs_le x hε
  have hμd : |μx - μy| ≤ δ := bnMean_input_close x y hn
  have hμyA : |μy| ≤ A := bnMean_abs_le y hn hAy
  have hsd : |sx - sy| ≤ (8 * A * δ) / (2 * ε * Real.sqrt ε) := bnIstd_input_close x y hn hε hAx hAy
  -- bnForward difference reduces to γ·(cx·sx − cy·sy); β cancels
  have hbf : bnForward n ε γ β x i - bnForward n ε γ β y i
      = γ * ((x i - μx) * sx - (y i - μy) * sy) := by
    simp only [bnForward, bnXhat, ← hμx, ← hμy, ← hsx, ← hsy]; ring
  rw [hbf, abs_mul]
  -- split cx·sx − cy·sy = (cx−cy)·sx + cy·(sx−sy)
  have hsplit : (x i - μx) * sx - (y i - μy) * sy
      = ((x i - μx) - (y i - μy)) * sx + (y i - μy) * (sx - sy) := by ring
  have hcd : |(x i - μx) - (y i - μy)| ≤ |x i - y i| + δ := by
    have he : (x i - μx) - (y i - μy) = (x i - y i) - (μx - μy) := by ring
    rw [he]; exact (abs_sub_le_add _ _).trans (add_le_add le_rfl hμd)
  have hcyA : |y i - μy| ≤ 2 * A := by
    have h := (abs_sub_le_add (y i) μy).trans (add_le_add (hAy i) hμyA); linarith
  have hcore : |(x i - μx) * sx - (y i - μy) * sy| ≤
      (|x i - y i| + δ) * (1 / Real.sqrt ε) + 2 * A * ((8 * A * δ) / (2 * ε * Real.sqrt ε)) := by
    rw [hsplit]
    refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
    · rw [abs_mul]
      exact mul_le_mul hcd hsxA (abs_nonneg _) (by positivity)
    · rw [abs_mul]
      exact mul_le_mul hcyA hsd (abs_nonneg _) (by linarith [hcyA, abs_nonneg (y i - μy)])
  calc |γ| * |(x i - μx) * sx - (y i - μy) * sy|
      ≤ |γ| * ((|x i - y i| + δ) * (1 / Real.sqrt ε)
              + 2 * A * ((8 * A * δ) / (2 * ε * Real.sqrt ε))) :=
        mul_le_mul_of_nonneg_left hcore (abs_nonneg _)

end Proofs
