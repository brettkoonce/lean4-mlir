import Mathlib.Data.Int.Log
import Mathlib.Algebra.Order.Archimedean.Real.Basic

/-! # `rndP` — round-to-nearest on the unbounded-exponent `p`-bit grid

The rounding operator behind the named float models (`binary32`, `fp8E4M3` in
`Binary32Instance.lean`) and the bf16 sharding lemmas (`DataParallel.SyncBf16`), with the
standard model `|rndP p x − x| ≤ 2⁻¹⁻ᵖ·|x|` proved, and its commuting with scaling by `2^z`
(`rndP_zpow_mul`, which the bf16 divisor step uses). A leaf on Mathlib alone, so a consumer that
needs only the operator does not import `FloatBridge`.
-/

namespace Proofs

/-- **Round-to-nearest on the unbounded-exponent `p`-bit-significand grid.**
    For `x ≠ 0` with binade exponent `e = Int.log 2 |x|` (i.e. `2^e ≤ |x| < 2^(e+1)`),
    round `x` to the nearest multiple of `2^(e−p)` — a significand of `p` fractional
    bits, every exponent available. Ties go toward +∞ (Mathlib's `round`); IEEE's
    ties-to-even differs only at ties, where the error bound is the same. Overflow and
    subnormals are idealized away, as in the standard model. -/
noncomputable def rndP (p : ℕ) (x : ℝ) : ℝ :=
  if x = 0 then 0 else
    (round (x / (2 : ℝ) ^ (Int.log 2 |x| - (p : ℤ))) : ℝ) *
      (2 : ℝ) ^ (Int.log 2 |x| - (p : ℤ))

@[simp] theorem rndP_zero (p : ℕ) : rndP p 0 = 0 := by simp [rndP]

/-- **The standard model, PROVED** (formerly the `ieeeRnd_err` axiom):
    `|rndP p x − x| ≤ 2⁻¹⁻ᵖ·|x|`. The grid spacing at `x` is `2^(e−p)`, nearest-rounding
    contributes half a step `2^(e−p−1)`, and `2^e ≤ |x|` turns that into the relative
    bound. Mathlib-only: `Int.zpow_log_le_self` + `abs_sub_round`. -/
theorem rndP_err (p : ℕ) (x : ℝ) :
    |rndP p x - x| ≤ ((2 : ℝ) ^ (p + 1))⁻¹ * |x| := by
  rcases eq_or_ne x 0 with hx | hx
  · simp [hx]
  · have hax : (0 : ℝ) < |x| := abs_pos.mpr hx
    unfold rndP
    rw [ite_eq_right hx]
    set e : ℤ := Int.log 2 |x| with he
    set s : ℝ := (2 : ℝ) ^ (e - (p : ℤ)) with hs
    have hs0 : (0 : ℝ) < s := zpow_pos (by norm_num) _
    have hkey : (round (x / s) : ℝ) * s - x = ((round (x / s) : ℝ) - x / s) * s := by
      rw [sub_mul, div_mul_cancel₀ x (ne_of_gt hs0)]
    rw [hkey, abs_mul, abs_of_pos hs0]
    have h1 : |(round (x / s) : ℝ) - x / s| ≤ 1 / 2 := by
      rw [abs_sub_comm]
      exact abs_sub_round (x / s)
    have h2 : (2 : ℝ) ^ e ≤ |x| := by
      exact_mod_cast Int.zpow_log_le_self (by norm_num) hax
    have hs_eq : (1 / 2 : ℝ) * s = ((2 : ℝ) ^ (p + 1))⁻¹ * (2 : ℝ) ^ e := by
      rw [hs, show (1 / 2 : ℝ) = (2 : ℝ) ^ (-1 : ℤ) by norm_num,
          ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0),
          ← zpow_natCast (2 : ℝ) (p + 1), ← zpow_neg,
          ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      congr 1
      push_cast
      ring
    calc |(round (x / s) : ℝ) - x / s| * s
        ≤ (1 / 2) * s := mul_le_mul_of_nonneg_right h1 (le_of_lt hs0)
      _ = ((2 : ℝ) ^ (p + 1))⁻¹ * (2 : ℝ) ^ e := hs_eq
      _ ≤ ((2 : ℝ) ^ (p + 1))⁻¹ * |x| := by
          exact mul_le_mul_of_nonneg_left h2 (by positivity)

-- ════════════════════════════════════════════════════════════════
-- § Scaling by a power of two
-- ════════════════════════════════════════════════════════════════

/-- **`Int.log b` shifts by `z` under scaling by `b^z`**, for any base `b > 1` and any integer
    exponent. -/
theorem int_log_zpow_mul {b : ℕ} (hb : 1 < b) (z : ℤ) {r : ℝ} (hr : 0 < r) :
    Int.log b ((b : ℝ) ^ z * r) = z + Int.log b r := by
  have hb0 : (0 : ℝ) < b := by exact_mod_cast zero_lt_one.trans hb
  have hbz : (0 : ℝ) < (b : ℝ) ^ z := zpow_pos hb0 z
  have hzr : 0 < (b : ℝ) ^ z * r := mul_pos hbz hr
  have hpow : ∀ y : ℤ, (b : ℝ) ^ (z + y) = (b : ℝ) ^ z * (b : ℝ) ^ y :=
    fun y => zpow_add₀ hb0.ne' z y
  apply le_antisymm
  · -- log (b^z r) < z + log r + 1, since b^z·r < b^(z + log r + 1)
    have hlt : (b : ℝ) ^ z * r < (b : ℝ) ^ (z + (Int.log b r + 1)) := by
      rw [hpow]
      exact mul_lt_mul_of_pos_left (Int.lt_zpow_succ_log_self hb r) hbz
    have := (Int.lt_zpow_iff_log_lt hb hzr).mp hlt
    omega
  · -- b^(z + log r) ≤ b^z·r
    apply (Int.zpow_le_iff_le_log hb hzr).mp
    rw [hpow]
    exact mul_le_mul_of_nonneg_left (Int.zpow_log_le_self hb hr) hbz.le

/-- **`Int.log` reads `|x|`, so the shift holds for either sign.** -/
theorem int_log_abs_zpow_mul {b : ℕ} (hb : 1 < b) (z : ℤ) {x : ℝ} (hx : x ≠ 0) :
    Int.log b |(b : ℝ) ^ z * x| = z + Int.log b |x| := by
  have hb0 : (0 : ℝ) < b := by exact_mod_cast zero_lt_one.trans hb
  rw [abs_mul, abs_of_pos (zpow_pos hb0 z)]
  exact int_log_zpow_mul hb z (abs_pos.mpr hx)

/-- **The repo's rounding model commutes with scaling by a power of two** — the grid at
    `2^z·x` is the grid at `x` scaled by `2^z`, because the exponent is unbounded. The
    exponent `z` is an integer, so this covers the multiplier `R = 2^k` and the divisor
    `1/R = 2^(-k)` alike. -/
theorem rndP_zpow_mul (p : ℕ) (z : ℤ) (x : ℝ) :
    rndP p ((2 : ℝ) ^ z * x) = (2 : ℝ) ^ z * rndP p x := by
  rcases eq_or_ne x 0 with hx | hx
  · simp [hx]
  have h2z : (2 : ℝ) ^ z ≠ 0 := zpow_ne_zero z two_ne_zero
  have hzx : (2 : ℝ) ^ z * x ≠ 0 := mul_ne_zero h2z hx
  have hlog : Int.log 2 |(2 : ℝ) ^ z * x| = z + Int.log 2 |x| := by
    have := int_log_abs_zpow_mul (b := 2) (by norm_num) z hx
    simpa using this
  unfold rndP
  rw [ite_eq_right hzx, ite_eq_right hx, hlog]
  have hs : (2 : ℝ) ^ (z + Int.log 2 |x| - (p : ℤ))
      = (2 : ℝ) ^ z * (2 : ℝ) ^ (Int.log 2 |x| - (p : ℤ)) := by
    rw [show z + Int.log 2 |x| - (p : ℤ) = z + (Int.log 2 |x| - (p : ℤ)) by ring,
        zpow_add₀ two_ne_zero]
  rw [hs, mul_div_mul_left _ _ h2z]
  ring

end Proofs
