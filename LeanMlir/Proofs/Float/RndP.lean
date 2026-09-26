import Mathlib.Data.Int.Log
import Mathlib.Algebra.Order.Archimedean.Real.Basic

/-! # `rndP` — round-to-nearest on the unbounded-exponent `p`-bit grid

The rounding operator behind the named float models (`binary32`, `fp8E4M3` in
`Binary32Instance.lean`) and the bf16 sharding lemmas (`DataParallel.SyncBf16`), with the
standard model `|rndP p x − x| ≤ 2⁻¹⁻ᵖ·|x|` proved. A leaf on Mathlib alone, so a consumer that
needs only the operator does not import `FloatBridge`.
-/

namespace Proofs

/-- **Round-to-nearest on the unbounded-exponent `p`-bit-significand grid.**
    For `x ≠ 0` with binade exponent `e = Int.log 2 |x|` (i.e. `2^e ≤ |x| < 2^(e+1)`),
    round `x` to the nearest multiple of `2^(e−p)` — a significand of `p` fractional
    bits, every exponent available. This is IEEE round-to-nearest minus overflow and
    subnormals, the standard-model idealization. -/
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

end Proofs
