import LeanMlir.Proofs.Architectures.LayerNorm

/-! # Swish's derivative is globally bounded: `|swish′| ≤ 2` — the window-free gain

`swish(x) = x·σ(x)` is what EfficientNet's MBConv blocks activate with, and its BACKWARD
scales the cotangent by the saved `swish′(x)` (`swish_has_vjp`, `LayerNorm.lean`). Until this
file the only bound the repo had on that factor came from the FORWARD's Lipschitz analysis —
`swishScalar_lipschitz_abs`'s `1 + A/4` at the pre-swish window `A` (`EnetFloatBridge.lean`) —
and B0's certified forward window is `2.580·10⁵⁵`, so one swish site charged `1.216·10⁵¹` to
the fold. That is the whole of what stood between EfficientNet-B0's input-gradient budget and
a number: `2.491·10⁴³¹` with the window-dependent bound, `7.640·10¹⁶⁹` with this one
(`b0_back_chain(ssw = 2)`, `scripts/float_budget_envelope.py`), against the ~`10²⁵³`
shape-dependent `norm_num` ceiling.

**The bound is elementary, and the sharpness does not matter.** `swish′ = σ + x·σ′` with

- `σ(x) ≤ 1`, since `1 + e^{−x} ≥ 1`;
- `σ′(x) = e^{−x}/(1+e^{−x})² ≤ e^{−|x|}` (`sigmoidDeriv_le_exp_neg_abs`) — the denominator is
  `≥ 1` for `x ≥ 0` and `≥ e^{−2x}` for `x < 0`, which is the whole case split;
- `|x|·e^{−|x|} ≤ 1` (`abs_mul_exp_neg_abs_le_one`), straight from `Real.add_one_le_exp`.

⚠ **The true global constant is ≈ 1.1** (`σ′ → 0` at both ends), and
`planning/float_budget_numbers.md` §3.4 recorded that getting it "needs the decay of `σ′`, i.e.
calculus" — which is true of the SHARP constant and was the wrong thing to cost. `11/10` puts
B0's backward at `10¹⁶⁷` and the crude `2` at `10¹⁷⁰`; both are statable, and the only property
that matters is that **the window does not appear**. No MVT, no derivative analysis, no sup.
That correction is §3.9's finding: *a cost estimate written down next to a bound is not
evidence, and it is exactly what stops anyone re-checking it.*

⚠ **Why it lives under `Architectures/` and not next to its consumers.** Same reason
`GeluSaturation.lean` does: it is pure real analysis about `swishScalar`, needing only
`LayerNorm.lean`'s `swishScalarDeriv` / `swishScalar_diff`, and its consumers are in the float
tier (`SEBackFloatBridge.lean`'s `Ssw`, `EfficientNetBackFloatBridge.lean`'s `swBe`/`swBd`).

⛔ **`swishScalar_lipschitz` is NOT wired into `floatClose_swish`, deliberately.** A global
`2`-Lipschitz constant is also a FORWARD improvement — `floatClose_swish`'s modulus is the `min`
of a multiplicative `(1 + A/4)·e` and an additive `A + e` (§3.4), and `2·e` beats both at B0's
windows, worth 8 orders on the forward budget (`8.408·10²¹⁰ → 3.679·10²⁰²`,
`b0_eval_chain(swish_lip = 2)`). Wiring it moves a committed number, which is its own commit.
-/

namespace Proofs

/-- **The closed form of `swishScalarDeriv`, split as `σ(x) + x·σ′(x)`** — the shape the bound
    is proved in, because each summand is bounded by `1` for a different reason. -/
theorem swishScalarDeriv_eq (x : ℝ) :
    swishScalarDeriv x =
      1 / (1 + Real.exp (-x)) + x * (Real.exp (-x) / (1 + Real.exp (-x)) ^ 2) := by
  unfold swishScalarDeriv swishScalar
  have hd0 : (0 : ℝ) < 1 + Real.exp (-x) := by positivity
  have hden : HasDerivAt (fun z : ℝ => 1 + Real.exp (-z)) (-Real.exp (-x)) x := by
    have h : HasDerivAt (fun z : ℝ => Real.exp (-z)) (-Real.exp (-x)) x := by
      simpa using (hasDerivAt_neg x).exp
    simpa using h.const_add (1 : ℝ)
  have hq : HasDerivAt (fun z : ℝ => z / (1 + Real.exp (-z)))
      ((1 * (1 + Real.exp (-x)) - x * (-Real.exp (-x))) / (1 + Real.exp (-x)) ^ 2) x :=
    (hasDerivAt_id x).div hden hd0.ne'
  rw [hq.deriv]
  field_simp
  ring

/-- `t·e^{−t} ≤ 1` at `t = |x|`, straight from `x + 1 ≤ exp x`. -/
private lemma abs_mul_exp_neg_abs_le_one (x : ℝ) : |x| * Real.exp (-|x|) ≤ 1 := by
  have h1 : |x| ≤ Real.exp |x| := by
    have := Real.add_one_le_exp |x|
    linarith
  have h2 : Real.exp |x| * Real.exp (-|x|) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  calc |x| * Real.exp (-|x|)
      ≤ Real.exp |x| * Real.exp (-|x|) :=
        mul_le_mul_of_nonneg_right h1 (Real.exp_pos _).le
    _ = 1 := h2

/-- **The logistic derivative decays: `σ′(x) = e^{−x}/(1+e^{−x})² ≤ e^{−|x|}`.** The denominator
    is `≥ 1` for `x ≥ 0` and `≥ (e^{−x})²` for `x < 0`; that two-line case split is the only
    place the saturation enters, and it is what makes `|x|·σ′(x)` bounded. -/
private lemma sigmoidDeriv_le_exp_neg_abs (x : ℝ) :
    Real.exp (-x) / (1 + Real.exp (-x)) ^ 2 ≤ Real.exp (-|x|) := by
  have he : (0:ℝ) < Real.exp (-x) := Real.exp_pos _
  rcases le_or_gt 0 x with hx | hx
  · rw [abs_of_nonneg hx]
    have h1 : (1:ℝ) ≤ (1 + Real.exp (-x)) ^ 2 := by nlinarith
    calc Real.exp (-x) / (1 + Real.exp (-x)) ^ 2
        ≤ Real.exp (-x) / 1 := div_le_div_of_nonneg_left he.le zero_lt_one h1
      _ = Real.exp (-x) := by ring
  · rw [abs_of_neg hx, neg_neg]
    have hsq : Real.exp (-x) ^ 2 ≤ (1 + Real.exp (-x)) ^ 2 := by nlinarith
    have hcancel : Real.exp (-x) / Real.exp (-x) ^ 2 = Real.exp x := by
      rw [div_eq_iff (by positivity), pow_two, ← Real.exp_add, ← Real.exp_add]
      congr 1; ring
    calc Real.exp (-x) / (1 + Real.exp (-x)) ^ 2
        ≤ Real.exp (-x) / Real.exp (-x) ^ 2 :=
          div_le_div_of_nonneg_left he.le (by positivity) hsq
      _ = Real.exp x := hcancel

/-- ⭐ **Global derivative bound for Swish: `|swish′(x)| ≤ 2`, at every `x`.** The bound the
    EfficientNet-B0 BACKWARD's `diagBack` slots take as their `Ssw`, replacing
    `swishScalar_lipschitz_abs`'s `1 + A/4` at the forward's certified window — `10⁴³¹ → 10¹⁶⁹`
    on that fold, and the difference between a number and none. -/
theorem swishScalarDeriv_abs_le (x : ℝ) : |swishScalarDeriv x| ≤ 2 := by
  have he : (0:ℝ) < Real.exp (-x) := Real.exp_pos _
  have hd0 : (0 : ℝ) < 1 + Real.exp (-x) := by positivity
  have hsig : |1 / (1 + Real.exp (-x))| ≤ 1 := by
    rw [abs_of_pos (by positivity), div_le_one hd0]
    linarith
  have hxs : |x * (Real.exp (-x) / (1 + Real.exp (-x)) ^ 2)| ≤ 1 := by
    have hpos : (0:ℝ) < Real.exp (-x) / (1 + Real.exp (-x)) ^ 2 := by positivity
    rw [abs_mul, abs_of_pos hpos]
    calc |x| * (Real.exp (-x) / (1 + Real.exp (-x)) ^ 2)
        ≤ |x| * Real.exp (-|x|) :=
          mul_le_mul_of_nonneg_left (sigmoidDeriv_le_exp_neg_abs x) (abs_nonneg x)
      _ ≤ 1 := abs_mul_exp_neg_abs_le_one x
  rw [swishScalarDeriv_eq]
  calc |1 / (1 + Real.exp (-x)) + x * (Real.exp (-x) / (1 + Real.exp (-x)) ^ 2)|
      ≤ |1 / (1 + Real.exp (-x))| + |x * (Real.exp (-x) / (1 + Real.exp (-x)) ^ 2)| :=
        abs_add_le _ _
    _ ≤ 1 + 1 := add_le_add hsig hxs
    _ = 2 := by norm_num

/-- **Swish is globally `2`-Lipschitz** — the mean-value consequence of
    `swishScalarDeriv_abs_le`, and the peer of `geluScalar_lipschitz`'s `3/2`.
    ⛔ Not consumed by `floatClose_swish`, on purpose: see the file header. -/
theorem swishScalar_lipschitz (x y : ℝ) :
    |swishScalar x - swishScalar y| ≤ 2 * |x - y| := by
  have h := Convex.norm_image_sub_le_of_norm_deriv_le (𝕜 := ℝ) (f := swishScalar)
    (s := Set.univ)
    (fun z _ => (swishScalar_diff z))
    (fun z _ => by
      rw [Real.norm_eq_abs]
      exact swishScalarDeriv_abs_le z)
    convex_univ (Set.mem_univ y) (Set.mem_univ x)
  simpa [Real.norm_eq_abs] using h

end Proofs
