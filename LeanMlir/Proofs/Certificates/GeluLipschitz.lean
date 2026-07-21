import LeanMlir.Proofs.Float.ViTFloatBridge
import LeanMlir.Proofs.Codegen.AdjointChainBridge

/-! # GELU is globally 3/2-Lipschitz — the saturation-aware gain

`floatClose_gelu`'s input-shift modulus is magnitude-polynomial
(`1 + √(2/π)/2·A·(1+3·0.044715·A²)`), which reaches ~400 at ConvNeXt's real
operating magnitudes (A ≈ 20) and inflates every GELU-bearing block budget by
that factor (probe §7 finding, planning/adjoint_chain.md). The TRUE constant
is ≈ 1.13: past the small-|x| region the `sech²` in `gelu′` decays like
`e^{−2√(2/π)|x|}` and beats the cubic growth. This file proves the clean
global bound `|gelu′| ≤ 3/2`, entirely from:

- `sech²u ≤ 4e^{−2|u|}` (cosh ≥ e^{|u|}/2),
- `|u(x)| ≥ √(2/π)·|x|` (the cubic inner is same-signed),
- the cubic Taylor lower bound on `exp`, and
- two exact facts: `(cs − ½)² ≥ 0` and `π ≤ 4` (which gives
  `6·0.044715 ≤ 4/(3π)`, the cubic-coefficient domination).

Products: `geluScalarDeriv_abs_le` (|gelu′| ≤ 3/2 pointwise),
`geluScalar_lipschitz` (the MVT step), `lipOnWindow_gelu` (the adjoint-chain
gain instance — global, window-free), and `floatClose_gelu_sat` (the
`floatClose_gelu` replacement with flat modulus `egelu + 3/2·e`). All reuse
`geluScalarDeriv_eq` / `geluScalar_diff` from `LayerNorm.lean`.
-/

namespace Proofs


/-- Cubic Taylor lower bound for `exp` on nonnegative arguments. -/
private lemma cubic_le_exp {x : ℝ} (hx : 0 ≤ x) :
    1 + x + x ^ 2 / 2 + x ^ 3 / 6 ≤ Real.exp x := by
  calc 1 + x + x ^ 2 / 2 + x ^ 3 / 6
      = ∑ i ∈ Finset.range 4, x ^ i / i.factorial := by
        simp [Finset.sum_range_succ, Nat.factorial]
    _ ≤ Real.exp x := Real.sum_le_exp_of_nonneg hx 4

/-- Saturation: `1 − tanh²y ≤ 4·e^{−2|y|}` (i.e. `sech²` decays exponentially). -/
private lemma one_sub_tanh_sq_le (y : ℝ) :
    1 - Real.tanh y ^ 2 ≤ 4 * Real.exp (-(2 * |y|)) := by
  have hcosh := Real.cosh_pos y
  have heq : 1 - Real.tanh y ^ 2 = 1 / Real.cosh y ^ 2 := by
    rw [Real.tanh_eq_sinh_div_cosh, div_pow]
    field_simp
    linarith [Real.cosh_sq_sub_sinh_sq y]
  have hge : Real.exp |y| / 2 ≤ Real.cosh y := by
    rw [← Real.cosh_abs, Real.cosh_eq]
    linarith [(Real.exp_pos (-|y|)).le]
  have h2 : (Real.exp |y| / 2) ^ 2 ≤ Real.cosh y ^ 2 :=
    pow_le_pow_left₀ (by positivity) hge 2
  have h3 : 1 / Real.cosh y ^ 2 ≤ 1 / ((Real.exp |y| / 2) ^ 2) :=
    one_div_le_one_div_of_le (by positivity) h2
  have h4 : 1 / ((Real.exp |y| / 2) ^ 2) = 4 * Real.exp (-(2 * |y|)) := by
    rw [div_pow, Real.exp_neg]
    rw [show Real.exp |y| ^ 2 = Real.exp (2 * |y|) by
      rw [pow_two, ← Real.exp_add, two_mul]]
    field_simp
    norm_num
  rw [heq]
  exact h3.trans_eq h4

/-- The polynomial-vs-exponential core: `2cs(1+3as²)·e^{−2cs} ≤ ½` for the
    gelu constants (`c = √(2/π)`, `a = 0.044715`, any `s ≥ 0`) — via the
    cubic Taylor bound, `(cs−½)² ≥ 0`, and `π ≤ 4`. -/
private lemma gelu_tail_le {s : ℝ} (hs : 0 ≤ s) :
    2 * Real.sqrt (2 / Real.pi) * s * (1 + 3 * 0.044715 * s ^ 2)
      * Real.exp (-(2 * (Real.sqrt (2 / Real.pi) * s))) ≤ 1 / 2 := by
  set c := Real.sqrt (2 / Real.pi) with hc
  have hc0 : 0 ≤ c := Real.sqrt_nonneg _
  have hc2 : c ^ 2 = 2 / Real.pi := Real.sq_sqrt (by positivity)
  have hcs0 : 0 ≤ c * s := mul_nonneg hc0 hs
  -- the polynomial side: 2cs(1+3as²) ≤ ½·e^{2cs}
  have hP : 2 * c * s * (1 + 3 * 0.044715 * s ^ 2)
      ≤ 1 / 2 * Real.exp (2 * (c * s)) := by
    have htaylor := cubic_le_exp (x := 2 * (c * s)) (by positivity)
    -- coefficient fact from π ≤ 4:  6a·c ≤ (2/3)c³  (as c·((2/3)c² − 6a) ≥ 0)
    have hcoef : 0 ≤ c * (2 / 3 * c ^ 2 - 6 * 0.044715) := by
      apply mul_nonneg hc0
      rw [hc2, sub_nonneg,
        show (2:ℝ) / 3 * (2 / Real.pi) = 4 / (3 * Real.pi) by
          field_simp; ring,
        le_div_iff₀ (by positivity)]
      nlinarith [Real.pi_le_four, Real.pi_pos]
    nlinarith [htaylor, sq_nonneg (c * s - 1 / 2),
      mul_nonneg hcoef (pow_nonneg hs 3)]
  calc 2 * c * s * (1 + 3 * 0.044715 * s ^ 2) * Real.exp (-(2 * (c * s)))
      ≤ 1 / 2 * Real.exp (2 * (c * s)) * Real.exp (-(2 * (c * s))) :=
        mul_le_mul_of_nonneg_right hP (Real.exp_pos _).le
    _ = 1 / 2 := by
        rw [mul_assoc, ← Real.exp_add, add_neg_cancel, Real.exp_zero, mul_one]

/-- **Global derivative bound for the tanh-form GELU: `|gelu′(x)| ≤ 3/2`.**
    Saturation-aware: past the small-|x| region the `sech²` factor decays like
    `e^{−2√(2/π)|x|}` and beats the cubic polynomial growth. -/
theorem geluScalarDeriv_abs_le (x : ℝ) : |geluScalarDeriv x| ≤ 3 / 2 := by
  rw [geluScalarDeriv_eq]
  set c := Real.sqrt (2 / Real.pi) with hc
  have hc0 : 0 ≤ c := Real.sqrt_nonneg _
  set u := c * (x + 0.044715 * x ^ 3) with hu
  set t := Real.tanh u with htdef
  have ht1 : -1 < t := Real.neg_one_lt_tanh u
  have ht2 : t < 1 := Real.tanh_lt_one u
  have hterm1 : |0.5 * (1 + t)| ≤ 1 := by
    rw [abs_of_nonneg (by linarith)]
    linarith
  have huabs : c * |x| ≤ |u| := by
    rw [hu, abs_mul, abs_of_nonneg hc0]
    have h1 : |x + 0.044715 * x ^ 3| = |x| * (1 + 0.044715 * x ^ 2) := by
      rw [show x + 0.044715 * x ^ 3 = x * (1 + 0.044715 * x ^ 2) by ring, abs_mul,
        abs_of_pos (by positivity : (0:ℝ) < 1 + 0.044715 * x ^ 2)]
    rw [h1]
    nlinarith [mul_nonneg (mul_nonneg hc0 (abs_nonneg x)) (sq_nonneg x)]
  have hsech : 1 - t ^ 2 ≤ 4 * Real.exp (-(2 * (c * |x|))) := by
    refine (one_sub_tanh_sq_le u).trans ?_
    have hmono : Real.exp (-(2 * |u|)) ≤ Real.exp (-(2 * (c * |x|))) :=
      Real.exp_le_exp.mpr (by linarith)
    linarith
  have hterm2 : |0.5 * x * ((1 - t ^ 2)
      * (c * (1 + 0.044715 * (3 * x ^ 2))))| ≤ 1 / 2 := by
    have h1t2 : (0:ℝ) ≤ 1 - t ^ 2 := by nlinarith
    have hK : (0:ℝ) < 1 + 0.044715 * (3 * x ^ 2) := by positivity
    have hB : (0:ℝ) ≤ (1 - t ^ 2) * (c * (1 + 0.044715 * (3 * x ^ 2))) :=
      mul_nonneg h1t2 (mul_nonneg hc0 hK.le)
    rw [abs_mul, abs_of_nonneg hB, abs_mul,
      show |(0.5:ℝ)| = 0.5 by norm_num]
    have hchain : 0.5 * |x| * ((1 - t ^ 2) * (c * (1 + 0.044715 * (3 * x ^ 2))))
        ≤ 0.5 * |x| * ((4 * Real.exp (-(2 * (c * |x|))))
            * (c * (1 + 0.044715 * (3 * x ^ 2)))) := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      exact mul_le_mul_of_nonneg_right hsech (mul_nonneg hc0 hK.le)
    refine hchain.trans ?_
    calc 0.5 * |x| * ((4 * Real.exp (-(2 * (c * |x|))))
            * (c * (1 + 0.044715 * (3 * x ^ 2))))
        = 2 * c * |x| * (1 + 3 * 0.044715 * |x| ^ 2)
            * Real.exp (-(2 * (c * |x|))) := by
          rw [sq_abs]; ring
      _ ≤ 1 / 2 := gelu_tail_le (abs_nonneg x)
  calc |0.5 * (1 + t) + 0.5 * x * ((1 - t ^ 2)
          * (c * (1 + 0.044715 * (3 * x ^ 2))))|
      ≤ |0.5 * (1 + t)| + |0.5 * x * ((1 - t ^ 2)
          * (c * (1 + 0.044715 * (3 * x ^ 2))))| := abs_add_le _ _
    _ ≤ 1 + 1 / 2 := add_le_add hterm1 hterm2
    _ = 3 / 2 := by norm_num

/-- **GELU is globally `3/2`-Lipschitz** — the saturation-aware constant that
    replaces `floatClose_gelu`'s magnitude-polynomial modulus (which reaches
    ~400 at ConvNeXt's operating magnitudes; the true constant is ≈1.13). -/
theorem geluScalar_lipschitz (x y : ℝ) :
    |geluScalar x - geluScalar y| ≤ 3 / 2 * |x - y| := by
  have h := Convex.norm_image_sub_le_of_norm_deriv_le (𝕜 := ℝ) (f := geluScalar)
    (s := Set.univ)
    (fun z _ => (geluScalar_diff z))
    (fun z _ => by
      rw [Real.norm_eq_abs]
      exact geluScalarDeriv_abs_le z)
    convex_univ (Set.mem_univ y) (Set.mem_univ x)
  simpa [Real.norm_eq_abs] using h

/-- **GELU has windowed gain `3/2`** — the adjoint-chain instance (in fact the
    gain is global: no window needed). -/
theorem lipOnWindow_gelu {n : Nat} (A : ℝ) :
    LipOnWindow A (3 / 2) (gelu n) := by
  intro u v e he _hu _hv hd i
  calc |gelu n u i - gelu n v i|
      = |geluScalar (u i) - geluScalar (v i)| := rfl
    _ ≤ 3 / 2 * |u i - v i| := geluScalar_lipschitz _ _
    _ ≤ 3 / 2 * e := by
        have := hd i
        linarith

/-- **`floatClose_gelu`, saturation-aware**: same rounding budget `egelu`, but
    the input-shift modulus is the flat `3/2·e` instead of the magnitude
    polynomial `(1 + √(2/π)/2·A·(1+3·0.044715·A²))·e` — at ConvNeXt's
    operating magnitudes (A ≈ 20) this tightens every GELU budget by ~×250. -/
theorem floatClose_gelu_sat {n : Nat} (fgelu : ℝ → ℝ) {egelu A : ℝ}
    (hegelu : 0 ≤ egelu) (hA : 0 ≤ A)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu) :
    FloatClose A (A + egelu) (gelu n) (fun v i => fgelu (v i))
      (fun e => egelu + 3 / 2 * e) := by
  refine ⟨(floatClose_gelu fgelu hegelu hA hg).1, fun vt va e _hva _hvt hd i => ?_⟩
  calc |fgelu (vt i) - gelu n va i|
      = |fgelu (vt i) - geluScalar (va i)| := rfl
    _ ≤ |fgelu (vt i) - geluScalar (vt i)|
        + |geluScalar (vt i) - geluScalar (va i)| :=
        abs_sub_le (fgelu (vt i)) (geluScalar (vt i)) (geluScalar (va i))
    _ ≤ egelu + 3 / 2 * e := by
        refine add_le_add (hg _) ((geluScalar_lipschitz _ _).trans ?_)
        have := hd i
        linarith

end Proofs
