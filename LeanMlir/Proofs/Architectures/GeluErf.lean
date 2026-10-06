import LeanMlir.Proofs.Foundation.Tensor
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-!
# The exact GELU: `x · Φ(x)`

GELU as Hendrycks and Gimpel define it, `gelu(x) = x · Φ(x)` with `Φ` the standard normal CDF.
This is the function PyTorch's `nn.GELU` and `jax.nn.gelu(approximate=False)` compute;
`Proofs.gelu` in `Activations` is its tanh approximation.

- `gaussPdf`, `gaussPhi`: the standard normal density `φ` and CDF `Φ(x) = ½ + ∫₀ˣ φ`, with
  `hasDerivAt_gaussPhi : Φ' = φ` by the fundamental theorem of calculus.
- `geluErf`, `geluErfScalarDeriv_eq`, `geluErfHasVJP`: the activation, its derivative
  `Φ(x) + x · φ(x)`, and its VJP.
- `erf`, `erfc`, `gaussPhi_eq_erf`, `gaussPhi_eq_erfc`: the error function and the two spellings of
  `Φ` through it, `½ (1 + erf(x/√2))` and `½ erfc(−x·√½)`.
- `geluErfScalar_eq_erfc`, `geluErfScalarDeriv_eq_erfc`: the forward and the derivative in the
  `erfc` spelling, the form in which `jax.nn.gelu(approximate=False)` and its `jax.vjp` compute
  them. The `erfc` spelling keeps the negative tail that `1 + erf` cancels away in floats.

Mathlib has no error function, so `erf` is defined here as `(2/√π) ∫₀ᶻ exp(−t²) dt`. The density
and the CDF are closed forms over the interval integral; `GeluErfGaussian` proves them equal to
Mathlib's `gaussianPDFReal 0 1` and to the `cdf` of `gaussianReal 0 1`.

## References

- Hendrycks & Gimpel 2016, *Gaussian Error Linear Units (GELUs)*. <https://arxiv.org/abs/1606.08415>
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The standard normal density and CDF
-- ════════════════════════════════════════════════════════════════

/-- **The standard normal density** `φ(x) = exp(−x²/2) / √(2π)`. -/
noncomputable def gaussPdf (x : ℝ) : ℝ :=
  Real.exp (-x ^ 2 / 2) / Real.sqrt (2 * Real.pi)

/-- `gaussPdf` is continuous. -/
@[fun_prop]
theorem gaussPdf_continuous : Continuous gaussPdf := by
  unfold gaussPdf; fun_prop

/-- **The standard normal CDF** `Φ(x) = ½ + ∫₀ˣ φ(t) dt`.

    The density is even and integrates to one, so the mass below zero is `½`
    (`integral_Iic_zero_gaussPdf`); the interval integral carries the rest, and makes `Φ' = φ`
    the fundamental theorem of calculus (`hasDerivAt_gaussPhi`). -/
noncomputable def gaussPhi (x : ℝ) : ℝ :=
  1 / 2 + ∫ t in (0 : ℝ)..x, gaussPdf t

/-- **`Φ' = φ`** — the fundamental theorem of calculus at the continuous density. -/
theorem hasDerivAt_gaussPhi (x : ℝ) : HasDerivAt gaussPhi (gaussPdf x) x :=
  (gaussPdf_continuous.integral_hasStrictDerivAt 0 x).hasDerivAt.const_add (1 / 2)

/-- `gaussPhi` is differentiable. Tagged for `fun_prop` so smoothness goals over the exact GELU
    dispatch. -/
@[fun_prop]
theorem gaussPhi_differentiable : Differentiable ℝ gaussPhi :=
  fun x => (hasDerivAt_gaussPhi x).differentiableAt

-- ════════════════════════════════════════════════════════════════
-- § GELU, exact
-- ════════════════════════════════════════════════════════════════

/-- **Exact GELU forward** — `gelu(x) = x · Φ(x)`. -/
noncomputable def geluErfScalar (x : ℝ) : ℝ :=
  x * gaussPhi x

/-- The elementwise exact GELU, applied componentwise to a vector. -/
noncomputable def geluErf (n : Nat) (x : Vec n) : Vec n :=
  fun i => geluErfScalar (x i)

/-- **Scalar derivative of `geluErfScalar`** — defined as Mathlib's `deriv`, as
    `geluScalarDeriv` is; its closed form is `geluErfScalarDeriv_eq`. -/
noncomputable def geluErfScalarDeriv (x : ℝ) : ℝ :=
  deriv geluErfScalar x

/-- The product rule on `x · Φ(x)`. -/
theorem hasDerivAt_geluErfScalar (x : ℝ) :
    HasDerivAt geluErfScalar (gaussPhi x + x * gaussPdf x) x := by
  have h : HasDerivAt (fun y : ℝ => y * gaussPhi y) (1 * gaussPhi x + x * gaussPdf x) x :=
    HasDerivAt.mul (hasDerivAt_id' x) (hasDerivAt_gaussPhi x)
  rw [one_mul] at h
  exact h

/-- **Closed form of `geluErfScalarDeriv`** — `gelu'(x) = Φ(x) + x · φ(x)`. -/
theorem geluErfScalarDeriv_eq (x : ℝ) :
    geluErfScalarDeriv x = gaussPhi x + x * gaussPdf x :=
  (hasDerivAt_geluErfScalar x).deriv

/-- Differentiability of `geluErfScalar` as a scalar function. -/
@[fun_prop]
lemma geluErfScalar_differentiable : Differentiable ℝ geluErfScalar :=
  fun x => (hasDerivAt_geluErfScalar x).differentiableAt

/-- Differentiability of `geluErf D` as a function on `Vec D`. -/
lemma geluErf_differentiable (D : Nat) : Differentiable ℝ (geluErf D) := by
  unfold geluErf; fun_prop

/-- **Partial derivative of the exact GELU** — diagonal, `pdiv_elementwise` at
    `geluErfScalar`. -/
theorem pdiv_geluErf (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (geluErf n) x i j =
    if i = j then geluErfScalarDeriv (x i) else 0 :=
  pdiv_elementwise geluErfScalar x (fun _ => geluErfScalar_differentiable _) i j

/-- **Exact GELU VJP**: elementwise multiply by the scalar derivative,

    `back(x, dy)_i = dy_i * geluErfScalarDeriv(x_i)`. -/
noncomputable def geluErfHasVJP (n : Nat) : HasVJP (geluErf n) where
  backward := fun x dy i => dy i * geluErfScalarDeriv (x i)
  correct := by
    intro x dy i
    simp [pdiv_geluErf, mul_comm]

/-- **Public correctness theorem for `geluErfHasVJP`**: the exact GELU backward (diagonal
scaling by `geluErfScalarDeriv`) equals the `pdiv`-contracted Jacobian. -/
theorem geluErfHasVJP_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (geluErfHasVJP n).backward x dy i =
    ∑ j : Fin n, pdiv (geluErf n) x i j * dy j :=
  (geluErfHasVJP n).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § The error function, and Φ through it
-- ════════════════════════════════════════════════════════════════

/-- **The error function** `erf(z) = (2/√π) ∫₀ᶻ exp(−t²) dt`. -/
noncomputable def erf (z : ℝ) : ℝ :=
  2 / Real.sqrt Real.pi * ∫ t in (0 : ℝ)..z, Real.exp (-t ^ 2)

/-- **The complementary error function** `erfc(z) = 1 − erf(z)`. -/
noncomputable def erfc (z : ℝ) : ℝ :=
  1 - erf z

/-- `erf` is odd: the integrand is even. -/
theorem erf_neg (z : ℝ) : erf (-z) = -erf z := by
  have h : (∫ t in (0 : ℝ)..(-z), Real.exp (-t ^ 2)) = -∫ t in (0 : ℝ)..z, Real.exp (-t ^ 2) := by
    have hneg := intervalIntegral.integral_comp_neg (a := 0) (b := z)
      (fun t : ℝ => Real.exp (-t ^ 2))
    simp only [neg_sq, neg_zero] at hneg
    rw [intervalIntegral.integral_symm, ← hneg]
  unfold erf
  rw [h]; ring

/-- **`Φ` through `erf`** — `Φ(x) = ½ (1 + erf(x/√2))`, by the substitution `t = √2 · s`. -/
theorem gaussPhi_eq_erf (x : ℝ) : gaussPhi x = 1 / 2 * (1 + erf (x / Real.sqrt 2)) := by
  have h2 : Real.sqrt 2 ≠ 0 := by positivity
  have hpi : Real.sqrt Real.pi ≠ 0 := by positivity
  have hfun : (fun t : ℝ => Real.exp (-(t / Real.sqrt 2) ^ 2)) = fun t => Real.exp (-t ^ 2 / 2) := by
    funext t
    rw [div_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2), neg_div]
  have hsub : (∫ t in (0 : ℝ)..x, Real.exp (-t ^ 2 / 2))
      = Real.sqrt 2 * ∫ s in (0 : ℝ)..(x / Real.sqrt 2), Real.exp (-s ^ 2) := by
    have h := intervalIntegral.integral_comp_div (a := 0) (b := x)
      (fun s : ℝ => Real.exp (-s ^ 2)) h2
    rw [zero_div, smul_eq_mul] at h
    rw [← h, hfun]
  unfold gaussPhi gaussPdf erf
  rw [intervalIntegral.integral_div, hsub, Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2)]
  field_simp

/-- **`Φ` through `erfc`** — `Φ(x) = ½ erfc(−x·√½)`. Equal to the `erf` spelling as real numbers
    (`erf` is odd); in floats this one keeps the negative tail, where `1 + erf` cancels. -/
theorem gaussPhi_eq_erfc (x : ℝ) : gaussPhi x = 0.5 * erfc (-x * Real.sqrt (1 / 2)) := by
  have hs : -x * Real.sqrt (1 / 2) = -(x / Real.sqrt 2) := by
    rw [one_div, Real.sqrt_inv]; ring
  rw [gaussPhi_eq_erf, erfc, hs, erf_neg]
  norm_num

/-- **The exact GELU forward as computed** — `gelu(x) = (0.5 · x) · erfc(−x · √½)`, the
    arithmetic of `jax.nn.gelu(approximate=False)`. -/
theorem geluErfScalar_eq_erfc (x : ℝ) :
    geluErfScalar x = 0.5 * x * erfc (-x * Real.sqrt (1 / 2)) := by
  rw [geluErfScalar, gaussPhi_eq_erfc]; ring

/-- **The exact GELU derivative as computed** — with `z = −x · √½`,

    `gelu'(x) = 0.5 · erfc(z) + (2/√π) · (0.5 · x) · exp(−z²) · √½`,

    the two terms `jax.vjp` of `jax.nn.gelu(approximate=False)` forms: `Φ(x)` through `erfc`, and
    `x · φ(x)` with the density written as the derivative of `erfc` at `z`. -/
theorem geluErfScalarDeriv_eq_erfc (x : ℝ) :
    geluErfScalarDeriv x =
      0.5 * erfc (-x * Real.sqrt (1 / 2))
      + 2 / Real.sqrt Real.pi * (0.5 * x) * Real.exp (-(-x * Real.sqrt (1 / 2)) ^ 2)
          * Real.sqrt (1 / 2) := by
  have h2 : Real.sqrt 2 ≠ 0 := by positivity
  have hpi : Real.sqrt Real.pi ≠ 0 := by positivity
  have hz : -(-x * Real.sqrt (1 / 2)) ^ 2 = -x ^ 2 / 2 := by
    rw [mul_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 1 / 2)]; ring
  rw [geluErfScalarDeriv_eq, gaussPhi_eq_erfc, gaussPdf, hz,
    Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2), one_div, Real.sqrt_inv]
  field_simp
  ring

end Proofs
