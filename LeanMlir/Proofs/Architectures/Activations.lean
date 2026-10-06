import LeanMlir.Proofs.Foundation.Tensor
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.SpecialFunctions.Sigmoid

/-!
# Smooth activations: GELU, Swish, sigmoid

The smooth elementwise activations the nets use, each with its scalar derivative, the closed
form the emitted backward computes, and its VJP:

- GELU (tanh approximation): `gelu`, `geluScalarDeriv_eq`, `geluHasVJP`; with it the `Real.tanh`
  derivative facts `differentiable_tanh` and `hasDerivAt_tanh`.
- Swish (SiLU): `swish`, `swishScalarDeriv_eq`, `hasDerivAt_swishScalar`, `swishHasVJP`.
- Sigmoid: `sigmoid`, `sigmoidScalarDeriv_eq`, `sigmoidHasVJP`.

GELU is `gelu(x) = x · Phi(x)`, where `Phi` is the CDF of the standard normal. `gelu` here is the
tanh approximation `0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`, the function `jax.nn.gelu`
computes by default; the exact form, PyTorch's `nn.GELU`, is `geluErf` in `GeluErf`. The two differ
by up to 4.7e-4.

Each Jacobian is diagonal (`pdiv_elementwise`), so each VJP is one line. ReLU and ReLU6, which
have kinks, live in `Foundation.MLP`.

## References

- Hendrycks & Gimpel 2016, *Gaussian Error Linear Units (GELUs)* (incl. tanh approximation). <https://arxiv.org/abs/1606.08415>
- Ramachandran, Zoph, Le 2017, *Searching for Activation Functions* (Swish). <https://arxiv.org/abs/1710.05941>
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § GELU
-- ════════════════════════════════════════════════════════════════

/-- **GELU forward** — Gaussian Error Linear Unit, tanh approximation.

    `gelu(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`

    What the `geluF` op emits; the exact `x · Φ(x)` is `geluErfScalar`, and `geluErfF`'s
    emit. -/
noncomputable def geluScalar (x : ℝ) : ℝ :=
  0.5 * x * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x^3)))

/-- The elementwise GELU, applied componentwise to a vector. -/
noncomputable def gelu (n : Nat) (x : Vec n) : Vec n :=
  fun i => geluScalar (x i)

/-- **Scalar derivative of `geluScalar`** — defined as Mathlib's `deriv`.

    `geluScalar` is the tanh approximation, so this is the derivative of
    that approximation; its closed form is `geluScalarDeriv_eq`. We define
    it via `deriv` rather than writing the closed form so the connection
    to `geluScalar` is automatic. -/
noncomputable def geluScalarDeriv (x : ℝ) : ℝ :=
  deriv geluScalar x

/-- **`Real.tanh` is differentiable everywhere** — bridge via
    `Real.tanh_eq_sinh_div_cosh` and `Real.cosh_pos`. Tagged for
    `fun_prop` so downstream gelu-style smoothness goals dispatch.
    (Mathlib has neither this nor `hasDerivAt_tanh` below for `Real.tanh`.) -/
@[fun_prop]
theorem differentiable_tanh : Differentiable ℝ Real.tanh := by
  have h_eq : Real.tanh = (fun x : ℝ => Real.sinh x / Real.cosh x) :=
    funext Real.tanh_eq_sinh_div_cosh
  rw [h_eq]
  intro x
  exact (Real.differentiable_sinh.differentiableAt).div
          Real.differentiable_cosh.differentiableAt
          (Real.cosh_pos x).ne'

/-- **Derivative of `Real.tanh`** — `tanh'(y) = 1 − tanh²(y)`, built from
    `tanh = sinh/cosh` via the quotient rule and `cosh² − sinh² = 1`, for the
    GELU closed-form derivative `geluScalarDeriv_eq`. -/
theorem hasDerivAt_tanh (y : ℝ) : HasDerivAt Real.tanh (1 - Real.tanh y ^ 2) y := by
  have h : Real.tanh = fun z => Real.sinh z / Real.cosh z := funext Real.tanh_eq_sinh_div_cosh
  rw [h]
  have hd := (Real.hasDerivAt_sinh y).div (Real.hasDerivAt_cosh y) (Real.cosh_pos y).ne'
  convert hd using 1
  simp only [div_pow]; field_simp

/-- **Closed form of `geluScalarDeriv`** — the analytic derivative of the
    tanh-approximation GELU. With `u = √(2/π)·(x + 0.044715·x³)` and `t = tanh u`,

    `gelu'(x) = 0.5·(1 + t) + 0.5·x·(1 − t²)·√(2/π)·(1 + 3·0.044715·x²)`.

    This is exactly the closed form the verified `geluBack` StableHLO emitter
    renders — so the emitted backward text is certified equal to `deriv geluScalar`
    (`swishScalarDeriv_eq` does the same for swish).
    Proof: assemble `HasDerivAt` for the polynomial inner, `tanh` via
    `hasDerivAt_tanh`, and the outer product, then `HasDerivAt.deriv`. -/
theorem geluScalarDeriv_eq (x : ℝ) :
    geluScalarDeriv x =
      0.5 * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x^3)))
      + 0.5 * x * ((1 - Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x^3))^2)
          * (Real.sqrt (2 / Real.pi) * (1 + 0.044715 * (3 * x^2)))) := by
  unfold geluScalarDeriv geluScalar
  have hpoly : HasDerivAt (fun z : ℝ => z + 0.044715 * z^3) (1 + 0.044715 * (3 * x^2)) x := by
    have h1 : HasDerivAt (fun z : ℝ => z) 1 x := hasDerivAt_id x
    have h2 : HasDerivAt (fun z : ℝ => 0.044715 * z^3) (0.044715 * (3 * x^2)) x :=
      (hasDerivAt_pow 3 x).const_mul 0.044715
    exact h1.add h2
  have hu : HasDerivAt (fun z : ℝ => Real.sqrt (2 / Real.pi) * (z + 0.044715 * z^3))
              (Real.sqrt (2 / Real.pi) * (1 + 0.044715 * (3 * x^2))) x :=
    hpoly.const_mul _
  have ht := (hasDerivAt_tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x^3))).comp x hu
  have h1pt := ht.const_add 1
  have hhalfx : HasDerivAt (fun z : ℝ => 0.5 * z) 0.5 x := by
    simpa using (hasDerivAt_id x).const_mul (0.5 : ℝ)
  have hg : HasDerivAt
      (fun z : ℝ => 0.5 * z * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (z + 0.044715 * z^3))))
      (0.5 * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x^3)))
        + 0.5 * x * ((1 - Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x^3))^2)
            * (Real.sqrt (2 / Real.pi) * (1 + 0.044715 * (3 * x^2))))) x :=
    hhalfx.mul h1pt
  rw [hg.deriv]

/-- Differentiability of `geluScalar` as a scalar function. -/
@[fun_prop]
lemma geluScalar_differentiable : Differentiable ℝ geluScalar := by
  unfold geluScalar; fun_prop

/-- Differentiability of `gelu D` as a function on `Vec D`. -/
lemma gelu_differentiable (D : Nat) : Differentiable ℝ (gelu D) := by
  unfold gelu; fun_prop

/-- **Partial derivative of GELU.**

    `gelu n` has diagonal Jacobian: each output coord depends only on
    the corresponding input coord via `geluScalar`. So
    `∂(gelu n y)_j / ∂y_i = (geluScalar' (y i))` if `i = j`, else `0` —
    `pdiv_elementwise` at `geluScalar`. -/
theorem pdiv_gelu (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (gelu n) x i j =
    if i = j then geluScalarDeriv (x i) else 0 :=
  pdiv_elementwise geluScalar x (fun _ => geluScalar_differentiable _) i j

/-- **GELU VJP**: elementwise multiply by the scalar derivative.

    `back(x, dy)_i = dy_i * geluScalarDeriv(x_i)`

    Same template as ReLU (`reluHasVJP`), Swish, h-swish. If your
    activation has a diagonal Jacobian, this is the only proof you
    need — "collapse the diagonal sum." -/
noncomputable def geluHasVJP (n : Nat) : HasVJP (gelu n) where
  backward := fun x dy i => dy i * geluScalarDeriv (x i)
  correct := by
    intro x dy i
    simp [pdiv_gelu, mul_comm]

/-- **Public correctness theorem for `geluHasVJP`**: the GELU
backward (diagonal scaling by `geluScalarDeriv`) equals the
`pdiv`-contracted Jacobian. -/
theorem geluHasVJP_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (geluHasVJP n).backward x dy i =
    ∑ j : Fin n, pdiv (gelu n) x i j * dy j :=
  (geluHasVJP n).correct x dy i

/-! ## The activation taxonomy is closed

Every activation function in every architecture in this repo is
elementwise -> diagonal Jacobian -> one-line VJP. Taking inventory:

| Activation | `pdiv_*` formula (at `j = i`)                       |
|------------|------------------------------------------------------|
| ReLU       | `1` if `x_i > 0`, else `0`                           |
| ReLU6      | `1` if `0 < x_i < 6`, else `0`                       |
| Swish      | `sigma(x_i) * (1 + x_i * (1 - sigma(x_i)))`          |
| h-swish    | piecewise: `0` / `(2x_i + 3)/6` / `1`                |
| h-sigmoid  | piecewise: `0` / `1/6` / `0`                         |
| GELU (tanh approx.) | `geluScalarDeriv_eq`                        |
| tanh       | `1 - tanh^2(x_i)`                                     |
| sigmoid    | `sigma(x_i) * (1 - sigma(x_i))`                       |

They all have the same proof shape (`pdiv_elementwise`, then collapse
the diagonal sum). This file instantiates it as `geluHasVJP`, `swishHasVJP` and
`sigmoidHasVJP`; ReLU's and ReLU6's (`reluHasVJP`, `relu6HasVJPAt`) are in `Foundation.MLP`,
and the other rows are not given
separate `HasVJP` instances.
-/

/-! ## Swish (a.k.a. SiLU)

`swish(x) = x * σ(x)`, where `σ(x) = 1 / (1 + exp(-x))` is the standard
logistic sigmoid. Used as the default activation in EfficientNet's
`MBConv` blocks. Same diagonal-Jacobian proof template as ReLU and GELU.
-/

/-- **Swish forward** — Sigmoid-Linear Unit (SiLU).

    `swish(x) = x / (1 + exp(-x)) = x · σ(x)`. Smooth everywhere
    (denominator is bounded below by 1 > 0). -/
noncomputable def swishScalar (x : ℝ) : ℝ :=
  x / (1 + Real.exp (-x))

/-- The elementwise Swish, applied componentwise to a vector. -/
noncomputable def swish (n : Nat) (x : Vec n) : Vec n :=
  fun i => swishScalar (x i)

/-- **Scalar derivative of `swishScalar`** — defined via Mathlib's
    `deriv`. The closed form is `σ(x)·(1 + x·(1 - σ(x)))` (`swishScalarDeriv_eq`);
    we define it as `deriv swishScalar` so the link to `swishScalar` is automatic. -/
noncomputable def swishScalarDeriv (x : ℝ) : ℝ :=
  deriv swishScalar x

/-- `swishScalar x = x · σ(x)` with Mathlib's logistic function `Real.sigmoid`. -/
theorem swishScalar_eq_mul_sigmoid : swishScalar = fun x => x * Real.sigmoid x := by
  funext x; simp [swishScalar, Real.sigmoid, div_eq_mul_inv]

/-- **Closed form of `swishScalarDeriv`**: `σ(x)·(1 + x·(1 − σ(x)))`, from
    `Real.hasDerivAt_sigmoid` — the formula the `swishBack` StableHLO emitter renders. -/
theorem swishScalarDeriv_eq (x : ℝ) :
    swishScalarDeriv x = Real.sigmoid x * (1 + x * (1 - Real.sigmoid x)) := by
  unfold swishScalarDeriv
  rw [swishScalar_eq_mul_sigmoid]
  refine ((hasDerivAt_id' x).mul (Real.hasDerivAt_sigmoid x)).deriv.trans ?_
  ring

/-- Differentiability of `swishScalar`. The denominator `1 + exp(-x)` is
    always positive, so the quotient is smooth everywhere. -/
@[fun_prop]
lemma swishScalar_differentiable : Differentiable ℝ swishScalar := by
  rw [swishScalar_eq_mul_sigmoid]; fun_prop

theorem hasDerivAt_swishScalar (x : ℝ) : HasDerivAt swishScalar (swishScalarDeriv x) x :=
  (swishScalar_differentiable x).hasDerivAt

/-- Differentiability of `swish D` as a function on `Vec D`. -/
lemma swish_differentiable (D : Nat) : Differentiable ℝ (swish D) := by
  unfold swish; fun_prop

@[fun_prop]
lemma swish_continuous (D : Nat) : Continuous (swish D) := (swish_differentiable D).continuous

/-- **Partial derivative of Swish** — diagonal Jacobian: each output coord
    depends only on the corresponding input coord via `swishScalar`
    (`pdiv_elementwise`). -/
theorem pdiv_swish (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (swish n) x i j =
    if i = j then swishScalarDeriv (x i) else 0 :=
  pdiv_elementwise swishScalar x (fun _ => swishScalar_differentiable _) i j

/-- **Swish VJP**: elementwise multiply by the scalar derivative.
    Same template as ReLU/GELU. The codegen emits the closed-form
    `σ(x)·(1 + x·(1 - σ(x)))` directly; `swishScalarDeriv_eq` equates it
    with `swishScalarDeriv = deriv swishScalar`. -/
noncomputable def swishHasVJP (n : Nat) : HasVJP (swish n) where
  backward := fun x dy i => dy i * swishScalarDeriv (x i)
  correct := by
    intro x dy i
    simp [pdiv_swish, mul_comm]

-- ════════════════════════════════════════════════════════════════
-- § Sigmoid activation (smooth, logistic)
-- ════════════════════════════════════════════════════════════════

/-- The logistic function `1 / (1 + e^{−x})` on one real. -/
noncomputable def sigmoidScalar (x : ℝ) : ℝ :=
  1 / (1 + Real.exp (-x))

/-- `sigmoidScalar` applied to each entry of a vector. -/
noncomputable def sigmoid (n : Nat) (x : Vec n) : Vec n :=
  fun i => sigmoidScalar (x i)

/-- The derivative of `sigmoidScalar` at `x`, as Mathlib's `deriv`. -/
noncomputable def sigmoidScalarDeriv (x : ℝ) : ℝ :=
  deriv sigmoidScalar x

/-- `sigmoidScalar` is Mathlib's logistic function `Real.sigmoid`. -/
theorem sigmoidScalar_eq_sigmoid : sigmoidScalar = Real.sigmoid := by
  funext x; simp [sigmoidScalar, Real.sigmoid]

/-- The closed form σ' = σ·(1 − σ). `sigmoidHasVJP`'s backward is stated with `deriv`; the
    emitted `sigmoidBack` text computes `σ(x)·(1 − σ(x))` — this is the equation between them,
    so it stays pinned in the axiom audit although no Lean proof consumes it. -/
theorem sigmoidScalarDeriv_eq (x : ℝ) :
    sigmoidScalarDeriv x = sigmoidScalar x * (1 - sigmoidScalar x) := by
  simp [sigmoidScalarDeriv, sigmoidScalar_eq_sigmoid, Real.deriv_sigmoid]

@[fun_prop]
lemma sigmoidScalar_differentiable : Differentiable ℝ sigmoidScalar := by
  rw [sigmoidScalar_eq_sigmoid]; exact differentiable_sigmoid

lemma sigmoid_differentiable (D : Nat) : Differentiable ℝ (sigmoid D) := by
  unfold sigmoid; fun_prop

theorem pdiv_sigmoid (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (sigmoid n) x i j =
    if i = j then sigmoidScalarDeriv (x i) else 0 :=
  pdiv_elementwise sigmoidScalar x (fun _ => sigmoidScalar_differentiable _) i j

/-- The VJP of elementwise `sigmoid`: `dy ⊙ σ'(x)`. -/
noncomputable def sigmoidHasVJP (n : Nat) : HasVJP (sigmoid n) where
  backward := fun x dy i => dy i * sigmoidScalarDeriv (x i)
  correct := by
    intro x dy i
    simp [pdiv_sigmoid, mul_comm]

end Proofs
