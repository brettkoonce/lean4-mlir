import LeanMlir.Proofs.Foundation.Tensor
import LeanMlir.Proofs.Architectures.BatchNorm
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.FDeriv.Prod
import Mathlib.Analysis.SpecialFunctions.Sigmoid

/-!
# LayerNorm & GELU

Two quick chapters that extend the activation and normalization families
to what ViT needs. Both are structural footnotes to existing chapters,
not new territory — which is itself the point.

## LayerNorm: BatchNorm on a different axis

BatchNorm reduces over `(batch, H, W)` for each channel. LayerNorm
reduces over the **feature** dimension for each `(batch, token)`.
**The 1D normalization primitive is literally the same function.** What
differs is the axis you slice along before applying it.

Concretely, for a 4D activation `x : Tensor4 B C H W`:
- BN computes `C` means/variances, each over `B · H · W` elements.
- LN computes `B · H · W` means/variances, each over `C` elements.

The mean/var/istd/xhat/affine math is identical. The consolidated
three-term backward is identical. Only the index being summed over
changes. In our `Vec n` formalism, BN and LN collapse to the same
function. This file just renames it to tell the reader "yes, really,
it's the same thing."

## GELU: another activation template

Gaussian Error Linear Unit: `gelu(x) = x · Phi(x)` where `Phi` is the CDF
of the standard normal. In practice everyone uses the tanh
approximation `gelu(x) ~ 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`
because it's faster than the exact erf form.

Same template as ReLU/Swish/h-swish: elementwise -> diagonal Jacobian.
Derivative is messier but it's still just a number you compute and
multiply. One more `pdiv_*` theorem, one more `HasVJP` instance.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § LayerNorm
-- ════════════════════════════════════════════════════════════════

/-- **LayerNorm forward** — renamed `bnForward` to make the book's
    claim unambiguous: this is the same function, operating on a
    different slice of the tensor.

    For a single "token's feature vector" `x : Vec n`:
    1. `mu = (1/n) sum_i x_i`                         — mean across features
    2. `sigma^2 = (1/n) sum_i (x_i - mu)^2`            — variance across features
    3. `istd = 1/sqrt(sigma^2 + eps)`
    4. `xhat_i = (x_i - mu) * istd`                    — normalized
    5. `y_i = gamma * xhat_i + beta`                    — affine

    The only semantic difference from BN: in LN, `gamma` and `beta` are
    per-feature (not per-channel), so they're full vectors. For the
    VJP math this doesn't matter — `gamma` and `beta` still just scale and
    shift the normalized output pointwise.

    MLIR (`MlirCodegen.emitLayerNormForward`):
    identical reduction structure to BN, just across a different axis. -/
noncomputable def layerNormForward (n : Nat) (ε : ℝ) (γ β : ℝ)
    (x : Vec n) : Vec n :=
  bnForward n ε γ β x

/-- **LayerNorm input gradient** — identical closed form to BN.

    `dx_i = (1/n) * istd * (n * dxhat_i - sum_j dxhat_j - xhat_i * sum_j xhat_j * dxhat_j)`

    where `dxhat_i = gamma * dy_i`.

    If you built `layerNorm_has_vjp` you'd discover it's `bn_has_vjp`
    with the exact same proof. Rather than restate, we just reuse:
-/
noncomputable def layerNorm_has_vjp (n : Nat) (ε γ β : ℝ) (hε : 0 < ε) :
    HasVJP (layerNormForward n ε γ β) := by
  -- layerNormForward is definitionally bnForward, so the BN VJP works as-is.
  show HasVJP (bnForward n ε γ β)
  exact bn_has_vjp n ε γ β hε

/-! ## Why this isn't a new chapter

The *practical* differences between BN and LN (batch dependence,
inference vs training, running statistics) are engineering concerns,
not VJP concerns. The backward pass is the same three-term formula
either way. This is a general lesson about formal work: engineering
distinctions often dissolve at the math level, and that's worth
making explicit. A reader who assumed BN and LN needed separate
proofs learns that the separation was an implementation artifact.

The same observation applies to:
- **RMSNorm**: LN with mean centering dropped. The closed-form has
  one fewer term (the `-sum_j dxhat_j` part), but the derivation is the
  same machinery.
- **GroupNorm**: LN applied to slices of the channel axis. Again,
  same primitive, different slicing.
- **InstanceNorm** (which is what the ResNet code actually uses):
  BN restricted to per-sample statistics. Literally the 1D primitive
  applied per `(sample, channel)`. Same function.

All four normalization variants share one `HasVJP` instance. The
taxonomy is "1D normalization + your choice of axis."
-/

-- ════════════════════════════════════════════════════════════════
-- § GELU
-- ════════════════════════════════════════════════════════════════

/-- **GELU forward** — Gaussian Error Linear Unit, tanh approximation.

    `gelu(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`

    Matches the MLIR codegen (which emits the tanh approximation rather
    than the exact `x · Φ(x)` erf form). No longer an axiom. -/
noncomputable def geluScalar (x : ℝ) : ℝ :=
  0.5 * x * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x^3)))

/-- The elementwise GELU, applied componentwise to a vector. -/
noncomputable def gelu (n : Nat) (x : Vec n) : Vec n :=
  fun i => geluScalar (x i)

/-- **Scalar derivative of `geluScalar`** — defined as Mathlib's `deriv`.

    Concretely, this is `Φ(x) + x · φ(x)` for the exact form, or the
    analytical derivative of the tanh approximation for our chosen
    `geluScalar`. We define it via `deriv` rather than writing the
    closed form so the connection to `geluScalar` is automatic.
    No longer an axiom. -/
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
lemma geluScalar_diff : Differentiable ℝ geluScalar := by
  unfold geluScalar; fun_prop

/-- Differentiability of `gelu D` as a function on `Vec D`. -/
lemma gelu_diff (D : Nat) : Differentiable ℝ (gelu D) := by
  unfold gelu; fun_prop

/-- **Partial derivative of GELU** — proved (planning/archive/VJP.md follow-up E).

    `gelu n` has diagonal Jacobian: each output coord depends only on
    the corresponding input coord via `geluScalar`. So
    `∂(gelu n y)_j / ∂y_i = (geluScalar' (y i))` if `i = j`, else `0` —
    `pdiv_elementwise` at `geluScalar`. -/
theorem pdiv_gelu (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (gelu n) x i j =
    if i = j then geluScalarDeriv (x i) else 0 :=
  pdiv_elementwise geluScalar x (fun _ => geluScalar_diff _) i j

/-- **GELU VJP**: elementwise multiply by the scalar derivative.

    `back(x, dy)_i = dy_i * geluScalarDeriv(x_i)`

    Same template as ReLU (`relu_has_vjp`), Swish, h-swish. If your
    activation has a diagonal Jacobian, this is the only proof you
    need — "collapse the diagonal sum." -/
noncomputable def gelu_has_vjp (n : Nat) : HasVJP (gelu n) where
  backward := fun x dy i => dy i * geluScalarDeriv (x i)
  correct := by
    intro x dy i
    simp [pdiv_gelu, mul_comm]

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
| GELU       | `Phi(x_i) + x_i * phi(x_i)`                           |
| tanh       | `1 - tanh^2(x_i)`                                     |
| sigmoid    | `sigma(x_i) * (1 - sigma(x_i))`                       |

They all have the same proof shape. Writing each as a separate `HasVJP`
instance is pure boilerplate. For the book, we show the template once
(ReLU, in `MLP.lean`) and assert that GELU follows the same pattern.
-/

/-- **Public correctness theorem for `gelu_has_vjp`**: the GELU
backward (diagonal scaling by `geluScalarDeriv`) equals the
`pdiv`-contracted Jacobian. -/
theorem gelu_has_vjp_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (gelu_has_vjp n).backward x dy i =
    ∑ j : Fin n, pdiv (gelu n) x i j * dy j :=
  (gelu_has_vjp n).correct x dy i

/-- **Public correctness theorem for `layerNorm_has_vjp`**: LayerNorm
reuses the BN proof template (LayerNorm is BN on a different axis), so
the contract is identical — backward equals the `pdiv`-contracted
Jacobian of `layerNormForward`. -/
theorem layerNorm_has_vjp_correct (n : Nat) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (layerNorm_has_vjp n ε γ β hε).backward x dy i =
    ∑ j : Fin n, pdiv (layerNormForward n ε γ β) x i j * dy j :=
  (layerNorm_has_vjp n ε γ β hε).correct x dy i

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
lemma swishScalar_diff : Differentiable ℝ swishScalar := by
  rw [swishScalar_eq_mul_sigmoid]; fun_prop

/-- Differentiability of `swish D` as a function on `Vec D`. -/
lemma swish_diff (D : Nat) : Differentiable ℝ (swish D) := by
  unfold swish; fun_prop

/-- **Partial derivative of Swish** — diagonal Jacobian: each output coord
    depends only on the corresponding input coord via `swishScalar`
    (`pdiv_elementwise`). -/
theorem pdiv_swish (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (swish n) x i j =
    if i = j then swishScalarDeriv (x i) else 0 :=
  pdiv_elementwise swishScalar x (fun _ => swishScalar_diff _) i j

/-- **Swish VJP**: elementwise multiply by the scalar derivative.
    Same template as ReLU/GELU. The codegen emits the closed-form
    `σ(x)·(1 + x·(1 - σ(x)))` directly; `swishScalarDeriv_eq` equates it
    with `swishScalarDeriv = deriv swishScalar`. -/
noncomputable def swish_has_vjp (n : Nat) : HasVJP (swish n) where
  backward := fun x dy i => dy i * swishScalarDeriv (x i)
  correct := by
    intro x dy i
    simp [pdiv_swish, mul_comm]

/-- **Public correctness theorem for `swish_has_vjp`**: diagonal scaling
    by `swishScalarDeriv` equals the `pdiv`-contracted Jacobian of
    `swish n`. -/
theorem swish_has_vjp_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (swish_has_vjp n).backward x dy i =
    ∑ j : Fin n, pdiv (swish n) x i j * dy j :=
  (swish_has_vjp n).correct x dy i

end Proofs
