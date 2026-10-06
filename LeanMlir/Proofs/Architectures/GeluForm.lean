import LeanMlir.Proofs.Architectures.Activations
import LeanMlir.Proofs.Architectures.GeluErf

/-!
# Either GELU

The GELU comes in two forms: the tanh approximation (`geluScalar`, the default of
`jax.nn.gelu`) and the exact `x · Φ(x)` (`geluErfScalar`, PyTorch's `nn.GELU`). `GeluForm` names
the choice. A net that uses the GELU, and every theorem about it, takes a `GeluForm` and holds
for both, so the renders that trained under the tanh form and the ones that train under the
exact form are instances of one statement.

- `GeluForm.scalar`, `GeluForm.scalarDeriv`: the scalar activation and its derivative (a
  `deriv`; the closed forms are `geluScalarDeriv_eq` and `geluErfScalarDeriv_eq`).
- `GeluForm.map`, `GeluForm.pdiv_map`, `GeluForm.hasVJP`: the activation on a vector, its diagonal
  Jacobian, its VJP.
- `GeluForm.map_tanh`, `GeluForm.map_erf`, `GeluForm.hasVJP_tanh`, `GeluForm.hasVJP_erf`: at each
  form these are `gelu` / `geluHasVJP` and `geluErf` / `geluErfHasVJP`, by `rfl`.

Nothing here cases on the form except `scalar` and its differentiability, so a definition built
on `GeluForm.map` unfolds the same way at either form.
-/

open Finset BigOperators

namespace Proofs

/-- **Which GELU**: the tanh approximation or the exact `x · Φ(x)`. -/
inductive GeluForm where
  /-- `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`, `geluScalar`. -/
  | tanh
  /-- `x · Φ(x)`, `geluErfScalar`. -/
  | erf
  deriving DecidableEq, Repr, Inhabited

namespace GeluForm

/-- The scalar activation of a form. -/
noncomputable def scalar : GeluForm → ℝ → ℝ
  | .tanh => geluScalar
  | .erf => geluErfScalar

@[simp] theorem scalar_tanh : scalar .tanh = geluScalar := rfl
@[simp] theorem scalar_erf : scalar .erf = geluErfScalar := rfl

/-- Either form is differentiable. -/
@[fun_prop]
theorem scalar_differentiable (gf : GeluForm) : Differentiable ℝ gf.scalar := by
  cases gf
  · exact geluScalar_differentiable
  · exact geluErfScalar_differentiable

/-- **Scalar derivative of a form** — Mathlib's `deriv`, as `geluScalarDeriv` and
    `geluErfScalarDeriv` are. -/
noncomputable def scalarDeriv (gf : GeluForm) (x : ℝ) : ℝ :=
  deriv gf.scalar x

theorem scalarDeriv_tanh (x : ℝ) : scalarDeriv .tanh x = geluScalarDeriv x := rfl
theorem scalarDeriv_erf (x : ℝ) : scalarDeriv .erf x = geluErfScalarDeriv x := rfl

/-- **The GELU of a form**, applied componentwise to a vector. -/
noncomputable def map (gf : GeluForm) (n : Nat) (x : Vec n) : Vec n :=
  fun i => gf.scalar (x i)

theorem map_tanh (n : Nat) : map .tanh n = gelu n := rfl
theorem map_erf (n : Nat) : map .erf n = geluErf n := rfl

/-- Differentiability of `gf.map D` as a function on `Vec D`. -/
lemma map_differentiable (gf : GeluForm) (D : Nat) : Differentiable ℝ (gf.map D) := by
  unfold map; fun_prop

/-- **Partial derivative of either GELU** — diagonal, `pdiv_elementwise` at `gf.scalar`. -/
theorem pdiv_map (gf : GeluForm) (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (gf.map n) x i j =
    if i = j then gf.scalarDeriv (x i) else 0 :=
  pdiv_elementwise gf.scalar x (fun _ => gf.scalar_differentiable _) i j

/-- **VJP of either GELU**: elementwise multiply by the scalar derivative,

    `back(x, dy)_i = dy_i * gf.scalarDeriv(x_i)`. -/
noncomputable def hasVJP (gf : GeluForm) (n : Nat) : HasVJP (gf.map n) where
  backward := fun x dy i => dy i * gf.scalarDeriv (x i)
  correct := by
    intro x dy i
    simp [pdiv_map, mul_comm]

theorem hasVJP_tanh (n : Nat) : hasVJP .tanh n = geluHasVJP n := rfl
theorem hasVJP_erf (n : Nat) : hasVJP .erf n = geluErfHasVJP n := rfl

/-- **Public correctness theorem for `GeluForm.hasVJP`**: the backward (diagonal scaling by
`gf.scalarDeriv`) equals the `pdiv`-contracted Jacobian. -/
theorem hasVJP_correct (gf : GeluForm) (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (gf.hasVJP n).backward x dy i =
    ∑ j : Fin n, pdiv (gf.map n) x i j * dy j :=
  (gf.hasVJP n).correct x dy i

end GeluForm

end Proofs
