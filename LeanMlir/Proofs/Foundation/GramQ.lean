import Mathlib.Basic.Real.Basic
import Mathlib.Data.Rat.BigOperators
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Algebra.BigOperators.Fin

/-! # Gram-matrix identities by one kernel check

The Lipschitz certificates bound `‖W‖₂` through Gram matrices `G = W Wᵀ` (and
`H = Gᵀ G` for the Schatten-8 step), and each instance has to establish
`∀ a b, G a b = ∑ j, W a j * W b j` for concrete rational data. Over `ℝ`, entry by
entry with `simp`/`norm_num`, that cost up to ~270 s per matrix. Here the data is `ℚ`,
the `ℝ` matrices are DEFINED as its cast (`castM`), and `gram_eq_of_check` turns one
kernel-evaluated `gramCheck` into the whole identity; `abs_le_of_check` does the same
for an entrywise bound `|A i j| ≤ c`.

`H = Gᵀ G` is the same statement with `W := fun a c => G c a`. -/

namespace Proofs

open scoped BigOperators

/-- A `ℚ` matrix read over `ℝ`. -/
def castM {m n : Nat} (A : Fin m → Fin n → ℚ) : Fin m → Fin n → ℝ := fun i j => (A i j : ℝ)

/-- `∑ j, W a j * W b j` over `ℚ`, as a sum the kernel can evaluate. -/
def rowDotQ {m n : Nat} (W : Fin m → Fin n → ℚ) (a b : Fin m) : ℚ :=
  (List.ofFn fun j => W a j * W b j).sum

/-- Every entry of `G` is the row inner product of `W` — the check `decide +kernel` runs. -/
def gramCheck {m n : Nat} (G : Fin m → Fin m → ℚ) (W : Fin m → Fin n → ℚ) : Bool :=
  (List.finRange m).all fun a => (List.finRange m).all fun b => decide (G a b = rowDotQ W a b)

/-- **A passing `gramCheck` is the Gram identity over `ℝ`.** -/
theorem gram_eq_of_check {m n : Nat} (G : Fin m → Fin m → ℚ) (W : Fin m → Fin n → ℚ)
    (h : gramCheck G W = true) :
    ∀ a b, castM G a b = ∑ j, castM W a j * castM W b j := by
  intro a b
  simp only [gramCheck, List.all_eq_true, List.mem_finRange, true_implies,
    decide_eq_true_eq] at h
  simp only [castM, h a b, rowDotQ, List.sum_ofFn, Rat.cast_sum, Rat.cast_mul]

/-- Every entry of `A` is at most `c` in absolute value — the check `decide +kernel` runs. -/
def absLeCheck {m n : Nat} (A : Fin m → Fin n → ℚ) (c : ℚ) : Bool :=
  (List.finRange m).all fun i => (List.finRange n).all fun j => decide (|A i j| ≤ c)

/-- **A passing `absLeCheck` bounds every entry over `ℝ`.** -/
theorem abs_le_of_check {m n : Nat} (A : Fin m → Fin n → ℚ) (c : ℚ)
    (h : absLeCheck A c = true) : ∀ i j, |castM A i j| ≤ (c : ℝ) := by
  intro i j
  simp only [absLeCheck, List.all_eq_true, List.mem_finRange, true_implies,
    decide_eq_true_eq] at h
  simp only [castM]
  rw [← Rat.cast_abs]
  exact Rat.cast_le.mpr (h i j)

end Proofs
