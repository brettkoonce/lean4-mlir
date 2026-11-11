module

public import LeanMlir.Proofs.Foundation.ListDot
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # `getD` bridges for certificate data in `List ℤ`

`Foundation/ListDot` turns one kernel-evaluated integer dot product into a `Fin n` sum of
`getD`-indexed reals. The trained-weight witnesses on the 28×28 image (`Trained/LinearDescent`,
`Trained/MlpDescent`) need four more bridges of the same kind, stated once here:

* `getD_of_forall` — a property of every entry (and of the default) holds of every `getD`, so a
  range fact checked over the list by `decide +kernel` gives the per-coordinate bound;
* `sum_getD` — the `Fin n` sum of the entries is the list sum;
* `sum_getD_abs_mul`, `sum_abs_div_mul` — the absolute-value dot product, in the scaled form the
  float dense-layer budget `FloatModel.denseErr` asks for.
-/

namespace Proofs

open scoped BigOperators

/-- A property of every entry and of the default holds of every `getD`. -/
theorem getD_of_forall (xs : List ℤ) (P : ℤ → Prop) (h0 : P 0) (h : ∀ z ∈ xs, P z) (i : ℕ) :
    P (xs.getD i 0) := by
  rw [List.getD_eq_getElem?_getD]
  cases hget : xs[i]? with
  | none => simpa using h0
  | some z => simpa using h z (List.mem_of_getElem? hget)

/-- `getD`-sum over `Fin n` = list sum. -/
theorem sum_getD : ∀ (n : ℕ) (xs : List ℤ), xs.length = n →
    (∑ j : Fin n, (xs.getD j 0 : ℝ)) = ((xs.sum : ℤ) : ℝ)
  | 0, xs, hx => by rw [List.length_eq_zero_iff.mp hx]; simp
  | n + 1, [], hx => by simp at hx
  | n + 1, a :: xs, hx => by
      rw [Fin.sum_univ_succ]
      simp only [List.getD_cons_zero, List.getD_cons_succ, Fin.val_succ, Fin.val_zero]
      rw [sum_getD n xs (by simpa using hx)]
      simp only [List.sum_cons]
      push_cast
      ring

/-- `getD`-dot of absolute values over `Fin n` = the list dot of the absolute-value lists. -/
theorem sum_getD_abs_mul : ∀ (n : ℕ) (ws xs : List ℤ), ws.length = n → xs.length = n →
    (∑ j : Fin n, |(ws.getD j 0 : ℝ)| * |(xs.getD j 0 : ℝ)|)
      = ((dotZ (ws.map fun z => |z|) (xs.map fun z => |z|) : ℤ) : ℝ)
  | 0, ws, xs, hw, hx => by
      rw [List.length_eq_zero_iff.mp hw, List.length_eq_zero_iff.mp hx]
      simp [dotZ]
  | n + 1, [], _, hw, _ => by simp at hw
  | n + 1, _ :: _, [], _, hx => by simp at hx
  | n + 1, a :: ws, b :: xs, hw, hx => by
      rw [Fin.sum_univ_succ]
      simp only [List.getD_cons_zero, List.getD_cons_succ, Fin.val_succ, Fin.val_zero]
      rw [sum_getD_abs_mul n ws xs (by simpa using hw) (by simpa using hx)]
      simp only [dotZ, List.map_cons, List.zipWith_cons_cons, List.sum_cons]
      push_cast [Int.cast_abs]
      ring

/-- The scaled form `denseErr` asks for: `Σ |w/c|·(|x/d| + 0)` from one kernel fact. -/
theorem sum_abs_div_mul {ws xs : List ℤ} {n : ℕ} (hw : ws.length = n) (hx : xs.length = n)
    {v : ℤ} (h : dotZ (ws.map fun z => |z|) (xs.map fun z => |z|) = v) {c d : ℝ}
    (hc : 0 < c) (hd : 0 < d) :
    (∑ j : Fin n, |(ws.getD j 0 : ℝ)/c| * (|(xs.getD j 0 : ℝ)/d| + 0)) = (v : ℝ)/(c*d) := by
  rw [← h, ← sum_getD_abs_mul n ws xs hw hx, Finset.sum_div]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [add_zero, abs_div, abs_div, abs_of_pos hc, abs_of_pos hd]
  field_simp

end Proofs
